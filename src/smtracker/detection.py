"""Detection-threshold optimizer, choose the threshold from data, not by eye.

The problem
-----------
Every single-molecule dataset starts with a detection threshold (u-track's
``alphaLocMax``, TrackMate's quality cut). Set it too low and background noise
enters as spurious detections, which then link into short spurious tracks and
bias the *slow* population upward: noise does not move. Set it too high and
genuine dim molecules are lost, biasing toward the bright and often the slow.
It is normally chosen by eye, which makes every downstream number rest on an
unrecorded judgement.

The criterion
-------------
Cells occupy a small part of the field. **Detections outside the cell outlines
are background**, and they measure the false-positive rate directly:

    background density  = N_outside / (area_outside · n_frames)
    expected false      = background density · area_inside · n_frames
    FDR(threshold)      = expected false inside / N_inside
    true positives      = N_inside − expected false inside

Sweeping the detection score then gives an FDR and a yield curve, and the
threshold can be chosen against a stated FDR target instead of an impression.
Nothing needs re-detecting: u-track records an amplitude (and its uncertainty)
per detection, and TrackMate records a quality, so the sweep is post-hoc.

Assumptions, stated because they are load-bearing
-------------------------------------------------
- Outside-cell detections are *representative* of the background inside cells.
  Cell interiors are usually brighter (autofluorescence, out-of-focus signal),
  so the estimate is a **lower bound** on the true FDR, the honest direction.
- Every cell in the field must be segmented. An unsegmented cell puts real
  molecules in the "outside" pool, inflating the apparent background and the
  FDR. ``threshold_sweep`` therefore reports the segmented area fraction so an
  implausible one is visible.
- The score must be monotone in "how likely this is real". Amplitude and
  quality both are.

PURITY vs COMPLETENESS: read this before acting on a recommendation
-------------------------------------------------------------------
The sweep can only see detections that the *upstream* detector already kept.
u-track's ``alphaLocMax`` cut has been applied before ``movieInfo`` is written,
so this measures **purity** (how clean the retained set is) and cannot measure
**completeness** (how many real molecules the upstream cut already discarded).

Consequence: if the recommendation lands on the lowest available score, the
answer is "everything you kept is clean", *not* "this is the optimum", the
optimum may lie below what you can still see. ``recommend_threshold`` detects
that case and says so.

The workflow that does find the optimum: re-run detection **deliberately
permissively** (a high ``alphaLocMax`` / low quality cut), then sweep here. Only
then does the curve span both sides of the trade-off.

This does not replace looking at the images. It replaces *guessing*, and it
records the choice with the number that justified it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cellmesh import CellMesh, points_in_polygon

__all__ = [
    "DetectionError",
    "ThresholdChoice",
    "DETECTION_COLUMNS",
    "read_utrack_detections",
    "detections_from_trackmate",
    "polygon_area",
    "sweep_for_scope",
    "threshold_sweep",
    "recommend_threshold",
]

DETECTION_COLUMNS = ["movie", "frame", "x", "y", "amp", "amp_std", "snr"]


class DetectionError(RuntimeError):
    """Raised when detections cannot be read or a threshold cannot be chosen."""


@dataclass
class ThresholdChoice:
    """A recommended detection threshold and the evidence behind it."""

    threshold: float
    score: str
    fdr: float
    target_fdr: float
    n_inside: int
    n_outside: int
    true_positives: float
    fraction_of_detections_kept: float
    segmented_area_fraction: float
    note: str = ""

    def summary(self) -> str:
        return (
            f"threshold {self.score} >= {self.threshold:.4g}: FDR ≈ "
            f"{self.fdr * 100:.1f} % (target {self.target_fdr * 100:.0f} %), "
            f"{self.n_inside} in-cell detections, ≈{self.true_positives:.0f} real, "
            f"keeping {self.fraction_of_detections_kept * 100:.0f} % of all "
            f"detections. {self.note}".strip()
        )


def read_utrack_detections(path: str | Path, *, movie: str | None = None) -> pd.DataFrame:
    """Read u-track's ``movieInfo`` detections (pre-linking) into a tidy table.

    ``movieInfo`` is one struct per frame with ``xCoord``/``yCoord``/``amp``,
    each an ``(N, 2)`` array of ``[value, uncertainty]``. Coordinates are in
    **pixels**, matching the raw stage and the cell outlines.
    """
    from scipy.io import loadmat

    path = Path(path)
    if not path.exists():
        raise DetectionError(f"detection file not found: {path}")
    movie = movie if movie is not None else path.parts[-4] if len(path.parts) >= 4 else path.stem

    mat = loadmat(path, struct_as_record=False, squeeze_me=True)
    if "movieInfo" not in mat:
        raise DetectionError(
            f"{path}: no 'movieInfo' variable "
            f"(found {[k for k in mat if not str(k).startswith('__')]})"
        )

    rows = []
    for frame_index, frame_info in enumerate(np.atleast_1d(mat["movieInfo"]), start=1):
        x = np.atleast_2d(np.asarray(getattr(frame_info, "xCoord", []), dtype=float))
        y = np.atleast_2d(np.asarray(getattr(frame_info, "yCoord", []), dtype=float))
        amp = np.atleast_2d(np.asarray(getattr(frame_info, "amp", []), dtype=float))
        if x.size == 0 or x.shape[-1] < 1:
            continue
        n = x.shape[0]
        amp_value = amp[:, 0] if amp.shape[0] == n and amp.shape[1] >= 1 else np.full(n, np.nan)
        amp_std = amp[:, 1] if amp.shape[0] == n and amp.shape[1] >= 2 else np.full(n, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(amp_std > 0, amp_value / amp_std, np.nan)
        for i in range(n):
            rows.append((movie, frame_index, x[i, 0], y[i, 0],
                         amp_value[i], amp_std[i], snr[i]))

    if not rows:
        raise DetectionError(f"{path}: movieInfo holds no detections")
    table = pd.DataFrame.from_records(rows, columns=DETECTION_COLUMNS)
    table.attrs["source"] = "utrack"
    table.attrs["n_frames"] = int(table["frame"].max())
    return table


def detections_from_trackmate(
    path: str | Path, *, movie: str | None = None, pixel_size_nm: float | None = None
) -> pd.DataFrame:
    """All TrackMate spots (before track filtering) as a detection table."""
    from .trackmate import TrackMateError, _parse, _pixels_per_unit, _read_spots
    from .trackmate import read_trackmate_metadata

    path = Path(path)
    movie = movie if movie is not None else path.stem
    try:
        root = _parse(path)
        model = root.find("Model")
        if model is None:
            raise TrackMateError(f"{path}: no <Model> element")
        metadata = read_trackmate_metadata(path)
        scale = _pixels_per_unit(metadata, pixel_size_nm, path)
        spots = _read_spots(model, path)
    except TrackMateError as exc:
        raise DetectionError(str(exc)) from exc

    # FIVE values a spot since the extra TrackMate features were read (`trackmate._read_spots`); this unpacked four
    # and raised, so the detection-threshold sweep could not run (found by the paper handoff, 12 Oct 2026)
    rows = [
        (movie, frame + 1, x * scale, y * scale, amp, np.nan, np.nan)
        for frame, x, y, amp, *_extras in spots.values()
    ]
    table = pd.DataFrame.from_records(rows, columns=DETECTION_COLUMNS)
    table.attrs["source"] = "trackmate"
    table.attrs["n_frames"] = int(table["frame"].max())
    return table


def polygon_area(polygon: np.ndarray) -> float:
    """Area of a simple polygon (shoelace), in the polygon's own units²."""
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise DetectionError(f"polygon must be (N>=3, 2), got {poly.shape}")
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def threshold_sweep(
    detections: pd.DataFrame,
    meshes: list[CellMesh],
    *,
    image_shape: tuple[int, int],
    score: str = "amp",
    thresholds=None,
    n_thresholds: int = 40,
) -> pd.DataFrame:
    """Sweep the detection score and estimate FDR and yield at each threshold.

    Parameters
    ----------
    detections
        Table from ``read_utrack_detections`` / ``detections_from_trackmate``.
    meshes
        Cell outlines for this movie (pixels), from ``smtracker.cellmesh``.
    image_shape
        ``(height, width)`` of the field in pixels: needed for the outside
        area. Available from ``utrack_meta`` as ``image_size_px``.
    score
        Column to threshold on (``"amp"``, ``"snr"``, …). Must be monotone in
        "how likely this detection is real".

    Returns
    -------
    pandas.DataFrame
        One row per threshold with ``n_inside``, ``n_outside``,
        ``background_density``, ``expected_false_inside``, ``fdr``,
        ``true_positives`` and ``fraction_kept``.
    """
    for column in ("x", "y", score):
        if column not in detections.columns:
            raise DetectionError(f"detections have no '{column}' column")
    if not meshes:
        raise DetectionError(
            "no cell outlines: the FDR estimate needs an in-cell/out-of-cell "
            "split, so an unsegmented movie cannot be optimized."
        )
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise DetectionError(f"image_shape must be positive, got {image_shape}")

    values = detections[score].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise DetectionError(f"no finite '{score}' values to threshold on")

    x = detections["x"].to_numpy(dtype=float)
    y = detections["y"].to_numpy(dtype=float)
    inside = np.zeros(len(detections), dtype=bool)
    for mesh in meshes:
        inside |= points_in_polygon(x, y, mesh.polygon)

    area_inside = float(sum(polygon_area(mesh.polygon) for mesh in meshes))
    area_total = float(height) * float(width)
    area_outside = area_total - area_inside
    if area_outside <= 0:
        raise DetectionError(
            f"segmented area ({area_inside:.0f} px²) is not smaller than the "
            f"field ({area_total:.0f} px²); the background cannot be estimated."
        )
    area_fraction = area_inside / area_total

    if thresholds is None:
        candidates = values[finite]
        thresholds = np.quantile(candidates, np.linspace(0.0, 0.98, n_thresholds))
        thresholds = np.unique(thresholds)

    rows = []
    for threshold in thresholds:
        keep = finite & (values >= threshold)
        n_inside = int(np.count_nonzero(keep & inside))
        n_outside = int(np.count_nonzero(keep & ~inside))
        background_density = n_outside / area_outside          # per px², all frames
        expected_false = background_density * area_inside
        fdr = expected_false / n_inside if n_inside > 0 else np.nan
        rows.append(
            {
                "threshold": float(threshold),
                "n_inside": n_inside,
                "n_outside": n_outside,
                "background_density_per_px2": background_density,
                "expected_false_inside": expected_false,
                "fdr": fdr,
                "true_positives": max(n_inside - expected_false, 0.0),
                "fraction_kept": float(np.count_nonzero(keep) / len(detections)),
            }
        )

    out = pd.DataFrame(rows)
    out.attrs["score"] = score
    out.attrs["area_inside_px2"] = area_inside
    out.attrs["area_total_px2"] = area_total
    out.attrs["segmented_area_fraction"] = area_fraction
    out.attrs["n_cells"] = len(meshes)
    if area_fraction > 0.5:
        out.attrs["warning"] = (
            f"cells cover {area_fraction * 100:.0f} % of the field, so the "
            "outside region is small and the background estimate is noisy."
        )
    return out


def recommend_threshold(
    sweep: pd.DataFrame, *, target_fdr: float = 0.05
) -> ThresholdChoice:
    """Lowest threshold meeting the FDR target: i.e. the most data kept safely.

    Choosing the *lowest* qualifying threshold maximises yield: raising it
    further only discards real molecules, and preferentially the dim ones.
    Raises if no threshold reaches the target, rather than returning the least
    bad option as though it passed.
    """
    if not 0 < target_fdr < 1:
        raise DetectionError(f"target_fdr must be in (0, 1), got {target_fdr}")
    required = {"threshold", "fdr", "n_inside", "true_positives"}
    if not required <= set(sweep.columns):
        raise DetectionError(f"not a threshold sweep; needs {sorted(required)}")

    usable = sweep[sweep["fdr"].notna() & (sweep["n_inside"] > 0)]
    qualifying = usable[usable["fdr"] <= target_fdr]
    if qualifying.empty:
        best = usable["fdr"].min() if not usable.empty else float("nan")
        raise DetectionError(
            f"no threshold reaches an FDR of {target_fdr:.0%}; the best achievable "
            f"is {best:.1%}. Either the detection settings are too permissive, or "
            "cells are unsegmented so real molecules are counted as background. "
            "Refusing to recommend a threshold that does not meet the target."
        )

    row = qualifying.sort_values("threshold").iloc[0]
    note = str(sweep.attrs.get("warning", ""))
    if float(row["threshold"]) <= float(usable["threshold"].min()):
        note = (
            "The recommendation is at the lowest available score. This measures "
            "purity only, so the true optimum may lie lower. Re-run detection "
            "with a permissive setting and sweep again. " + note
        ).strip()
    return ThresholdChoice(
        threshold=float(row["threshold"]),
        score=str(sweep.attrs.get("score", "amp")),
        fdr=float(row["fdr"]),
        target_fdr=float(target_fdr),
        n_inside=int(row["n_inside"]),
        n_outside=int(row["n_outside"]),
        true_positives=float(row["true_positives"]),
        fraction_of_detections_kept=float(row["fraction_kept"]),
        segmented_area_fraction=float(sweep.attrs.get("segmented_area_fraction", np.nan)),
        note=note,
    )

def sweep_for_scope(scope, *, target_fdr: float = 0.05, movie: str | None = None):
    """Detection-threshold sweep for one condition of a loaded project.

    THE CATALOGUE'S SHAPE, over `threshold_sweep`, which takes detections, meshes and an
    image shape -- none of which the canonical table carries, because detections are what
    exists BEFORE linking and the table is what exists after.

    IT MEASURES PURITY, NOT COMPLETENESS, and that limit is structural rather than a
    shortcoming of this wrapper: the sweep only sees detections the upstream detector
    already kept, so it cannot say how many real molecules were discarded before it
    looked. Where the recommendation lands on the lowest score available it says so.

    IT ALSO NEEDS CELL OUTLINES, which is why it belongs in the analysis stage and not
    beside the tracking settings: outlines come from segmentation, which runs AFTER
    tracking. It cannot help you choose a threshold; it can tell you what the one you
    used was worth.
    """
    from .readers import read_tracks

    root = getattr(scope, "root", None)
    if root is None:
        raise DetectionError(
            f"{scope.condition}: this analysis reads the tracker output directly, and "
            f"the project did not record where that condition lives")
    meshes = [m for m in (getattr(scope.project, "meshes", None) or [])
              if movie is None or getattr(m, "movie", None) == movie]
    if not meshes:
        raise DetectionError(
            f"{scope.condition}: no cell outlines. The false-positive rate is measured "
            f"from detections OUTSIDE the cells, so without outlines there is nothing "
            f"to measure against: segment this condition first")
    try:
        raw = read_tracks(Path(root))
    except Exception as exc:                              # noqa: BLE001 - named
        raise DetectionError(
            f"{scope.condition}: could not re-read the tracker output ({exc})")
    chosen = str(movie) if movie else str(
        sorted(raw["movie"].dropna().unique())[0]) if "movie" in raw else None
    here = raw[raw["movie"].astype(str) == chosen] if chosen else raw
    mine = [m for m in meshes if getattr(m, "movie", None) == chosen] or meshes

    shape = getattr(scope.project.params, "image_shape", None)
    if not shape:
        # THE FIELD'S SIZE decides the out-of-cell AREA, which is the denominator of the
        # whole estimate. Guessed, every FDR would be wrong by the ratio of the guess.
        raise DetectionError(
            f"{scope.condition}: the image shape is not recorded, and the false-positive "
            f"rate is a density over the field's area, so a guess would scale every "
            f"number by the size of the guess")
    sweep = threshold_sweep(here, mine, image_shape=tuple(shape))
    got = recommend_threshold(sweep, target_fdr=float(target_fdr))
    return got
