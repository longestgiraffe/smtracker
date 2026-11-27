"""Cell autofluorescence, calibrated on a WT strain, Tier 3 of the intensity ladder.

What the legacy does, read from the source rather than remembered.
`autofluorescenceDecay.m` fits, per cell and over frames from the intensity peak to the
end of the movie, the BACKGROUND mean against that cell's own mean:

    y = mean over pixels OUTSIDE every cell        (dilated by 2 px)
    x = mean over pixels INSIDE this cell          (dilated by 2 px)
    polyfit(x, y, 1) and polyfit(x, y, 2), then the coefficients are AVERAGED over cells

`autofluorescenceCorrection` (a subfunction of `firstFrames.m`) then inverts it per cell:
the cell's own end-of-movie state fixes the constant,
``c = bkg_end - cf1*auto_end^2 - cf2*auto_end``, and the background measured at the FIRST
frame is solved back through the quadratic for the autofluorescence that must have been
there. Two non-negative roots means the quadratic is ambiguous for that cell, and the
legacy falls back to the linear fit; that fallback is reproduced here and COUNTED, since
how often it fires says how well the quadratic is determined.

Why this needs the raw movie. The correction anchors to a bleached end and a lit first
frame, and the stacks u-track saw are CROPPED to the single-molecule phase where neither
exists. MEASURED on the WT control: the cropped stack decays 1.6 % over 2001 frames --
nothing to fit -- while the RAW decays **34 to 44 %** (1307 to 737 counts on series001).
So Tier 3 was never blocked on data, only on reading `raw/`.

The WT control has no dark frames, and that is fine here. All three movies were recorded
with the laser already on (`rawmovie.detect_dark_frames` reports no step), so they cannot
supply an instrumental offset -- but this calibration never needs one. It correlates two
quantities measured in the SAME frames, so any constant common to both is absorbed into
the fitted intercept.

The model is only valid at its own laser power. Autofluorescence scales with
excitation, `movieData.mat` does not record power, and the experimenter's controls are
named for it. `laser_power_mw` therefore has no default and a mismatch is REFUSED, exactly
as in `spot_intensity`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "AutofluorescenceError",
    "AutofluorescenceModel",
    "DILATE_PX",
    "PEAK_SEARCH_FRAMES",
    "cell_masks",
    "fit_autofluorescence",
    "cell_autofluorescence",
]


class AutofluorescenceError(RuntimeError):
    """Raised when an autofluorescence calibration cannot be made honestly."""


#: `autofluorescenceDecay.m` dilates every mask by 2 px (`bwExtend(bw, 2)`) before taking
#: means, so a cell's own halo is not counted as background. Kept verbatim.
DILATE_PX = 2

#: The legacy hunts the intensity peak within the first 200 frames (`if j<200`) and fits
#: from there to the end: before the peak the field is still rising, and a rise fitted
#: together with the decay is two different regimes in one polynomial.
PEAK_SEARCH_FRAMES = 200

#: Below this many cells the averaged coefficients are not a calibration at all.
#:
#: DELIBERATELY LOW, AND THE REAL CAVEAT IS REPORTED RATHER THAN ENFORCED. The WT control
#: holds **7 segmented cells across its 3 movies** (3 / 3 / 1) -- not the hundreds a
#: first estimate assumed, because these 128x128 fields carry about 3 cells each, the
#: same density as the MSD data (~84 cells per condition over 30 movies). A hard bound at
#: 10 would simply refuse the only control that exists, which is not a judgement this
#: code should make. So the floor sits where a MEAN is meaningless, and
#: `AutofluorescenceModel` carries `n_cells` and the per-cell coefficient SPREAD so the
#: thinness is visible wherever the model is.
MIN_CELLS = 5


@dataclass(frozen=True)
class AutofluorescenceModel:
    """Averaged per-cell coefficients relating background to cell intensity."""

    quadratic: np.ndarray                 # [p2, p1, p0], highest power first (polyfit)
    linear: np.ndarray                    # [p1, p0]
    laser_power_mw: float
    source: str
    n_cells: int
    n_movies: int
    r2_quadratic: float = float("nan")    # median per-cell R^2
    r2_linear: float = float("nan")
    #: SD of each coefficient ACROSS CELLS, divided by |mean|. With few cells this is the
    #: number that says whether an average means anything: coefficients that agree make a
    #: calibration, coefficients that scatter make a mean of idiosyncrasies.
    spread_quadratic: np.ndarray | None = None
    spread_linear: np.ndarray | None = None
    #: The CELL-INTENSITY range the coefficients were actually fitted over, and the
    #: BACKGROUND range that implies. Applying the model outside these is extrapolation,
    #: and on this control that is not a fine distinction -- see `cell_autofluorescence`.
    cell_range: tuple | None = None
    background_range: tuple | None = None
    #: Which of the legacy's two preprocessing steps actually ran. A model fitted on raw
    #: counts is not the same quantity as one fitted on `(raw - offset) / ratio`, so this
    #: has to travel with the coefficients rather than be assumed.
    corrections: tuple = ()
    #: Why a step did not run, per movie. A silent skip reads as "not needed".
    notes: tuple = ()

    def __post_init__(self):
        if self.laser_power_mw is None or not np.isfinite(self.laser_power_mw):
            raise AutofluorescenceError(
                "laser_power_mw has no default: autofluorescence scales with excitation "
                "and movieData.mat does not record the power. State the WT control's")
        if self.n_cells < MIN_CELLS:
            raise AutofluorescenceError(
                f"{self.n_cells} cells is below the {MIN_CELLS} an averaged calibration "
                "needs; the coefficients would be one or two cells' idiosyncrasies")

    def stamp(self) -> str:
        spread = ""
        if self.spread_linear is not None:
            spread = (" | per-cell CV lin "
                      + "/".join(f"{v:.2f}" for v in np.atleast_1d(self.spread_linear)))
        applied = ", ".join(self.corrections) if self.corrections else "NONE (raw counts)"
        return (f"autofluorescence | {self.laser_power_mw:g} mW | {self.n_cells} cells "
                f"in {self.n_movies} movies | preprocessing: {applied} | "
                f"R^2 quad {self.r2_quadratic:.3f} lin {self.r2_linear:.3f}{spread} "
                f"| from {self.source}")

    def check_matches(self, laser_power_mw: float) -> None:
        if not np.isclose(float(laser_power_mw), self.laser_power_mw, rtol=1e-6):
            raise AutofluorescenceError(
                f"calibration is {self.laser_power_mw:g} mW, dataset declares "
                f"{float(laser_power_mw):g} mW. Autofluorescence scales with excitation, "
                "so this model does not transfer")


def cell_masks(meshes, shape, *, dilate_px: int = DILATE_PX):
    """Boolean mask per cell, plus the "outside every cell" mask, on the sensor grid.

    Coordinates are 1-BASED from Oufti/MATLAB, as everywhere else in this project.
    """
    from scipy.ndimage import binary_dilation

    from .cellmesh import points_in_polygon

    height, width = shape
    yy, xx = np.mgrid[1:height + 1, 1:width + 1]      # 1-based, matching the outlines
    flat_x, flat_y = xx.ravel().astype(float), yy.ravel().astype(float)

    per_cell, union = [], np.zeros((height, width), dtype=bool)
    for cell in meshes:
        contour = getattr(cell, "polygon", None)
        if contour is None or len(contour) < 3:
            continue
        inside = points_in_polygon(flat_x, flat_y, np.asarray(contour, dtype=float))
        mask = inside.reshape(height, width)
        if dilate_px:
            mask = binary_dilation(mask, iterations=int(dilate_px))
        if not mask.any():
            continue
        per_cell.append(mask)
        union |= mask
    if not per_cell:
        raise AutofluorescenceError("no cell produced a mask on this frame shape")
    return per_cell, ~union


def _masked_means(movie, masks, *, chunk: int = 400, offset=None,
                  flat_field=None) -> np.ndarray:
    """Mean inside each mask, for every frame. Shape ``(n_masks, n_frames)``.

    Chunked and expressed as one matrix product per chunk: a 4000-frame movie against
    ~80 masks is 4000 x 16 384 x 80, which is a second of BLAS and would be minutes as a
    Python loop over frames and masks.
    """
    flat = np.stack([m.ravel().astype(np.float64) for m in masks])       # (k, pixels)
    counts = flat.sum(axis=1)
    if np.any(counts == 0):
        raise AutofluorescenceError("a mask covers no pixels")
    out = np.empty((len(masks), len(movie)), dtype=np.float64)
    for start in range(0, len(movie), chunk):
        stop = min(start + chunk, len(movie))
        block = np.asarray(movie._frames[start:stop], dtype=np.float64)
        # `(dat_ - offset) ./ ratio`, as autofluorescenceDecay.m:49 and firstFrames.m:24
        # do, and in that order -- dividing first would scale the offset by the
        # illumination profile as though it were signal.
        if offset is not None:
            block = block - offset
        if flat_field is not None:
            block = block / flat_field
        block = block.reshape(stop - start, -1)                          # (n, pixels)
        out[:, start:stop] = (flat @ block.T) / counts[:, None]
    return out


def fit_autofluorescence(wt_root, *, laser_power_mw: float,
                         calibration=None,
                         dilate_px: int = DILATE_PX,
                         peak_search: int = PEAK_SEARCH_FRAMES,
                         movies: list | None = None) -> AutofluorescenceModel:
    """Calibrate on a WT (unlabelled) strain: background against cell intensity.

    ``wt_root`` is the condition folder: ``raw/*.nd2`` for the frames and
    ``cell_meshes/*.mat`` for the outlines. Returns coefficients averaged over every
    cell in every movie, as the legacy does.

    ``calibration`` is a `spot_intensity.IntensityCalibration`, and passing it is what
    makes this match the source. Both sides of the legacy work on corrected frames:
    `autofluorescenceDecay.m:49` and `firstFrames.m:24` each compute
    ``dat = (dat_ - offset) ./ ratio`` before taking any mean. A first version of this
    module used RAW counts on both sides: internally consistent, and not the same
    quantity, so its coefficients could not be compared with anything the legacy
    produced. Left optional so the raw-count behaviour stays reachable, but the corrected
    form is what reproduces the source.
    """
    from .cellmesh import read_project_cell_meshes
    from .rawmovie import instrumental_offset, read_raw_movie

    root = Path(wt_root)
    meshes_by_movie = read_project_cell_meshes(root, allow_unsegmented=True)
    raws = sorted((root / "raw").glob("*.nd2"))
    if not raws:
        raise AutofluorescenceError(f"no raw movies under {root / 'raw'}")
    if movies:
        wanted = set(movies)
        raws = [p for p in raws if p.stem in wanted]

    quad, lin, r2q, r2l = [], [], [], []
    seen_x, seen_y = [], []          # the range the coefficients are actually valid over
    skipped_offset, shape_mismatch = [], []
    used_movies = 0
    for path in raws:
        cells = meshes_by_movie.get(path.stem)
        if not cells:
            continue
        movie = read_raw_movie(path)
        shape = (movie.meta.height, movie.meta.width)
        per_cell, outside = cell_masks(cells, shape, dilate_px=dilate_px)

        # THE OFFSET IS PER MOVIE, from its own dark frames -- `autofluorescenceDecay.m`
        # calls `bkgInstrumental(dat_)` on the stack in hand rather than importing one.
        # Only `ratio` comes from outside, and only where its geometry matches: the WT
        # control is 128x128 while the no-cell control is 126x130, and a flat field is a
        # map of the SENSOR, so it cannot be stretched onto a different crop.
        offset = None
        try:
            offset, _found = instrumental_offset(movie)
        except Exception as exc:                       # noqa: BLE001 - named below
            skipped_offset.append(f"{path.stem}: {exc.__class__.__name__}")
        flat = None
        if calibration is not None and calibration.flat_field is not None:
            if calibration.flat_field.shape == shape:
                flat = calibration.flat_field
            else:
                shape_mismatch.append(
                    f"{path.stem} is {shape[0]}x{shape[1]} but the illumination "
                    f"reference is {calibration.flat_field.shape[0]}x"
                    f"{calibration.flat_field.shape[1]}")
        traces = _masked_means(movie, per_cell + [outside], offset=offset,
                               flat_field=flat)
        cell_traces, background = traces[:-1], traces[-1]

        # THE PEAK, per the legacy: the frame of maximum intensity within the first
        # `peak_search` frames. Before it the field is still rising, and fitting a rise
        # together with a decay is two regimes in one polynomial.
        head = min(peak_search, cell_traces.shape[1])
        start = int(np.argmax(cell_traces[0, :head]))

        for trace in cell_traces:
            x, y = trace[start:], background[start:]
            if x.size < 4 or np.ptp(x) <= 0:
                continue                       # a flat cell constrains no polynomial
            try:
                cq = np.polyfit(x, y, 2)
                cl = np.polyfit(x, y, 1)
            except Exception:                  # noqa: BLE001 - singular, skip this cell
                continue
            if not (np.all(np.isfinite(cq)) and np.all(np.isfinite(cl))):
                continue
            quad.append(cq)
            lin.append(cl)
            seen_x.extend((float(np.min(x)), float(np.max(x))))
            seen_y.extend((float(np.min(y)), float(np.max(y))))
            r2q.append(_r2(y, np.polyval(cq, x)))
            r2l.append(_r2(y, np.polyval(cl, x)))
        used_movies += 1

    if not quad:
        raise AutofluorescenceError(
            f"no cell in {root} produced a usable fit; the WT control cannot calibrate")
    quad_stack, lin_stack = np.vstack(quad), np.vstack(lin)

    def _cv(stack):
        mean = np.mean(stack, axis=0)
        sd = np.std(stack, axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(mean) > 0, sd / np.abs(mean), np.inf)

    return AutofluorescenceModel(
        quadratic=np.mean(quad_stack, axis=0), linear=np.mean(lin_stack, axis=0),
        laser_power_mw=float(laser_power_mw), source=str(root),
        n_cells=len(quad), n_movies=used_movies,
        r2_quadratic=float(np.median(r2q)), r2_linear=float(np.median(r2l)),
        spread_quadratic=_cv(quad_stack), spread_linear=_cv(lin_stack),
        cell_range=(float(np.min(seen_x)), float(np.max(seen_x))),
        background_range=(float(np.min(seen_y)), float(np.max(seen_y))),
        corrections=tuple(
            (["offset"] if not skipped_offset else [])
            + (["flat_field"] if (calibration is not None and not shape_mismatch)
               else [])),
        notes=tuple(skipped_offset + shape_mismatch))


def cell_autofluorescence(model: AutofluorescenceModel, *, auto_end: float,
                          bkg_end: float, bkg_first: float,
                          laser_power_mw: float,
                          allow_extrapolation: bool = False) -> dict:
    """One cell's autofluorescence at the FIRST frame, from the background measured there.

    Reproduces `autofluorescenceCorrection`: the cell's own end-of-movie state fixes the
    constant, then the first frame's background is solved back through the quadratic.

    The LEGACY'S fallback is kept and counted. Two non-negative roots means the quadratic
    does not determine which branch this cell is on, and the legacy drops to the linear
    fit. Reporting WHICH was used matters: a calibration that falls back constantly is a
    linear calibration wearing a quadratic's name.
    """
    model.check_matches(laser_power_mw)

    # EXTRAPOLATION IS REFUSED, and on this data that is not a fine distinction.
    # MEASURED: the WT control was recorded with the laser ALREADY ON, so it never saw
    # the bright initial transient -- its own peak is 1307 counts where a sample's first
    # lit frame reads 12 494 of background and ~29 000 inside a cell. Inverting a model
    # fitted over cell intensities of 700 to 1300 at a background of 12 494 returns
    # ~55 000, which EXCEEDS THE WHOLE CELL SIGNAL: autofluorescence is a part of what
    # the cell emits, never more than all of it. Measured on 8 real cells, the prediction
    # exceeded the observed intensity in 8 of 8, by 1.21x to 2.24x.
    #
    # So the number is refused rather than returned with a caveat. The remedy is a
    # control recorded with the laser OFF at the start, so it spans the range the
    # correction is applied in; MORE MOVIES OF THE SAME KIND WOULD NOT HELP.
    if model.background_range is not None and not allow_extrapolation:
        low, high = model.background_range
        if not (low <= float(bkg_first) <= high):
            raise AutofluorescenceError(
                f"background at the first frame is {float(bkg_first):.0f}, outside the "
                f"{low:.0f} to {high:.0f} this model was fitted over, so inverting it would "
                "extrapolate. The WT control never covers the bright start of a sample: "
                "re-record it with the laser off at the start, not more movies of the "
                "same kind. Pass allow_extrapolation=True only to inspect the failure")
    a, b, _ = (float(v) for v in model.quadratic)
    c = float(bkg_end) - a * float(auto_end) ** 2 - b * float(auto_end)
    roots = np.roots([a, b, c - float(bkg_first)]) if a else np.array([])
    real = np.array([r.real for r in np.atleast_1d(roots)
                     if abs(np.imag(r)) < 1e-9 and r.real >= 0])

    if real.size == 1:
        return {"autofluorescence": float(real[0]), "branch": "quadratic",
                "n_roots": int(real.size)}

    # ambiguous (or no real root) -> the linear model, exactly as the legacy does
    la, lb = (float(v) for v in model.linear)
    if la == 0:
        raise AutofluorescenceError("linear coefficient is zero; no fallback exists")
    lc = float(bkg_end) - la * float(auto_end)
    value = (float(bkg_first) - lc) / la
    return {"autofluorescence": float(value), "branch": "linear",
            "n_roots": int(real.size)}


def _r2(y: np.ndarray, fitted: np.ndarray) -> float:
    resid = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - resid / total if total > 0 else float("nan")
