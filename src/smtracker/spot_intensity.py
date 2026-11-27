"""Integrated spot intensity out of the raw frames, the legacy `spotIntensity` quantity.

Why this is a separate module from `intensity`. That one holds u-track's fitted Gaussian
`amp`, which is a peak height with no background removed and no illumination correction.
This one integrates photons under a ROI. They are different quantities that must never be
converted into one another, and keeping them in one file is how that mistake gets made.

What unblocked it. The module head of `intensity` still says the legacy pipeline has no
input because the project ships `phase/` and `sum/` only. **That is true of the small
reference project and NOT of the real dataset**, where every movie carries its cropped
single-molecule stack under `tif/<movie>/<movie>/*.tif` (1001 to 1501 frames, 126x130)
plus the untouched acquisition in `raw/<movie>.nd2`.

The LEGACY'S temporal background does not apply here, and that is a property of the data.
`spotIntensity.m` takes frames 1 to 20 as instrumental background "while the laser is
off". MEASURED on one lab movie, the frame mean is flat across the whole stack --
1086 at frame 1, 1109 at frame 20, 1129 at frame 1001 -- and frame 1 carries the brightest
pixel in the movie (5963). There is no laser-off period, because **the stacks u-track saw
are CROPPED to the single-molecule phase**: the laser-off frames and the bleach-down are
in the `.nd2` and were removed before tracking. Taking frames 1-20 here would call twenty
frames of ordinary signal-bearing background a dark reference.

So the background comes from measurements instead of from a convention, and there are two
of them, which is better than the one the legacy had:

  NO-cell control   a field with no cells at all, same optics and same laser power. A
                    direct measurement of camera offset + illumination + stray light.
  OUTSIDE-CELL      the annulus around each spot, which is contemporaneous with the
                    signal and needs no separate recording. This is what the legacy's own
                    `bkgSub`/`bkgCell` do spatially.

They are independent, so agreement between them validates both and disagreement is itself
a finding about drift.

The crop is defined by a physical state, not a frame count, which is what makes intensity
comparable across movies at all: every stack is cut where the field reaches single
molecules, so 1001 and 1501 frames describe the same regime for different bleach
durations. The frame count is therefore NOT a confound, but anything anchored to the END
of a movie must be resolved per movie -- a global "last frame" index would read a
different point of the bleaching curve in each one.

TIER 3 IS BLOCKED ON THE `.nd2` reader, not on data, and the reason is the crop again.
`autofluorescenceCorrection` (a subfunction of `firstFrames.m`) models background as
`bkg = cf1*cell^2 + cf2*cell + c`, fixes `c` from each cell's OWN end-of-movie state, and
then solves that quadratic for the cell's autofluorescence at the first frame using the
background measured there. It therefore needs a laser-on transient at one end and a
bleached field at the other. MEASURED, no stack has either -- frame means first to last:
WT 3610 755 -> 743 over 2001 frames (1.6 %), no-cell 652 -> 632, a labelled sample 1086 -> 1129
(it RISES). The WT control is cropped exactly as the samples are, so the calibration
cannot be fitted from it and could not be applied if it were. The full acquisition is in
`raw/<movie>.nd2` and that is what Tier 3 waits on.

Laser power has no default. The correction is only valid at the power it was recorded at,
`movieData.mat` does not store it, and the experimenter's own controls are named for it
("15mw laser"). A calibration whose declared power differs from the dataset's is REFUSED
rather than silently applied, on the same grounds as `sigma_stamp`: a correction without
its provenance is not a correction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "SpotIntensityError",
    "INNER_DIAMETER_PX",
    "OUTER_DIAMETER_PX",
    "MIN_LOCALIZATIONS",
    "IntensityCalibration",
    "movie_frame_paths",
    "flat_field_from_no_cell",
    "pseudo_flat_field",
    "integrated_intensity",
    "integrate_spot",
    "illumination_ratio",
    "intensity_for_scope",
]


class SpotIntensityError(RuntimeError):
    """Raised when an integrated intensity cannot be produced honestly."""


#: `spotIntensity.m:15-16`: `d = 6` is the inner ROI DIAMETER in pixels and `D = 2*d` the
#: outer one. Kept as diameters so they read the same as the source; every use halves them.
INNER_DIAMETER_PX = 6.0
OUTER_DIAMETER_PX = 12.0

#: `spotIntensity.m:4`: `minLength = 3`, the shortest track it will integrate.
MIN_LOCALIZATIONS = 3

#: How many frames from the END of a movie form the illumination reference. The legacy
#: `unevenIllu` averages "Nf frames at the end", where everything is bleached so what is
#: left is illumination alone. Resolved PER MOVIE because the stacks differ in length.
FLAT_FIELD_FRAMES = 100

#: Smoothing used ONLY to locate the normalising peak, in pixels. The field itself is
#: NEVER smoothed, and separating those two was the point.
#:
#: THE DEFECT IS IN THE PEAK, NOT IN THE FIELD. `unevenIllu` divides by the raw maximum,
#: an extreme-value statistic over ~16 000 pixels, so it is inflated by whatever shot
#: noise survived averaging -- and by a different amount depending on how many frames
#: were averaged. Only the PEAK needs to be robust; smoothing the whole field to get
#: there also destroys the gradient the field exists to record. Measured on the no-cell
#: control, reproducibility between a 50-frame and an 800-frame reference against how
#: much of the real 16.4 per cent falloff survives:
#:
#:     normalisation                 50 vs 800    worst px   falloff kept
#:     legacy raw maximum               2.195%       5.47%          16.4%
#:     peak of a smoothed copy          0.663%       3.36%          16.4%   <- shipped
#:     smooth the field as well         0.095%       0.39%          15.2%
#:
#: The last row looks best and is not: it buys its number by deleting 1.2 points of the
#: 16.4 per cent falloff, which is 7 per cent of the signal. Its apparent advantage was
#: also an artefact of how it was first checked -- smoothed against smoothed, so the
#: over-smoothing cancelled on both sides and could not show up.
#:
#: What is left at 0.663 per cent is SHOT NOISE, which is zero-mean and averages down by
#: the square root of the 29-pixel ROI to about 0.12 per cent on an integrated value.
PEAK_SMOOTH_PX = 4.0


@dataclass(frozen=True)
class IntensityCalibration:
    """What was measured, from where, at what power, and which corrections are present.

    A Tier-1 (uncorrected) number and a Tier-3 (fully corrected) one are different
    quantities. This carries which one a value is, so the two cannot be mixed in a table
    and read as one column.
    """

    laser_power_mw: float
    source: str
    flat_field: np.ndarray | None = None
    offset: np.ndarray | None = None
    corrections: tuple[str, ...] = ()
    n_movies: int = 0
    n_frames: int = 0
    #: 0 restores the legacy raw-maximum normalisation; see `_normalise`.
    smooth_px: float = PEAK_SMOOTH_PX

    def __post_init__(self):
        if self.laser_power_mw is None or not np.isfinite(self.laser_power_mw):
            raise SpotIntensityError(
                "laser_power_mw has no default and must be given: the correction is only "
                "valid at the power it was recorded at, and movieData.mat does not store "
                "it. Read it from the acquisition settings")
        if self.laser_power_mw <= 0:
            raise SpotIntensityError(
                f"laser_power_mw must be positive, got {self.laser_power_mw}")

    @property
    def tier(self) -> int:
        """1 = raw integration, 2 = + illumination/offset, 3 = + autofluorescence."""
        if "autofluorescence" in self.corrections:
            return 3
        if "flat_field" in self.corrections or "offset" in self.corrections:
            return 2
        return 1

    def stamp(self) -> str:
        present = ", ".join(self.corrections) if self.corrections else "NONE"
        peak = ("legacy raw-max normalisation" if not self.smooth_px
                else f"peak from a {self.smooth_px:g} px smoothed copy (field unsmoothed)")
        return (f"tier {self.tier} | {self.laser_power_mw:g} mW | corrections: {present}"
                f" | {peak} | from {self.source} "
                f"({self.n_movies} movies, {self.n_frames} frames)")

    def check_matches(self, laser_power_mw: float) -> None:
        """REFUSE a calibration recorded at a different power. Not a warning: applying
        the wrong illumination curve produces a plausible number that is simply wrong."""
        if not np.isclose(float(laser_power_mw), self.laser_power_mw, rtol=1e-6):
            raise SpotIntensityError(
                f"calibration was recorded at {self.laser_power_mw:g} mW but this dataset "
                f"declares {float(laser_power_mw):g} mW. Autofluorescence and illumination "
                "both scale with power, so the correction does not transfer. Record a "
                "control at the dataset's power, or state the dataset's true power")


def movie_frame_paths(movie_dir) -> list[Path]:
    """The cropped single-molecule stack for one movie, in frame order.

    Layout is `<movie_dir>/<movie name>/*_c1_t####.tif`, the stack sits in a folder that
    repeats the movie's own name, beside `movieData.mat`. Sorted by the numeric frame
    index in the name rather than lexically, so `t1000` cannot sort before `t999`.
    """
    movie_dir = Path(movie_dir)
    inner = movie_dir / movie_dir.name
    root = inner if inner.is_dir() else movie_dir
    hits = list(root.glob("*_c1_t*.tif"))
    if not hits:
        hits = [p for p in root.glob("*.tif") if "_t" in p.stem]
    if not hits:
        raise SpotIntensityError(f"no frame .tif files under {root}")

    def index(path: Path) -> int:
        tail = path.stem.rsplit("_t", 1)[-1]
        digits = "".join(c for c in tail if c.isdigit())
        if not digits:
            raise SpotIntensityError(f"cannot read a frame index from {path.name}")
        return int(digits)

    return sorted(hits, key=index)


def _read_stack_tail(movie_dir, n_frames: int) -> np.ndarray:
    """Mean image over the LAST ``n_frames`` frames of one movie.

    Per movie, never by a global frame index: the stacks are 1001 to 2001 frames long
    because the bleach-down before the crop took different amounts of time, so a fixed
    index would land at a different point of each movie's curve.
    """
    import tifffile

    paths = movie_frame_paths(movie_dir)
    if n_frames < 1:
        raise SpotIntensityError(f"n_frames must be >= 1, got {n_frames}")
    tail = paths[-min(n_frames, len(paths)):]
    total = None
    for path in tail:
        frame = tifffile.imread(path).astype(np.float64)
        total = frame if total is None else total + frame
    return total / len(tail)


def _movie_dirs(root) -> list[Path]:
    """Every movie folder under a condition root, found by its `movieData.mat`."""
    root = Path(root)
    hits = sorted(p.parent for p in root.glob("**/movieData.mat")
                  if "backups" not in p.parts)
    if not hits:
        raise SpotIntensityError(f"no movies (no movieData.mat) under {root}")
    return hits


def _normalise(offset: np.ndarray, smooth_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Illumination reference and its ratio-to-peak, with the peak taken ROBUSTLY.

    The legacy divides by the single brightest pixel and that pixel is mostly noise.
    `unevenIllu` forms "the ratio between the intensity of every pixel and its maximum".
    A raw maximum over ~16 000 pixels is an extreme-value statistic, so it is inflated by
    whatever shot noise the averaging left behind, and by a DIFFERENT amount depending
    on how many frames were averaged. Measured on the no-cell control:

        raw max              50 frames 705.24    800 frames 688.65   (2.4 % apart)
        max after smoothing  50 frames 683.51    800 frames 683.52   (identical)

    So the whole correction's scale moved with the analyst's choice of N, which is a
    property of the microscope being read off a decision about processing. Taking the
    peak of the SMOOTHED reference instead makes 50 frames reproduce 800 frames to
    0.095 per cent mean (0.390 per cent worst pixel), against 2.193 / 5.14 before.

    ``smooth_px = 0`` restores the legacy behaviour verbatim.
    """
    if smooth_px and smooth_px > 0:
        from scipy.ndimage import gaussian_filter

        peak = float(np.max(gaussian_filter(offset, float(smooth_px))))
    else:
        peak = float(np.max(offset))                  # the legacy's own choice
    if not np.isfinite(peak) or peak <= 0:
        raise SpotIntensityError(
            f"illumination reference has a non-positive maximum ({peak}); no ratio exists")
    # THE FIELD IS RETURNED UNSMOOTHED. Only the divisor came from the smoothed copy.
    return offset, offset / peak


def flat_field_from_no_cell(root, *, laser_power_mw: float,
                            n_frames: int = FLAT_FIELD_FRAMES,
                            smooth_px: float = PEAK_SMOOTH_PX,
                            ) -> IntensityCalibration:
    """Illumination profile and instrumental offset from a NO-CELL control.

    A field with no cells measures camera offset + illumination + stray light and nothing
    else, which is what the legacy wanted frames 1-20 to give it and, on this data, they
    do not (see the module head).

    Two products from one recording:
      offset       the mean image itself, in camera counts, subtract it.
      flat_field   that image divided by its own maximum, so values run up to 1, divide
                   by it, exactly as `unevenIllu` specifies.

    The order is fixed and matters: ``(raw - offset) / flat_field``. Dividing first would
    scale the offset by the illumination profile as though it were signal.
    """
    dirs = _movie_dirs(root)
    frames = [_read_stack_tail(d, n_frames) for d in dirs]
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise SpotIntensityError(
            f"no-cell movies disagree about frame shape: {sorted(shapes)}. They cannot be "
            "averaged into one illumination profile")
    reference, flat = _normalise(np.mean(frames, axis=0), smooth_px)
    return IntensityCalibration(
        laser_power_mw=float(laser_power_mw),
        source=str(root),
        offset=reference,
        flat_field=flat,
        corrections=("offset", "flat_field"),
        n_movies=len(dirs),
        n_frames=n_frames,
        smooth_px=float(smooth_px),
    )


def pseudo_flat_field(root, *, laser_power_mw: float,
                      n_frames: int = FLAT_FIELD_FRAMES,
                      smooth_px: float = PEAK_SMOOTH_PX,
                      shape: tuple | None = None) -> IntensityCalibration:
    """Illumination profile from the SAMPLE movies, as a cross-check on the control.

    Per-pixel MEDIAN across movies, not mean: cells sit in a different place in every
    field, so the median suppresses them while the mean would smear them into the
    profile. It needs no separate recording and it is CONTEMPORANEOUS with the data,
    which the no-cell control recorded months later is not, the illumination profile
    belongs to the alignment of the day.

    Agreement between this and :func:`flat_field_from_no_cell` validates both; a
    disagreement says the alignment moved, which is a finding rather than a nuisance.
    """
    dirs = _movie_dirs(root)
    if len(dirs) < 3:
        raise SpotIntensityError(
            f"a median across movies needs at least 3 movies, found {len(dirs)}; with "
            "fewer, cells survive into the profile")
    frames = [_read_stack_tail(d, n_frames) for d in dirs]  # noqa: F841 - regrouped below
    # THE SPATIAL CROP VARIES TOO, not just the temporal one: one lab project holds 26 movies
    # at (126, 130) and 3 at (124, 128). A flat field is a map of the SENSOR, so frames of
    # different shape cannot be averaged into one -- and they must not be silently
    # centre-cropped together either, because nothing here knows where each crop sat on
    # the chip, and a misregistered illumination map invents a gradient rather than
    # removing one. So one shape is chosen, named, and the rest are reported as skipped.
    shapes = {}
    for d, f in zip(dirs, frames):
        shapes.setdefault(f.shape, []).append((d, f))
    if shape is None:
        shape = max(shapes, key=lambda k: len(shapes[k]))
    if shape not in shapes:
        raise SpotIntensityError(
            f"no movie has shape {shape}; present: "
            + ", ".join(f"{k} x{len(v)}" for k, v in sorted(shapes.items(), key=str)))
    chosen = shapes[shape]
    skipped = sum(len(v) for k, v in shapes.items() if k != shape)
    dirs = [d for d, _ in chosen]
    frames = [f for _, f in chosen]
    reference, flat = _normalise(np.median(np.stack(frames), axis=0), smooth_px)
    return IntensityCalibration(
        laser_power_mw=float(laser_power_mw),
        source=(f"{root} (pseudo-flat-field, median of {len(dirs)} movies at "
                f"{shape}" + (f", {skipped} skipped for a different shape)" if skipped
                              else ")")),
        offset=reference,
        flat_field=flat,
        corrections=("offset", "flat_field"),
        n_movies=len(dirs),
        n_frames=n_frames,
        smooth_px=float(smooth_px),
    )


def _disc_masks(shape, cx: float, cy: float, inner_r: float, outer_r: float):
    """Boolean inner disc and background annulus around one localisation.

    An exact distance test, not the LEGACY'S 25-POINT polygon. `Roi.m` builds the circle
    as `sin/cos(0:pi/12:2*pi)`, a 25-gon, and `poly2mask` rasterises that, an artefact of
    the MATLAB API rather than a choice about which pixels belong to a spot. A 25-gon of
    radius 3 px undershoots the circle's area by about 1 per cent, uniformly, so it
    changes the scale by a constant and not the comparison; the exact test is used because
    there is no reason to reproduce a rasterisation quirk.

    Coordinates arrive 1-BASED from u-track/MATLAB, so one is subtracted here, the same
    `PIXEL_ORIGIN_OFFSET` the viewer applies. Getting this wrong shifts every ROI by a
    pixel, which is a third of the inner radius.

    A local window, not a FULL-FRAME grid. The obvious form allocates two arrays the size
    of the whole image for every localisation; at ~42 000 localisations on one condition
    that is ~42 000 x 126 x 130 elements of pure waste, and the ROI touches at most a
    13x13 patch. The window is clipped at the sensor edge, and the returned slice says
    where it sits so the caller indexes the image once.
    """
    height, width = shape
    px, py = cx - 1.0, cy - 1.0                       # u-track/MATLAB is 1-based
    reach = int(np.ceil(outer_r)) + 1
    y0, y1 = max(0, int(np.floor(py)) - reach), min(height, int(np.ceil(py)) + reach + 1)
    x0, x1 = max(0, int(np.floor(px)) - reach), min(width, int(np.ceil(px)) + reach + 1)
    if y0 >= y1 or x0 >= x1:
        empty = np.zeros((0, 0), dtype=bool)
        return empty, empty, (slice(0, 0), slice(0, 0))
    yy = np.arange(y0, y1)[:, None]
    xx = np.arange(x0, x1)[None, :]
    dist2 = (xx - px) ** 2 + (yy - py) ** 2
    inner = dist2 <= inner_r ** 2
    annulus = (dist2 > inner_r ** 2) & (dist2 <= outer_r ** 2)
    return inner, annulus, (slice(y0, y1), slice(x0, x1))


def integrate_spot(image: np.ndarray, cx: float, cy: float, inner_r: float, outer_r: float):
    """One localisation's integrated intensity on one (already corrected) frame: the inner disc's sum minus the
    MEDIAN of the annulus times the disc's pixel count. ``cx``/``cy`` 1-BASED (u-track/MATLAB, `_disc_masks`).
    Returns ``(integrated, background, n_inner, n_annulus)``, or None where the spot lies off the sensor edge.
    THE ONE place a spot is integrated: `integrated_intensity` calls it."""
    inner, annulus, box = _disc_masks(image.shape, cx, cy, inner_r, outer_r)
    n_in = int(inner.sum())
    if n_in == 0 or not annulus.any():
        return None
    patch = image[box]
    background = float(np.median(patch[annulus]))
    return float(patch[inner].sum()) - background * n_in, background, n_in, int(annulus.sum())


def illumination_ratio(reference: np.ndarray, smooth_px: float = PEAK_SMOOTH_PX) -> tuple:
    """``(offset, flat_field)`` from an illumination reference image, the arithmetic `flat_field_from_no_cell` and
    `pseudo_flat_field` share (`_normalise`): the reference itself is the offset, its ratio to a robust peak the
    flat field, applied as ``(raw - offset) / flat_field``."""
    return _normalise(np.asarray(reference, dtype=float), smooth_px)


def integrated_intensity(raw_table: pd.DataFrame, movie_root, *,
                         calibration: IntensityCalibration,
                         laser_power_mw: float,
                         inner_diameter_px: float = INNER_DIAMETER_PX,
                         outer_diameter_px: float = OUTER_DIAMETER_PX,
                         min_localizations: int = MIN_LOCALIZATIONS,
                         movies: list | None = None) -> pd.DataFrame:
    """One row per localisation: integrated photons under the spot, background removed.

    ``raw_table`` is the PIXEL-stage table (`x`, `y`, `frame`, `movie`, `uid`), not the
    canonical µm one, because the ROI lives on the detector grid.

    Per localisation: sum the inner disc, estimate the background as the MEDIAN of the
    surrounding annulus (the legacy uses a mean over a cell-masked region; the median is
    robust to a neighbouring molecule drifting into the annulus, which at this density
    happens), and subtract background × area.

    Returns `integrated` in camera counts after `(raw - offset) / flat_field`, plus the
    background level and the pixel counts, so a suspicious value can be traced.
    """
    import tifffile

    calibration.check_matches(laser_power_mw)
    for column in ("movie", "frame", "x", "y"):
        if column not in raw_table.columns:
            raise SpotIntensityError(f"missing column {column!r}: this needs the raw "
                                     "pixel table, not the canonical µm one")
    if min_localizations < 1:
        raise SpotIntensityError("min_localizations must be >= 1")

    inner_r = float(inner_diameter_px) / 2.0
    outer_r = float(outer_diameter_px) / 2.0
    if outer_r <= inner_r:
        raise SpotIntensityError(
            f"outer diameter ({outer_diameter_px}) must exceed inner ({inner_diameter_px})"
            " or the background annulus is empty")

    table = raw_table
    if "uid" in table.columns:
        keep = table.groupby("uid")["frame"].transform("size") >= min_localizations
        table = table[keep]
    if table.empty:
        raise SpotIntensityError(
            f"no track reaches {min_localizations} localisations; nothing to integrate")

    by_name = {p.name: p for p in _movie_dirs(movie_root)}
    rows = []
    wanted = set(movies) if movies else None
    for movie, part in table.groupby("movie", sort=True):
        name = str(movie).split("_", 1)[-1] if str(movie) not in by_name else str(movie)
        if wanted is not None and name not in wanted:
            continue
        if name not in by_name:
            raise SpotIntensityError(
                f"no movie folder for {movie!r} under {movie_root}; "
                f"have {sorted(by_name)[:5]}...")
        paths = movie_frame_paths(by_name[name])
        for frame_no, hits in part.groupby("frame", sort=True):
            index = int(frame_no) - 1                 # frames are 1-based
            if index < 0 or index >= len(paths):
                raise SpotIntensityError(
                    f"{movie} frame {frame_no} is outside the stack of {len(paths)} "
                    "frames: the table and the images disagree about this movie")
            image = tifffile.imread(paths[index]).astype(np.float64)
            if calibration.offset is not None:
                if calibration.offset.shape != image.shape:
                    raise SpotIntensityError(
                        f"calibration is {calibration.offset.shape} but {movie} frame "
                        f"{frame_no} is {image.shape}")
                image = image - calibration.offset
            if calibration.flat_field is not None:
                image = image / calibration.flat_field
            for row in hits.itertuples():
                got = integrate_spot(image, row.x, row.y, inner_r, outer_r)
                if got is None:
                    continue                          # spot lies off the sensor edge
                integrated, background, n_in, n_annulus = got
                rows.append({
                    "movie": movie, "frame": int(frame_no),
                    "uid": getattr(row, "uid", None),
                    "x": float(row.x), "y": float(row.y),
                    "integrated": integrated,
                    "background": background,
                    "n_inner_px": n_in, "n_annulus_px": n_annulus,
                })

    if not rows:
        raise SpotIntensityError("no localisation produced an intensity")
    out = pd.DataFrame(rows)
    out.attrs["calibration"] = calibration.stamp()
    out.attrs["tier"] = calibration.tier
    out.attrs["inner_diameter_px"] = float(inner_diameter_px)
    out.attrs["outer_diameter_px"] = float(outer_diameter_px)
    out.attrs["min_localizations"] = int(min_localizations)
    return out

def intensity_for_scope(scope, *, laser_power_mw: float, movie: str | None = None,
                        tier: int = 1):
    """Integrated spot intensity for one condition of a loaded project.

    THE CATALOGUE'S SHAPE, over `integrated_intensity`, which takes a RAW table, a movie
    folder and a calibration. None of those survive into the canonical table: by the
    time a table exists the acquisition it came from is a path nobody kept, which is
    exactly why this module sat built, tested and reachable from nothing.

    ONE MOVIE AT A TIME, and by name. The integration reads every frame of a stack, so a
    condition of thirty movies is minutes; a panel that quietly did all of them would be
    the frozen window this project has already had twice. The default is the first movie
    in the condition, and it says which it took.

    LASER POWER HAS NO DEFAULT and never will: it appears in no sample file, the tiers
    scale with it, and a guessed calibration produces a number that looks like a
    measurement. `IntensityCalibration` refuses a mismatch for the same reason.
    """
    import pandas as pd

    from .grouping import load_utrack_project
    from .readers import read_tracks

    root = getattr(scope, "root", None)
    if root is None:
        raise SpotIntensityError(
            f"{scope.condition}: this analysis reads the raw acquisition beside the "
            f"tracks, and the project did not record where that condition lives")
    root = Path(root)

    # THE RAW TABLE, RE-READ. `Project` keeps the canonical table and not the raw one,
    # and the raw one is what carries each spot's frame and its fitted amplitude.
    try:
        raw = read_tracks(root)
    except Exception as exc:                              # noqa: BLE001 - named
        raise SpotIntensityError(
            f"{scope.condition}: could not re-read the tracker output ({exc})")
    if "movie" not in getattr(raw, "columns", ()):
        raise SpotIntensityError(f"{scope.condition}: the tracker output has no movie "
                                 f"column, so a stack cannot be matched to it")
    movies = sorted(str(m) for m in raw["movie"].dropna().unique())
    if not movies:
        raise SpotIntensityError(f"{scope.condition}: no movies in the tracker output")
    chosen = str(movie) if movie else movies[0]
    if chosen not in movies:
        raise SpotIntensityError(
            f"{scope.condition}: no movie called {chosen!r}. It has: "
            + ", ".join(movies[:6]) + (" ..." if len(movies) > 6 else ""))

    here = raw[raw["movie"].astype(str) == chosen]
    params = getattr(scope.project, "params", None)
    calibration = IntensityCalibration(
        laser_power_mw=float(laser_power_mw),
        pixel_size_nm=float(getattr(params, "pixel_size_nm", 0.0) or 0.0) or None,
    ) if "IntensityCalibration" in globals() else None

    got = integrated_intensity(here, root, calibration=calibration,
                               laser_power_mw=float(laser_power_mw), tier=int(tier))
    if isinstance(got, pd.DataFrame):
        got.attrs["movie"] = chosen
        got.attrs["of_movies"] = len(movies)
        # SAID ON THE RESULT, because a panel showing one movie's intensities under a
        # condition's name is the "two datasets, one name" shape this project records.
        got.attrs["caveat"] = (
            f"one movie ({chosen}) of {len(movies)} in {scope.condition}; integration "
            f"reads every frame, so this is per movie by design")
    return got
