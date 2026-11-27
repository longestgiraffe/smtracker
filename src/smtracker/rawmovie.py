"""The RAW acquisition, before anyone cropped it, and it comes from two microscopes.

Why this exists. `tif/<movie>/<movie>/*.tif` is the stack u-track saw: CROPPED to the
single-molecule phase, so the laser-off frames and the whole bleach-down are gone. Every
legacy quantity that anchors to the start or the end of an acquisition -- the
instrumental background, the bleaching step, the autofluorescence correction -- needs
what was cut away, and that lives in `raw/`.

Two microscopes, two formats, one interface. One Nikon writes `.nd2` (NIS); the other
runs MetaMorph and writes `.stk` (plus a `.nd` experiment descriptor). `read_raw_movie`
dispatches on the file, exactly as `readers.read_tracks` dispatches u-track against
TrackMate, and everything downstream is source-agnostic. MetaMorph needs no new
dependency: `.stk` is a TIFF variant and `tifffile` already decodes it, including its
`stk_metadata`.

The LEGACY'S `Ni = 1, Nf = 20` dark window is a guess, and on this data it is OFTEN
WRONG. `spotIntensity.m` takes frames 1 to 20 as "instrumental background while the
laser is off". Measured on protein A, the number of genuinely dark frames before the laser
fires is not 20 and is not even stable:

    condition   dark frames   laser on at   frame mean, dark -> lit
    Malate           32            33            508  ->  13702
    LB               12            13            507  ->  11771
    Glucose           1             2            507  ->  13266

On Glucose the legacy window averages nineteen lit frames at ~13 000 counts into the
dark reference and reports about 7 000 where the truth is 507 -- a fourteenfold
overestimate of the instrumental floor. On LB it swallows eight. Only Malate happens to
be safe, which is how a hard-coded window survives review: it works on the file someone
checked. So the dark region is DETECTED here and the count is reported, never assumed.

Surveyed over all 119 raw movies (`legacy_out/tables/raw_dark_frames.csv`), because the
experimenter switches the laser ON BY HAND and said so:

    condition   movies   with dark   no dark   n_dark min / med / max
    Glu + Mal      30        30         0            5 / 15 / 50
    Glucose        30        14        16            1 / 12 / 44
    LB             30        30         0            3 / 13 / 32
    Malate         29        16        13            3 / 14 / 32

29 movies have no dark region at all -- the laser was already on when recording
started -- and of the 90 that do, 67 have fewer than 20 dark frames. So the legacy's
fixed window is right for 23 of 119, about 19 per cent. A movie with no step is REPORTED
as such and sent to the no-cell control rather than given a boundary it does not have.

The dark level is **506.0 to 510.4 counts** across all 119 movies -- a 0.9 per cent
spread, which is what camera dark current alone should look like. An earlier reading of
506 to 700 was an ARTEFACT of the fixed-step detector counting partly lit frames as dark
in the dim movies; with the per-movie threshold it disappears. So one movie's dark level
IS a reasonable stand-in for another's, which is what the 29 movies carrying none need.

One dark frame is a level, not a map. Glucose gives a single dark frame. Averaged
over ~16 000 pixels that is a perfectly good scalar offset, and it is nowhere near enough
for a PER-PIXEL dark image. The two are separated below and the per-pixel form refuses
when the frames cannot support it, rather than returning a noisy array that looks like a
calibration.

Pixel size from this file is not trusted. The `.nd2` reports 0.16 µm, which is the
Andor DU-897's 16 µm pixel at 100x with no account of the ~1.5x tube lens; the true value
is 106 nm and `movieData.mat` has it. Confirmed independently by the cells, which are
1.01 µm wide at 106 nm and an impossible 1.52 µm at 160 nm. `RawMovieMeta` therefore
carries the file's value under a name that says what it is and `pixel_size_nm` is not
among its fields, so nothing can take it by accident.
"""
from __future__ import annotations


import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chart_text import fold_words

__all__ = [
    "RawMovieError",
    "RawMovieMeta",
    "RawMovie",
    "read_raw_movie",
    "detect_dark_frames",
    "same_movie",
    "movie_number",
    "instrumental_offset",
    "find_crop_offset",
    "SCALING_SETTINGS",
    "stk_settings",
    "ACQUISITION_SUFFIXES",
    "discover_raw_movies",
    "condition_folder_of",
    "known_movie_names",
    "movie_name",
    "movie_names",
    "movie_named_crops",
    "claimed_twice",
    "acquisition_time",
    "compare_settings",
    "check_recorded_interval",
    "frame_interval",
    "require_comparable",
    "MIN_FRAMES_FOR_PIXEL_MAP",
    "LIT_FRACTION_OF_RANGE",
    "MIN_CONTRAST",
]


class RawMovieError(RuntimeError):
    """Raised when a raw acquisition cannot be read or interpreted honestly."""


#: Below this many dark frames, a PER-PIXEL offset map is refused and only a scalar
#: level is offered. A single frame's per-pixel noise is the full shot noise of one
#: exposure; averaged over the frame it is negligible, so the scalar survives where the
#: map does not. Glucose supplies exactly one, so this is not a hypothetical bound.
MIN_FRAMES_FOR_PIXEL_MAP = 10

#: How far above the dark floor a frame must sit to count as LIT, as a fraction of the
#: movie's OWN dynamic range. Scaled per movie, not fixed, and that is the whole point.
#:
#: A FIXED STEP FACTOR CANNOT WORK, and the first version's 3x was measured on labelled
#: samples alone. mNeonGreen makes the laser-on jump enormous -- 507 to 13 266 counts,
#: 26x in a single frame -- while the WT control is UNLABELLED, so it climbs only to
#: about 2.6x and does it gradually: its largest single-frame rise is 1.55 to 2.38x, all
#: below 3, and the detector reported "no dark region" for a movie whose first ten frames
#: sit at 507 counts. Third instance in this project of a number measured in one regime
#: being applied to another.
#:
#: A fraction of the movie's own range fixes it because the dark plateau is quiet in
#: absolute terms (sd ~1 count) whatever the sample is, while the lit signal is hundreds
#: of sd above it. MEASURED thresholds and the dark counts they give:
#:     Sample A     floor 504, ceiling 13702 -> lit above 768   -> 32 dark frames
#:     Sample B     floor 507, ceiling 13266 -> lit above 762   ->  1 dark frame
#:     WT series001 floor 507, ceiling  1307 -> lit above 523   -> 15 dark frames
LIT_FRACTION_OF_RANGE = 0.02

#: The lit portion must exceed the dark floor by at least this factor for a step to be
#: claimed at all. Without it a movie with no step -- the laser already on, so the trace
#: is a smooth decay -- would have its dimmest leading frames called "dark".
MIN_CONTRAST = 1.3


@dataclass(frozen=True)
class RawMovieMeta:
    """Acquisition settings as the FILE reports them, with the untrusted parts named."""

    n_frames: int
    height: int
    width: int
    source: str                                   # "nd2" | "stk" | "tiff"
    exposure_ms: float | None = None
    em_gain: float | None = None
    acquired: str | None = None                   # the instrument's own date string
    camera: str | None = None
    objective: str | None = None
    #: The EMCCD multiplier. It scales every count directly (140x here), so a calibration
    #: taken at a different one does not transfer, the same argument as laser power, and
    #: unlike laser power this IS recorded, so it is verified rather than declared.
    conversion_gain: str | None = None
    readout_speed: str | None = None
    #: WHAT THE FILE CLAIMS, which for the .nd2 here is the bare objective magnification
    #: with no tube lens. NOT called `pixel_size_nm`, so it cannot be mistaken for the
    #: analysis value -- that comes from `movieData.mat` via `utrack_meta`.
    reported_pixel_size_nm_UNTRUSTED: float | None = None
    #: **THE OBJECTIVE, AS THE INSTRUMENT RECORDS IT.** `movieData.mat` is u-track's
    #: file, so a project tracked with TrackMate will never have one and the acquisition
    #: is the ONLY source for these. NA is recorded outright
    #: (`objectiveNumericalAperture`); it is instrument-recorded like the frame interval,
    #: not derived like the pixel size, so it is trusted on the same grounds.
    numerical_aperture: float | None = None
    objective_magnification: float | None = None
    objective_name: str | None = None
    #: **THE SENSOR PIXEL, RECOVERED FROM THE FILE'S OWN ERROR.** NIS computes its
    #: calibration as `camera pixel / magnification`, dropping the tube lens, which is
    #: why it claims 160 nm where the truth is 106. Multiplying the claim back BY the
    #: magnification therefore returns the honest quantity: 0.16 um x 100 = **16 um**,
    #: the Andor DU-897's real pixel. So the file cannot give the pixel SIZE but it can
    #: give the camera pixel, and then only the tube lens is missing.
    camera_pixel_um: float | None = None
    #: The FILTER's emission centre, which is NOT the dye's emission maximum. This file
    #: records 540 nm for a YFPHQ filter while the mNeonGreen in it peaks at 517, and the
    #: PSF wants the DYE. Kept under a name that says which, and never auto-filled.
    filter_emission_nm: float | None = None
    #: The interval between frames, MEASURED from the instrument's own per-frame
    #: timestamps where the format records them. This is the tau every D scales with,
    #: and it is not the same thing as the exposure: a 20 ms exposure can sit in a
    #: 20.472 ms frame, and this project has already had to correct that pair twice.
    #: ``None`` when the file records no timestamps, in which case it must be supplied.
    frame_interval_ms: float | None = None
    #: How ``frame_interval_ms`` was obtained, so a measured value and a nominal one are
    #: never confused: "per-frame timestamps" | "exposure (nominal)" | None.
    frame_interval_source: str | None = None
    extra: dict = field(default_factory=dict)

    def stamp(self) -> str:
        bits = [f"{self.n_frames} frames", f"{self.height}x{self.width}", self.source]
        if self.exposure_ms is not None:
            bits.append(f"{self.exposure_ms:g} ms")
        if self.em_gain is not None:
            bits.append(f"EM {self.em_gain:g}")
        if self.acquired:
            bits.append(self.acquired)
        return " | ".join(bits)


@dataclass
class RawMovie:
    """A raw acquisition: its frames and what the instrument said about them."""

    path: Path
    meta: RawMovieMeta
    _frames: np.ndarray

    def __len__(self) -> int:
        return int(self.meta.n_frames)

    def frame(self, index: int) -> np.ndarray:
        """One frame, 0-based."""
        if not 0 <= index < len(self):
            raise RawMovieError(
                f"frame {index} outside 0..{len(self) - 1} for {self.path.name}")
        return np.asarray(self._frames[index], dtype=np.float64)

    def frame_means(self, upto: int | None = None) -> np.ndarray:
        """Mean of each frame: the trace the dark detector reads."""
        stop = len(self) if upto is None else min(int(upto), len(self))
        block = np.asarray(self._frames[:stop], dtype=np.float64)
        return block.reshape(stop, -1).mean(axis=1)


# ------------------------------------------------------------------ the readers
def _read_nd2(path: Path) -> RawMovie:
    try:
        import nd2
    except ImportError as exc:                     # pragma: no cover - env dependent
        raise RawMovieError(
            "reading .nd2 needs the `nd2` package (pure Python, bundles cleanly): "
            "py -m pip install nd2") from exc

    with nd2.ND2File(str(path)) as handle:
        # LAZY. `asarray()` pulls the whole acquisition into memory -- 4500 x 126 x 130
        # uint16 is 147 MB per movie -- and the two things this module does most often,
        # reading metadata and finding the laser-on step, touch a few hundred frames at
        # most. `to_dask` keeps the file backed on disk and materialises only what is
        # indexed; `RawMovie.frame` and `frame_means` both go through np.asarray, which
        # computes just the slice they ask for.
        data = handle.to_dask()
        sizes = dict(handle.sizes)
        text = dict(handle.text_info or {})
        try:
            voxel = handle.voxel_size()
            reported_nm = float(voxel.x) * 1000.0
        except Exception:                          # noqa: BLE001 - optional
            reported_nm = None
        objective = text.get("optics")
        interval, interval_source = _nd2_frame_interval(handle)
        optics = _nd2_objective(handle, reported_nm)

    description = str(text.get("description", ""))
    exposure = _search_float(r"Exposure:\s*([\d.]+)\s*ms", description)
    gain = _search_float(r"Multiplier:\s*([\d.]+)", description)
    camera = _search_str(r"Camera Type:\s*([^\n|]+)", description)
    conversion = _search_str(r"Conversion Gain:\s*([^\n|]+)", description)
    readout = _search_str(r"Readout Speed:\s*([^\n|]+)", description)
    if data.ndim != 3:
        raise RawMovieError(
            f"{path.name}: expected a single-channel time stack, got shape {data.shape} "
            f"with sizes {sizes}")
    return RawMovie(path, RawMovieMeta(
        n_frames=int(data.shape[0]), height=int(data.shape[1]),
        width=int(data.shape[2]), source="nd2", exposure_ms=exposure, em_gain=gain,
        acquired=_clean(text.get("date")), camera=_clean(camera),
        objective=_clean(objective), reported_pixel_size_nm_UNTRUSTED=reported_nm,
        conversion_gain=_clean(conversion), readout_speed=_clean(readout),
        frame_interval_ms=interval, frame_interval_source=interval_source,
        extra={"sizes": sizes}, **optics), data)


def _nd2_objective(handle, reported_nm):
    """The objective's own record: NA, magnification, and the sensor pixel behind it.

    This is the only source for a trackmate project. `movieData.mat` is u-track's
    file, so anything tracked here will never have one, and NA (which sets the PSF and
    therefore the localisation floor) would otherwise have to be typed from memory.

    NA is read, not derived. `objectiveNumericalAperture` is recorded by the instrument,
    the same standing as the frame interval; the pixel SIZE is derived by NIS from a
    magnification that drops the tube lens, which is why that one stays untrusted.

    The emission here is the filter's, not the dye's (540 nm for a YFPHQ filter holding
    mNeonGreen, which peaks at 517) so it is returned under a name that says so and is
    never used to fill the emission box.
    """
    out = {}
    try:
        channel = handle.metadata.channels[0]
    except Exception:                              # noqa: BLE001 - optional metadata
        return out
    scope = getattr(channel, "microscope", None)
    if scope is not None:
        aperture = getattr(scope, "objectiveNumericalAperture", None)
        if aperture:
            out["numerical_aperture"] = float(aperture)
        magnification = getattr(scope, "objectiveMagnification", None)
        if magnification:
            out["objective_magnification"] = float(magnification)
            if reported_nm:
                # The claim x the magnification recovers the SENSOR pixel, because the
                # claim is exactly `sensor / magnification` with the tube lens dropped.
                out["camera_pixel_um"] = reported_nm / 1000.0 * float(magnification)
        name = getattr(scope, "objectiveName", None)
        if name:
            out["objective_name"] = str(name)
    emission = getattr(getattr(channel, "channel", None), "emissionLambdaNm", None)
    if emission:
        out["filter_emission_nm"] = float(emission)
    return out


def _nd2_frame_interval(handle):
    """The REAL frame interval out of a `.nd2`, measured rather than nominal.

    This project recorded that `.nd2` has no PER-FRAME timing, and that was wrong.
    The note read "the Nikon file offers only a NOMINAL 20 ms exposure, and this project
    has twice had to correct a 20.0 that was really 20.472", and the 20.4717 was sitting
    in the file the whole time, in two independent places:

      * `experiment[0].parameters.periodDiff.avg`, the loop's own measured period, and
      * `events()["Time [s]"]`, one timestamp per frame.

    MEASURED: one lab movie gives **20.4717 ms** from both, sd 0.0037 over 4500
    frames: exactly the hand-corrected value. protein I gives **51.7477 ms** where the
    exposure SETTING says 50. So the nominal number is wrong in both projects, by 2.4 %
    and 3.5 %, and both were recoverable without asking anybody.

    The exposure setting is NOT the frame interval: it is how long the shutter is open,
    and the readout adds to it. Never substitute one for the other.
    """
    # THE LOOP PARAMETER FIRST: it is the instrument's own average over the whole
    # acquisition, so it does not depend on reading 4500 event rows.
    try:
        for loop in handle.experiment or ():
            period = getattr(getattr(loop, "parameters", None), "periodDiff", None)
            average = getattr(period, "avg", None)
            if average and float(average) > 0:
                return float(average), "nd2 TimeLoop periodDiff"
    except Exception:                              # noqa: BLE001 - fall through
        pass
    try:
        import numpy as np

        rows = handle.events()
        times = [row.get("Time [s]") for row in rows] if rows else []
        times = [t for t in times if t is not None]
        if len(times) > 2:
            gaps = np.diff(np.asarray(times, dtype=float)) * 1000.0
            return float(np.median(gaps)), "nd2 per-frame timestamps"
    except Exception:                              # noqa: BLE001 - optional
        pass
    return None, None


def check_recorded_interval(recorded_s, movie_path, *, tolerance=0.01) -> dict:
    """Compare a recorded frame interval against what the acquisition itself says.

    A wrong frame interval is invisible downstream: every D scales as 1/tau, the
    fractions do not move at all, and nothing comes out looking wrong. This project has
    corrected 20.0 to 20.472 by hand twice; now that `.nd2` timing is read, the check is
    free wherever an acquisition sits beside the record.

    Found on real data: `D:/data/project/condition/bleaching/crops.json` records
    **20 ms** where the acquisition measures **51.7477 ms**, a factor of **2.59** on
    every D that folder would produce.

    Returns a verdict; raises nothing, because a mismatch is the USER's to resolve (the
    recorded value may be the deliberate one) and a refusal here would block a window on
    a number it cannot arbitrate.
    """
    out = {"recorded_s": None if recorded_s is None else float(recorded_s),
           "measured_s": None, "source": None, "agrees": None, "ratio": None,
           "note": ""}
    try:
        meta = read_raw_movie(movie_path).meta
    except Exception as exc:                       # noqa: BLE001 - reported, not raised
        out["note"] = (f"could not read {Path(movie_path).name} to check the frame "
                       f"interval: {type(exc).__name__}: {exc}")
        return out
    if meta.frame_interval_ms is None:
        out["note"] = (f"{Path(movie_path).name} records no per-frame timing, so the "
                       f"recorded interval cannot be checked against it")
        return out
    measured = float(meta.frame_interval_ms) / 1000.0
    out["measured_s"] = measured
    out["source"] = meta.frame_interval_source
    if recorded_s is None:
        out["note"] = f"acquisition measures {measured * 1000:.4f} ms"
        return out
    ratio = float(recorded_s) / measured if measured else float("inf")
    out["ratio"] = ratio
    out["agrees"] = abs(ratio - 1.0) <= tolerance
    if out["agrees"]:
        out["note"] = (f"frame interval agrees with the acquisition "
                       f"({measured * 1000:.4f} ms, {meta.frame_interval_source})")
    else:
        out["note"] = (
            f"Frame interval mismatch: recorded {float(recorded_s) * 1000:g} ms, but "
            f"{Path(movie_path).name} measures {measured * 1000:.4f} ms "
            f"({meta.frame_interval_source}). Every D scales as 1/tau, so this is a "
            f"factor of {1 / ratio:.3g} on every diffusion coefficient, and nothing "
            f"downstream can detect it, because the fractions do not move.")
    return out


def _read_stk(path: Path) -> RawMovie:
    """MetaMorph `.stk`: a TIFF variant, so `tifffile` already decodes it."""
    import tifffile

    with tifffile.TiffFile(str(path)) as handle:
        data = handle.asarray()
        stk = {}
        try:
            stk = dict(handle.stk_metadata or {})
        except Exception:                          # noqa: BLE001 - not every .stk has it
            stk = {}
    if data.ndim == 2:
        data = data[None, ...]
    if data.ndim != 3:
        raise RawMovieError(f"{path.name}: expected a time stack, got shape {data.shape}")

    exposure = stk.get("Exposure") or stk.get("ExposureTime")
    if exposure is not None:
        exposure = float(np.ravel(exposure)[0])

    (exposure, em_gain, conversion, interval,
     interval_source) = stk_settings(stk, exposure)

    created = stk.get("DateTime") or stk.get("CreateTime")
    return RawMovie(path, RawMovieMeta(
        n_frames=int(data.shape[0]), height=int(data.shape[1]),
        width=int(data.shape[2]), source="stk", exposure_ms=exposure,
        em_gain=em_gain,
        conversion_gain=conversion,
        acquired=_clean(str(created) if created is not None else None),
        camera=_clean(stk.get("CameraName")),
        frame_interval_ms=interval, frame_interval_source=interval_source,
        extra={"stk_keys": sorted(stk)[:20]}), data)



def frame_interval(path):
    """``(ms, source)``: a raw acquisition's MEASURED frame interval, from the file's metadata only (never a frame;
    `read_raw_movie` reads a whole `.stk`). `.nd2`: its time loop, else its per-frame times (`_nd2_frame_interval`);
    `.stk`: the average spacing of its per-plane timestamps (`stk_settings`). ``(None, None)`` when the file holds
    no timing -- the exposure is the shutter, not the interval, and is never returned as one. Raises
    `RawMovieError` when the file cannot be opened."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".nd2":
        try:
            import nd2
        except ImportError as exc:                 # pragma: no cover - env dependent
            raise RawMovieError("reading .nd2 needs the `nd2` package") from exc
        try:
            with nd2.ND2File(str(path)) as handle:
                return _nd2_frame_interval(handle)
        except Exception as exc:                   # noqa: BLE001 - re-raised, named
            raise RawMovieError(f"{path.name}: cannot read its metadata: {exc}") from exc
    if suffix == ".stk":
        import tifffile

        try:
            with tifffile.TiffFile(str(path)) as handle:
                stk = dict(handle.stk_metadata or {})
        except Exception as exc:                   # noqa: BLE001 - re-raised, named
            raise RawMovieError(f"{path.name}: cannot read its metadata: {exc}") from exc
        *_, interval, source = stk_settings(stk)
        return (interval, source) if source == "per-frame timestamps" else (None, None)
    return None, None


def exposure_and_interval(path) -> tuple:
    """``(exposure_ms, interval_ms)`` of a raw acquisition, from its metadata only: how long the camera was open each
    frame (the exposure SETTING) and the MEASURED frame interval (`frame_interval`), each None where the file holds
    none. The two differ where the camera waits between exposures (one user's movies: 20 ms open in each 25.0 ms frame), and
    the jump angles' chance copies that (`hmm_angles.exposure_for`). Raises `RawMovieError` when the file cannot be
    opened."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".nd2":
        try:
            import nd2
        except ImportError as exc:                 # pragma: no cover - env dependent
            raise RawMovieError("reading .nd2 needs the `nd2` package") from exc
        try:
            with nd2.ND2File(str(path)) as handle:
                text = dict(handle.text_info or {})
                interval, _source = _nd2_frame_interval(handle)
        except Exception as exc:                   # noqa: BLE001 - re-raised, named
            raise RawMovieError(f"{path.name}: cannot read its metadata: {exc}") from exc
        exposure = _search_float(r"Exposure:\s*([\d.]+)\s*ms", str(text.get("description", "")))
        return exposure, interval
    if suffix == ".stk":
        import tifffile

        try:
            with tifffile.TiffFile(str(path)) as handle:
                stk = dict(handle.stk_metadata or {})
        except Exception as exc:                   # noqa: BLE001 - re-raised, named
            raise RawMovieError(f"{path.name}: cannot read its metadata: {exc}") from exc
        exposure = stk.get("Exposure") or stk.get("ExposureTime")
        if exposure is not None:
            exposure = float(np.ravel(exposure)[0])
        exposure, _gain, _conversion, interval, source = stk_settings(stk, exposure)
        return exposure, (interval if source == "per-frame timestamps" else None)
    return None, None


def stk_settings(stk: dict, exposure=None):
    """The acquisition settings MetaMorph actually writes, out of a `.stk`'s metadata.

    Pure, so it is testable without a `.stk` on disk - which matters, because the
    microscope drives here are external and not always mounted.

    Two things the plain keys miss. MetaMorph stores the per-plane settings as FREE
    TEXT, one string per frame, and the numeric keys (`Exposure`, `Gain`) are usually
    absent; the fields are newline-separated, e.g.
    ``Exposure: 20 ms / Binning: 1 / Digitizer: 16 bits / Gain: Gain 0 /
    EMCCD Gain: 141``.

    And `DatetimeCreated` holds one timestamp per frame, so the frame interval can be
    MEASURED rather than taken from the exposure. They are not the same number: a 20 ms
    exposure can sit in a 20.472 ms frame, and every D scales with the frame interval.
    Measured on the LytE project, whose `movieData.mat` omits `timeInterval_`
    altogether: 20.000 ms in all six groups, which is what made it analysable.

    Returns ``(exposure_ms, em_gain, conversion_gain, interval_ms, interval_source)``.
    """
    import numpy as np

    # `x or []` cannot be used here: these values are numpy arrays and the truth
    # test on one raises rather than falling back.
    raw_descriptions = stk.get("PlaneDescriptions")
    descriptions = np.ravel([] if raw_descriptions is None else raw_descriptions)
    description = str(descriptions[0]) if descriptions.size else ""
    if exposure is None:
        exposure = _search_float(r"Exposure:\s*([\d.]+)\s*ms", description)
    em_gain = _search_float(r"EMCCD Gain:\s*([\d.]+)", description)
    if em_gain is None:
        em_gain = _first_float(stk.get("Gain"))
    conversion = _clean(_search_str(r"(?<!EMCCD )Gain:\s*(Gain \d+)", description))

    interval = interval_source = None
    raw_stamps = stk.get("DatetimeCreated")
    stamps = np.ravel([] if raw_stamps is None else raw_stamps)
    if stamps.size > 1:
        # THE AVERAGE SPACING, first stamp to last over the frames between. MetaMorph stamps whole
        # milliseconds, so the MEDIAN step was the nearest whole ms, not the interval: protein E/Glucose read 25.000
        # where its 6000 frames span 25.298 ms each (u-track recorded 25.15), one user's series002 26.000 where they
        # span 25.824 (measured 20 Oct 2026). Every D scales with it.
        times = np.asarray(stamps, dtype="datetime64[ns]")
        span = (times[-1] - times[0]).astype("timedelta64[ns]").astype(float)
        interval = float(span / (times.size - 1) / 1e6)
        interval_source = "per-frame timestamps"
    elif exposure is not None:
        # NAMED AS NOMINAL, because it is the exposure and not the interval. A caller
        # that cannot tell them apart will quote a D that is wrong by the readout gap.
        interval, interval_source = float(exposure), "exposure (nominal)"
    return exposure, em_gain, conversion, interval, interval_source


def _read_tiff(path: Path) -> RawMovie:
    """A plain multi-page TIFF, the lowest common denominator."""
    import tifffile

    data = tifffile.imread(str(path))
    if data.ndim == 2:
        data = data[None, ...]
    if data.ndim != 3:
        raise RawMovieError(f"{path.name}: expected a time stack, got shape {data.shape}")
    return RawMovie(path, RawMovieMeta(
        n_frames=int(data.shape[0]), height=int(data.shape[1]),
        width=int(data.shape[2]), source="tiff"), data)


_READERS = {".nd2": _read_nd2, ".stk": _read_stk, ".tif": _read_tiff,
            ".tiff": _read_tiff}

#: What each reader needs at RUN time, so availability can be reported rather than
#: discovered when a run fails. `.stk` is a TIFF variant, which is why the MetaMorph
#: scope costs no dependency beyond the one already required.
_READER_REQUIRES = {".nd2": "nd2", ".stk": "tifffile", ".tif": "tifffile",
                    ".tiff": "tifffile"}


def readable_formats() -> dict:
    """Which raw formats THIS INSTALL can actually read, and what is missing.

    A packaged application must be able to answer this BEFORE a run, not during one.
    ``nd2`` is an optional extra (`pip install .[raw]`) and can be absent from a frozen
    bundle without anything failing at import (the import is lazy) so the first sign
    would otherwise be a user's acquisition refusing to open.

    Returns ``{suffix: {"available": bool, "requires": str, "why": str}}``.
    """
    import importlib.util

    out = {}
    for suffix, module in sorted(_READER_REQUIRES.items()):
        try:
            spec = importlib.util.find_spec(module)
            ok = spec is not None
            why = "" if ok else f"{module} is not installed"
        except (ImportError, ValueError) as exc:      # pragma: no cover - defensive
            ok, why = False, f"{module}: {exc}"
        out[suffix] = {"available": bool(ok), "requires": module, "why": why}
    return out


def describe_formats() -> str:
    """`readable_formats` as text, for the CLI and for a support request."""
    got = readable_formats()
    lines = ["raw acquisition formats this install can read:"]
    for suffix, info in got.items():
        mark = "yes" if info["available"] else "NO "
        note = "" if info["available"] else f"   <- {info['why']}"
        lines.append(f"  {suffix:6s} {mark}  (needs {info['requires']}){note}")
    missing = [s for s, i in got.items() if not i["available"]]
    if missing:
        lines.append("")
        lines.append(f"Missing: {', '.join(missing)}. .nd2 files come from Nikon NIS-Elements and "
                     ".stk files from MetaMorph;")
        lines.append("install with:  pip install 'smtracker[raw]'")
    return "\n".join(lines)


def read_raw_movie(path, *, source: str | None = None) -> RawMovie:
    """Read a raw acquisition from either microscope into one interface.

    ``source`` forces a reader ("nd2" | "stk" | "tiff"); by default the suffix decides.
    An unknown suffix is NAMED rather than guessed at, because silently handing a file
    to the wrong decoder produces an array rather than an error.
    """
    path = Path(path)
    if not path.exists():
        raise RawMovieError(f"no such raw movie: {path}")
    if source is not None:
        reader = {"nd2": _read_nd2, "stk": _read_stk, "tiff": _read_tiff}.get(source)
        if reader is None:
            raise RawMovieError(
                f"source must be 'nd2', 'stk' or 'tiff', got {source!r}")
    else:
        reader = _READERS.get(path.suffix.lower())
        if reader is None:
            raise RawMovieError(
                f"{path.name}: no reader for {path.suffix!r}. Supported: "
                f"{', '.join(sorted(_READERS))}: .nd2 from Nikon NIS-Elements, .stk "
                "from MetaMorph. Pass source= to force one")
    return reader(path)


# ----------------------------------------------------------- the dark detector
def detect_dark_frames(movie: RawMovie, *, search: int = 600,
                       lit_fraction: float = LIT_FRACTION_OF_RANGE,
                       min_contrast: float = MIN_CONTRAST) -> dict:
    """How many frames at the start were taken with the laser OFF.

    Detected, not assumed, because the legacy's fixed 20 is wrong on most of this data
    (see the module head: 32, 12 and 1 dark frames on Malate, LB and Glucose). Returns
    ``n_dark``, the frame the laser fires on (1-based), the dark level and the step size,
    so a caller can see WHY the boundary landed where it did.

    The step is a factor of ~26 here, so the threshold is not delicate; a movie with no
    step at all is reported as such rather than being given a boundary it does not have.
    """
    means = movie.frame_means(upto=min(search, len(movie)))
    if means.size < 2:
        raise RawMovieError(f"{movie.path.name}: too few frames to find a dark region")
    floor = float(means.min())
    ceiling = float(means.max())
    if floor <= 0:
        raise RawMovieError(f"{movie.path.name}: non-positive frame mean")

    contrast = ceiling / floor
    if contrast < float(min_contrast):
        return {"n_dark": 0, "laser_on_frame": 1, "dark_level": float("nan"),
                "step_factor": contrast, "found_step": False, "searched": int(means.size),
                "threshold": float("nan"),
                "reason": (f"no off-to-on step in the first {means.size} frames "
                           f"(the brightest frame only {fold_words(contrast, 2, of=True)} the darkest, needs "
                           f"{fold_words(min_contrast, of=True)}): "
                           "either the laser was already on when recording started, or "
                           "it was switched on later than the search window")}

    # A THRESHOLD SCALED TO THIS MOVIE'S OWN RANGE. See LIT_FRACTION_OF_RANGE: the step
    # is 26x on a labelled sample and 2.6x on the unlabelled control, so no fixed factor
    # covers both, while a small fraction of each movie's own contrast covers both easily.
    threshold = floor + float(lit_fraction) * (ceiling - floor)
    lit = means > threshold
    n_dark = int(np.argmax(lit)) if lit.any() else int(means.size)
    if n_dark == 0:
        return {"n_dark": 0, "laser_on_frame": 1, "dark_level": float("nan"),
                "step_factor": contrast, "found_step": False, "searched": int(means.size),
                "threshold": threshold,
                "reason": ("the first frame is already lit, so the laser was already on "
                           "when recording started and this movie carries no dark "
                           "reference")}
    return {
        "n_dark": n_dark,
        "laser_on_frame": n_dark + 1,                  # 1-based, as the frames are
        "dark_level": float(means[:n_dark].mean()),
        "step_factor": contrast,
        "threshold": threshold,
        "found_step": True,
        "searched": int(means.size),
        "reason": (f"laser fires at frame {n_dark + 1}; dark {means[:n_dark].mean():.0f}, "
                   f"the brightest frame {fold_words(contrast, 1, of=True)} the darkest"),
    }


def instrumental_offset(movie: RawMovie, *, per_pixel: bool = False,
                        min_frames: int = MIN_FRAMES_FOR_PIXEL_MAP,
                        lit_fraction: float = LIT_FRACTION_OF_RANGE,
                        min_contrast: float = MIN_CONTRAST):
    """The camera's dark level, from the frames taken before the laser fired.

    ``per_pixel=False`` (default) returns a scalar, which one frame supports perfectly
    well: its mean is over ~16 000 pixels. ``per_pixel=True`` returns a dark IMAGE and
    REFUSES below ``min_frames``, because a map built from one exposure is that
    exposure's shot noise wearing a calibration's name.
    """
    found = detect_dark_frames(movie, lit_fraction=lit_fraction,
                               min_contrast=min_contrast)
    n_dark = found["n_dark"]
    if not n_dark:
        raise RawMovieError(
            f"{movie.path.name}: no dark frames before the laser fired, so this "
            "acquisition carries no instrumental background. Use a no-cell control "
            f"instead (detector saw a step factor of {found['step_factor']:.1f})")
    if not per_pixel:
        return float(found["dark_level"]), found
    if n_dark < int(min_frames):
        raise RawMovieError(
            f"{movie.path.name}: {n_dark} dark frame(s), below the {min_frames} a "
            "per-pixel dark map needs. The scalar level is still available "
            "(per_pixel=False) and one frame supports it: its mean is over the whole "
            "sensor. Do not build a map from this")
    block = np.stack([movie.frame(i) for i in range(n_dark)])
    return block.mean(axis=0), found


# ------------------------------------------------------------------- utilities
def _clean(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _search_float(pattern: str, text: str):
    hit = re.search(pattern, text)
    return float(hit.group(1)) if hit else None


def _search_str(pattern: str, text: str):
    hit = re.search(pattern, text)
    return hit.group(1).strip() if hit else None


def _first_float(value):
    if value is None:
        return None
    try:
        return float(np.ravel(value)[0])
    except Exception:                              # noqa: BLE001 - metadata is untidy
        return None


def find_crop_offset(movie: RawMovie, cropped_means, *, probe: int = 24,
                     tolerance: float = 0.02) -> dict:
    """Where the CROPPED stack sits inside this raw acquisition.

    The cropped stack is what u-track saw and it carries no record of where it was cut
    from. But the two hold the same frames, so their per-frame means match exactly at the
    right offset and nowhere else -- the bleaching decay is monotonic, so the trace is a
    fingerprint rather than a repeating pattern. Recovering the offset by MATCHING is
    therefore exact, where assuming "the last N frames" is a guess that happens to be
    right on the movie someone checked.

    MEASURED on one lab movie: raw frame 4500 has mean 1129.3 and the cropped
    stack's last frame has mean 1129.3, so that crop is the final 1001 of 4500.

    ``cropped_means`` is the mean of each frame of the cropped stack (see
    `spot_intensity.movie_frame_paths` to list them). ``probe`` evenly spaced frames are
    compared, which is ample against a monotonic trace and keeps this cheap.

    Returns the offset, the residual at it, and the runner-up, so a caller can see that
    the match is unambiguous rather than merely best.
    """
    cropped = np.asarray(cropped_means, dtype=np.float64)
    if cropped.ndim != 1 or cropped.size < 2:
        raise RawMovieError("cropped_means must be a 1-D trace of at least 2 frames")
    n_raw, n_crop = len(movie), int(cropped.size)
    if n_crop > n_raw:
        raise RawMovieError(
            f"the cropped stack has {n_crop} frames and the raw only {n_raw}; these are "
            "not the same acquisition")

    raw_means = movie.frame_means()
    idx = np.unique(np.linspace(0, n_crop - 1, min(probe, n_crop)).round().astype(int))
    want = cropped[idx]
    scale = float(np.mean(np.abs(want))) or 1.0

    offsets = np.arange(0, n_raw - n_crop + 1)
    residuals = np.array([np.mean(np.abs(raw_means[o + idx] - want)) for o in offsets])
    best = int(np.argmin(residuals))
    order = np.argsort(residuals)
    runner = int(order[1]) if residuals.size > 1 else best
    rel = float(residuals[best]) / scale
    return {
        "offset": int(offsets[best]),
        "first_raw_frame": int(offsets[best]) + 1,     # 1-based
        "residual": float(residuals[best]),
        "relative_residual": rel,
        "runner_up_residual": float(residuals[runner]),
        "unambiguous": bool(rel < tolerance
                            and residuals[runner] > 4 * max(residuals[best], 1e-9)),
        "is_tail": bool(int(offsets[best]) == n_raw - n_crop),
        "n_raw": n_raw, "n_cropped": n_crop,
    }


#: The settings that SCALE COUNTS, so a calibration taken under different ones does not
#: transfer. Exposure sets how many photons arrive, the EM multiplier amplifies them, and
#: the conversion gain sets electrons per ADU, change any and the same molecule reads a
#: different number. MEASURED across all 127 movies here (samples plus both controls):
#: EM 140, "Gain 3" and 20 ms in every single one, so on THIS data the calibration does
#: transfer. Readout speed is deliberately NOT in this list: it sets read NOISE rather
#: than signal scale. One movie differs on it (one movie at 10 MHz against
#: 17 MHz everywhere else, 25.02.2026) and that is reported rather than refused.
#:
#: MEASURED rather than reasoned, on all 28 movies of its condition: that movie's slow fraction
#: sits dead centre of the other 27 (z = -0.01), and its
#: localisation sigma is 22.0 nm against 18.8 +/- 3.9 (z = +0.81) -- SLIGHTLY HIGHER, not
#: lower. The "lower read noise makes it marginally better" prediction is not visible in
#: the data; whatever readout speed contributes is swamped by movie-to-movie variation.
#: Both sit well inside the spread, so the movie is sound and the mechanism story was
#: simply not supported.
SCALING_SETTINGS = ("exposure_ms", "em_gain", "conversion_gain")


def compare_settings(a: RawMovieMeta, b: RawMovieMeta) -> dict:
    """Do two acquisitions share the settings that scale counts?

    Returns the differing fields, split into those that INVALIDATE a shared calibration
    and those that are merely worth knowing. Laser power is absent from both because the
    file does not record it: that one stays a declared, no-default parameter.
    """
    scaling, informational = {}, {}
    for field in SCALING_SETTINGS:
        x, y = getattr(a, field, None), getattr(b, field, None)
        if x != y:
            scaling[field] = (x, y)
    for field in ("readout_speed", "camera", "objective"):
        x, y = getattr(a, field, None), getattr(b, field, None)
        if x != y:
            informational[field] = (x, y)
    return {"comparable": not scaling, "scaling_differs": scaling,
            "informational_differs": informational}


def require_comparable(a: RawMovieMeta, b: RawMovieMeta) -> None:
    """Refuse two acquisitions whose count scale differs. Not a warning: the same
    molecule reads a different number, and every downstream intensity silently shifts."""
    got = compare_settings(a, b)
    if not got["comparable"]:
        detail = "; ".join(f"{k}: {x!r} vs {y!r}"
                           for k, (x, y) in got["scaling_differs"].items())
        raise RawMovieError(
            "these acquisitions do not share the settings that scale counts, so a "
            f"calibration from one cannot be applied to the other: {detail}")


#: Suffixes that hold a real ACQUISITION rather than a companion image. A `.tif` can be
#: either, an already-extracted stack, or the phase snapshot saved beside the movie,
#: which is exactly the ambiguity `discover_raw_movies` exists to resolve.
ACQUISITION_SUFFIXES = frozenset({".nd2", ".stk"})

#: The folders whose FILE NAMES are the movies' own names, IN ORDER OF AUTHORITY: the
#: cell outlines first, because they are what the analysis matches a movie against, then
#: the phase images they are segmented from, the prepared stacks, and last the sum images,
#: which this tool itself writes and so may carry a name it gave before this rule existed.
NAMED_COMPANIONS = (("cell_meshes", ".mat"), ("phase", ".tif"), ("tif", ".tif"),
                    ("sum", ".tif"))
#: Folders a movie's files sit in BELOW its condition folder.
_BELOW_CONDITION = ("raw", "trackmate", "bleaching")
#: Never a movie, though it sits beside the outlines (u-track's own project file).
_NOT_MOVIES = frozenset({"movieList"})


def condition_folder_of(path):
    """The condition folder a movie's file belongs to: the parent, or one level up from
    `raw/`, `trackmate/` or `bleaching/`."""
    from pathlib import Path

    parent = Path(path).parent
    return parent.parent if parent.name.lower() in _BELOW_CONDITION else parent


#: How long one listing of a condition's companion names is reused, in seconds.
#: A TIME, NOT A FOLDER TIMESTAMP: exFAT (the portable SSD one user's data is on) does not
#: move a folder's modified time when a file is added, so a cache keyed on it would serve
#: a stale listing for good. Two seconds spares the panel listing four folders per movie
#: per refresh, and nothing in a session renames a movie faster than that.
_KNOWN_TTL_S = 2.0
_KNOWN_CACHE: dict = {}


def _list_known_names(folder) -> tuple:
    out = []
    from .cellmesh import CELL_MESH_FOLDER, mesh_folder

    for sub, suffix in NAMED_COMPANIONS:
        where = mesh_folder(folder) if sub == CELL_MESH_FOLDER else folder / sub
        names = set()
        if where.is_dir():
            names = {p.stem for p in where.iterdir()
                     if p.is_file() and p.suffix.lower() == suffix
                     and not p.name.startswith("._")}
            if sub == "phase":
                names = {_snapshot_name(n) for n in names}
        elif sub == "phase":
            names = _loose_snapshot_names(folder)
        out.append(frozenset(names - _NOT_MOVIES))
    return tuple(out)


def _snapshot_name(stem: str) -> str:
    """``series001.nd2`` (the stem of an export that kept the suffix) is ``series001``."""
    from .segmentation.channels import snapshot_field

    return snapshot_field(stem + ".tif")


def _loose_snapshot_names(folder) -> set:
    """The names the phase snapshots lying BESIDE the movies carry, where no ``phase/``
    folder exists (ruled 18 Oct 2026: either layout; handoff 18 Oct, place 4).

    They are what `discover_raw_movies` leaves out as images, never movies: a TIFF of an
    acquisition's own name, and a one-page TIFF beside acquisitions -- MetaMorph's
    ``series001.tif`` beside ``series001_YFPLaser.stk``, whose name is the one this vote
    exists for. A folder without acquisitions gives nothing, so ``.tif`` movies never vote.
    Unreadable gives nothing too: the other folders still vote."""
    try:
        _kept, images = discover_raw_movies(folder)
    except (RawMovieError, OSError):
        return set()
    return {_snapshot_name(p.stem) for p in images
            if p.suffix.lower() in (".tif", ".tiff") and not p.name.startswith("._")}


def _known_by_folder(folder) -> tuple:
    """The companion names, one set per `NAMED_COMPANIONS` folder, in its order."""
    import time
    from pathlib import Path

    folder = Path(folder)
    now = time.monotonic()
    hit = _KNOWN_CACHE.get(str(folder))
    if hit is not None and now - hit[0] < _KNOWN_TTL_S:
        return hit[1]
    names = _list_known_names(folder)
    _KNOWN_CACHE[str(folder)] = (now, names)
    return names


def known_movie_names(folder) -> frozenset:
    """Every movie name the condition folder already uses (`NAMED_COMPANIONS`)."""
    return frozenset().union(*_known_by_folder(folder))


#: A movie's SERIES NUMBER in the names its files go by: a numbered snapshot ``001``, ``series001``, MetaMorph's
#: ``series001_YFPLaser`` (the illumination appended), ``series0010`` (four digits). Nothing else matches -- a
#: frame file ``series001_c1_t0001`` would, but is never among the names compared.
_SERIES_NUMBER = re.compile(r"^(?:series)?(\d{1,4})(?:_[A-Za-z][^\s()]*)?$", re.IGNORECASE)


def movie_number(name) -> int | None:
    """The series number ``name`` carries AS A WHOLE NAME (`_SERIES_NUMBER`), or None.

    Not `grouping.series_number`, which reads the FIRST run of digits for a replicate day and is right to be lenient
    there: pairing two FILES by it would call ``20260904_series001`` movie 20260904 and make ``series002 (rep1)`` one
    of a pair. A name that is not a series name has no number here, and is paired by its name or not at all."""
    match = _SERIES_NUMBER.match(str(name).strip())
    return int(match.group(1)) if match else None


def same_movie(name, candidates) -> str | None:
    """``name`` among ``candidates``, else the ONE candidate with the same series number, else None.

    Asked 23 Oct 2026 ("make it clear that the analysis tool finds movies, phase and sum image"): one user's
    MetaMorph folders name the phase snapshot ``001.tif`` while the movie, its sum image and its tracks are
    ``series001_YFPLaser``, so a lookup by name paired nothing -- the outlines window found no sum for any field
    and its outlines would have been filed under ``001``, which the analysis never reads. Two candidates with one
    number are never chosen between: that is a folder to look at, not a guess to make.
    """
    names = [str(c) for c in candidates]
    if str(name) in names:
        return str(name)
    number = movie_number(name)
    if number is None:
        return None
    hits = sorted({c for c in names if movie_number(c) == number})
    return hits[0] if len(hits) == 1 else None


def movie_names(stems, folder) -> dict:
    """`{file stem: movie name}` for SIBLING files of one condition folder.

    MetaMorph names a file after the acquisition AND its illumination setting:
    `series001_YFPLaser.stk`, while the lab macro's prepared stack, the phase and sum
    images and the cell outline are all `series001`. Named after its file, the movie
    matched nothing: crop ranges and tracks were filed as `series001_YFPLaser`, and the
    analysis found no outline for any movie (measured on `a lab's MetaMorph folders`, 32 of 32).

    So a stem that is an existing name plus `_<something>` IS that movie. The candidates
    are the stem itself and every existing name it extends; the companion folders are
    asked IN ORDER OF AUTHORITY (`NAMED_COMPANIONS`) and the first one holding exactly one
    candidate decides. That order matters on one user's own folders: this tool had already
    written `sum/series001_YFPLaser.tif`, so the old name existed too, and only the
    outlines (`series001.mat`) say which name the analysis needs. A folder with no
    companions keeps every stem (it names its companions after them later), a folder
    holding BOTH candidates in its most authoritative set keeps the stem, and two files
    that would land on ONE name both keep their own stems -- this never merges two
    acquisitions into a movie.
    """
    from collections import Counter

    layers = _known_by_folder(folder)
    every = frozenset().union(*layers)
    out = {}
    for stem in stems:
        candidates = {stem} | {name for name in every if stem.startswith(name + "_")}
        out[stem] = stem
        for names in layers:
            hits = candidates & names
            if len(hits) == 1:
                out[stem] = next(iter(hits))
                break
            if len(hits) > 1:
                break
    counts = Counter(out.values())
    return {stem: (name if counts[name] == 1 else stem) for stem, name in out.items()}


def claimed_twice(stems, folder) -> dict:
    """`{movie name: [stems]}` for every movie that MORE THAN ONE sibling file claims.

    `movie_names` never merges two files into one movie, so each keeps its stem. For two
    ACQUISITIONS that is right; for two TRACK FILES it hides the fact that one movie was
    tracked twice -- found on one user's folders, where a run under the old naming left
    `trackmate/series001_YFPLaser.xml` beside the new `series001.xml`. Kept apart, the old
    one became a twelfth movie with no outline. A reader of track files asks this first.
    """
    from collections import defaultdict

    layers = _known_by_folder(folder)
    every = frozenset().union(*layers)
    claims = defaultdict(list)
    for stem in stems:
        candidates = {stem} | {name for name in every if stem.startswith(name + "_")}
        claimed = stem
        for names in layers:
            hits = candidates & names
            if len(hits) == 1:
                claimed = next(iter(hits))
                break
            if len(hits) > 1:
                break
        claims[claimed].append(stem)
    return {name: sorted(group) for name, group in claims.items() if len(group) > 1}


def movie_name(path) -> str:
    """The movie a raw acquisition, a prepared stack or a TrackMate XML IS
    (`movie_names`), from its path. Never raises: an unreadable folder keeps the stem."""
    from pathlib import Path

    path = Path(path)
    try:
        siblings = set(_sibling_stems(path.parent, path.suffix.lower()))
        siblings.add(path.stem)
        return movie_names(sorted(siblings), condition_folder_of(path))[path.stem]
    except OSError:
        return path.stem


#: ``{(folder, suffix): (time, stems)}`` -- the siblings `movie_name` compares against.
_SIBLING_CACHE: dict = {}


def _sibling_stems(folder, suffix: str) -> frozenset:
    """The stems of every ``suffix`` file in ``folder``, on the same two-second listing
    cache as the companions (`_known_by_folder`).

    MEASURED: `movie_name` listed and STAT'ed every sibling on every call, so a window
    naming 32 movies on each redraw made about a thousand stats per redraw, and on one user's
    exFAT USB drive that froze the preparation window for 0.5-1 s at a time. A listing
    is a property of the folder, not of the call.
    """
    import time

    key = (str(folder), suffix)
    now = time.monotonic()
    hit = _SIBLING_CACHE.get(key)
    if hit is not None and now - hit[0] < _KNOWN_TTL_S:
        return hit[1]
    stems = frozenset(p.stem for p in folder.iterdir()
                      if p.is_file() and p.suffix.lower() == suffix)
    _SIBLING_CACHE[key] = (now, stems)
    return stems


def movie_named_crops(crops: dict, crops_path) -> dict:
    """Crop ranges keyed by MOVIE name. A set saved before `movie_names` existed keys a
    MetaMorph file's stem (`series001_YFPLaser`); read through this it keys `series001`
    and is found again, so nobody re-crops. A key that already is a movie name wins over
    an old stem for the same movie."""
    from pathlib import Path

    folder = condition_folder_of(Path(crops_path))
    raw = folder / "raw" if (folder / "raw").is_dir() else folder
    try:
        stems = [p.stem for p in raw.iterdir()
                 if p.is_file() and p.suffix.lower() in ACQUISITION_SUFFIXES]
    except OSError:
        return dict(crops)
    names = movie_names(stems, folder)
    out = {key: span for key, span in crops.items() if key not in names
           or names[key] == key}
    for key, span in crops.items():
        out.setdefault(names.get(key, key), span)
    return out


def discover_raw_movies(folder, *, report=None):
    """The raw movies in a folder, with companion images left out.

    One scan, three callers. `movieprep.prepare_batch`, `prep_gui` and `tracking_gui`
    each listed the folder themselves with the same rule (every file whose suffix a
    reader claims) and that rule is wrong for the layout a microscope actually leaves
    behind:

        series001.nd2   the acquisition, 4500 frames
        series001.tif   the phase snapshot, ONE page

    `.tif` is readable, so all three listed **twenty** movies where there are ten, and
    the two entries share a stem, so the crop mapping and the outputs collide as well.
    Measured on `D:/data/project/condition`: ten `.nd2` of 4500 frames beside ten
    single-page `.tif` of the same names.

    The rule is stem collision, not "tiffs are not movies", because a `.tif` genuinely
    can be a movie: an already-extracted stack is how TrackMate data and the prepared
    stacks arrive. A `.tif` is dropped only when an ACQUISITION of the same stem sits
    beside it, where it can only be a companion. A folder of `.tif` stacks alone is
    untouched.

    `report` (optional callable) receives one line per skipped file, so the omission is
    stated rather than silent: a movie that vanishes from a listing reads as missing
    data.

    TWO MORE SHAPES, found on one user's folders (12 Oct 2026), where the crop stage gave
    stray images the "range" 1-1 and u-track would have been handed one-frame movies:
    ``series001.nd2.tif`` (an export that kept the acquisition's suffix in its name) is
    that acquisition's companion like ``series001.tif``; and in a folder that HOLDS
    acquisitions, a ONE-PAGE image (``S.tif``, ``series00keineahnung.tif``) is never a
    movie, since a single frame cannot be tracked. A folder of `.tif` alone is still not
    opened at all, and a file that cannot be read as a TIFF is kept so its reader fails
    by name.
    """
    from pathlib import Path

    folder = Path(folder)
    if not folder.is_dir():
        raise RawMovieError(f"not a folder: {folder}")

    readable = [p for p in sorted(folder.iterdir())
                if p.is_file() and p.suffix.lower() in _READERS]
    acquisitions = {p.stem for p in readable
                    if p.suffix.lower() in ACQUISITION_SUFFIXES}

    kept, skipped, single = [], [], []
    for path in readable:
        if path.suffix.lower() not in ACQUISITION_SUFFIXES:
            inner = Path(path.stem)
            if path.stem in acquisitions or (inner.suffix.lower() in ACQUISITION_SUFFIXES
                                             and inner.stem in acquisitions):
                skipped.append(path)
                continue
            if acquisitions and _one_page(path):
                single.append(path)
                continue
        kept.append(path)

    if skipped and report is not None:
        report(f"{len(skipped)} companion image(s) beside a raw acquisition of the "
               f"same name, not treated as movies: "
               + ", ".join(p.name for p in skipped[:6])
               + (" ..." if len(skipped) > 6 else ""))
    if single and report is not None:
        report(f"{len(single)} single image(s) beside raw acquisitions, not treated as "
               f"movies (one page cannot be tracked): "
               + ", ".join(p.name for p in single[:6])
               + (" ..." if len(single) > 6 else ""))
    return kept, skipped + single


def _one_page(path) -> bool:
    """True when ``path`` is a TIFF holding ONE image; False for anything else, including
    a file that is not a readable TIFF (its reader then fails by name).

    Counted from the image SERIES, never `len(pages)`: tifffile indexes only the first
    page of a contiguously written stack, so a 3-frame stack reads as one page."""
    try:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            frames = 1
            for axis, size in zip(series.axes, series.shape):
                if axis not in "YXS":
                    frames *= int(size)
            return frames == 1
    except Exception:                                  # noqa: BLE001 - kept, read later
        return False


#: The date layouts an instrument has been seen to write, tried in order. NIS writes its
#: locale's form ("25.06.2025  16:05:28", two spaces) and a US install writes
#: "6/25/2025 4:05:28 PM"; the separator tells the two apart, so neither is a guess.
_DATE_FORMATS = ("%d.%m.%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S",
                 "%Y-%m-%d %H:%M:%S")


def _parse_instrument_date(text):
    """A datetime from an instrument's date string, or None where the layout is unknown."""
    import datetime as _dt

    if text is None:
        return None
    flat = " ".join(str(text).split())
    for fmt in _DATE_FORMATS:
        try:
            return _dt.datetime.strptime(flat, fmt)
        except ValueError:
            continue
    try:
        return _dt.datetime.fromisoformat(flat)
    except ValueError:
        return None


def acquisition_time(path):
    """When the instrument started recording this movie, from the file's own metadata.

    Reads metadata only, never a frame: ~15 ms per `.nd2` once the package is loaded.

    THIS IS THE DATE THE REPLICATE DAYS NEED, AND `movieData.mat` OFTEN DOES NOT HOLD IT.
    u-track records ``acquisitionDate_`` only when it was given one; otherwise it holds
    ``createTime_``, the moment the movie was IMPORTED. Measured on one lab project, all 29
    movies were imported on 9 March 2026 -- series041-050 at 07:36, series001-009 at 08:18,
    series031-040 at 15:19 -- while the `.nd2` files say 25.06.2025, 04.03.2026 and
    25.02.2026. Clustered by import time, two sessions eight months apart became one day.

    Returns None where the file records no date or writes it in a layout not listed in
    `_DATE_FORMATS`: an unread date is a missing date, never a guessed one. Raises
    `RawMovieError` where the file cannot be opened, or `.nd2` support is not installed.
    """
    import datetime as _dt

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".nd2":
        try:
            import nd2
        except ImportError as exc:                 # pragma: no cover - env dependent
            raise RawMovieError("reading .nd2 needs the `nd2` package") from exc
        try:
            with nd2.ND2File(str(path)) as handle:
                text = dict(handle.text_info or {})
        except Exception as exc:                   # noqa: BLE001 - re-raised, named
            raise RawMovieError(f"{path.name}: cannot read its metadata: {exc}") from exc
        return _parse_instrument_date(text.get("date"))
    if suffix == ".stk":
        import tifffile

        try:
            with tifffile.TiffFile(str(path)) as handle:
                stk = dict(handle.stk_metadata or {})
        except Exception as exc:                   # noqa: BLE001 - re-raised, named
            raise RawMovieError(f"{path.name}: cannot read its metadata: {exc}") from exc
        # One timestamp per plane; the first is when recording began.
        per_plane = stk.get("DatetimeCreated")
        if per_plane is not None and len(per_plane):
            first = np.datetime64(np.ravel(per_plane)[0], "us").astype(_dt.datetime)
            if isinstance(first, _dt.datetime):
                return first
        created = stk.get("CreateTime")
        if isinstance(created, _dt.datetime):
            return created
        return _parse_instrument_date(created)
    return None
