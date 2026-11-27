"""The optics that set the PSF: numerical aperture and emission wavelength.

Why they belong in the tracking stage. Together they fix the width of a single
molecule's image, and that width sets two things this pipeline currently takes on faith:

* the detection radius. TrackMate's LoG responds best when its scale matches the spot,
  and the radius here has been 3 px because that is what was used, not because anything
  derived it.
* the localisation floor. `AcquisitionParams` already turns a PSF width into the floor
  below which a `D` is a bound rather than a measurement, and on this data that decides
  whether the slow state can be quoted at all: at sigma 30 nm the reference D sits 4.7x
  above the floor, at 76 nm below it.

u-track records both and this project already reads them (`numAperture_`,
`emissionWavelength_`, and the `psfSigma_` u-track computed from them). A raw movie has
no such file, so for the raw path they have to be entered - which is exactly why they
need a place in the window rather than a constant in the code.

Nothing here is a default. `LABEL_PRESETS` fills the emission wavelength for a chosen
label and says where the number came from; it does not choose the label, and every value
it writes stays editable. A preset that silently picked a fluorophore would be deciding
the localisation floor on the user's behalf.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "OpticsError",
    "PSF_SIGMA_COEFFICIENT",
    "psf_sigma_nm",
    "detection_radius_px",
    "pixel_size_nm",
    "pixel_size_from_project",
    "LABEL_PRESETS",
    "preset_names",
    "preset",
]


class OpticsError(ValueError):
    """Raised when the optics cannot give an honest PSF width."""


#: sigma_PSF = C * lambda / NA. The literature offers several C for the Gaussian that
#: best approximates an Airy pattern - 0.21 is the common one for the 2-D fit (Zhang
#: 2007), and larger values appear where the fit is to a different criterion.
#:
#: CALIBRATED AGAINST u-track'S OWN NUMBER RATHER THAN CHOSEN FROM A PAPER. For this
#: project's data u-track stores psfSigma_ = 0.9317 px at 106.0 nm, i.e. 98.8 nm, with
#: NA 1.49 and mNeonGreen emission 517 nm. That gives
#:     C = 98.8 * 1.49 / 517 = 0.285
#: so the coefficient below reproduces the value the tracker actually used, and
#: `psf_sigma_nm` is therefore consistent with every floor already recorded here. Where
#: u-track HAS recorded psfSigma_, prefer it - this is for the raw path, which has none.
PSF_SIGMA_COEFFICIENT = 0.285


def psf_sigma_nm(*, emission_nm: float, numerical_aperture: float,
                 coefficient: float = PSF_SIGMA_COEFFICIENT) -> float:
    """The Gaussian PSF width in nanometres, from the optics.

    Raises rather than returning a number for a non-physical input: an NA of zero or a
    negative wavelength would produce a floor, and a floor decides whether a diffusion
    coefficient may be quoted.
    """
    if not emission_nm or emission_nm <= 0:
        raise OpticsError(f"emission wavelength must be positive, got {emission_nm!r}")
    if not numerical_aperture or numerical_aperture <= 0:
        raise OpticsError(
            f"numerical aperture must be positive, got {numerical_aperture!r}")
    return float(coefficient) * float(emission_nm) / float(numerical_aperture)


def detection_radius_px(*, emission_nm: float, numerical_aperture: float,
                        pixel_size_nm: float, sigmas: float = 3.0) -> float:
    """A LoG detection radius derived from the optics instead of assumed.

    ``sigmas`` is how much of the PSF the radius should span. THREE is not a physical
    constant: it is the value that reproduces the 3.0 px this project has been using on
    its own optics (sigma 98.8 nm, pixel 106 nm gives 2.80 px), so switching to the
    derived radius does not silently change results on existing data while making the
    number move correctly for a different dye or objective.

    A red dye at 646 nm on the same scope gives 3.50 px; at NA 1.2 it gives 4.34. Those
    are the differences a fixed 3 px cannot express.
    """
    if not pixel_size_nm or pixel_size_nm <= 0:
        raise OpticsError(f"pixel size must be positive, got {pixel_size_nm!r}")
    sigma = psf_sigma_nm(emission_nm=emission_nm,
                         numerical_aperture=numerical_aperture)
    return float(sigmas) * sigma / float(pixel_size_nm)


@dataclass(frozen=True)
class LabelPreset:
    """One labelling chemistry, and what about it the pipeline needs to know."""

    name: str
    emission_nm: float
    #: What changes about the EXPERIMENT, not just the wavelength. Shown beside the
    #: preset because these are the parts a number cannot carry.
    note: str
    #: Photoactivated labels arrive sparsely over the whole acquisition instead of
    #: bleaching down to single molecules, so the crop rules - which look for a
    #: photobleaching decay - do not describe them.
    photoactivated: bool = False


#: EMISSION MAXIMA, and nothing else pretends to be measured. A dye's actual emission in
#: a given filter set differs; these are starting values to be edited, which is why the
#: window shows the resulting PSF width rather than hiding it.
LABEL_PRESETS = (
    LabelPreset("mNeonGreen", 517.0,
                "The fluorophore the PSF coefficient is calibrated against: u-track recorded "
                "psfSigma 0.932 px at NA 1.49."),
    LabelPreset("GFP / mGFP", 509.0, "Dimmer and less photostable than mNeonGreen; "
                                     "expect shorter tracks at the same power."),
    LabelPreset("mCherry", 610.0, "Longer wavelength, so a wider PSF and a higher "
                                  "localisation floor at the same NA."),
    LabelPreset("HaloTag JF549", 549.0,
                "Chemical dye. Far brighter and more photostable than a fluorescent "
                "protein, so tracks run longer and the localisation floor drops: the "
                "slow-state D may become measurable where it was a bound."),
    LabelPreset("HaloTag JF646", 646.0,
                "Red chemical dye. Brightest of these and the widest PSF; needs the "
                "detection radius to grow with it, which a fixed 3 px cannot do."),
    LabelPreset("SNAP TMR", 580.0, "Chemical dye, similar considerations to JF549."),
    LabelPreset("PAmCherry (PALM)", 595.0,
                "Photoactivated: molecules appear sparsely throughout the acquisition "
                "rather than bleaching down to single molecules, so a crop chosen from "
                "a bleaching curve does not describe this experiment.",
                photoactivated=True),
    LabelPreset("mEos3.2 (PALM)", 580.0,
                "Photoactivated, green-to-red converting. Same caveat as PAmCherry: the "
                "single-molecule regime is created by the activation laser, not by "
                "bleaching, so the crop rules do not apply.",
                photoactivated=True),
)


def preset_names() -> tuple[str, ...]:
    return tuple(item.name for item in LABEL_PRESETS)


def preset(name: str) -> LabelPreset:
    for item in LABEL_PRESETS:
        if item.name == name:
            return item
    raise OpticsError(f"no label preset named {name!r}; have {preset_names()}")


def pixel_size_nm(*, camera_pixel_um: float, magnification: float,
                  tube_lens: float = 1.0) -> float:
    """The sample-plane pixel size, from the camera and the magnification.

    This is not derivable from NA, and the confusion is worth naming because u-track
    looks as though it does derive it. u-track derives ``psfSigma_`` from NA and the
    emission wavelength: the width of a molecule's image. It does not derive
    ``pixelSize_``: that is a stored field the user supplies, exactly as here, because
    no optical formula can recover it from the image. It depends on the camera's
    physical pixel and on every magnifying element between sample and sensor, none of
    which the light itself reports.

        pixel size = camera pixel / (objective x tube lens)

    the tube lens is the term that gets dropped, and dropping it is exactly the error
    this project's `.nd2` files make. An Andor DU-897 has 16 um pixels; at 100x that
    gives 160 nm, which is what those files claim. With this microscope's ~1.5x tube
    lens it is 16000/150 = **106.7 nm**, within 0.7 per cent of the 106.0 recorded in
    `movieData.mat`, and confirmed independently by the cells themselves, which are
    1.01 um wide at 106 nm and an impossible 1.52 um at 160.

    So a value computed here should agree with `movieData.mat` where one exists; where
    none does, this is how to get it, and the tube lens is the number to check first.
    """
    if not camera_pixel_um or camera_pixel_um <= 0:
        raise OpticsError(
            f"camera pixel size must be positive, got {camera_pixel_um!r}")
    if not magnification or magnification <= 0:
        raise OpticsError(f"magnification must be positive, got {magnification!r}")
    if not tube_lens or tube_lens <= 0:
        raise OpticsError(f"tube lens factor must be positive, got {tube_lens!r}")
    return 1000.0 * float(camera_pixel_um) / (float(magnification) * float(tube_lens))


def pixel_size_from_project(folder) -> tuple[float | None, str]:
    """``(nm, where it came from)``: the pixel size this folder already records.

    Why this exists: The number was never typed into u-track. It was typed once into the
    Fiji preparation macro, which writes

        run("Properties...", "... unit=[um] pixel_width=0.106000 ...")

    and bakes the calibration into the multi-page TIFF. u-track read it from there, and
    `movieData.mat` records the result as 106.0 nm. So on a prepared project the value
    is already on disk twice and asking for it again is asking the user to re-derive
    something they settled years ago.

    Two sources, in this order, and the order is the trust order:

    1. ``movieData.mat``: what the TRACKER used, and therefore what every existing
       number in this project was computed with.
    2. the multi-page prepared TIFF's ``XResolution`` with ``unit`` in microns.

    NOT the `.nd2`, ever: it reports 160 nm, the bare objective magnification with the
    ~1.5x tube lens dropped, and reading it would rescale every D by 2.28x. And NOT the
    single-frame TIFFs u-track actually reads, measured, those carry `(72, 1)` inch,
    the library default, because the calibration did not survive the frame export.

    Returns ``(None, reason)`` when neither source is present, so a caller can fall back
    to the camera calculator rather than to a guess.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return None, f"not a folder: {folder}"

    from .utrack_meta import MOVIE_DATA_FILE

    for candidate in sorted(folder.glob(f"**/{MOVIE_DATA_FILE}"))[:1]:
        try:
            from .utrack_meta import read_movie_metadata

            meta = read_movie_metadata(candidate)
        except Exception:                              # noqa: BLE001 - try the TIFF
            break
        return float(meta.pixel_size_nm), f"movieData.mat ({candidate.parent.name})"

    for candidate in sorted(folder.glob("tif/*.tif"))[:1]:
        value = _pixel_size_from_tiff(candidate)
        if value is not None:
            return value, f"the calibration in {candidate.name}"
    for candidate in sorted(folder.glob("*.tif"))[:1]:
        value = _pixel_size_from_tiff(candidate)
        if value is not None:
            return value, f"the calibration in {candidate.name}"

    return None, ("no movieData.mat and no calibrated TIFF here: enter it, or work it "
                  "out from the camera and the magnification")


#: ImageJ's ResolutionUnit values. 1 = none (ImageJ then reads its own `unit`), 2 = inch,
#: 3 = centimetre. A `(72, 1)` resolution with unit 2 is the LIBRARY DEFAULT and means
#: "uncalibrated": measured on this project's exported single frames, which lost the
#: macro's calibration.
_UNCALIBRATED = ((72, 1), (1, 1))


def _pixel_size_from_tiff(path) -> float | None:
    """Pixel size in nm from an ImageJ-calibrated TIFF, or None."""
    try:
        import tifffile
    except ImportError:
        return None
    try:
        with tifffile.TiffFile(path) as handle:
            meta = getattr(handle, "imagej_metadata", None) or {}
            unit = str(meta.get("unit", "")).lower()
            tag = handle.pages[0].tags.get("XResolution")
            if tag is None or tuple(tag.value) in _UNCALIBRATED:
                return None
            numerator, denominator = tag.value
            if not numerator:
                return None
            per_unit = float(numerator) / float(denominator)
    except Exception:                                  # noqa: BLE001 - absent, not wrong
        return None
    if per_unit <= 0:
        return None
    if _imagej_unit(unit) not in ("um", "\u00b5m", "\u03bcm", "micron", "microns"):
        return None
    return 1000.0 / per_unit


def _imagej_unit(unit: str) -> str:
    """ImageJ's unit, readable. ImageJ ESCAPES non-ASCII in its TIFF description, so the
    lab macro's micron arrives as the six characters `\\u00B5m`, not as `\u00b5m`. Measured on
    One user's MetaMorph projects: every prepared stack carried a correct
    106.0 nm in exactly that form and was read as "no calibrated TIFF here"."""
    unit = str(unit).strip()
    if "\\u" in unit:
        try:
            unit = unit.encode("ascii", "backslashreplace").decode("unicode_escape")
        except UnicodeDecodeError:
            pass
    return unit.lower()
