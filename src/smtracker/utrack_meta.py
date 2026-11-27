"""Acquisition and tracking settings, read from u-track's ``movieData.mat``.

``CLAUDE.md`` says τ must be *confirmed from acquisition metadata* because every
diffusion coefficient scales with it. This module removes the guesswork: it
reads the numbers u-track itself recorded, so ``AcquisitionParams`` is
constructed from the experiment rather than from a remembered constant.

What is read (from the ``MovieData`` class object, via ``_matlab_mcos``)
-----------------------------------------------------------------------
Acquisition: ``pixelSize_`` (nm), ``timeInterval_`` (s), ``numAperture_``,
emission wavelength, PSF sigma (px), frame count and image size, image type
(e.g. Widefield) and fluorophore. These come from the objective/camera settings
entered at tracking time: that is exactly where the user sets NA → widefield →
pixel size, and where τ is either read from the movie or entered by hand.

Tracking: ``maxSearchRadius`` (px) (the **detection ceiling**) plus
``minTrackLen``, the gap-closing ``timeWindow``, and the merge/split flag.

Why the tracking settings matter downstream
-------------------------------------------
- ``max_search_radius_px`` sets the detection ceiling r²/(4τ): molecules faster
  than this are truncated or lost, so it must be *reported*, never assumed.
- ``min_track_len`` is u-track's own pre-filter. Our analysis-stage
  ``min_localizations`` filter is therefore usually **redundant** for u-track
  data: it is kept as a configurable safety net for tracks that arrive
  unfiltered (e.g. TrackMate), and ``check_length_filter_redundancy`` says
  plainly whether it will do anything.
- ``gap_time_window == 1`` means no gap closing, i.e. tracks are contiguous.
- ``merge_split == 0`` means every compound track has a single segment.

Localization precision is **not** stored by u-track, so
``localization_precision_nm`` stays ``None`` unless the caller supplies it; the
localization floor is then reported as unknown rather than invented.

Fail loud: a missing ``MovieData``, an absent pixel size or time interval, or
series that disagree on acquisition settings all raise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._matlab_mcos import McosError, McosUnvalidatedVersion, ObjectRef, read_mcos_objects
from .canonical import AcquisitionParams

__all__ = [
    "MetadataError",
    "MissingAcquisitionError",
    "STATED_FILE",
    "stated_values",
    "MovieMetadata",
    "read_movie_metadata",
    "read_project_metadata",
    "acquisition_params_from_project",
    "check_length_filter_redundancy",
    "check_project_layout",
    "stack_agrees",
]

#: u-track writes one of these per movie folder.
MOVIE_DATA_FILE = "movieData.mat"
#: A value the EXPERIMENTER states for movies whose u-track record left it empty (`stated_values`), beside the
#: movies: ``{"pixel_size_nm": 106.0, "movies": ["series002"], "stated_by": ..., "stated_on": ..., "why": ...}``.
#: Asked 20 Oct 2026 ("can u add the pixel size for series002"): `19.03.2025` free/series002 was tracked with
#: u-track's pixel size empty, and u-track's MovieData is a MATLAB class object this program reads but never writes.
STATED_FILE = "acquisition.json"
#: What it may state: only the pixel size. The frame interval comes from the camera (`rawmovie.frame_interval`),
#: which measures it, where the app takes it from.
_STATABLE = ("pixel_size_nm",)


def stated_values(movie_data_path) -> dict:
    """``{"pixel_size_nm": ..., "file": ...}`` stated for this movie in an `STATED_FILE` in its own folder, the
    folder above (``tif/``) or the condition's (the one holding ``tif/``); ``{}`` when none names it. A file that
    cannot be read, or states something else or a value that is not a positive number, is REFUSED by name: a
    stated value is used only because a person wrote it down, so a garbled one is never half-read."""
    import json

    movie_dir = Path(movie_data_path).parent
    for folder in (movie_dir, movie_dir.parent, movie_dir.parent.parent):
        record = folder / STATED_FILE
        if not record.is_file():
            continue
        try:
            payload = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise MetadataError(f"{record}: cannot be read as a stated acquisition ({exc})") from exc
        if movie_dir.name not in [str(m) for m in payload.get("movies", [])]:
            continue
        out = {}
        for key in _STATABLE:
            if key in payload:
                value = _number_or_none(payload[key])
                if value is None or value <= 0:
                    raise MetadataError(f"{record}: {key} must be a positive number, got {payload[key]!r}")
                out[key] = float(value)
        unknown = sorted(set(payload) - set(_STATABLE) - {"movies", "stated_by", "stated_on", "why"})
        if unknown:
            raise MetadataError(f"{record}: states {unknown}, which cannot be stated (only {list(_STATABLE)})")
        if out:
            out["file"] = str(record)
            return out
    return {}


class MetadataError(RuntimeError):
    """Raised when u-track metadata is missing, unreadable or inconsistent."""


class MissingAcquisitionError(MetadataError):
    """Movies whose MovieData lacks a value every D scales with: the pixel size, or a frame interval that no camera
    file beside the movie gives either. ``missing`` is ``[(movie, field, path)]`` for EVERY such movie of the
    folder read, so a window can ask once whether to leave them out (`project.load_project(leave_out=...)`; asked
    20 Oct 2026, "if they print no frame interval set -> ask if movie is ignored")."""

    def __init__(self, message: str, missing):
        super().__init__(message)
        self.missing = list(missing)


@dataclass(frozen=True)
class MovieMetadata:
    """Acquisition + tracking settings for one movie, as recorded by u-track."""

    movie: str
    source: Path
    pixel_size_nm: float
    frame_interval_s: float
    num_aperture: float | None = None
    emission_wavelength_nm: float | None = None
    psf_sigma_px: float | None = None
    n_frames: int | None = None
    image_size_px: tuple[int, int] | None = None
    image_type: str | None = None
    fluorophore: str | None = None
    max_search_radius_px: float | None = None
    min_track_len: int | None = None
    gap_time_window: int | None = None
    merge_split: bool | None = None
    prob_dim: int | None = None
    original_path: str | None = None
    #: what u-track RECORDED, when ``frame_interval_s`` was taken from the camera instead (`read_movie_metadata`)
    recorded_frame_interval_s: float | None = None
    #: values the experimenter STATED for this movie where u-track's record was empty (`stated_values`), with the file
    stated: dict | None = None

    @property
    def frame_interval_ms(self) -> float:
        return self.frame_interval_s * 1000.0

    def acquisition_params(
        self, *, localization_precision_nm: float | None = None
    ) -> AcquisitionParams:
        """Build ``AcquisitionParams`` from what the microscope actually recorded.

        ``localization_precision_nm`` is not stored by u-track. Left as ``None``,
        σ falls back to the THEORETICAL value, which here rests on the optical PSF
        width u-track *did* record (``psfSigma``), carried through rather than
        recomputed, since reimplementing u-track's fitted high-NA approximation
        would be a guess while the file already holds its output. Pass a value only
        for a lab-configured constant or a stationary-reference measurement.
        """
        return AcquisitionParams(
            pixel_size_nm=self.pixel_size_nm,
            frame_interval_ms=self.frame_interval_ms,
            localization_precision_nm=localization_precision_nm,
            psf_sigma_px=self.psf_sigma_px,
            search_radius_px=self.max_search_radius_px,
        )

    def summary(self) -> str:
        return (
            f"{self.movie}: pixel {self.pixel_size_nm:.4g} nm, "
            f"τ {self.frame_interval_ms:g} ms, NA {self.num_aperture}, "
            f"search radius {self.max_search_radius_px} px, "
            f"u-track minTrackLen {self.min_track_len}"
        )


def read_movie_metadata(path: str | Path, *, movie: str | None = None,
                        frame_interval_ms: float | None = None) -> MovieMetadata:
    """Read one ``movieData.mat``.

    ``path`` may be the file itself or the movie folder containing it. ``frame_interval_ms`` is the interval the
    CAMERA measured (`rawmovie.frame_interval`): it is used in place of u-track's, which is then not required and is
    kept as ``recorded_frame_interval_s``.
    """
    path = Path(path)
    if path.is_dir():
        path = path / MOVIE_DATA_FILE
    if not path.exists():
        raise MetadataError(f"u-track movieData not found: {path}")
    movie = movie if movie is not None else path.parent.name

    # A MATLAB RELEASE NOT YET SEEN (an MCOS version outside the validated list) is read with every layout check
    # still on, and kept only if its decoded frame count and image size match the movie's own stack -- two
    # independent numbers the value cells must hold in the right places (asked 20 Oct 2026: "is it now fixed that
    # every matlab version opens?", after version 4 stopped one user's project).
    unvalidated = None
    try:
        objects = read_mcos_objects(path)
    except McosUnvalidatedVersion as exc:
        unvalidated = exc
        try:
            objects = read_mcos_objects(path, unvalidated=True)
        except McosError as inner:
            raise MetadataError(f"{path}: cannot decode the MovieData object: {exc} Read anyway, its layout "
                                f"fails: {inner}") from inner
    except McosError as exc:
        raise MetadataError(f"{path}: cannot decode the MovieData object: {exc}") from exc

    movie_data = next((o for o in objects if o.class_name == "MovieData"), None)
    if movie_data is None:
        found = sorted({o.class_name for o in objects})
        raise MetadataError(f"{path}: no MovieData object (found classes: {found})")

    # A PIXEL SIZE STATED BESIDE THE MOVIE (`stated_values`) fills an EMPTY record only, never a recorded one
    stated = {} if _number_or_none(movie_data.get("pixelSize_")) is not None else stated_values(path)
    pixel_size = (stated["pixel_size_nm"] if "pixel_size_nm" in stated
                  else _require_number(movie_data.get("pixelSize_"), "pixelSize_", path))
    recorded = _number_or_none(movie_data.get("timeInterval_"))
    interval = (float(frame_interval_ms) / 1000.0 if frame_interval_ms is not None
                else _require_number(movie_data.get("timeInterval_"), "timeInterval_", path))

    channel = next((o for o in objects if o.class_name == "Channel"), None)
    tracking = _tracking_params(objects)

    size = movie_data.get("imSize_")
    image_size = (
        (int(size[0]), int(size[1])) if size is not None and len(list(size)) >= 2 else None
    )
    if unvalidated is not None:
        n_frames = _int_or_none(movie_data.get("nFrames_"))
        agrees, where = stack_agrees(path.parent, movie, n_frames, image_size)
        if not agrees:
            raise MetadataError(
                f"{path}: cannot decode the MovieData object: {unvalidated} Read anyway, it could not be "
                f"confirmed against the movie's own stack ({where}), so it is not used.")
        import warnings

        warnings.warn(
            f"{path}: MCOS version {unvalidated.version} (a MATLAB release not validated here) was read: its "
            f"layout checks, and its {n_frames} frames of {image_size[0]} x {image_size[1]} px match {where}.",
            RuntimeWarning, stacklevel=2)

    return MovieMetadata(
        movie=movie,
        source=path,
        pixel_size_nm=pixel_size,
        frame_interval_s=interval,
        num_aperture=_number_or_none(movie_data.get("numAperture_")),
        emission_wavelength_nm=(
            _number_or_none(channel.get("emissionWavelength_")) if channel else None
        ),
        psf_sigma_px=_number_or_none(channel.get("psfSigma_")) if channel else None,
        n_frames=_int_or_none(movie_data.get("nFrames_")),
        image_size_px=image_size,
        image_type=_str_or_none(channel.get("imageType_")) if channel else None,
        fluorophore=_str_or_none(channel.get("fluorophore_")) if channel else None,
        original_path=_str_or_none(movie_data.get("movieDataPath_")),
        recorded_frame_interval_s=recorded if frame_interval_ms is not None else None,
        stated=stated or None,
        **tracking,
    )


def stack_agrees(movie_folder, movie: str, n_frames, image_size) -> tuple[bool, str]:
    """Whether a MovieData's frame count and image size are those of the movie's own frames: u-track's per-frame
    copy (``<movie>/<movie>/*.tif`` in the movie folder) or the stack beside it (``tif/<movie>.tif``). Returns
    ``(agrees, what was compared or why nothing could be)``; reads headers only."""
    if n_frames is None or image_size is None:
        return False, "the record holds no frame count or image size to compare"
    folder = Path(movie_folder)
    frames = sorted(p for p in (folder / movie).glob("*.tif*") if p.is_file()) if (folder / movie).is_dir() else []
    stack = folder.parent / f"{movie}.tif"
    try:
        import tifffile

        if frames:
            with tifffile.TiffFile(frames[0]) as tif:
                shape = tuple(tif.pages[0].shape[:2])
            found, where = (len(frames), shape), f"u-track's {len(frames)} frames in {folder / movie}"
        elif stack.is_file():
            with tifffile.TiffFile(stack) as tif:
                found = (len(tif.pages), tuple(tif.pages[0].shape[:2]))
            where = f"{stack.name}, {found[0]} frames of {found[1][0]} x {found[1][1]} px"
        else:
            return False, f"no stack beside it ({folder / movie}/*.tif or {stack})"
    except Exception as exc:                                      # noqa: BLE001 - named, never swallowed
        return False, f"the stack could not be read ({type(exc).__name__}: {exc})"
    if found != (int(n_frames), tuple(image_size)):
        return False, (f"the record says {n_frames} frames of {image_size[0]} x {image_size[1]} px, "
                       f"{where}")
    return True, where


def _tracking_params(objects) -> dict:
    """Pull search radius / minTrackLen / gap window from the TrackingProcess."""
    empty = {
        "max_search_radius_px": None,
        "min_track_len": None,
        "gap_time_window": None,
        "merge_split": None,
        "prob_dim": None,
    }
    proc = next((o for o in objects if o.class_name == "TrackingProcess"), None)
    if proc is None:
        return empty
    params = proc.get("funParams_")
    if params is None:
        return empty

    gap = getattr(params, "gapCloseParam", None)
    out = dict(empty)
    out["prob_dim"] = _int_or_none(getattr(params, "probDim", None))
    if gap is not None:
        out["min_track_len"] = _int_or_none(getattr(gap, "minTrackLen", None))
        out["gap_time_window"] = _int_or_none(getattr(gap, "timeWindow", None))
        ms = getattr(gap, "mergeSplit", None)
        out["merge_split"] = bool(ms) if ms is not None else None

    radii = []
    cost_matrices = getattr(params, "costMatrices", None)
    if cost_matrices is not None:
        for cm in _iterate(cost_matrices):
            p = getattr(cm, "parameters", None)
            r = _number_or_none(getattr(p, "maxSearchRadius", None)) if p else None
            if r is not None:
                radii.append(r)
    if radii:
        # Linking and gap-closing can differ; the ceiling is set by the largest.
        out["max_search_radius_px"] = max(radii)
    return out


def read_project_metadata(
    root: str | Path, *, pattern: str = f"**/{MOVIE_DATA_FILE}",
    frame_intervals_ms: dict | None = None, leave_out=(),
) -> list[MovieMetadata]:
    """Read every ``movieData.mat`` under a project folder (skipping backups).

    ``frame_intervals_ms`` ``{movie: ms}`` are camera intervals used in place of u-track's
    (`read_movie_metadata`); movies named in ``leave_out`` are not read. A movie lacking a value every D scales
    with is collected, not raised at the first: one :class:`MissingAcquisitionError` names them all."""
    root = Path(root)
    if not root.is_dir():
        raise MetadataError(f"project folder not found: {root}")
    from .dayfolders import namer

    known_as = namer(root)                        # ``series001 (19.03.2025)`` in day folders reusing names
    hits = [p for p in sorted(root.glob(pattern)) if "backups" not in p.parts]
    hits = [p for p in hits if known_as(p.parent) not in set(leave_out)]
    if not hits:
        raise MetadataError(f"no {MOVIE_DATA_FILE} under {root}")
    measured = frame_intervals_ms or {}
    out, missing = [], []
    for p in hits:
        name = known_as(p.parent)
        try:
            out.append(read_movie_metadata(p, movie=name, frame_interval_ms=measured.get(name)))
        except MissingAcquisitionError as exc:
            missing += [(name, field, where) for _movie, field, where in exc.missing]
    if missing:
        names = ", ".join(f"{movie} ({field.rstrip('_')})" for movie, field, _path in missing)
        raise MissingAcquisitionError(
            f"{root}: {len(missing)} movie(s) were tracked without a value every D scales with: {names}. "
            "Set it in u-track, or leave the movie out.", missing)
    return out


def acquisition_params_from_project(
    root: str | Path,
    *,
    localization_precision_nm: float | None = None,
    tolerance: float = 1e-6,
    frame_intervals_ms: dict | None = None,
    leave_out=(),
) -> AcquisitionParams:
    """One ``AcquisitionParams`` for a whole project, verified consistent.

    Fails loud if series disagree on pixel size, τ, or search radius, a mixed
    project must not be pooled into a single set of diffusion coefficients
    without the analyst knowing.
    """
    metas = read_project_metadata(root, frame_intervals_ms=frame_intervals_ms, leave_out=leave_out)
    # tau MAY differ across movies: that is a mixed-rate project, not an error. It
    # is carried per movie and `AcquisitionParams.tau_s` then refuses to hand out a
    # scalar, so nothing can pool it by accident. Pixel size and search radius are
    # still fatal to mix: they change the geometry and the detection ceiling with no
    # per-movie route through the estimators.
    tau_by_movie = {m.movie: m.frame_interval_ms for m in metas
                    if m.frame_interval_ms is not None}
    from .canonical import same_setting

    for field_name in ("pixel_size_nm", "max_search_radius_px"):
        values = {getattr(m, field_name) for m in metas}
        numeric = [v for v in values if v is not None]
        # RELATIVE, as across conditions (`canonical.same_setting`): an absolute 1e-6 refused 106.000003 against 106.0
        if numeric and (max(numeric) - min(numeric) > tolerance and not same_setting(max(numeric), min(numeric))):
            detail = ", ".join(f"{m.movie}={getattr(m, field_name)}" for m in metas)
            raise MetadataError(
                f"movies disagree on {field_name}: {detail}. "
                "Pooling them would mix acquisition settings."
            )
    first = metas[0]
    params = first.acquisition_params(
        localization_precision_nm=localization_precision_nm)
    from dataclasses import replace

    return replace(params, frame_interval_ms_by_movie=tau_by_movie)


def check_length_filter_redundancy(
    metas: list[MovieMetadata], min_localizations: int
) -> str:
    """State whether the analysis-stage length filter can do anything.

    u-track already applies ``minTrackLen``; when that is at least as strict as
    ``min_localizations`` our filter is a no-op and should be *expected* to be
    one when comparing against the frozen MATLAB reference.
    """
    lens = {m.min_track_len for m in metas if m.min_track_len is not None}
    if not lens:
        return (
            f"u-track minTrackLen unknown; the min_localizations={min_localizations} "
            "filter may or may not remove tracks."
        )
    smallest = min(lens)
    spread = f"minTrackLen={smallest}" if len(lens) == 1 else f"minTrackLen in {sorted(lens)}"
    if smallest >= min_localizations:
        return (
            f"u-track already pre-filtered tracks ({spread}), so the analysis-stage "
            f"min_localizations={min_localizations} filter is redundant here and "
            "should remove nothing."
        )
    return (
        f"u-track {spread} is looser than min_localizations={min_localizations}; "
        "the analysis-stage filter will remove tracks."
    )


def check_project_layout(metas: list[MovieMetadata]) -> list[str]:
    """Compare each movie's **folder name** with what u-track recorded.

    u-track stores an absolute ``movieDataPath_`` from wherever it was run. That
    path is *not* a layout requirement: analysis is often run in whatever folder
    the data sits in, and the folders above the movie get renamed or dropped when
    a project is moved. So the parent folders are deliberately **not** checked,
    only the movie folder's own name, which is what identifies the movie
    throughout the analysis and must therefore stay stable.

    Returns a list of human-readable discrepancies (empty when names match).
    Reporting, not raising: a moved or renamed project is still analysable.
    """
    notes: list[str] = []
    for m in metas:
        if not m.original_path:
            notes.append(f"{m.movie}: u-track recorded no movieDataPath_")
            continue
        recorded = m.original_path.replace("\\", "/").rstrip("/").split("/")
        actual = list(m.source.resolve().parent.parts)
        if recorded[-1] != actual[-1]:
            notes.append(
                f"{m.movie}: u-track recorded the movie folder as '{recorded[-1]}' "
                f"but it is '{actual[-1]}'. Movie identity comes from this name, "
                "so check the grouping."
            )
    return notes


def _iterate(value):
    try:
        return list(value)
    except TypeError:
        return [value]


def _require_number(value, name: str, path: Path) -> float:
    number = _number_or_none(value)
    if number is None:
        raise MissingAcquisitionError(
            f"{path}: '{name}' is missing or empty in the MovieData. It must be set "
            "in u-track (objective NA / pixel size, frame interval). SMTracker does not "
            "guess it, since every D depends on it.",
            [(Path(path).parent.name, name, Path(path))])
    return number


def _number_or_none(value) -> float | None:
    if value is None or isinstance(value, (str, ObjectRef, list)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # reject NaN


def _int_or_none(value) -> int | None:
    number = _number_or_none(value)
    return int(number) if number is not None else None


def _str_or_none(value) -> str | None:
    return value if isinstance(value, str) and value else None
