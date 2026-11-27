"""u-track ``tracksFinal`` → tidy raw track table.

Reads a u-track output ``.mat`` and returns a tidy, *raw* table of localizations:
one row per (track, frame) with ``x, y`` in **pixels** and the original **frame**
number. This module is deliberately faithful to the source: it performs **no**
unit conversion and **no** track-length filtering. Those live in exactly one
place downstream (the canonical-table / observables step), per the project's
"one source of truth" rule (see ``CLAUDE.md``).

u-track format (invariants)
---------------------------
``tracksFinal`` is a struct array, one element per *compound* track. Relevant
fields:

- ``tracksCoordAmpCG`` : ``[nSegments x 8*nFrames]``. Each row is one track
  segment; per frame the 8 packed values are
  ``[x, y, z, amp, dx, dy, dz, dAmp]``. Gaps / frames outside a segment's
  existence are ``NaN``. Columns 1:2 (0-based 0,1) are ``x, y`` in **pixels**;
  columns 5:6 (0-based 4,5) are ``dx, dy``, the per-localization coordinate
  standard deviations in **pixels**, carried through to the raw table. They are
  the only per-spot precision estimate u-track stores, and they are a lower bound
  on the localization σ (formal fit error only: no drift, sample motion or
  PSF-model error), so they bracket σ from below and are never a measured σ.
- ``seqOfEvents`` : ``[nEvents x 4]`` = ``[frame, type(1=start,2=end),
  segmentIndex, mergeSplitPartner(NaN if none)]``. The minimum frame over all
  events is the compound track's first frame; the ``tracksCoordAmpCG`` columns
  span from that frame forward.

Compound tracks with merges/splits have several segments (rows); each segment
becomes its own track here, id ``"<compound>.<segment>"`` (1-based, matching
MATLAB indexing).

Fail-loud: a missing variable, an empty ``tracksFinal``, or a track that yields
no valid localization raises rather than silently returning empty/zero data.

Trackers: this reader is u-track specific. A TrackMate XML/CSV importer will be
added later as a sibling adapter (``read_trackmate_xml``) that produces the SAME
``RAW_COLUMNS`` table, so everything downstream stays tracker-agnostic. Existing
reference data is u-track and stays supported permanently; TrackMate is for future
data (see the plan, Phase 1 / Phase 5).

Status: validated on the real u-track sample project (the reference project, via
``tests/test_ingest_realdata.py``) and on synthetic fixtures that mirror the
layout (``tests/test_ingest.py``). The v7.3 (mat73) path is a normalizing
fallback and still needs validation on a real v7.3 file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = ["movie", "track_id", "frame", "x", "y", "amp", "dx", "dy"]
#: Grouping columns left empty at ingest and filled by ``smtracker.grouping``.
#: ``protein`` is the molecular species; the *fit group* is normally per protein,
#: because a state count is a property of the molecule, not of the experiment set
#: (see ``smtracker.compare``).
GROUPING_PLACEHOLDERS = ["protein", "condition", "day", "cell"]

# Packed stride of tracksCoordAmpCG: [x, y, z, amp, dx, dy, dz, dAmp] per frame.
_PACK_STRIDE = 8
_IX_X, _IX_Y, _IX_AMP = 0, 1, 3
#: u-track's per-localization coordinate STANDARD DEVIATIONS, in pixels. Carried
#: through because they are the only per-spot precision estimate the tracker stores.
#: They are a LOWER BOUND on the localization σ (they are the fit's formal error and
#: exclude drift, sample motion and PSF-model error), so they bracket σ from below and
#: must never be presented as a measured σ. Kept as NaN when the file has none.
_IX_DX, _IX_DY = 4, 5


#: Magic number of an AppleDouble resource-fork sidecar (macOS writes ``._name``
#: beside every file when copying to a non-HFS volume, USB sticks, exFAT drives,
#: network shares). They carry the extension of the file they shadow, so a ``*.mat``
#: glob picks them up, and they are NOT MATLAB files: scipy fails on them with the
#: unhelpful "Mat 4 mopt wrong format, byteswapping problem?".
_APPLEDOUBLE_MAGIC = bytes((0x00, 0x05, 0x16, 0x07))


def is_sidecar(path) -> bool:
    """True for a macOS AppleDouble sidecar: metadata masquerading as data.

    Checked by NAME (``._something``) and confirmed by the magic number, so a real
    file that merely starts with ``._`` is not discarded. Discovery skips these the
    same way it skips ``backups/``: they are never data, and letting one reach a
    reader produces an error about byte order that names neither the file nor the
    cause.
    """
    path = Path(path)
    if not path.name.startswith("._"):
        return False
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == _APPLEDOUBLE_MAGIC
    except OSError:                                    # unreadable → not our data
        return True


class IngestError(RuntimeError):
    """Raised when a u-track .mat cannot be parsed into valid tracks."""


class EmptyTrackFileError(IngestError):
    """The tracker ran and produced NO tracks for this movie.

    A DIFFERENT condition from "cannot be parsed", and it needs its own type
    because the right response differs. An empty ``tracksFinal`` is DATA (the
    movie had nothing trackable (no cells in frame, bleached, out of focus)) so a
    project loader should skip it and SAY SO. An unexpected layout or an all-NaN
    file is a FAULT, and skipping it would hide a parsing bug behind a plausible
    "that movie was empty".

    Never caught silently: whoever skips one must record which movies were skipped,
    because a condition that quietly lost a third of its movies is a different
    experiment from the one the analyst thinks they ran.
    """


def read_utrack_mat(
    path: str | Path,
    *,
    var_name: str = "tracksFinal",
    movie: str | None = None,
) -> pd.DataFrame:
    """Read a u-track ``.mat`` into a tidy raw track table.

    Parameters
    ----------
    path
        Path to the u-track output ``.mat``.
    var_name
        Name of the tracks variable in the file (default ``"tracksFinal"``).
        If absent and the file holds exactly one non-metadata variable, that one
        is used.
    movie
        Identifier for this movie/file, written to the ``movie`` column. Defaults
        to the file stem. Grouping (condition/day/cell) is filled in later.

    Returns
    -------
    pandas.DataFrame
        Columns ``movie, track_id, frame, x, y, amp`` plus empty placeholder
        columns ``condition, day, cell``. ``x, y`` are in **pixels**; ``frame``
        is the original integer frame number. Sorted by ``(track_id, frame)``.

    Raises
    ------
    IngestError
        If the file/variable is missing or holds no valid tracks.
    """
    path = Path(path)
    if not path.exists():
        raise IngestError(f"u-track file not found: {path}")
    movie = movie if movie is not None else path.stem

    tracks = _load_tracks_variable(path, var_name)
    if len(tracks) == 0:
        # The tracker ran and found nothing. Distinguished from a parse failure so
        # a caller can skip-and-record rather than treat it as a broken file.
        raise EmptyTrackFileError(
            f"{path}: '{var_name}' is empty: the tracker produced no tracks for "
            "this movie. The movie had nothing trackable, so skip it and record that."
        )
    records = _parse_compound_tracks(tracks, movie=movie)

    if not records:
        raise IngestError(
            f"{path}: '{var_name}' holds {len(tracks)} track(s) but 0 valid "
            "localizations were parsed. This is not an empty movie: the layout is "
            "unexpected or every coordinate is NaN, and skipping it would hide a "
            "parsing fault."
        )

    df = pd.DataFrame.from_records(records, columns=RAW_COLUMNS)
    for col in GROUPING_PLACEHOLDERS:
        df[col] = pd.Series([pd.NA] * len(df), dtype="object")
    df = df.astype({"frame": "int64", "x": "float64", "y": "float64",
                    "amp": "float64", "dx": "float64", "dy": "float64"})
    return df.sort_values(["track_id", "frame"], kind="stable").reset_index(drop=True)


def _load_tracks_variable(path: Path, var_name: str):
    """Load the tracks struct array via scipy.io, falling back to mat73 (v7.3).

    Returns a list of per-compound-track objects, each exposing
    ``tracksCoordAmpCG`` and ``seqOfEvents`` (either as scipy ``mat_struct``
    attributes or dict keys, normalized by the callers below).
    """
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - scipy is a hard dependency
        raise IngestError("scipy is required to read u-track .mat files") from exc

    try:
        mat = loadmat(path, struct_as_record=False, squeeze_me=True)
    except NotImplementedError:
        # v7.3 files are HDF5; scipy.io cannot read them.
        return _load_tracks_variable_v73(path, var_name)

    key = _resolve_var_name(mat, var_name, path)
    raw = mat[key]
    # squeeze_me collapses a 1-element struct array to a scalar mat_struct.
    arr = np.atleast_1d(raw)
    return list(arr.ravel())


def _load_tracks_variable_v73(path: Path, var_name: str):
    """v7.3 (HDF5) fallback via mat73. Needs validation on a real v7.3 file."""
    try:
        import mat73
    except ImportError as exc:
        raise IngestError(
            f"{path} looks like a MATLAB v7.3 file; install 'mat73' to read it "
            "(pip install mat73)."
        ) from exc

    data = mat73.loadmat(path)
    key = _resolve_var_name(data, var_name, path)
    entries = data[key]
    # mat73 returns a struct array as a dict of lists (field -> list per element).
    if isinstance(entries, dict):
        n = len(next(iter(entries.values())))
        return [_V73Track(entries, i) for i in range(n)]
    return list(np.atleast_1d(entries).ravel())


class _V73Track:
    """Adapter exposing mat73 struct-of-arrays element as attribute access."""

    def __init__(self, fields: dict, i: int):
        self._fields = fields
        self._i = i

    def __getattr__(self, name: str):
        try:
            return self._fields[name][self._i]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _resolve_var_name(mapping, var_name: str, path: Path) -> str:
    if var_name in mapping:
        return var_name
    user_keys = [k for k in mapping if not str(k).startswith("__")]
    if len(user_keys) == 1:
        return user_keys[0]
    raise IngestError(
        f"{path}: variable '{var_name}' not found. Available: {user_keys}"
    )


def _parse_compound_tracks(tracks, *, movie: str) -> list[tuple]:
    """Turn each compound track's segments into (movie, id, frame, x, y, amp) rows."""
    records: list[tuple] = []
    for i, t in enumerate(tracks):
        coords = _get_field(t, "tracksCoordAmpCG")
        seq = _get_field(t, "seqOfEvents")
        if coords is None or seq is None:
            raise IngestError(
                f"{movie}: compound track {i + 1} is missing "
                "tracksCoordAmpCG or seqOfEvents."
            )

        coords = np.atleast_2d(np.asarray(coords, dtype=float))
        seq = np.atleast_2d(np.asarray(seq, dtype=float))
        if coords.shape[1] % _PACK_STRIDE != 0:
            raise IngestError(
                f"{movie}: compound track {i + 1} has tracksCoordAmpCG width "
                f"{coords.shape[1]}, not a multiple of {_PACK_STRIDE}."
            )

        frame0 = int(np.nanmin(seq[:, 0]))
        n_frames = coords.shape[1] // _PACK_STRIDE
        frames = frame0 + np.arange(n_frames)

        for s in range(coords.shape[0]):
            packed = coords[s].reshape(n_frames, _PACK_STRIDE)
            x = packed[:, _IX_X]
            y = packed[:, _IX_Y]
            amp = packed[:, _IX_AMP]
            dx = packed[:, _IX_DX]
            dy = packed[:, _IX_DY]
            valid = ~np.isnan(x) & ~np.isnan(y)
            if not valid.any():
                # A segment with no localizations is unexpected; skip but don't
                # fabricate data. (Overall emptiness is caught by the caller.)
                continue
            track_id = f"{i + 1}.{s + 1}"
            for f, xx, yy, aa, ddx, ddy in zip(frames[valid], x[valid], y[valid],
                                               amp[valid], dx[valid], dy[valid]):
                records.append((movie, track_id, int(f), float(xx), float(yy),
                                float(aa), float(ddx), float(ddy)))
    return records


def _get_field(obj, name: str):
    """Read a field whether obj is a scipy mat_struct or a dict-like."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


@dataclass(frozen=True)
class ReportedSigma:
    """The tracker's own per-localisation uncertainty, summarised.

    u-track fits each spot and writes the fit's coordinate uncertainty into
    ``tracksCoordAmpCG`` columns 5:6. This module has read them into ``dx``/``dy``
    since Phase 1 and nothing looked at the distribution until it was needed to
    settle whether the slow-state D is a number or a bound.

    A LOWER BOUND, not a measurement of the microscope. A fit's formal covariance
    knows photon noise and background; it knows nothing about stage drift, PSF model
    mismatch, or a molecule out of the focal plane. That is why this becomes the
    ``"tracker_reported"`` provenance state and never ``"measured"``, which is
    reserved for a stationary reference sample.
    """

    median_nm: float
    p10_nm: float
    p90_nm: float
    n: int
    n_movies: int

    def caption(self) -> str:
        return (f"sigma {self.median_nm:.1f} nm (tracker-reported, a lower bound); "
                f"p10-p90 {self.p10_nm:.1f}-{self.p90_nm:.1f} nm over {self.n:,} "
                f"localisations in {self.n_movies} movie(s)")


def reported_sigma(raw: pd.DataFrame, *, pixel_size_nm: float) -> ReportedSigma:
    """Summarise ``dx``/``dy`` as a localisation sigma in nanometres.

    Fails loud rather than returning a default. A tracker that does not record a
    per-spot uncertainty (TrackMate does not) leaves these columns absent or empty,
    and a silent fallback there would put a theoretical sigma behind a
    ``"tracker_reported"`` label, which is the one thing the provenance state exists
    to prevent.
    """
    missing = [c for c in ("dx", "dy") if c not in raw.columns]
    if missing:
        raise IngestError(
            f"no per-localisation uncertainty in this table (missing {missing}). "
            "u-track records dx/dy; TrackMate does not, so a tracker-reported sigma "
            "is unavailable and the theoretical one must be used and stamped as such")
    if not pixel_size_nm > 0:
        raise IngestError(f"pixel_size_nm must be > 0, got {pixel_size_nm}")

    values = pd.concat([raw["dx"], raw["dy"]]).to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise IngestError(
            "dx/dy are present but hold no positive finite value, so there is no "
            "uncertainty to summarise")
    values = values * float(pixel_size_nm)
    return ReportedSigma(
        median_nm=float(np.median(values)),
        p10_nm=float(np.percentile(values, 10)),
        p90_nm=float(np.percentile(values, 90)),
        n=int(values.size),
        n_movies=int(raw["movie"].nunique()) if "movie" in raw.columns else 0,
    )
