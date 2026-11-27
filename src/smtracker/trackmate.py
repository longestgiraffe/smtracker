"""TrackMate XML/CSV → the same raw track table as u-track.

The second tracker, never a replacement for the first. u-track stays canonical
(the reference data is u-track and must remain readable forever); TrackMate is
added **alongside** it, and both produce the identical ``ingest.RAW_COLUMNS``
table so nothing downstream knows or cares which tracker produced the data.

Units: the one real trap
-------------------------
TrackMate records coordinates in whatever the image calibration says, and
declares it in ``Model/@spatialunits``. That is often **µm**, whereas this
pipeline's raw stage is defined in **pixels** (because cell outlines from
Oufti/MicrobeTracker are in pixels, and unit conversion happens exactly once, in
``smtracker.canonical``). So a µm-calibrated TrackMate file must be converted
*back* to pixels here, which is precisely what the legacy did
(``cellTracks.m``: ``x(:,2:3)*(1000/pixel_size)`` for ``spaceUnits == 'µm'``).

Converting requires the pixel size. If the file is µm-calibrated the reader
takes it from ``Settings/ImageData/@pixelwidth`` when present, else from the
caller, and **fails loud** rather than assuming, because a wrong pixel size
silently rescales every displacement and therefore every D.

Track structure
---------------
TrackMate stores tracks as a graph of ``<Edge>`` elements between spot IDs, so a
track may branch (splits/merges). Branching tracks are decomposed into linear
segments and each segment becomes its own track, id ``"<track>.<segment>"``,
exactly the convention the u-track reader uses for compound tracks, so the two
sources stay indistinguishable downstream.

Filtered tracks
---------------
TrackMate's ``<FilteredTracks>`` lists the tracks that survived the user's
filters: what the operator actually saw. Those are used by default; pass
``filtered_only=False`` to read everything, which is reported either way.

Fail loud: an unparseable file, no spots, no tracks, or a unit conversion that
cannot be performed all raise rather than returning a partial table.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .ingest import GROUPING_PLACEHOLDERS, RAW_COLUMNS

__all__ = [
    "read_tracking_settings",
    "describe_tracking_settings",
    "linking_radius_px",
    "FRAME_ORIGIN_OFFSET",
    "TrackMateError",
    "TrackMateMetadata",
    "read_trackmate_xml",
    "read_trackmate_csv",
    "read_trackmate_metadata",
]

#: Spatial-unit strings TrackMate writes for micrometres.
_MICRON_UNITS = {"µm", "um", "micron", "microns", "micrometer", "micrometre"}
_PIXEL_UNITS = {"pixel", "pixels", "px", ""}


#: TrackMate counts frames from ZERO; u-track counts from ONE, being MATLAB. The raw
#: table is a single schema that must not mean different things depending on which
#: tracker filled it, so TrackMate's frames are shifted here and u-track's are left alone.
#:
#: MEASURED on the same movie: u-track reported frames 1..1001 and a TrackMate run over
#: the identical stack reported 0..999. Nothing in the project documented a convention
#: and no test pinned one, so the mismatch was latent rather than active -- but
#: `spot_intensity.integrated_intensity` already indexes the raw stack as `frame - 1`,
#: so TrackMate data would have read one frame early, silently, and every intensity
#: would have come from the wrong image. Same shape as the `step_frames` lesson: an
#: off-by-one looks like a plausible time offset rather than a bug.
FRAME_ORIGIN_OFFSET = 1


class TrackMateError(RuntimeError):
    """Raised when a TrackMate file cannot be parsed into valid tracks."""


@dataclass(frozen=True)
class TrackMateMetadata:
    """Acquisition settings recorded in a TrackMate XML, if present."""

    spatial_units: str
    time_units: str
    pixel_width: float | None = None
    pixel_height: float | None = None
    time_interval: float | None = None
    n_frames: int | None = None
    image_filename: str | None = None

    @property
    def is_micron_calibrated(self) -> bool:
        return self.spatial_units.strip().lower() in _MICRON_UNITS

    @property
    def is_pixel_calibrated(self) -> bool:
        return self.spatial_units.strip().lower() in _PIXEL_UNITS


def read_trackmate_metadata(path: str | Path) -> TrackMateMetadata:
    """Read units and calibration from a TrackMate XML without parsing spots."""
    root = _parse(path)
    model = root.find("Model")
    if model is None:
        raise TrackMateError(f"{path}: no <Model> element: is this a TrackMate XML?")

    image = root.find("./Settings/ImageData")
    return TrackMateMetadata(
        spatial_units=model.get("spatialunits", ""),
        time_units=model.get("timeunits", ""),
        pixel_width=_float_or_none(image.get("pixelwidth")) if image is not None else None,
        pixel_height=_float_or_none(image.get("pixelheight")) if image is not None else None,
        time_interval=_float_or_none(image.get("timeinterval")) if image is not None else None,
        n_frames=_int_or_none(image.get("nframes")) if image is not None else None,
        image_filename=image.get("filename") if image is not None else None,
    )


#: Spot features worth carrying when the run computed them. Measured on a real 7.11.1
#: run of 946 spots: SNR is the one that earns its place - its median was 0.825, i.e. the
#: median detection sat BELOW the local noise, which is the purity question answered per
#: spot rather than inferred from where detections fall relative to cell outlines.
#:
#: NOT included, and why: ELLIPSE_* came back as exactly the detection radius on every
#: spot (the LoG detector fits no shape), and MIN/MAX/STD add nothing over the mean and
#: the SNR.
_EXTRA_SPOT_FEATURES = ("QUALITY", "SNR_CH1", "CONTRAST_CH1",
                        "MEAN_INTENSITY_CH1", "TOTAL_INTENSITY_CH1")

#: The columns those land in. `total_intensity` is deliberately NOT called an integrated
#: intensity: TrackMate sums the ROI without subtracting background, so it is not the
#: quantity `spot_intensity.integrated_intensity` computes.
_EXTRA_SPOT_COLUMNS = ("quality", "snr", "contrast",
                       "mean_intensity", "total_intensity")


#: The nested blocks TrackMate writes inside ``Settings/TrackerSettings``. Flat
#: attributes are read from the element itself; these carry the numbers that decide
#: linking, and they are one level down.
_TRACKER_BLOCKS = ("Linking", "GapClosing", "TrackSplitting", "TrackMerging")


def read_tracking_settings(path) -> dict:
    """The DETECTOR and TRACKER settings the XML was produced with.

    The file has always carried them and the table did not. Asked directly ("do the
    trackmate table contain the tracking settings?") and the answer was no: `.attrs`
    held the source, the units and the pixel scale, while the radius, the threshold, the
    linking distance and the frame gap sat unread in ``Settings/DetectorSettings`` and
    ``Settings/TrackerSettings`` a few lines further down the same file this module was
    already parsing.

    That matters here more than it would elsewhere. This project has measured that the
    linking radius SETS the fast-state D once the population is quick enough
    (`trackmate_run.max_measurable_D`), that the frame gap is a TIME rather than a frame
    count, and that a threshold is a property of one session's noise. A table whose
    numbers depend on three settings it does not record cannot be checked later, and
    `run_batch`'s live stamp does not help a folder of XMLs reopened months on, which is
    exactly when the question gets asked.

    Returns a flat dict, values left as STRINGS exactly as written, plus the parsed
    numbers this project reasons about under well-known keys (``radius_px``,
    ``threshold``, ``linking_max_distance_px``, ``max_frame_gap``,
    ``allow_gap_closing``). Missing blocks are simply absent rather than defaulted: an
    XML from another TrackMate version may not carry them, and a default here would be
    this function's opinion wearing the file's authority.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(str(path)).getroot()
    settings = root.find("./Settings")
    out: dict = {}
    if settings is None:
        return out

    detector = settings.find("./DetectorSettings")
    if detector is not None:
        out.update({f"detector.{k}": v for k, v in detector.attrib.items()})
    tracker = settings.find("./TrackerSettings")
    if tracker is not None:
        out.update({f"tracker.{k}": v for k, v in tracker.attrib.items()})
        for block in _TRACKER_BLOCKS:
            node = tracker.find("./" + block)
            if node is not None:
                out.update({f"tracker.{block}.{k}": v
                            for k, v in node.attrib.items()})

    def _num(key, cast=float):
        raw = out.get(key)
        if raw is None:
            return None
        try:
            return cast(raw)
        except (TypeError, ValueError):
            return None

    friendly = {
        "radius_px": _num("detector.RADIUS"),
        "threshold": _num("detector.THRESHOLD"),
        "detector_name": out.get("detector.DETECTOR_NAME"),
        "tracker_name": out.get("tracker.TRACKER_NAME"),
        "linking_max_distance_px": _num("tracker.Linking.LINKING_MAX_DISTANCE"),
        "gap_closing_max_distance_px": _num(
            "tracker.GapClosing.GAP_CLOSING_MAX_DISTANCE"),
        "max_frame_gap": _num("tracker.GapClosing.MAX_FRAME_GAP", int),
    }
    allow = out.get("tracker.GapClosing.ALLOW_GAP_CLOSING")
    if allow is not None:
        friendly["allow_gap_closing"] = str(allow).strip().lower() == "true"
    out.update({k: v for k, v in friendly.items() if v is not None})
    return out


def describe_tracking_settings(settings: dict) -> str:
    """One line naming what produced a table, for a caption or a log.

    Says what is MISSING rather than filling it in, an XML that records no tracker
    block is a different situation from one that records a gap of zero, and a caption
    that cannot tell them apart is worse than one that says so.
    """
    if not settings:
        return "tracking settings: not recorded in this XML"
    bits = []
    if settings.get("detector_name"):
        bits.append(str(settings["detector_name"]))
    if settings.get("radius_px") is not None:
        bits.append(f"radius {settings['radius_px']:g} px")
    if settings.get("threshold") is not None:
        bits.append(f"threshold {settings['threshold']:g}")
    if settings.get("linking_max_distance_px") is not None:
        bits.append(f"linking {settings['linking_max_distance_px']:g} px")
    if settings.get("allow_gap_closing") is False:
        bits.append("no gap closing")
    elif settings.get("max_frame_gap") is not None:
        bits.append(f"gap {settings['max_frame_gap']} frames")
    return "tracking settings: " + ", ".join(bits) if bits else (
        "tracking settings: present but none recognised")


def read_trackmate_xml(
    path: str | Path,
    *,
    movie: str | None = None,
    pixel_size_nm: float | None = None,
    filtered_only: bool = True,
) -> pd.DataFrame:
    """Read a TrackMate XML into the shared raw track table (**pixels**, frames).

    Parameters
    ----------
    path
        TrackMate XML (the "Export to XML"/full model file, not the simplified
        spot table: use ``read_trackmate_csv`` for that).
    movie
        Identifier for the ``movie`` column; defaults to the file stem.
    pixel_size_nm
        Needed only when the file is µm-calibrated and does not record its own
        ``pixelwidth``. Never guessed.
    filtered_only
        Use TrackMate's ``<FilteredTracks>`` (default), the tracks the operator
        actually kept.

    Returns
    -------
    pandas.DataFrame
        ``ingest.RAW_COLUMNS`` plus empty grouping placeholders, identical in
        shape and units to ``ingest.read_utrack_mat``.
    """
    path = Path(path)
    movie = movie if movie is not None else path.stem
    root = _parse(path)
    model = root.find("Model")
    if model is None:
        raise TrackMateError(f"{path}: no <Model> element: is this a TrackMate XML?")

    metadata = read_trackmate_metadata(path)
    scale = _pixels_per_unit(metadata, pixel_size_nm, path)

    spots = _read_spots(model, path)
    edges_by_track = _read_tracks(model, path, filtered_only=filtered_only)

    records = []
    spot_ids = []
    for track_id, edges in edges_by_track.items():
        for segment_index, chain in enumerate(_linear_segments(edges), start=1):
            for spot_id in chain:
                spot = spots.get(spot_id)
                if spot is None:
                    raise TrackMateError(
                        f"{path}: track {track_id} references unknown spot {spot_id}"
                    )
                frame, x, y, amp, extras = spot
                # dx, dy: u-track stores a per-localization coordinate SD; the
                # TrackMate model does not export one, so they are NaN here, an
                # ABSENT measurement, never a zero (a zero would read as perfect
                # precision and silently sharpen any σ derived downstream). Each
                # spot's own 2D Gaussian fit fills them where it was made (below).
                records.append(
                    (movie, f"{track_id}.{segment_index}", frame,
                     x * scale, y * scale, amp, np.nan, np.nan)
                    + tuple(extras[name] for name in _EXTRA_SPOT_FEATURES)
                )
                spot_ids.append(spot_id)

    if not records:
        raise TrackMateError(
            f"{path}: parsed 0 localizations. The file may hold no tracks, or "
            f"{'all tracks were filtered out' if filtered_only else 'the model is empty'}."
        )

    df = pd.DataFrame.from_records(
        records, columns=list(RAW_COLUMNS) + list(_EXTRA_SPOT_COLUMNS))
    # EACH SPOT'S PRECISION, where SMTracker fitted the spots of this run (`spotfit`, beside the XML): the fit's formal
    # SD in pixels into dx, dy, as u-track writes its own; a spot the fit could not place stays NaN, never a zero.
    from .spotfit import read_sidecar

    fitted = read_sidecar(path)
    if fitted is not None:
        by_id = fitted.set_index("spot_id")
        ids = pd.Index(spot_ids)
        df["dx"] = by_id["sigma_x_px"].reindex(ids).to_numpy(dtype=float)
        df["dy"] = by_id["sigma_y_px"].reindex(ids).to_numpy(dtype=float)
    for column in GROUPING_PLACEHOLDERS:
        df[column] = pd.Series([pd.NA] * len(df), dtype="object")
    df = df.astype({"frame": "int64", "x": "float64", "y": "float64",
                    "amp": "float64", "dx": "float64", "dy": "float64"})
    df = df.sort_values(["track_id", "frame"], kind="stable").reset_index(drop=True)
    # WHAT `amp` ACTUALLY IS HERE, stamped so it cannot be compared with u-track's by
    # accident. u-track's `amp` is a FITTED GAUSSIAN AMPLITUDE, background subtracted;
    # TrackMate's QUALITY is the LoG filter response. Different quantities, and
    # `total_intensity` is not the bridge either - TrackMate does not subtract the
    # background from it.
    df.attrs["source"] = "trackmate"
    df.attrs["amp_source"] = "trackmate QUALITY (LoG response), not an intensity"
    df.attrs["precision_source"] = ("each spot's 2D Gaussian fit (spotfit, beside the XML)" if fitted is not None
                                    else None)
    # WHAT PRODUCED THIS TABLE TRAVELS WITH IT. The radius and the frame gap decide what
    # the numbers downstream can mean, and they were sitting unread in the same file.
    settings = read_tracking_settings(path)
    df.attrs["tracking_settings"] = settings
    df.attrs["spatial_units"] = metadata.spatial_units
    df.attrs["pixels_per_unit"] = scale
    # THE LINKING DISTANCE IN PIXELS, the condition's search radius (ruled 4 Nov 2026, D10 follow-up): the setting is
    # in the file's own calibration, so it is converted as the coordinates are
    df.attrs["linking_radius_px"] = linking_radius_px(settings, scale)
    return df


def linking_radius_px(settings: dict, pixels_per_unit: float) -> float | None:
    """TrackMate's LINKING_MAX_DISTANCE in PIXELS, the search radius the analysis's ceiling is drawn at (ruled 4 Nov
    2026: "read TrackMate's linking distance from its XML as that condition's search radius"), or None where the XML
    records none. ``settings`` is `read_tracking_settings`' (its ``linking_max_distance_px`` is in the FILE's units,
    pixels for SMTracker's own runs, which force 1 pixel = 1 unit); ``pixels_per_unit`` converts it as the
    coordinates are (`_pixels_per_unit`). The gap-closing distance spans several frames and sets no one-frame
    ceiling, so it is not read here."""
    value = (settings or {}).get("linking_max_distance_px")
    if value is None:
        return None
    radius = float(value) * float(pixels_per_unit)
    return radius if np.isfinite(radius) and radius > 0 else None


def read_trackmate_csv(
    path: str | Path,
    *,
    movie: str | None = None,
    pixel_size_nm: float | None = None,
    spatial_units: str = "pixel",
) -> pd.DataFrame:
    """Read a TrackMate spot-table CSV export into the shared raw table.

    Expects TrackMate's exported columns ``TRACK_ID``, ``POSITION_X``,
    ``POSITION_Y``, ``FRAME`` (case-insensitive), optionally a quality or
    intensity column for ``amp``. ``spatial_units`` says how the coordinates are
    calibrated, since a CSV carries no unit declaration.
    """
    path = Path(path)
    movie = movie if movie is not None else path.stem
    try:
        table = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
        raise TrackMateError(f"{path}: cannot read CSV: {exc}") from exc

    # TrackMate CSVs sometimes carry 3 extra header rows of units/descriptions.
    lookup = {str(c).strip().upper(): c for c in table.columns}
    required = ("TRACK_ID", "POSITION_X", "POSITION_Y", "FRAME")
    missing = [name for name in required if name not in lookup]
    if missing:
        raise TrackMateError(
            f"{path}: missing column(s) {missing}. Found: {list(table.columns)[:12]}"
        )

    frame = pd.to_numeric(table[lookup["FRAME"]], errors="coerce") + FRAME_ORIGIN_OFFSET
    x = pd.to_numeric(table[lookup["POSITION_X"]], errors="coerce")
    y = pd.to_numeric(table[lookup["POSITION_Y"]], errors="coerce")
    track = table[lookup["TRACK_ID"]].astype(str)
    amp_column = next(
        (lookup[name] for name in ("QUALITY", "MEAN_INTENSITY_CH1", "MEAN_INTENSITY",
                                   "TOTAL_INTENSITY_CH1") if name in lookup),
        None,
    )
    amp = (
        pd.to_numeric(table[amp_column], errors="coerce")
        if amp_column else pd.Series(np.nan, index=table.index)
    )

    valid = frame.notna() & x.notna() & y.notna() & track.notna() & (track != "nan")
    if not valid.any():
        raise TrackMateError(f"{path}: no rows with a usable TRACK_ID, FRAME and position")

    metadata = TrackMateMetadata(spatial_units=spatial_units, time_units="")
    scale = _pixels_per_unit(metadata, pixel_size_nm, path)

    df = pd.DataFrame(
        {
            "movie": movie,
            "track_id": track[valid].to_numpy(),
            "frame": frame[valid].to_numpy(),
            "x": x[valid].to_numpy() * scale,
            "y": y[valid].to_numpy() * scale,
            "amp": amp[valid].to_numpy(),
            # see the XML path: TrackMate exports no per-spot coordinate SD, so
            # dx/dy are ABSENT (NaN), never zero.
            "dx": np.nan,
            "dy": np.nan,
        }
    )
    for column in GROUPING_PLACEHOLDERS:
        df[column] = pd.Series([pd.NA] * len(df), dtype="object")
    df = df.astype({"frame": "int64", "x": "float64", "y": "float64",
                    "amp": "float64", "dx": "float64", "dy": "float64"})
    df = df.sort_values(["track_id", "frame"], kind="stable").reset_index(drop=True)
    df.attrs["source"] = "trackmate"
    df.attrs["spatial_units"] = spatial_units
    df.attrs["pixels_per_unit"] = scale
    return df


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------

def _parse(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise TrackMateError(f"TrackMate file not found: {path}")
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise TrackMateError(f"{path}: not valid XML: {exc}") from exc


def _pixels_per_unit(metadata: TrackMateMetadata, pixel_size_nm, path) -> float:
    """Factor converting the file's coordinates to PIXELS."""
    if metadata.is_pixel_calibrated:
        return 1.0
    if not metadata.is_micron_calibrated:
        raise TrackMateError(
            f"{path}: unrecognised spatial units {metadata.spatial_units!r}. "
            "Expected pixels or µm. SMTracker does not guess, since a wrong choice "
            "rescales every displacement."
        )
    # µm-calibrated: convert back to pixels for the raw stage.
    micron_per_pixel = metadata.pixel_width
    if micron_per_pixel is None or micron_per_pixel <= 0:
        if pixel_size_nm is None or pixel_size_nm <= 0:
            raise TrackMateError(
                f"{path}: coordinates are in µm but the file records no usable "
                "pixel width, and no pixel_size_nm was given. The raw table is "
                "defined in pixels, so this conversion cannot be skipped or "
                "guessed: every D depends on it."
            )
        micron_per_pixel = pixel_size_nm / 1000.0
    return 1.0 / micron_per_pixel


def _read_spots(model, path) -> dict[str, tuple[int, float, float, float, dict]]:
    all_spots = model.find("AllSpots")
    if all_spots is None:
        raise TrackMateError(f"{path}: no <AllSpots> element")
    spots: dict[str, tuple[int, float, float, float, dict]] = {}
    for spot in all_spots.iter("Spot"):
        spot_id = spot.get("ID")
        if spot_id is None:
            continue
        frame = _int_or_none(spot.get("FRAME"))
        if frame is not None:
            frame += FRAME_ORIGIN_OFFSET
        x = _float_or_none(spot.get("POSITION_X"))
        y = _float_or_none(spot.get("POSITION_Y"))
        if frame is None or x is None or y is None:
            continue
        amp = next(
            (
                _float_or_none(spot.get(name))
                for name in ("QUALITY", "MEAN_INTENSITY_CH1", "MEAN_INTENSITY",
                             "TOTAL_INTENSITY_CH1")
                if spot.get(name) is not None
            ),
            None,
        )
        # THE EXTRA FEATURES, when the run asked TrackMate to compute them. Until the
        # analyzers were registered every XML carried QUALITY and nothing else, so these
        # are absent on older files and simply come back NaN.
        extras = {name: _float_or_none(spot.get(name))
                  for name in _EXTRA_SPOT_FEATURES}
        spots[spot_id] = (frame, x, y, np.nan if amp is None else amp, extras)
    if not spots:
        raise TrackMateError(f"{path}: <AllSpots> contains no usable spots")
    return spots


def _read_tracks(model, path, *, filtered_only: bool) -> dict[str, list[tuple[str, str]]]:
    all_tracks = model.find("AllTracks")
    if all_tracks is None:
        raise TrackMateError(f"{path}: no <AllTracks> element")

    keep = None
    if filtered_only:
        filtered = model.find("FilteredTracks")
        if filtered is not None:
            keep = {
                element.get("TRACK_ID")
                for element in filtered.iter("TrackID")
                if element.get("TRACK_ID") is not None
            }

    out: dict[str, list[tuple[str, str]]] = {}
    for track in all_tracks.iter("Track"):
        track_id = track.get("TRACK_ID")
        if track_id is None:
            continue
        if keep is not None and track_id not in keep:
            continue
        edges = [
            (edge.get("SPOT_SOURCE_ID"), edge.get("SPOT_TARGET_ID"))
            for edge in track.iter("Edge")
            if edge.get("SPOT_SOURCE_ID") and edge.get("SPOT_TARGET_ID")
        ]
        if edges:
            out[track_id] = edges

    if not out:
        raise TrackMateError(
            f"{path}: no tracks with edges"
            + (" survived <FilteredTracks>" if filtered_only else "")
        )
    return out


def _linear_segments(edges: list[tuple[str, str]]) -> list[list[str]]:
    """Split a possibly-branching track graph into linear chains.

    Mirrors the u-track compound-track convention: each linear run becomes one
    track, so branching never silently merges two lineages into one trajectory.
    """
    successors: dict[str, list[str]] = {}
    predecessors: dict[str, list[str]] = {}
    nodes = set()
    for source, target in edges:
        successors.setdefault(source, []).append(target)
        predecessors.setdefault(target, []).append(source)
        nodes.update((source, target))

    # A chain starts at a node with no predecessor, or just after a branch.
    starts = [n for n in nodes if not predecessors.get(n)]
    if not starts:                      # a cycle (should not happen): break it
        starts = [sorted(nodes)[0]]

    segments: list[list[str]] = []
    queue = list(starts)
    seen_edges: set[tuple[str, str]] = set()
    while queue:
        node = queue.pop()
        chain = [node]
        while True:
            following = successors.get(node, [])
            if len(following) != 1:
                # 0 = end of chain; >1 = branch, so start a new segment at each
                for nxt in following:
                    if (node, nxt) not in seen_edges:
                        seen_edges.add((node, nxt))
                        queue.append(nxt)
                break
            nxt = following[0]
            if (node, nxt) in seen_edges or len(predecessors.get(nxt, [])) > 1:
                # a merge point also terminates the chain
                if (node, nxt) not in seen_edges:
                    seen_edges.add((node, nxt))
                    queue.append(nxt)
                break
            seen_edges.add((node, nxt))
            chain.append(nxt)
            node = nxt
        if len(chain) >= 1:
            segments.append(chain)
    return [s for s in segments if len(s) >= 2] or segments


def _float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
