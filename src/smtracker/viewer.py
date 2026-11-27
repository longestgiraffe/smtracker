"""The bridge from the canonical table to napari layers, and nothing else.

Why this module is data-first
-----------------------------
Everything here that *computes* is pure numpy/pandas and importable without Qt:
``track_layer_data``, ``outline_shapes``, ``background_image`` and
``ViewerBundle`` can be built and tested headless. Only ``launch`` and
``selftest`` touch napari, and they import it lazily. That keeps the analysis
package installable and testable on a machine with no GUI stack, and it means
the viewer's geometry is covered by ordinary tests rather than by clicking.

Every downstream view: track gallery, interactive fit overlay, QC dashboard,
is expected to consume ``ViewerBundle`` rather than re-deriving layer data.

Coordinate conventions, which are where viewers usually go wrong
----------------------------------------------------------------
- The canonical table is in **µm**; images are in **pixels**. Layer data is
  converted back to pixels here, using the same ``pixel_size_nm`` the analysis
  used, so an overlay can never silently drift from the analysis.
- napari indexes arrays as **(row, column) = (y, x)**, so track coordinates are
  emitted ``(y, x)``, not ``(x, y)``.
- u-track's coordinates are **1-based** (MATLAB); numpy/napari are 0-based, so
  ``PIXEL_ORIGIN_OFFSET`` is subtracted. A one-pixel overlay shift is exactly
  the kind of error that looks like "close enough" and is actually a bug.
- The ``Tracks`` layer wants ``[track_id, t, y, x]`` with an *integer* track id,
  so string uids are mapped to indices and the mapping is kept on the bundle.

Invariants
----------
- Fails loud when a movie image is missing rather than showing tracks over a
  blank canvas: a viewer that invents a background is worse than no viewer.
- Layer data is never computed twice: the bundle carries it.
- No napari import at module import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .canonical import CANONICAL_COLUMNS, AcquisitionParams

__all__ = [
    "ViewerError",
    "PIXEL_ORIGIN_OFFSET",
    "BACKGROUND_KINDS",
    "track_layer_data",
    "outline_shapes",
    "background_image",
    "ViewerBundle",
    "TrackHit",
    "nearest_track",
    "build_bundle",
    "launch",
    "populate_layers",
    "repopulate",
    "hide_layer_controls",
    "selftest",
]

#: MATLAB/u-track coordinates are 1-based; numpy and napari are 0-based.
PIXEL_ORIGIN_OFFSET = 1.0

#: Static backgrounds available in a u-track project when the raw stack is not
#: shipped with it. ``phase`` is the transmitted-light image the cells were
#: segmented on; ``sum`` is the fluorescence projection.
BACKGROUND_KINDS = ("phase", "sum")


class ViewerError(RuntimeError):
    """Raised when a viewer bundle cannot be built honestly."""


def track_layer_data(
    canonical: pd.DataFrame, *, movie: str | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    """``(data, properties, uid_to_id)`` for a napari ``Tracks`` layer.

    ``data`` is ``(N, 4)``: ``[track_id, frame, y_px, x_px]``. ``properties``
    carries whatever per-localization columns exist (``state``, ``posterior``,
    ``confined``, ``cell``, …) so the layer can be coloured by any of them
    without recomputing anything.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise ViewerError(f"not a canonical table (missing {missing})")
    params = canonical.attrs.get("params")
    if not isinstance(params, AcquisitionParams):
        raise ViewerError(
            "canonical table carries no AcquisitionParams, so µm cannot be "
            "converted back to pixels for display"
        )

    frame = canonical if movie is None else canonical[canonical["movie"] == movie]
    if frame.empty:
        raise ViewerError(f"no localizations for movie {movie!r}")
    frame = frame.sort_values(["uid", "frame"], kind="stable")

    uids = list(dict.fromkeys(frame["uid"]))
    uid_to_id = {uid: i for i, uid in enumerate(uids)}
    scale = params.um_per_pixel

    data = np.column_stack(
        [
            frame["uid"].map(uid_to_id).to_numpy(dtype=float),
            frame["frame"].to_numpy(dtype=float),
            frame["y_um"].to_numpy(dtype=float) / scale - PIXEL_ORIGIN_OFFSET,
            frame["x_um"].to_numpy(dtype=float) / scale - PIXEL_ORIGIN_OFFSET,
        ]
    )

    extras = [c for c in ("state", "posterior", "confined", "track_class",
                          "cell", "condition", "day", "l_norm", "d_norm")
              if c in frame.columns]
    properties = {c: frame[c].to_numpy() for c in extras}
    properties["uid"] = frame["uid"].to_numpy()
    return data, properties, uid_to_id


def outline_shapes(
    meshes, *, movie: str | None = None, keyed: bool = False
):
    """Cell outlines as ``(M, 2)`` ``(y, x)`` pixel arrays for a Shapes layer.

    Outlines are already in raw pixels (cells are segmented on the image), so
    only the origin offset and the axis order change here. ``keyed=True``
    returns ``{cell_id: polygon}`` instead of a list, which is what highlighting
    a *selected* track's cell needs.
    """
    # `read_project_cell_meshes` returns {movie: [CellMesh, ...]}; a plain
    # sequence of meshes is accepted too, so callers need not unpack it.
    if isinstance(meshes, dict):
        meshes = [m for values in meshes.values() for m in values]
    selected = [m for m in meshes if movie is None or m.movie == movie]
    if not selected:
        raise ViewerError(f"no cell outlines for movie {movie!r}")
    shapes = {} if keyed else []
    for mesh in selected:
        contour = np.asarray(mesh.polygon, dtype=float)
        if contour.ndim != 2 or contour.shape[1] != 2:
            raise ViewerError(f"cell {mesh.cell_id} has a malformed contour")
        polygon = contour[:, ::-1] - PIXEL_ORIGIN_OFFSET        # (x,y) -> (y,x)
        if keyed:
            shapes[f"{mesh.movie}:{mesh.cell_id}"] = polygon
        else:
            shapes.append(polygon)
    return shapes


def _background_path(root, movie: str, kind: str):
    """Where ``movie``'s ``kind`` image is, or None -- BOTH LAYOUTS and both namings.

    `Path(root) / kind / f"{movie}.tif"` was the whole rule, and it fails two ways that
    the rest of this project already handles:

    - the phase snapshot may lie LOOSE in the condition folder rather than in `phase/`
      (ruled 18 Oct 2026: "they can be either in phase subfolder or in main folder");
    - the file may carry a stray acquisition suffix an export kept, `series001.nd2.tif`,
      which is field `series001` (`channels.snapshot_field`, since 14 Oct).

    So it RAISED on exactly the folders that keep their snapshots loose, and because this
    is the one source of the built-in Phase and Sum backgrounds, the failure surfaced as
    "Images & tracks doesn't find the phase image" with a blank canvas.

    `channels.resolve_channels` is asked rather than reimplemented: it knows the folder
    aliases, the loose layout and the field naming, and it is the module the segmentation
    side already trusts. The plain path is tried FIRST so the common case costs one stat
    rather than a directory listing.
    """
    direct = Path(root) / kind / f"{movie}.tif"
    if direct.is_file():
        return direct
    try:
        from .segmentation.channels import resolve_channels

        resolution = resolve_channels(Path(root), (kind,))
        found = resolution.image_path(kind, movie)
    except Exception:                       # noqa: BLE001 - a reader, never fatal here
        return None
    return Path(found) if found is not None else None


def background_image(root: Path, movie: str, *, kind: str = "phase") -> np.ndarray:
    """A static background for ``movie``.

    The reference project ships the phase image and the fluorescence sum
    projection, not the raw 1001-frame stack, so a viewer built on it shows
    tracks over a projection. That is stated rather than hidden: if the file is
    absent this raises, because drawing tracks over a blank canvas invites
    reading the overlay as if it were registered to something.
    """
    if kind not in BACKGROUND_KINDS:
        raise ViewerError(f"kind must be one of {BACKGROUND_KINDS}, got {kind!r}")
    from .dayfolders import on_disk

    root, movie = on_disk(root, movie)          # a day folder's movie (``series001 (rep1)``) where it is
    path = _background_path(root, movie, kind)
    if path is None:
        raise ViewerError(
            f"no {kind} image for {movie} under {root}: looked in a {kind}/ folder and "
            f"beside the movies. The raw movie stack is not part of this project; pass "
            f"an explicit image or another kind."
        )
    try:
        import tifffile
    except ImportError as exc:                                  # pragma: no cover
        raise ViewerError("tifffile is required to read movie images") from exc
    return np.asarray(tifffile.imread(path))


@dataclass(frozen=True)
class TrackHit:
    """What a click resolved to. ``None`` is returned instead when nothing is near."""

    track_id: int
    uid: str
    frame: int
    distance_px: float


def nearest_track(
    bundle: "ViewerBundle",
    y_px: float,
    x_px: float,
    *,
    frame: float | None = None,
    max_distance_px: float = 3.0,
    frame_tolerance: float | None = None,
) -> TrackHit | None:
    """Resolve a click at ``(y_px, x_px)`` to a track: pure numpy, no Qt.

    This is the whole selection mechanism, deliberately kept out of the widget so
    it can be tested by asserting that a synthetic click on a known track returns
    that track, rather than by clicking and squinting.

    ``frame`` restricts the search to localizations near that time point, which
    is what makes selection usable at all: cells here hold several tracks whose
    2-D paths overlap, and without the time restriction a click lands on whatever
    happens to be closest in projection. ``None`` searches every frame.

    Returns ``None`` rather than a far-away guess when nothing is within
    ``max_distance_px``: a viewer that always selects *something* teaches the
    user to distrust it.
    """
    if bundle.tracks.size == 0:
        return None
    if max_distance_px <= 0:
        raise ViewerError(f"max_distance_px must be > 0, got {max_distance_px}")

    data = bundle.tracks
    candidates = np.ones(len(data), dtype=bool)
    if frame is not None:
        tolerance = 0.5 if frame_tolerance is None else frame_tolerance
        candidates = np.abs(data[:, 1] - frame) <= tolerance
        if not candidates.any():
            return None

    index = np.flatnonzero(candidates)
    distances = np.hypot(data[index, 2] - y_px, data[index, 3] - x_px)
    best = int(index[np.argmin(distances)])
    distance = float(distances.min())
    if distance > max_distance_px:
        return None

    track_id = int(data[best, 0])
    id_to_uid = {i: uid for uid, i in bundle.uid_to_id.items()}
    return TrackHit(
        track_id=track_id,
        uid=id_to_uid[track_id],
        frame=int(data[best, 1]),
        distance_px=distance,
    )


@dataclass
class ViewerBundle:
    """Everything a napari window needs, computed once, headless-constructible."""

    movie: str
    tracks: np.ndarray
    properties: dict[str, np.ndarray]
    uid_to_id: dict[str, int]
    outlines: list[np.ndarray] = field(default_factory=list)
    outlines_by_cell: dict[str, np.ndarray] = field(default_factory=dict)
    images: dict[str, np.ndarray] = field(default_factory=dict)
    pixel_size_nm: float | None = None

    @property
    def image(self) -> np.ndarray | None:
        """The first background image, or ``None``. Kept for callers that only
        need "an" image (the QC dashboard's field shape, for instance)."""
        return next(iter(self.images.values()), None)

    @property
    def image_kind(self) -> str | None:
        return next(iter(self.images), None)

    @property
    def n_tracks(self) -> int:
        return len(self.uid_to_id)

    @property
    def n_localizations(self) -> int:
        return len(self.tracks)

    def track_rows(self, uid: str) -> np.ndarray:
        """The track's own ``(N, 4)`` ``[track_id, frame, y, x]`` rows.

        This is Tracks-layer shaped on purpose: a *highlighted* track has to be
        drawn as a Tracks layer so it appears, moves and fades with the time
        slider exactly like every other track. Drawn as a static path it would
        hang on screen through the whole movie.
        """
        if uid not in self.uid_to_id:
            raise ViewerError(f"no track {uid!r} in this bundle")
        rows = self.tracks[self.tracks[:, 0] == self.uid_to_id[uid]]
        return rows[np.argsort(rows[:, 1])]

    def track_path(self, uid: str) -> np.ndarray:
        """The track's ``(M, 2)`` ``(y, x)`` pixel path, in frame order.

        Used to draw the *selected* track prominently. Pure lookup on the
        bundle's own array, so the widget does no geometry.
        """
        if uid not in self.uid_to_id:
            raise ViewerError(f"no track {uid!r} in this bundle")
        return self.track_rows(uid)[:, 2:4]

    def cell_of(self, uid: str) -> str | None:
        """Which cell a track belongs to, from the properties riding along."""
        cells = self.properties.get("cell")
        uids = self.properties.get("uid")
        if cells is None or uids is None:
            return None
        match = np.flatnonzero(uids == uid)
        return str(cells[match[0]]) if match.size else None

    def cell_outline(self, uid: str) -> np.ndarray | None:
        """The outline of the cell a track sits in, ready to highlight."""
        cell = self.cell_of(uid)
        return self.outlines_by_cell.get(cell) if cell else None

    def first_frame(self, uid: str) -> int:
        """The frame a track starts on: what the time slider should jump to.

        Pure lookup on the bundle's own array, so the widget does no arithmetic.
        """
        if uid not in self.uid_to_id:
            raise ViewerError(f"no track {uid!r} in this bundle")
        rows = self.tracks[self.tracks[:, 0] == self.uid_to_id[uid]]
        return int(rows[:, 1].min())

    def summary(self) -> str:
        background = (
            ", ".join(f"{kind} {image.shape}" for kind, image in self.images.items())
            if self.images else "no background image"
        )
        return (
            f"{self.movie}: {self.n_tracks} tracks / {self.n_localizations} "
            f"localizations, {len(self.outlines)} cell outlines, {background}"
        )


def build_bundle(
    canonical: pd.DataFrame,
    *,
    movie: str,
    meshes=None,
    root: Path | None = None,
    image_kinds=("phase", "sum"),
) -> ViewerBundle:
    """Assemble one movie's layers. Everything optional degrades loudly.

    ``image_kinds`` loads **every** available background, so the phase image and
    the fluorescence sum projection both appear in the layer list and can be
    toggled: they answer different questions (where is the cell vs where did
    the signal actually come from). A kind that is missing is skipped rather
    than raising, but at least one must load if any were asked for, so a project
    with no images at all still fails loudly.
    """
    tracks, properties, uid_to_id = track_layer_data(canonical, movie=movie)
    outlines = outline_shapes(meshes, movie=movie) if meshes else []
    outlines_by_cell = (
        outline_shapes(meshes, movie=movie, keyed=True) if meshes else {}
    )

    images: dict[str, np.ndarray] = {}
    if root is not None:
        if isinstance(image_kinds, str):
            image_kinds = (image_kinds,)
        errors = []
        for kind in image_kinds:
            try:
                images[kind] = background_image(Path(root), movie, kind=kind)
            except ViewerError as exc:
                errors.append(str(exc))
        if not images:
            raise ViewerError(
                "no background image could be loaded for "
                f"{movie!r}: " + " | ".join(errors)
            )

    params = canonical.attrs.get("params")
    return ViewerBundle(
        movie=movie,
        tracks=tracks,
        properties=properties,
        uid_to_id=uid_to_id,
        outlines=outlines,
        outlines_by_cell=outlines_by_cell,
        images=images,
        pixel_size_nm=params.pixel_size_nm if params else None,
    )


def launch(
    bundle: ViewerBundle,
    *,
    color_by: str | None = None,
    show: bool = True,
    editable: bool = False,
    show_layer_controls: bool = False,
):
    """Open a napari window on ``bundle``. Imports napari lazily.

    ``editable=False`` (the default) opens **read-only**, and that is a
    correctness measure rather than tidiness: the cell outlines came from Oufti
    and *define* the cell filter and the bootstrap units. A viewer that lets a
    stray drag nudge a polygon invites a silent mismatch between what is on
    screen and what was analysed. So the Shapes layer is made non-editable and
    non-interactive, and the tracks layer (the thing you came to look at) is
    the selected one, which is also what puts napari's editing toolbar away.

    Returns the viewer so callers (and the packaging smoke test) can inspect or
    close it without a running event loop.
    """
    import napari

    viewer = napari.Viewer(title=f"SMTracker · {bundle.movie}", show=show)
    populate_layers(viewer, bundle, color_by=color_by, editable=editable,
                    show_layer_controls=show_layer_controls)
    return viewer


def populate_layers(viewer, bundle: ViewerBundle, *, color_by: str | None = None,
                    editable: bool = False, show_layer_controls: bool = False):
    """Add ``bundle``'s image/cells/tracks layers to an existing ``viewer``.

    Extracted from :func:`launch` so switching movie/condition can REUSE the same
    napari window (one GL context for the session) via :func:`repopulate`, instead
    of opening a second window, which churns the shared vispy GL context and
    crashes on some drivers (``GL_INVALID_OPERATION``). Returns the tracks layer.
    """
    for kind, array in bundle.images.items():
        # Every background goes in the layer list so they can be toggled: the
        # phase image shows where the cell is, the fluorescence sum shows where
        # the signal came from, and those are different questions. Only the
        # first is visible initially so the window opens on one thing.
        layer = viewer.add_image(array, name=kind, colormap="gray")
        layer.interactive = editable
        layer.visible = kind == bundle.image_kind
    if bundle.outlines:
        shapes = viewer.add_shapes(
            bundle.outlines, shape_type="polygon", name="cells",
            edge_color="cyan", face_color="transparent", edge_width=0.3,
        )
        shapes.editable = editable
        shapes.interactive = editable
    if color_by is not None and color_by not in bundle.properties:
        raise ViewerError(
            f"cannot colour by {color_by!r}; available: "
            f"{sorted(bundle.properties)}"
        )
    tracks = viewer.add_tracks(
        bundle.tracks, properties=bundle.properties, name="tracks",
        color_by=color_by or "track_id", tail_length=10,
    )
    # Select the tracks layer last so it is the active one: napari shows the
    # controls of whatever is selected, and the cells layer's controls are an
    # editing toolbar for a segmentation that must not be edited here.
    viewer.layers.selection.active = tracks
    if not show_layer_controls:
        hide_layer_controls(viewer)
    return tracks


def repopulate(viewer, bundle: ViewerBundle, *, color_by: str | None = None,
               editable: bool = False, show_layer_controls: bool = False):
    """Replace ALL layers in ``viewer`` with ``bundle``'s: IN PLACE, one window.

    This is how the track viewer switches movie/condition without opening a second
    napari window: clear the layers and re-add from the new bundle in the SAME GL
    context. Stable where reopening a window is not. Returns the tracks layer.
    """
    viewer.title = f"SMTracker · {bundle.movie}"
    viewer.layers.clear()
    return populate_layers(viewer, bundle, color_by=color_by, editable=editable,
                           show_layer_controls=show_layer_controls)


def hide_layer_controls(viewer) -> bool:
    """Hide napari's layer-**controls** dock, keeping the layer **list**.

    The controls panel is a per-layer editing toolbar; this tool is for looking
    at tracks, so it is clutter on open. The layer *list* stays, because
    toggling between the phase image and the fluorescence sum is a real part of
    reading a track.

    Returns whether it succeeded. This reaches into a private napari attribute,
    so it is isolated here and failure is reported rather than raised: a
    cosmetic panel is not worth taking the window down for, but a silent no-op
    would be untraceable when a napari upgrade renames it.
    """
    qt_viewer = getattr(getattr(viewer, "window", None), "_qt_viewer", None)
    dock = getattr(qt_viewer, "dockLayerControls", None)
    if dock is None:
        return False
    dock.setVisible(False)
    return True


def selftest(*, verbose: bool = True) -> bool:
    """Construct a napari viewer with synthetic layers and tear it down.

    This exists for the **packaging** smoke test: a frozen build can run it with
    no data and no display interaction to prove that Qt, vispy and napari were
    bundled correctly. Failing here at Phase 7, with the whole UI built on top,
    would be far more expensive than failing here now.
    """
    import napari

    rng = np.random.default_rng(0)
    frames = np.repeat(np.arange(10), 3)
    ids = np.tile(np.arange(3), 10)
    tracks = np.column_stack(
        [ids, frames, rng.normal(30, 5, 30), rng.normal(30, 5, 30)]
    )
    viewer = napari.Viewer(show=False)
    try:
        viewer.add_image(rng.random((64, 64)), name="synthetic")
        viewer.add_tracks(tracks, name="synthetic tracks")
        ok = len(viewer.layers) == 2
    finally:
        viewer.close()
    if verbose:
        print(
            f"napari {napari.__version__} selftest: "
            f"{'OK' if ok else 'FAILED'} ({len(tracks)} points, 2 layers)"
        )
    return ok
