"""Cell outlines (Oufti / MicrobeTracker) → per-cell track assignment.

This is where the ``cell`` column comes from, and it is not cosmetic: the legacy
analysis **only fits tracks that lie inside a segmented cell**, and honest error
bars bootstrap over cells. Getting this wrong changes the numbers.

The legacy rule (``SMT_GUI/@SMT/cellTracks.m``)
----------------------------------------------
For every track and every cell outline::

    [in, on] = inpolygon(x, y, model(:,1), model(:,2));
    if all(in | on)   % EVERY localization inside or on the outline
        <track belongs to this cell>

Tracks not fully inside any cell are **dropped from the analysis entirely**.
Boundary points count as inside. Coordinates are raw camera **pixels**, so
assignment happens on the raw table, before the µm conversion.

Only the **first frame's** outlines are used (``cellList.meshData{1}`` in
``dataImport.m``): cells are assumed not to move during the movie.

Deliberate divergences
----------------------
- If a track is fully inside **more than one** outline (overlapping segmentation),
  the legacy code writes it into *each* such cell, counting the same track twice.
  Here it is assigned to the first match and the ambiguity is reported, so a
  track is never double-counted.
- Legacy forces NaN localizations to count as "inside". Our ingest drops NaN
  rows outright, so there is nothing to force; the effect is the same.

Fail loud: a missing mesh file, an empty ``cellList``, or a movie with no
outlines raises rather than silently analysing unsegmented data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "assignment_funnel",
    "format_funnel",
    "CellMeshError",
    "CellMesh",
    "read_cell_meshes",
    "read_project_cell_meshes",
    "write_cell_meshes",
    "assign_cells",
    "points_in_polygon",
    "TRACKER_PIXEL_ORIGIN",
    "ALIGNMENT_WARN_PX",
    "to_tracker_origin",
    "outline_alignment",
    "alignment_mismatch",
    "mark_pixel_origin",
]

#: Folder holding one ``<movie>.mat`` of outlines per movie, as Oufti and this program write it.
CELL_MESH_FOLDER = "cell_meshes"
#: The same folder as people name it by hand: ``cell meshes``, ``cellmeshes``, ``cell-meshes``, any case (a user's
#: folder has ``cell meshes`` in every condition; asked 20 Oct 2026, "can u fix that all of these are
#: found?"). Read through `mesh_folder`, never by spelling the name.
_MESH_NAME = re.compile(r"^cell[\s_-]?meshes$", re.IGNORECASE)


def is_mesh_folder_name(name: str) -> bool:
    """Whether a folder NAME is an outlines folder, however it is spelled (`_MESH_NAME`)."""
    return bool(_MESH_NAME.match(str(name)))


def mesh_folder(folder, name: str = CELL_MESH_FOLDER) -> Path:
    """``folder``'s outlines folder whatever its spelling (``cell_meshes`` first, else ``cell meshes`` /
    ``cellmeshes`` / ``cell-meshes``, any case), else ``folder/cell_meshes``: where new outlines are written, so
    a hand-named folder is written INTO, never joined by a second one. Another ``name`` (``cell_meshes_auto``) is
    taken literally."""
    folder = Path(folder)
    exact = folder / name
    if exact.is_dir() or name != CELL_MESH_FOLDER:
        return exact
    try:
        for child in sorted(folder.iterdir()):
            if child.is_dir() and is_mesh_folder_name(child.name):
                return child
    except OSError:
        pass
    return exact

# Tolerance for "on the boundary", in pixels. Oufti outlines are sub-pixel
# floats; this only decides ties exactly on an edge.
_ON_EDGE_TOL = 1e-9

#: How much wider an outline the near-edge count tries (pixels), beyond the tolerance in use:
#: a DROPPED track every point of which lies within this of an outline is counted, so QC can
#: say how many tracks a tight outline costs (`assign_cells`, ``n_tracks_near_edge``).
NEAR_EDGE_PX = 1.0


class CellMeshError(RuntimeError):
    """Raised when cell outlines are missing or cannot be parsed."""


#: U-TRACK'S OWN PROJECT FILES, WHICH ARE NOT OUTLINES AND SOMETIMES SIT BESIDE THEM.
#: `movieList.mat` holds an `ML` variable and `movieData.mat` an `MD`; neither carries
#: `cellList`. Found in a real project's `cell_meshes/` folder, where it took the WHOLE
#: application down at startup before any window opened -- the outline reader rightly
#: refuses a file with no `cellList`, and the discovery glob had handed it one.
#:
#: SKIPPED BY NAME, NOT BY SNIFFING FOR `cellList`. Skipping anything that lacks the
#: variable would also silently skip a genuinely corrupt outline file, which is exactly
#: the failure the loud error exists to surface. These two names are known to be a
#: DIFFERENT KIND OF FILE; anything else missing `cellList` still raises.
NOT_OUTLINE_STEMS = frozenset({"movielist", "moviedata"})


@dataclass(frozen=True)
class CellMesh:
    """One segmented cell outline, in camera pixels.

    ``polygon`` is the closed contour (Oufti's ``model``). ``mesh`` is Oufti's
    rib array ``[x_left, y_left, x_right, y_right]`` per cross-section, which is
    what defines the cell's **centerline**, required for the cell-coordinate
    projection in ``smtracker.cellcoords``. It is ``None`` for outlines that
    carry only a contour.
    """

    movie: str
    cell_id: str
    polygon: np.ndarray          # (N, 2) closed contour, pixels
    mesh: np.ndarray | None = None   # (N, 4) ribs, pixels
    #: Where the file puts a pixel's centre (`pixelOrigin`): 0 (the image's own index, what SynR and the
    #: segmentation write) or 1 (MATLAB's, as Oufti and u-track); None when the file does not record it.
    pixel_origin: int | None = None

    @property
    def has_centerline(self) -> bool:
        return self.mesh is not None and len(self.mesh) >= 2

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        x, y = self.polygon[:, 0], self.polygon[:, 1]
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())


def read_cell_meshes(
    path: str | Path, *, movie: str | None = None, on_empty: str = "raise"
) -> list[CellMesh]:
    """Read one Oufti/MicrobeTracker ``.mat`` of cell outlines.

    Handles both layouts: Oufti (``cellList`` a struct with ``meshData``) and
    MicrobeTracker (``cellList`` a cell array). Only the first frame is used.

    ``on_empty`` controls what happens when the file holds no outlines at all
    (a movie that was never segmented): ``"raise"`` (default, fail loud) or
    ``"empty"`` to return ``[]`` so the caller can report and skip it.
    """
    from scipy.io import loadmat

    path = Path(path)
    if not path.exists():
        raise CellMeshError(f"cell mesh file not found: {path}")
    movie = movie if movie is not None else path.stem

    mat = loadmat(path, struct_as_record=False, squeeze_me=True)
    if "cellList" not in mat:
        raise CellMeshError(
            f"{path}: no 'cellList' variable "
            f"(found {[k for k in mat if not str(k).startswith('__')]})"
        )
    cell_list = mat["cellList"]

    if on_empty not in ("raise", "empty"):
        raise CellMeshError(f"on_empty must be 'raise' or 'empty', got {on_empty!r}")

    if hasattr(cell_list, "meshData"):          # Oufti
        frames = np.atleast_1d(cell_list.meshData)
        if frames.size == 0:
            return _no_outlines(path, movie, on_empty, "cellList.meshData is empty")
        # meshData is {frames}{cells}, but scipy's squeeze_me collapses a
        # single-frame file so the array IS the cell list. Distinguish by
        # content: a cell entry is a struct carrying 'model'/'mesh'.
        entries = frames if _is_cell_entry(frames[0]) else np.atleast_1d(frames[0])
        ids = np.atleast_1d(getattr(cell_list, "cellId", []))
        first_frame_ids = ids if _is_flat_ids(ids, len(entries)) else (
            np.atleast_1d(ids[0]) if ids.size else []
        )
    else:                                        # MicrobeTracker
        entries = np.atleast_1d(cell_list)
        if entries.size and isinstance(entries[0], np.ndarray):
            entries = np.atleast_1d(entries[0])
        first_frame_ids = []

    origin = _recorded_origin(mat, path)
    meshes: list[CellMesh] = []
    for i, entry in enumerate(entries):
        polygon = _polygon_of(entry)
        if polygon is None:
            continue
        ribs = _ribs_of(entry)
        cell_id = (
            str(int(first_frame_ids[i]))
            if len(first_frame_ids) > i and np.isscalar(first_frame_ids[i])
            else str(i + 1)
        )
        meshes.append(
            CellMesh(movie=movie, cell_id=cell_id, polygon=polygon, mesh=ribs, pixel_origin=origin)
        )

    if not meshes:
        return _no_outlines(path, movie, on_empty, "parsed 0 cell outlines")
    return meshes


def _recorded_origin(mat: dict, path) -> int | None:
    """The file's ``pixelOrigin`` (0 or 1), or None when it records none (Oufti's files, and every file written
    before 22 Oct 2026)."""
    if "pixelOrigin" not in mat:
        return None
    value = np.asarray(mat["pixelOrigin"]).ravel()
    if value.size != 1 or float(value[0]) not in (0.0, 1.0):
        raise CellMeshError(f"{path}: pixelOrigin must be 0 or 1, got {mat['pixelOrigin']!r}")
    return int(value[0])


# -- the pixel ORIGIN: where a pixel's centre lies, 0 or 1 -------------------------------- #
#: Where each tracker puts a pixel's centre. MEASURED on SynR (22 Oct 2026) against the rendered image: u-track
#: +0.97 px (MATLAB counts from 1), TrackMate 0.00 (it counts from 0). Track tables from other tools: not known,
#: so never shifted.
TRACKER_PIXEL_ORIGIN = {"utrack": 1, "trackmate": 0}
#: An alignment this far off in both axes is a pixel-origin mismatch, not chance (`outline_alignment`).
ALIGNMENT_WARN_PX = 0.5


def to_tracker_origin(meshes: dict, tracker: str) -> tuple:
    """``(meshes, shifts)``: outlines that RECORD their origin moved onto ``tracker``'s (`TRACKER_PIXEL_ORIGIN`);
    unrecorded ones, and every outline when the tracker's origin is not known, untouched. ``shifts`` is
    ``{movie: px}`` for the movies moved.

    WHY ONLY RECORDED ONES. Found 22 Oct 2026: SynR's true outlines are written in the image's own pixels (0) and
    its u-track tracks sat +1.08 +- 0.09 px off them; but the real projects measured (ten u-track, four TrackMate)
    agree with their outlines within half a pixel, each segmented in its own tracker's convention -- and an
    unrecorded file does not say which. Moving those on a guess would break the ones that are right.
    `outline_alignment` measures every project instead, and QC says it."""
    from dataclasses import replace

    target = TRACKER_PIXEL_ORIGIN.get(tracker)
    if target is None:
        return meshes, {}
    out, shifts = {}, {}
    for movie, cells in meshes.items():
        moved = []
        for cell in cells:
            if cell.pixel_origin is None or cell.pixel_origin == target:
                moved.append(cell)
                continue
            d = float(target - cell.pixel_origin)
            moved.append(replace(cell, polygon=np.asarray(cell.polygon, float) + d,
                                 mesh=None if cell.mesh is None else np.asarray(cell.mesh, float) + d,
                                 pixel_origin=target))
            shifts[movie] = d
        out[movie] = moved
    return out, shifts


def outline_alignment(raw, meshes: dict, *, reach_px: float = 6.0, min_points: int = 30) -> dict | None:
    """Do the tracks and the outlines share a pixel origin? MEASURED, before the cell filter: for each cell, the
    mean offset of the localisations within ``reach_px`` of its outline from the outline's AREA centroid; averaged
    over cells, each weighing the same. Molecules spread evenly in their cells give ~0; one pixel of origin between
    tracker and outline gives ~+-1 in both x and y (SynR with u-track: +1.08; real projects: -0.5 to +0.5).

    ``raw`` has ``movie``, ``x``, ``y`` (pixels, as read); ``meshes`` ``{movie: [CellMesh]}`` in the same names.
    Returns ``{"x", "y", "se_x", "se_y", "cells"}`` (px), or None with fewer than three cells to measure."""
    from matplotlib.path import Path as MPath

    offsets = []
    for movie, part in raw.groupby(raw["movie"].astype(str), sort=False):
        cells = meshes.get(movie)
        if not cells:
            continue
        xy = part[["x", "y"]].to_numpy(float)
        for cell in cells:
            poly = np.asarray(cell.polygon, float)[:, :2]
            if len(poly) < 3:
                continue
            path = MPath(poly)
            near = xy[path.contains_points(xy, radius=reach_px) | path.contains_points(xy, radius=-reach_px)]
            if len(near) < min_points:
                continue
            x, y = poly[:, 0], poly[:, 1]
            cross = x * np.roll(y, -1) - np.roll(x, -1) * y
            area = cross.sum() / 2.0
            if area == 0:
                continue
            centroid = np.array([((x + np.roll(x, -1)) * cross).sum(),
                                 ((y + np.roll(y, -1)) * cross).sum()]) / (6.0 * area)
            offsets.append(near.mean(axis=0) - centroid)
    if len(offsets) < 3:
        return None
    o = np.asarray(offsets)
    se = o.std(axis=0, ddof=1) / np.sqrt(len(o))
    return {"x": float(o[:, 0].mean()), "y": float(o[:, 1].mean()), "se_x": float(se[0]), "se_y": float(se[1]),
            "cells": int(len(o))}


def alignment_mismatch(alignment: dict | None) -> bool:
    """True when ``alignment`` (`outline_alignment`) is a pixel-origin mismatch: both axes at least
    `ALIGNMENT_WARN_PX` off, the same way, each beyond three standard errors."""
    if not alignment:
        return False
    x, y = alignment["x"], alignment["y"]
    return (abs(x) >= ALIGNMENT_WARN_PX and abs(y) >= ALIGNMENT_WARN_PX and x * y > 0
            and abs(x) > 3 * alignment["se_x"] and abs(y) > 3 * alignment["se_y"])


def _no_outlines(path: Path, movie: str, on_empty: str, why: str) -> list[CellMesh]:
    if on_empty == "empty":
        return []
    raise CellMeshError(
        f"{path}: {why}. Movie '{movie}' appears unsegmented, and the analysis only "
        "uses tracks inside a cell. "
        "Pass on_empty='empty' (or allow_unsegmented=True) to skip it."
    )


def _is_cell_entry(entry) -> bool:
    """True if this is a single cell (a struct with an outline), not a frame."""
    return getattr(entry, "model", None) is not None or getattr(entry, "mesh", None) is not None


def _is_flat_ids(ids: np.ndarray, n_cells: int) -> bool:
    """True if ``cellId`` is already a flat list of ids for this frame's cells."""
    return ids.size == n_cells and all(np.isscalar(v) or np.ndim(v) == 0 for v in ids)


def _ribs_of(entry) -> np.ndarray | None:
    """Oufti's ``mesh``: ``[x_left, y_left, x_right, y_right]`` per cross-section."""
    mesh = getattr(entry, "mesh", None)
    if mesh is None:
        return None
    mesh = np.asarray(mesh, dtype=float)
    if mesh.ndim == 2 and mesh.shape[1] == 4 and mesh.shape[0] >= 2:
        return mesh
    return None


def _polygon_of(entry) -> np.ndarray | None:
    """Extract the outline polygon from one cell entry (``model``, else ``mesh``)."""
    model = getattr(entry, "model", None)
    if model is not None:
        model = np.asarray(model, dtype=float)
        if model.ndim == 2 and model.shape[0] >= 3 and model.shape[1] >= 2:
            return model[:, :2]

    # Fall back to the 4-column mesh: [x_left, y_left, x_right, y_right] per rib.
    mesh = getattr(entry, "mesh", None)
    if mesh is not None:
        mesh = np.asarray(mesh, dtype=float)
        if mesh.ndim == 2 and mesh.shape[1] == 4 and mesh.shape[0] >= 2:
            left = mesh[:, 0:2]
            right = mesh[::-1, 2:4]
            return np.vstack([left, right])
    return None


@dataclass(frozen=True)
class OutlineCoverage:
    """Which movies of one condition already have cell outlines.

    IT PARSES NOTHING AND CREATES NOTHING. Every count is a directory listing, matching
    `movieprep.condition_progress`, which opens no movie for the same reason: an overview
    that had to read its inputs would be the wait it exists to explain. Reading the
    outlines themselves is `read_project_cell_meshes`, and it is far from free.

    Creating the folder is deliberately NOT done here. `organise` names `cell_meshes/` as
    still-to-come rather than making it, because an empty folder turns
    `read_project_cell_meshes`' clear refusal into a subtler failure further down.

    WHY A TRACKING WINDOW WANTS THIS. Outlines are what let the detection threshold be
    chosen from the false-positive rate OUTSIDE cells (`smtracker.detection`, which
    refuses without meshes), what make linking ambiguity an IN-CELL number, and what turn
    the cell-filter loss into a pre-flight check instead of a post-mortem. Measured on
    one lab condition: whole-field ambiguity reads 36.6 per cent against an in-cell 3.3, and
    the difference is one movie -- series003 -- whose signal is three quarters outside the
    cells. Without outlines that movie is invisible until after the condition is tracked.
    """

    folder: Path | None
    movies: tuple[str, ...]
    with_outlines: tuple[str, ...]

    @property
    def n_movies(self) -> int:
        return len(self.movies)

    @property
    def n_with_outlines(self) -> int:
        return len(self.with_outlines)

    @property
    def missing(self) -> tuple[str, ...]:
        have = set(self.with_outlines)
        return tuple(m for m in self.movies if m not in have)

    @property
    def complete(self) -> bool:
        return bool(self.movies) and not self.missing

    def caption(self) -> str:
        if self.folder is None:
            return "no cell_meshes/ folder yet"
        if not self.movies:
            return "no movies to match outlines against"
        movie_word = "movie" if self.n_movies == 1 else "movies"
        if self.complete:
            return f"outlines for all {self.n_movies} {movie_word}"
        return f"outlines for {self.n_with_outlines} of {self.n_movies} {movie_word}"


def outline_file(meshes_dir, field) -> Path:
    """THE outline file of ``field`` in ``meshes_dir``: the one under its own name, else the ONE file there with the
    same movie number (`rawmovie.same_movie`), else a new file under its own name.

    One answer for the segmenter, the editor and the analysis (reported 25 Oct 2026: MetaMorph names a movie and its
    sum image ``series0013_YFPLaser`` past series009 and its phase snapshot ``series013_YFPLaser``; the segmenter
    filed its outlines under the first, the editor did not find them under the second and saved the hand-drawn
    outlines there as a new file, and the analysis read the automatic file). A field whose outlines are filed under
    another name for the same movie is read and WRITTEN there, so one movie never gets two files."""
    from .ingest import is_sidecar
    from .rawmovie import same_movie

    meshes_dir = Path(meshes_dir)
    own = meshes_dir / f"{field}.mat"
    if own.exists() or not meshes_dir.is_dir():
        return own
    stems = [p.stem for p in meshes_dir.glob("*.mat")
             if not is_sidecar(p) and p.stem.lower() not in NOT_OUTLINE_STEMS]
    hit = same_movie(str(field), stems)
    return meshes_dir / f"{hit}.mat" if hit is not None else own


def outline_twins(meshes, movies) -> dict:
    """``{movie: [file stems]}`` for each movie TWO or more outline files belong to: one under its own name and one
    under another name with its number, or two under other names. Read by `meshes_by_movie_name`, which refuses them."""
    from .rawmovie import movie_number

    names = [str(m) for m in movies]
    taken = set(names)
    out = {}
    for movie in names:
        number = movie_number(movie)
        if number is None:
            continue
        mine = sorted(k for k in meshes if k not in taken and movie_number(k) == number)
        if (movie in meshes and mine) or len(mine) > 1:
            out[movie] = ([movie] if movie in meshes else []) + mine
    return out


def meshes_by_movie_name(meshes: dict, movies) -> dict:
    """``meshes`` keyed by the TRACKS' movie names: an outline file named otherwise -- ``001.mat`` for the movie
    ``series001_YFPLaser``, from a snapshot named by its number -- is taken under the movie's name when it is the
    ONE file no movie claims by name that carries the movie's series number (`rawmovie.same_movie`).

    Asked 23 Oct 2026 ("make it clear that the analysis tool finds movies, phase and sum image" of one user's
    MetaMorph folders): `assign_cells` pairs by name, so those outlines paired nothing and every movie would have
    been left out as unsegmented. Never merges two files into one movie, never guesses between two."""
    from dataclasses import replace

    from .rawmovie import same_movie

    names = [str(m) for m in movies]
    # TWO FILES FOR ONE MOVIE ARE REFUSED, BY NAME (25 Oct 2026): taking the one under the movie's own name left a
    # hand-corrected twin unread without a word. Which one is meant is the user's to say.
    twins = outline_twins(meshes, names)
    if twins:
        listed = "; ".join(f"{m}: {', '.join(f + '.mat' for f in files)}" for m, files in twins.items())
        raise CellMeshError(f"two outline files for one movie: {listed}. Keep the one you mean in cell_meshes "
                            f"and move the other out.")
    out = dict(meshes)
    unclaimed = [k for k in meshes if k not in names]
    for movie in names:
        if movie in out:
            continue
        hit = same_movie(movie, unclaimed)
        if hit is not None:
            out[movie] = [replace(cell, movie=movie) for cell in out.pop(hit)]
            unclaimed.remove(hit)
    return out


def outline_coverage(root, movies=None, *, folder: str = CELL_MESH_FOLDER
                     ) -> OutlineCoverage:
    """Which of ``movies`` have an outline file under ``<root>/<folder>``.

    ``movies`` are movie NAMES (``series001``), the same identifier used everywhere else
    here; pass them in rather than letting this rediscover them, because a second
    discovery is a second opinion about what the condition contains and the two would
    eventually disagree. With ``movies=None`` it reports what the folder holds and
    matches nothing against it.

    Files are skipped by the same two rules `read_project_cell_meshes` uses -- AppleDouble
    sidecars, and u-track's own project files by name -- so the two cannot disagree about
    what counts as an outline.
    """
    root = Path(root)
    mesh_dir = mesh_folder(root, folder)
    if not mesh_dir.is_dir():
        # ONE LEVEL DOWN, never `**`. `read_project_cell_meshes` searches the whole tree
        # because it is opening a project someone points it at; this is an OVERVIEW called
        # per condition, where two things go wrong with a recursive search: it walks an
        # arbitrarily large tree, when the contract is that this costs a directory
        # listing, and it will happily report a DIFFERENT condition's outlines as this
        # one's. Verified: pointed at a home directory the recursive version found an
        # unrelated `cell_meshes` and matched `series001` against it.
        found = sorted(mesh_folder(q, folder) for q in root.iterdir()
                       if q.is_dir() and mesh_folder(q, folder).is_dir())
        mesh_dir = found[0] if found else None

    names = tuple(str(m) for m in (movies or ()))
    if mesh_dir is None:
        return OutlineCoverage(folder=None, movies=names, with_outlines=())

    from .ingest import is_sidecar

    from .rawmovie import same_movie

    stems = {q.stem for q in mesh_dir.glob("*.mat")
             if not is_sidecar(q) and q.stem.lower() not in NOT_OUTLINE_STEMS}
    # by name, or the one outline with the movie's series number (`meshes_by_movie_name`, as the loader pairs them)
    have = (tuple(m for m in names if m in stems or same_movie(m, stems - set(names)) is not None)
            if names else tuple(sorted(stems)))
    return OutlineCoverage(folder=mesh_dir, movies=names or tuple(sorted(stems)),
                           with_outlines=have)


#: What outlines unlock at tracking time, named so an offer to make them can say why
#: rather than reading as a chore. Each entry is short enough to put on one line.
OUTLINES_UNLOCK = (
    "the detection threshold measured against the false-positive rate outside cells, "
    "instead of the adaptive rule alone",
    "linking ambiguity as an IN-CELL number - whole-field reads 36.6 % on "
    "one lab condition where in-cell reads 3.3 %",
    "the cell-filter loss per movie before tracking, not after - one movie of a lab "
    "condition discards 75 % of its localisations",
)


def read_project_cell_meshes(
    root: str | Path,
    *,
    folder: str = CELL_MESH_FOLDER,
    allow_unsegmented: bool = True,
) -> dict[str, list[CellMesh]]:
    """Read every ``<movie>.mat`` of outlines under ``<root>/<folder>``.

    A movie whose file holds no outlines is mapped to an empty list and RECORDED
    under ``.attrs["unsegmented_movies"]`` by :func:`assign_cells`; every track of
    such a movie is then dropped. One unsegmented series out of thirty is an
    ordinary occurrence and should not stop a project from loading.

    The default used to be fail-loud, on the reasoning that a movie silently
    contributing nothing must never be mistaken for one that contributed. That
    reasoning still holds: what changed is that it is no longer silent:
    :func:`assignment_funnel` reports the unsegmented movies and the discarded
    track counts, so the information is surfaced without being fatal. Pass
    ``allow_unsegmented=False`` to restore the hard failure.
    """
    root = Path(root)
    # REPLICATE DAYS AS SUB-FOLDERS (`dayfolders`): each day folder's outlines, under the movie names the loader
    # gives them (``series001 (rep1)``). A day folder without outlines is refused by name, as a condition is.
    from dataclasses import replace as _replace

    from .dayfolders import day_folders, movie_name, numbered_folders

    folders = day_folders(root)
    if folders is not None:
        out = {}
        for day_folder in folders:
            for movie, cells in read_project_cell_meshes(
                    day_folder, folder=folder, allow_unsegmented=allow_unsegmented).items():
                name = movie_name(movie, day_folder.name)
                out[name] = [_replace(cell, movie=name) for cell in cells]
        return out
    # NUMBERED DAY FOLDERS (`dayfolders.numbered_folders`: `LB/LB-1`, `LB-2`, `LB-3`, series numbered on): every
    # day's outlines under the movies' own names. Only the first folder found was read, so days 2 and 3 were
    # refused as unsegmented (`a lab folder`, 20 Oct 2026). A day folder without outlines is refused by name.
    numbered = numbered_folders(root)
    if numbered is not None:
        return _merged([read_project_cell_meshes(day_folder, folder=folder, allow_unsegmented=allow_unsegmented)
                        for day_folder in numbered], [day_folder / folder for day_folder in numbered])
    mesh_dir = mesh_folder(root, folder)
    if not mesh_dir.is_dir():
        # Tolerate the folder living somewhere under root -- EVERY such folder, as if their files lay in one: the
        # first one found alone dropped the others' movies as unsegmented.
        candidates = outline_folders(root, folder=folder)
        if not candidates:
            raise CellMeshError(f"no '{folder}' folder under {root}")
        if len(candidates) > 1:
            return _merged([read_project_cell_meshes(c.parent, folder=folder, allow_unsegmented=allow_unsegmented)
                            for c in candidates], candidates)
        mesh_dir = candidates[0]

    # macOS AppleDouble sidecars (``._series011.mat``) shadow every file when a Mac
    # copies to a non-HFS volume. They carry the .mat extension, so this glob finds
    # them, and scipy fails on them with "Mat 4 mopt wrong format, byteswapping
    # problem?": an error naming neither the file nor the cause. Never data.
    from .ingest import is_sidecar

    candidates = [p for p in mesh_dir.glob("*.mat") if not is_sidecar(p)]
    skipped = sorted(p.name for p in candidates if p.stem.lower() in NOT_OUTLINE_STEMS)
    files = sorted(p for p in candidates if p.stem.lower() not in NOT_OUTLINE_STEMS)
    if not files:
        detail = f" (skipped {', '.join(skipped)}: u-track project files, not outlines)"
        raise CellMeshError(f"no .mat outline files in {mesh_dir}"
                            + (detail if skipped else ""))
    on_empty = "empty" if allow_unsegmented else "raise"
    return {p.stem: read_cell_meshes(p, movie=p.stem, on_empty=on_empty) for p in files}


def outline_folders(root: str | Path, *, folder: str = CELL_MESH_FOLDER) -> list[Path]:
    """Every outline folder a condition's movies are read from (`read_project_cell_meshes`), for whatever must
    know which files those are (the canonical cache's key): ``<root>/cell_meshes``, each replicate-day folder's
    (`dayfolders.replicate_folders`), else every one under ``root`` outside u-track's ``backups/``."""
    from .dayfolders import replicate_folders

    import os

    root = Path(root)
    days = replicate_folders(root)
    if days is not None:
        return [mesh_folder(day, folder) for day in days]
    if mesh_folder(root, folder).is_dir():
        return [mesh_folder(root, folder)]
    found = []
    for here, subs, _files in os.walk(root):
        keep = []
        for name in subs:
            if name in ("backups", "TrackingPackage") or name.startswith("."):
                continue
            if name == folder or (folder == CELL_MESH_FOLDER and is_mesh_folder_name(name)):
                found.append(Path(here) / name)            # an outlines folder: never entered
            else:
                keep.append(name)
        subs[:] = keep
    return sorted(found)


def _merged(parts: list, where: list) -> dict:
    """Several folders' outlines as one, refusing a movie found in two: which one is its outline is not a choice
    to make in silence."""
    out, origin = {}, {}
    for part, folder in zip(parts, where):
        for movie, cells in part.items():
            if movie in out:
                raise CellMeshError(f"outlines for {movie} in both {origin[movie]} and {folder}; keep one")
            out[movie], origin[movie] = cells, folder
    return out


# ── WRITING OUTLINES BACK ────────────────────────────────────────────────────
#
# THE ONE CAPABILITY THIS MODULE LACKED. Reading has always lived here; writing
# lived in the segmentation repository (`hada/meshio.build_mat`) and again, a third
# time, inside the mesh editor. The editor is being folded into SMTracker, so the
# format now has ONE implementation and the round trip is closed:
# `read_cell_meshes` → edit → `write_cell_meshes` → `read_cell_meshes`.
#
# Every dtype below was MEASURED on this project's own Oufti files rather than
# chosen: `mesh` and `model` are float32 in the real files, not float64, so writing
# float64 would produce a file no Oufti ever wrote.

#: `mesh` and `model` are **float32** in every real Oufti file here, so that is what
#: is written. MEASURED: all 51 cells of the reference project read, written and read
#: back differ by **exactly 0**: Oufti's own values are already float32, so nothing
#: is
#: lost. A float64 mesh built here (the editor's own capsules) costs **1.9e-6 px**,
#: i.e. 0.0002 nm, and the second cycle moves it by exactly 0, so repeated
#: open-edit-save does not drift.
OUFTI_COORD_DTYPE = np.float32

#: Padding, in pixels, around the outline's bounding box when filling Oufti's ``box``.
#:
#: OUFTI'S OWN `box` IS NOT RECONSTRUCTIBLE AND THIS DOES NOT PRETEND TO BE IT.
#: Measured over the 51 cells of the reference project, the gap between the outline's
#: bounding box and the recorded `box` runs 0 to 20 px on the left and top and **0 to
#: 78 px** on the right and bottom, so it is the segmentation's working region rather
#: than a property of the finished outline. What is written here is an honest padded
#: bounding box; 20 is the most common padding Oufti used and is inside the observed
#: range. Nothing in SMTracker reads this field.
OUFTI_BOX_PAD_PX = 20


def write_cell_meshes(
    path: str | Path,
    meshes,
    *,
    image_shape: tuple[int, int] | None = None,
    overwrite: bool = False,
    allow_empty: bool = False,
    pixel_origin="keep",
) -> Path:
    """Write :class:`CellMesh` outlines to one Oufti-compatible ``.mat``.

    ``pixel_origin`` records where the coordinates put a pixel's centre (`to_tracker_origin`): 0 for outlines made
    from an image's own pixels (SynR, the segmentation), 1 for MATLAB's; "keep" (the default) keeps what the file
    being replaced recorded, or records nothing -- an editor writes back the coordinates it read; None records
    nothing.

    ``meshes`` is an iterable of :class:`CellMesh`, the same objects
    :func:`read_cell_meshes` returns, so an editor reads, changes and writes back
    without a second representation of a cell existing anywhere.

    ``image_shape`` is ``(height, width)`` and is written into ``p.imsize`` when
    given. It is OPTIONAL because the real Oufti files in this project do not carry
    it at all; supply it if the file may be reopened in Oufti without its image.

    Refusals, each because the alternative fails later and further away:

    * **no cells**, UNLESS ``allow_empty=True``, see below;
    * **cells from more than one movie**, the reader takes the movie name from the
      FILENAME, so a mixed file silently relabels half its cells;
    * **a non-integer ``cell_id``**: Oufti's ``cellId`` is numeric, and renumbering
      to fit would move a track from ``series001:7`` to ``series001:3`` in every
      downstream table that keys on it;
    * **an existing file without** ``overwrite=True``: hand-corrected outlines are
      not recoverable once replaced.

    ``allow_empty=True`` writes a file with an empty ``cellList``, which means "this
    field was looked at and has no usable cells". That is a REAL curation outcome and
    the editor now needs it: "sometimes i wanna be able to delte ALL meshes from a
    series file ... make it so that even an emtpy mat is what it is -> no cell mesh".

    It is opt-in rather than the default because an empty list reaching here by accident
    (a filter that matched nothing, a parse that returned nothing) would otherwise
    silently erase a field's outlines. The caller has to mean it.

    Nothing downstream breaks, checked rather than assumed: `read_cell_meshes` has taken
    ``on_empty="empty"`` since it was written, `read_project_cell_meshes` passes exactly
    that (``allow_unsegmented=True``, the default), and `assign_cells` then records the
    movie under ``.attrs["unsegmented_movies"]`` and drops its tracks. The refusal's
    original reason ("the reader refuses a file holding none") describes the reader's
    non-default mode only, and was overtaken when unsegmented movies became survivable.

    Returns the path written.
    """
    from scipy.io import savemat

    path = Path(path)
    meshes = list(meshes)
    if not meshes and not allow_empty:
        raise CellMeshError(
            f"refusing to write {path}: no cell outlines, and allow_empty is not set. "
            "An empty list arriving here by accident would erase a field's outlines "
            "silently. Pass allow_empty=True to record 'this field has no cells' "
            "deliberately."
        )

    movies = sorted({m.movie for m in meshes})
    if len(movies) > 1:
        raise CellMeshError(
            f"refusing to write {path}: outlines from {len(movies)} movies "
            f"({', '.join(movies)}). One file is one movie: the reader takes the "
            "movie name from the filename, so a mixed file would relabel cells."
        )

    if path.exists() and not overwrite:
        raise CellMeshError(
            f"{path} exists. Pass overwrite=True to replace it; outlines you "
            "corrected by hand cannot be recovered once overwritten."
        )

    cell_ids = []
    for m in meshes:
        try:
            cell_ids.append(int(str(m.cell_id)))
        except (TypeError, ValueError):
            raise CellMeshError(
                f"cell id {m.cell_id!r} in movie {m.movie!r} is not an integer, and "
                "Oufti's cellId is numeric. Renumbering here would move this cell to "
                "a different key in every table that identifies a cell as "
                "'movie:cell'."
            ) from None

    fields = ["mesh", "model", "birthframe", "divisions", "ancestors", "descendants",
              "timelapse", "algorithm", "polarity", "stage", "box"]
    dtype = np.dtype([(f, object) for f in fields])

    frame = np.empty((1, len(meshes)), dtype=object)
    for i, m in enumerate(meshes):
        polygon = np.asarray(m.polygon, dtype=float)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] < 2:
            raise CellMeshError(
                f"cell {m.movie}:{m.cell_id} has a {polygon.shape} outline; an Oufti "
                "'model' is (N, 2) with at least 3 points."
            )
        ribs = m.mesh
        if ribs is not None:
            ribs = np.asarray(ribs, dtype=float)
            if ribs.ndim != 2 or ribs.shape[1] != 4 or ribs.shape[0] < 2:
                raise CellMeshError(
                    f"cell {m.movie}:{m.cell_id} has a {ribs.shape} mesh; Oufti ribs "
                    "are (N, 4) [x_left, y_left, x_right, y_right] with N >= 2. "
                    "Pass mesh=None for a contour-only cell."
                )

        entry = np.zeros((1, 1), dtype=dtype)
        entry["model"][0, 0] = polygon[:, :2].astype(OUFTI_COORD_DTYPE)
        # A cell Oufti could not mesh carries a SCALAR 0, not an empty array, which
        # is what `_ribs_of` already reads back as "no centerline".
        entry["mesh"][0, 0] = (np.zeros((1, 1), dtype=OUFTI_COORD_DTYPE)
                               if ribs is None else ribs.astype(OUFTI_COORD_DTYPE))
        entry["birthframe"][0, 0] = np.array([[1.0]])
        entry["divisions"][0, 0] = np.zeros((0, 0))
        entry["ancestors"][0, 0] = np.zeros((0, 0))
        entry["descendants"][0, 0] = np.zeros((0, 0))
        entry["timelapse"][0, 0] = np.array([[0.0]])
        entry["algorithm"][0, 0] = np.array([[4.0]])
        entry["polarity"][0, 0] = np.array([[0.0]])
        entry["stage"][0, 0] = np.array([[1.0]])
        entry["box"][0, 0] = _oufti_box(polygon, image_shape)
        frame[0, i] = entry

    mesh_data = np.empty((1, 1), dtype=object)
    mesh_data[0, 0] = frame
    ids = np.empty((1, 1), dtype=object)
    ids[0, 0] = np.asarray(cell_ids, dtype=OUFTI_COORD_DTYPE).reshape(1, -1)

    params = {}
    if image_shape is not None:
        # int32 specifically: Oufti does `ones(int32(H), int32(W))`, which other
        # integer widths do not satisfy on every MATLAB version.
        params["imsize"] = np.asarray(image_shape, dtype=np.int32).reshape(1, 2)

    payload = {
        "p": params,
        "coefPCA": np.zeros((0, 0)),
        "weights": np.zeros((0, 0)),
        "mCell": np.zeros((0, 0)),
        "rawPhaseFolder": np.zeros((0, 0)),
        "cellList": {"meshData": mesh_data, "cellId": ids},
        "shiftframes": np.zeros((0, 0)),
        "shiftfluo": np.zeros((2, 2)),
        "cellListN": np.array([[float(len(meshes))]]),
    }
    if pixel_origin == "keep":
        pixel_origin = None
        if path.exists():
            try:
                from scipy.io import loadmat

                pixel_origin = _recorded_origin(loadmat(path, variable_names=["pixelOrigin"]), path)
            except (OSError, ValueError, CellMeshError):
                pixel_origin = None
    if pixel_origin is not None:
        if pixel_origin not in (0, 1):
            raise CellMeshError(f"pixel_origin must be 0 or 1, got {pixel_origin!r}")
        payload["pixelOrigin"] = np.array([[float(pixel_origin)]])
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, payload, do_compression=True)
    return path


def mark_pixel_origin(path, origin: int) -> Path:
    """Record ``origin`` (0 or 1) in an existing outline file, its coordinates untouched: for files whose origin is
    KNOWN but was never written (SynR's true outlines before 22 Oct 2026, made in the image's own pixels: 0)."""
    from scipy.io import loadmat, savemat

    if origin not in (0, 1):
        raise CellMeshError(f"origin must be 0 or 1, got {origin!r}")
    path = Path(path)
    mat = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    if "cellList" not in mat:
        raise CellMeshError(f"{path}: not an outline file (no cellList)")
    for key, value in mat.items():
        # an EMPTY struct (this writer's `p` without an image size) reads back as [[None]], which savemat refuses
        if isinstance(value, np.ndarray) and value.dtype == object and all(v is None for v in value.ravel()):
            mat[key] = {}
    mat["pixelOrigin"] = np.array([[float(origin)]])
    savemat(path, mat, do_compression=True)
    return path


def _oufti_box(polygon: np.ndarray, image_shape: tuple[int, int] | None) -> np.ndarray:
    """Oufti's ``box`` = ``[x, y, width, height]``, 1-based, padded and clipped."""
    x0 = int(np.floor(polygon[:, 0].min())) - OUFTI_BOX_PAD_PX
    y0 = int(np.floor(polygon[:, 1].min())) - OUFTI_BOX_PAD_PX
    x1 = int(np.ceil(polygon[:, 0].max())) + OUFTI_BOX_PAD_PX
    y1 = int(np.ceil(polygon[:, 1].max())) + OUFTI_BOX_PAD_PX
    x0, y0 = max(x0, 1), max(y0, 1)
    if image_shape is not None:
        height, width = int(image_shape[0]), int(image_shape[1])
        x1, y1 = min(x1, width), min(y1, height)
    return np.array([[x0, y0, x1 - x0 + 1, y1 - y0 + 1]], dtype=float)


def prepare_polygon(polygon: np.ndarray) -> tuple:
    """A polygon's edges and bounding box, computed ONCE.

    `assign_cells` tests every track against every cell in its movie, and the
    edge arrays are a property of the CELL -- so recomputing them per track is
    the same `np.roll` over the same outline thousands of times. Profiled on a
    30-series project: 40 893 calls to `points_in_polygon`, of which 163 572
    `np.roll` calls costing 2.0 s of the 8.5 s the whole assignment took.
    """
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise CellMeshError(f"polygon must be (N>=3, 2), got {poly.shape}")
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    return (x1, y1, x2, y2,
            x1.min() - _ON_EDGE_TOL, x1.max() + _ON_EDGE_TOL,
            y1.min() - _ON_EDGE_TOL, y1.max() + _ON_EDGE_TOL)


def in_bounds(x, y, prepared: tuple, pad: float = 0.0) -> np.ndarray:
    """Which points fall in the polygon's bounding box, widened by ``pad`` on every side.
    Cheap, and EXACT as a pre-filter: a point outside the box is outside the polygon,
    always -- and a point more than ``pad`` outside it is more than ``pad`` from it."""
    _x1, _y1, _x2, _y2, xlo, xhi, ylo, yhi = prepared
    return (x >= xlo - pad) & (x <= xhi + pad) & (y >= ylo - pad) & (y <= yhi + pad)


def near_outline(x, y, prepared: tuple, tolerance: float) -> np.ndarray:
    """Which points lie within ``tolerance`` (pixels) of the outline: the distance to the
    nearest point of any edge SEGMENT, so a corner is rounded rather than extended."""
    x = np.asarray(x, dtype=float)[:, None]
    y = np.asarray(y, dtype=float)[:, None]
    x1, y1, x2, y2 = prepared[:4]
    dx, dy = x2 - x1, y2 - y1
    length2 = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(length2 > 0, ((x - x1) * dx + (y - y1) * dy) / length2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    d2 = (x - (x1 + t * dx)) ** 2 + (y - (y1 + t * dy)) ** 2
    return d2.min(axis=1) <= tolerance * tolerance


def points_in_prepared(x, y, prepared: tuple) -> np.ndarray:
    """:func:`points_in_polygon` against an already-prepared polygon."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x1, y1, x2, y2 = prepared[:4]
    px = x[:, None]
    py = y[:, None]

    # On an edge: collinear and within the segment's span.
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    within = (
        (np.minimum(x1, x2) - _ON_EDGE_TOL <= px) & (px <= np.maximum(x1, x2) + _ON_EDGE_TOL)
        & (np.minimum(y1, y2) - _ON_EDGE_TOL <= py) & (py <= np.maximum(y1, y2) + _ON_EDGE_TOL)
    )
    on_edge = (np.abs(cross) <= _ON_EDGE_TOL * np.maximum(1.0, np.abs(cross).max())) & within
    on = on_edge.any(axis=1)

    # Crossing number: count edges crossing the upward ray from the point.
    straddles = (y1 > py) != (y2 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = (x2 - x1) * (py - y1) / (y2 - y1) + x1
    crossings = straddles & (px < x_cross)
    inside = (crossings.sum(axis=1) % 2).astype(bool)

    return inside | on


def points_in_polygon(x, y, polygon: np.ndarray) -> np.ndarray:
    """MATLAB ``inpolygon``'s ``in | on``: inside counts, boundary counts.

    Ray casting (crossing number) plus an explicit on-edge test, so a
    localization sitting exactly on the outline is included, as in the legacy.

    The arithmetic lives in :func:`points_in_prepared`; this is the one-shot
    form, for a caller with a single polygon and no loop to hoist out of.
    """
    return points_in_prepared(x, y, prepare_polygon(polygon))


def assign_cells(
    raw: pd.DataFrame,
    meshes: dict[str, list[CellMesh]],
    *,
    drop_unassigned: bool = True,
    require_all_inside: bool = True,
    tolerance_px: float = 0.0,
) -> pd.DataFrame:
    """Assign each track to the cell it lies in; optionally drop the rest.

    Parameters
    ----------
    raw
        Raw (pixel) track table with ``movie, track_id, x, y``.
    meshes
        ``movie -> [CellMesh]``, from ``read_project_cell_meshes``.
    drop_unassigned
        ``True`` (default, legacy behaviour) removes tracks that are not fully
        inside any cell. ``False`` keeps them with ``cell = <NA>`` so they can
        be inspected.
    require_all_inside
        ``True`` (default, legacy) demands *every* localization be inside.
        ``False`` assigns on the majority of localizations instead.
    tolerance_px
        ``0`` (default) is the legacy filter, exactly. Above it, a localization also
        counts as inside when it lies within this distance of the outline, so a track
        is kept when it lies inside the outline WIDENED by it. Asked for (13 Oct 2026,
        "Add a tolerance option, default off") after SynR stage 2 measured the bias a
        tight outline puts on the fractions: segmented outlines about 11 per cent too
        small read SynR's bound fraction as 42 per cent against a truth of 31, because
        the filter drops the free molecules that reach the membrane, and a 1 px
        tolerance read 34 (`docs/synthetic_synr.md`). Where two widened outlines both
        hold a track, it goes to the cell holding more of its points strictly inside.
        The tracks the tolerance ADDED are counted per movie (``by_tolerance`` in
        ``cell_filter_by_movie``) and in total (``n_tracks_added_by_tolerance``).

    Whatever the tolerance, the DROPPED tracks that an outline `NEAR_EDGE_PX` wider would
    keep are counted per movie (``cell_filter_near_edge_by_movie``) and in total
    (``n_tracks_near_edge``): the cost of a tight outline, which QC states beside the
    in-cell share. Measured here because nowhere else can: the dropped tracks have no rows
    in the canonical table.

    Returns
    -------
    pandas.DataFrame
        A copy with ``cell`` filled (``"<movie>:<cell_id>"``). ``.attrs`` record
        the track counts before/after and how many tracks were ambiguous.
    """
    for col in ("movie", "track_id", "x", "y"):
        if col not in raw.columns:
            raise CellMeshError(f"raw table is missing '{col}'")
    if not meshes:
        raise CellMeshError("no cell outlines supplied")
    tol = float(tolerance_px)
    if not np.isfinite(tol) or tol < 0:
        raise CellMeshError(f"the outline tolerance must be 0 or more pixels, got {tolerance_px!r}")

    missing = sorted(set(pd.unique(raw["movie"])) - set(meshes))
    if missing:
        raise CellMeshError(
            f"no cell outlines for movies: {missing}. Analysing them would mix "
            "segmented and unsegmented movies."
        )

    # EACH OUTLINE PREPARED ONCE, not once per track. Every track in a movie is
    # tested against every cell in it, so the edge arrays -- a property of the
    # CELL -- were being rebuilt thousands of times. See `prepare_polygon`.
    prepared = {movie: [(m.cell_id, prepare_polygon(m.polygon)) for m in cells]
                for movie, cells in meshes.items()}

    # POSITIONS, NOT SUB-FRAMES. Iterating `groupby` hands back a DataFrame per
    # group, and this loop wants two float arrays: on a 30-series project that
    # was 5 857 DataFrames built and thrown away, and pandas' own bookkeeping
    # (`_chop`, `__finalize__`, a deepcopy of the attrs each time) came to more
    # than the geometry it was wrapping. `.indices` gives the row positions in
    # one pass and the columns are taken once, as numpy.
    xs = raw["x"].to_numpy(dtype=float)
    ys = raw["y"].to_numpy(dtype=float)
    groups = raw.groupby(["movie", "track_id"], sort=False).indices

    assignment: dict[tuple, str] = {}
    ambiguous = 0
    #: (movie, track) kept only because of the tolerance: its cell does not hold it strictly.
    widened: set = set()
    for (movie, track_id), rows in groups.items():
        x = xs[rows]
        y = ys[rows]
        if tol > 0:
            hit = _widened_match(x, y, prepared[movie], tol, require_all_inside)
            if hit is not None:
                cell_id, n_matched, strict = hit
                assignment[(movie, track_id)] = f"{movie}:{cell_id}"
                ambiguous += int(n_matched > 1)
                if not strict:
                    widened.add((movie, track_id))
            continue
        matches = []
        for cell_id, prep in prepared[movie]:
            # THE BOUNDING BOX FIRST, and it is EXACT rather than approximate: a
            # point outside the box is outside the polygon, so a box test can
            # only ever rule a cell OUT. Most (track, cell) pairs are nowhere
            # near each other -- a movie holds several cells and a track sits in
            # at most one -- and this replaces a points x vertices matrix with
            # four comparisons for all of them.
            box = in_bounds(x, y, prep)
            if require_all_inside:
                if not box.all():          # a point outside the box is outside
                    continue
            elif box.mean() <= 0.5:        # inside-polygon is a SUBSET of this
                continue
            inside = points_in_prepared(x, y, prep)
            ok = inside.all() if require_all_inside else inside.mean() > 0.5
            if ok:
                matches.append(cell_id)
                if len(matches) > 1:
                    break
        if matches:
            assignment[(movie, track_id)] = f"{movie}:{matches[0]}"
            if len(matches) > 1:
                ambiguous += 1

    # THE NEAR-EDGE COUNT, over the dropped tracks only (few, and only their own movie's cells).
    probe = tol + NEAR_EDGE_PX
    near_edge: dict[str, int] = {str(m): 0 for m in prepared}
    for (movie, track_id), rows in groups.items():
        if (movie, track_id) in assignment:
            continue
        if _widened_match(xs[rows], ys[rows], prepared[movie], probe, require_all_inside) is not None:
            near_edge[str(movie)] = near_edge.get(str(movie), 0) + 1

    out = raw.copy()
    keys = list(zip(out["movie"], out["track_id"]))
    out["cell"] = pd.Series([assignment.get(k, pd.NA) for k in keys], index=out.index, dtype="object")

    n_before = out.groupby(["movie", "track_id"], sort=False).ngroups
    # PER MOVIE, RECORDED HERE BECAUSE HERE IS THE ONLY PLACE IT EXISTS. The canonical
    # table is POST-filter, so the tracks this drops leave no trace in it and the loss
    # cannot be reconstructed downstream -- only the condition TOTAL survives, on attrs.
    # A per-condition total cannot answer "which movie", which is the question a QC panel
    # is opened to answer: one lab project's worst case, a single movie,
    # discarded 75.4 per cent of its localisations against a 9.4 per cent median and was
    # invisible in its condition's average.
    _per_movie = out.groupby("movie", sort=False).apply(
        lambda g: pd.Series({
            "found": int(g.groupby("track_id", sort=False).ngroups),
            "kept": int(g.loc[g["cell"].notna()]
                        .groupby("track_id", sort=False).ngroups),
        }),
        include_groups=False,
    )
    by_movie = {str(m): {"found": int(r["found"]), "kept": int(r["kept"])}
                for m, r in _per_movie.iterrows()}
    if tol > 0:
        # ONLY WITH A TOLERANCE, so the legacy record keeps its exact shape.
        for movie in by_movie:
            by_movie[movie]["by_tolerance"] = 0
        for movie, _track in widened:
            by_movie[str(movie)]["by_tolerance"] += 1
    if drop_unassigned:
        out = out.loc[out["cell"].notna()]
        if len(out) == 0:
            raise CellMeshError(
                "no track lies fully inside a cell outline. Check that the outlines "
                "and the tracks use the same coordinates (both in raw pixels)."
            )
    n_after = out.groupby(["movie", "track_id"], sort=False).ngroups

    out.attrs["n_tracks_before_cell_filter"] = int(n_before)
    out.attrs["n_tracks_after_cell_filter"] = int(n_after)
    out.attrs["n_tracks_ambiguous_cell"] = int(ambiguous)
    out.attrs["n_cells"] = int(sum(len(v) for v in meshes.values()))
    out.attrs["unsegmented_movies"] = sorted(m for m, v in meshes.items() if not v)
    out.attrs["cell_filter_by_movie"] = by_movie
    out.attrs["cell_filter_tolerance_px"] = tol
    out.attrs["n_tracks_added_by_tolerance"] = len(widened)
    out.attrs["cell_filter_near_edge_px"] = probe
    out.attrs["cell_filter_near_edge_by_movie"] = near_edge
    out.attrs["n_tracks_near_edge"] = int(sum(near_edge.values()))
    return out.reset_index(drop=True)


def _widened_match(x, y, cells, tol: float, require_all_inside: bool):
    """The cell whose outline, widened by ``tol``, holds the track: ``(cell_id, how many
    cells did, held strictly)``, or None. Of several, the one holding most points strictly
    inside -- the cell the track is IN, not the neighbour whose widened edge it grazes."""
    best = None
    n_matched = 0
    for cell_id, prep in cells:
        box = in_bounds(x, y, prep, pad=tol)
        if require_all_inside:
            if not box.all():
                continue
        elif box.mean() <= 0.5:
            continue
        inside = points_in_prepared(x, y, prep)
        held = inside.copy()
        if not held.all():
            held[~inside] = near_outline(x[~inside], y[~inside], prep, tol)
        ok = held.all() if require_all_inside else held.mean() > 0.5
        if not ok:
            continue
        n_matched += 1
        strict = inside.all() if require_all_inside else inside.mean() > 0.5
        score = int(inside.sum())
        if best is None or score > best[1]:
            best = (cell_id, score, strict)
    if best is None:
        return None
    return best[0], n_matched, best[2]


def assignment_funnel(assigned: pd.DataFrame, meshes: dict | None = None,
                      *, um_per_pixel: float | None = None) -> dict:
    """What the cell filter kept, what it discarded, and WHERE the discards were.

    The cell filter is the largest silent data loss in the pipeline. On well-behaved
    data it removes ~5-10 % of tracks; on data where the label sits outside cells it
    has removed **59 %**, with no message. A funnel that is not printed is a funnel
    nobody checks, and two conditions can lose very different fractions, which makes
    any comparison between them a comparison of what survived.

    ``distance_to_outline`` classifies the discards by how far the track's centroid
    lies from the nearest outline vertex, which distinguishes the three cases that
    need different responses:

    - **at the boundary** (< ~2 px): a track that strayed over the edge, or a
      genuinely peripheral molecule. Relaxing ``require_all_inside`` recovers these.
    - **near a cell** (2-10 px): most likely an unsegmented cell. Fix the
      segmentation, do not fix the analysis.
    - **far from any cell** (> 20 px, i.e. > 2 µm): extracellular. This is a real
      observation about the sample, not a defect, and discarding it silently throws
      away the evidence.

    Pass ``meshes`` to get the distance breakdown; without it only the counts are
    returned. Requires the ``cell`` column and the raw pixel ``x, y``.
    """
    out = {
        "n_tracks_before": int(assigned.attrs.get("n_tracks_before_cell_filter", 0)),
        "n_tracks_after": int(assigned.attrs.get("n_tracks_after_cell_filter", 0)),
        "n_ambiguous": int(assigned.attrs.get("n_tracks_ambiguous_cell", 0)),
        "n_cells": int(assigned.attrs.get("n_cells", 0)),
        "unsegmented_movies": list(assigned.attrs.get("unsegmented_movies", [])),
    }
    if "cell" in assigned.columns:
        keys = assigned.groupby(["movie", "track_id"], sort=False)["cell"].first()
        out["n_tracks_before"] = out["n_tracks_before"] or int(keys.size)
        out["n_tracks_after"] = int(keys.notna().sum())
    out["n_dropped"] = out["n_tracks_before"] - out["n_tracks_after"]
    out["dropped_fraction"] = (out["n_dropped"] / out["n_tracks_before"]
                               if out["n_tracks_before"] else float("nan"))

    if meshes is None or "cell" not in assigned.columns:
        return out
    lost = assigned[assigned["cell"].isna()]
    if lost.empty:
        out["distance_to_outline"] = {}
        return out
    cent = lost.groupby(["movie", "track_id"])[["x", "y"]].mean().reset_index()
    dists = []
    for movie, g in cent.groupby("movie"):
        polys = [np.asarray(m.polygon, dtype=float) for m in meshes.get(movie, [])]
        if not polys:
            dists.extend([float("inf")] * len(g))
            continue
        xy = g[["x", "y"]].to_numpy()
        best = np.full(len(g), np.inf)
        for poly in polys:
            d = np.hypot(poly[None, :, 0] - xy[:, 0, None],
                         poly[None, :, 1] - xy[:, 1, None]).min(axis=1)
            best = np.minimum(best, d)
        dists.extend(best.tolist())
    d = np.asarray(dists, dtype=float)
    edges = [(0, 2, "at the boundary"), (2, 5, "just outside"),
             (5, 10, "near a cell"), (10, 20, "between cells"),
             (20, np.inf, "far from any cell")]
    out["distance_to_outline"] = {
        label: float(((d >= lo) & (d < hi)).mean()) for lo, hi, label in edges}
    out["distance_units"] = ("pixels" if um_per_pixel is None
                             else f"pixels ({um_per_pixel * 1000:.0f} nm each)")
    return out


def format_funnel(funnel: dict) -> str:
    """One printable block. Loud when the loss is large, because it usually is not
    looked at until a comparison has already been made on the survivors."""
    lines = [
        f"cell-filter funnel: {funnel['n_tracks_before']} tracks -> "
        f"{funnel['n_tracks_after']} assigned, {funnel['n_dropped']} dropped "
        f"({100 * funnel['dropped_fraction']:.1f} %)"]
    if funnel["dropped_fraction"] > 0.20:
        lines.append("  WARNING: over a fifth of the tracks were discarded. Two "
                     "conditions losing different fractions cannot be compared "
                     "without accounting for what survived.")
    if funnel.get("unsegmented_movies"):
        lines.append(f"  movies with no outlines: {funnel['unsegmented_movies']}")
    for label, frac in (funnel.get("distance_to_outline") or {}).items():
        lines.append(f"    {label:<20} {100 * frac:>5.1f} % of the discards")
    return "\n".join(lines)
