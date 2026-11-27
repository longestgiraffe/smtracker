"""Loading a MULTI-CONDITION project, several condition folders as one table.

The layout this exists for
--------------------------
``CLAUDE.md`` states that a fit group is "one subfolder of the loaded project",
and that protein and condition are interchangeable labels for such a subfolder.
The single-condition loader was only ever exercised against one folder, where
movie names happened to be globally unique. A real comparison is not like that::

    <root>/glucose/tif/series001/TrackingPackage/tracks/*.mat
    <root>/malate/ tif/series001/TrackingPackage/tracks/*.mat

Both conditions number their movies from 001, so ``series001`` is ambiguous and
``load_utrack_project`` rightly refuses it ("movie names must be unique within a
project"). Each condition folder is, however, *itself* a complete project: its
own ``tif/``, ``cell_meshes/``, ``phase/``, ``sum/`` and ``movieData.mat``.

So this loads each condition as its own project and **qualifies the movie name**
with its condition before merging. Because ``uid`` is ``"<movie>:<track_id>"``
and ``cell`` is ``"<movie>:<cell_id>"``, qualifying the movie makes tracks and
cells unique across the whole comparison automatically: no other identifier
needs touching, and nothing downstream has to learn about conditions.

Acquisition parameters are CHECKED, not assumed
-----------------------------------------------
Every D scales with 1/τ and with pixel size squared, so comparing two conditions
acquired differently would produce a difference that is pure units. The
per-condition ``AcquisitionParams`` are compared and a mismatch **raises**,
this is exactly the kind of thing that silently produces a beautiful result.

Invariants
----------
- Movie names are qualified ``"<condition>_<movie>"``; the original name and its
  root are kept in ``Project.movie_source`` so images can still be found.
- Cell meshes are re-keyed to match, so ``assign_cells`` still works unchanged.
- Conditions with differing pixel size fail loud: it rescales every coordinate
  and the cell geometry has no per-movie route, so that is two projects.
- Conditions with differing frame interval load, with τ carried per movie and a
  warning. They are a STRATUM, not one number: at 20.3 vs 30 ms the detection
  ceiling differs by ~48 % and the blur differs, so a comparison across strata is
  confounded by acquisition even when every D is computed correctly.
- Conditions LINKED at different search radii load, with the radius carried per
  condition (``AcquisitionParams.search_radius_px_by_condition``) and a warning
  naming each (`radius_mix_text`, on a cached read too): every ceiling and
  crowding check reads its condition's own (tester D10, ruled 4 Nov 2026).
- A condition folder with no u-track results fails loud rather than vanishing.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import pickle
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from .canonical import AcquisitionParams, to_canonical
from .cellmesh import (assign_cells, meshes_by_movie_name, outline_alignment, read_project_cell_meshes,
                       to_tracker_origin)
from .grouping import (PROJECT_SOURCES, TRACKMATE_GLOB, UTRACK_GLOB,
                       detect_project_source, load_table_project, load_trackmate_project,
                       load_utrack_project, sources_present)
from .tracktable import TABLE_FOLDER, TABLE_GLOB, TABLE_SUFFIXES
from .utrack_meta import acquisition_params_from_project

__all__ = [
    "ProjectError",
    "Project",
    "discover_conditions",
    "tracker_options",
    "load_project",
    "movie_frames",
    "with_exclusions",
    "with_track_cap",
    "qualified_movie",
]

#: Bump when the CANONICAL/ingest logic changes in a way that alters the parsed
#: table (px→µm, the length filter, cell assignment) OR the stored
#: ``AcquisitionParams``. The canonical cache is keyed on the INPUT FILES, not the
#: code commit (like the warm cache), so this manual version is what invalidates a
#: stale parse after a logic change.
#:
#: 2, ``AcquisitionParams`` gained ``psf_sigma_px`` (the theoretical σ's optical
#:     basis) and ``dx, dy`` reached the raw table. A v1 entry pickled before that
#:     restores with ``psf_sigma_px=None``, which silently demotes σ from the PSF
#:     basis (98.8 nm) to the pixelation one (30.6 nm), a 3.2× change in the
#:     localization floor served from cache with no sign that anything happened.
#:     Exactly the silent default the σ provenance state exists to prevent, so the
#:     old entries must MISS rather than be reused.
#: 3, the canonical table gained a ``tau_s`` column (this row's own frame interval,
#:     so a mixed-rate project computes every D with the right τ). A v2 entry has no
#:     such column, so a cache hit would hand back a table that silently lacks the
#:     one thing a mixed project needs. Second instance of the same lesson: A CACHE
#:     keyed on inputs is keyed on half the problem: the code is an input.
#: 8 -> 9, sigma now comes from the tracker's own per-localisation uncertainty where
#:     the tracker records one, instead of from the PSF width. A v8 entry carries the
#:     theoretical value, which is 5.7x larger on this data and moves every floor, so
#:     the old entries must MISS rather than be reused. Same lesson as rows 2 and 3.
#: 9 -> 10 TWICE, INDEPENDENTLY, WHICH IS WHY THIS ROW IS 11. Two branches added a
#:     per-movie PRE-FILTER track count within a fortnight, each bumping to 10 for its own
#:     reason, so neither flavour of "10" describes what a v10 entry actually contains --
#:     an entry written by one of them is missing what the other reads. The reasons, both
#:     kept because they are the same defect seen from two ends:
#:
#:     * the onboarding confirm screen fell back to ``found = kept`` where the count was
#:       absent and reported a **100 PER CENT keep rate on a project whose real rate is
#:       70** -- a wrong number on the screen whose whole job is to be checked before
#:       anything is trusted;
#:     * the QC dashboard could not show a movie the cell filter EMPTIED, because every
#:       panel is drawn from the post-filter table and such a movie has no rows in it. The
#:       reference data has one: `series009`, **376 tracks found, 0 kept**.
#:
#:     Both are now served by ONE count (`cell_filter_by_movie`, recorded where the filter
#:     runs), so a v9 or v10 entry must MISS rather than be reused. Same lesson as rows 2,
#:     3 and 8: A CACHE KEYED ON INPUTS IS KEYED ON HALF THE PROBLEM, THE CODE IS AN INPUT.
#:   * 12: the table records the tracker each condition was read from
#:     (``attrs["trackers"]``), now that a condition tracked twice is a choice.
#:   * 13: the cell filter counts the dropped tracks an outline 1 px wider would keep
#:     (``cell_filter_near_edge_by_movie``), which QC states; a v12 entry has no such count
#:     and would show a tight-outline project as unmeasured.
#:   * 14: track tables are a third tracker, and the key lists every tracker's files and
#:     crops.json (it listed only TrackMate's XML for anything not u-track).
#:   * 15: replicate days in NUMBERED folders (`dayfolders.numbered_folders`: `LB/LB-1`, `-2`, `-3`) take their
#:     day from the folder and their outlines from every day folder; the first outline folder found was all
#:     that was read, and the key listed none of them.
#:   * 16: a movie's frame interval may come from its CAMERA file (`frame_interval_from`), and movies can be left
#:     out before they are read (`leave_out`); both are in the key, with the camera files.
#:   * 17: outlines that record their pixel origin are moved onto the tracker's, and the alignment of tracks and
#:     outlines is measured into the table's attrs (`cellmesh.to_tracker_origin`, `outline_alignment`).
#:   * 18: an outline named by its movie's series number (``001.mat`` for ``series001_YFPLaser``) pairs with the
#:     movie (`cellmesh.meshes_by_movie_name`): a load that found none of one user's outlines now finds them.
#:   * 19: the length floor records the tracks it removed, per condition (``attrs["length_floor_removed"]``), so QC
#:     can draw them; a v18 entry has no record and QC would call the filter free on TrackMate data.
#:   * 20: a condition with ONE stated frame interval (TrackMate, a track table) carries it per movie when the
#:     project has rates per movie or its stated rates differ (`_check_params_agree`): a v19 entry of two TrackMate
#:     conditions at different rates holds the second at the first's tau in its ``tau_s`` column.
#:   * 21: each condition's OWN recorded acquisition (pixel size, stated frame interval, search radius, PSF) is kept
#:     in ``attrs["acquisition_by_condition"]`` for the Confirm page (tester B1, 3 Nov 2026): the merged ``params``
#:     hold the first condition's, and a project linked at 5 and 8 px read 5 px for both. Display only; a v20 entry
#:     has no record and the page would show the project-wide values again.
#:   * 22: the ANALYSIS carries the search radius per condition where the conditions were linked at different radii
#:     (``AcquisitionParams.search_radius_px_by_condition``, tester D10, ruled 4 Nov 2026): a v21 entry of a project
#:     linked at 5 and 8 px holds params with the first condition's 5 px for all, and every ceiling and crowding
#:     check read from it would be drawn at 5 px again.
#:   * 23: a TrackMate condition's search radius is its XMLs' linking distance (`_with_linking_radius`, ruled 4 Nov
#:     2026): a v22 entry holds a TrackMate condition with no radius, so no ceiling and no crowding check.
#:   * 24: a u-track condition carries each spot's own precision (`canonical.LOC_SIGMA_COLUMN`, 6 Nov 2026, the
#:     per-spot precision fit): a v23 entry has no such column, and the per-spot fit would refuse a project that has one.
CANONICAL_CACHE_VERSION = 24


def _canonical_cache_key(available, *, day_maps, localization_precision_nm,
                         min_localizations, series_per_day, allow_unsegmented,
                         require_all_inside=True, source="utrack",
                         pixel_size_nm=None, frame_interval_ms=None,
                         skip_unsegmented=False, outline_tolerance_nm=0.0,
                         frame_interval_from="record", leave_out=None, design=None):
    """A content key for the parsed project: the input FILES + the load options.

    Keyed on each tracks / mesh / movieData ``.mat`` file's (path, size, mtime),
    not its bytes, so it is fast: plus the options that change the parsed table
    (localization precision, the length filter, the day mapping, the conditions).
    An SSD move that resets mtimes simply forces one rebuild; a real edit to any
    input file changes its size/mtime and invalidates cleanly.
    """
    files = []
    for name in sorted(available):
        root = Path(available[name])
        # EVERY TRACKER'S FILES AND BOTH CALIBRATION RECORDS. The list once held only
        # TrackMate's XML for anything that was not u-track, so an edited track table,
        # the u-track half of a mixed project, or a corrected pixel size in crops.json
        # left the key -- and the table served -- unchanged (found 13 Oct 2026 adding the
        # third tracker; `CANONICAL_CACHE_VERSION` 14).
        patterns = (UTRACK_GLOB, TRACKMATE_GLOB, TABLE_GLOB, "cell_meshes/*.mat",
                    "**/movieData.mat", f"**/{CROPS_FILENAME}", "**/acquisition.json")
        # OUTLINES READ FROM ANYWHERE BUT `cell_meshes/`: each day folder's (`dayfolders`), or every folder of them
        # below the condition -- the ones `read_project_cell_meshes` reads, so editing day 2's outlines is a miss
        from .cellmesh import outline_folders

        patterns += tuple(f"{p.relative_to(root).as_posix()}/*.mat" for p in outline_folders(root)
                          if p != root / "cell_meshes")
        # THE CAMERA FILES the frame intervals are read from (`camera_files`), when they are
        if frame_interval_from == "camera":
            for movie, raw in sorted(camera_files(root).items()):
                try:
                    st = raw.stat()
                except OSError:
                    continue
                files.append([name, raw.relative_to(root).as_posix(), st.st_size, st.st_mtime_ns])
        for pattern in patterns:
            for p in sorted(root.glob(pattern)):
                try:
                    st = p.stat()
                except OSError:
                    continue
                rel = str(p.relative_to(root)).replace("\\", "/")
                files.append([name, rel, st.st_size, st.st_mtime_ns])
    day = {c: getattr(m, "movie_to_day", m) for c, m in (day_maps or {}).items()}
    # ``require_all_inside`` IS PART OF THE KEY, and leaving it out is the trap this
    # whole fix exists around. It changes which tracks survive the cell filter, so two
    # runs differing only in it produce different canonical tables. A key that ignored it
    # would hand the second run the first one's table -- the same class of failure as the
    # day-map cache entry that served one project as days [1, 4, 5] months after the guard
    # against it was added.
    obj = {"files": files, "loc_nm": localization_precision_nm,
           "min_loc": min_localizations, "series_per_day": series_per_day,
           "allow_unseg": allow_unsegmented, "conditions": sorted(available),
           "day_maps": day, "require_all_inside": bool(require_all_inside),
           # THE TRACKER IS AN INPUT, so it is in the key. u-track and TrackMate on the
           # same movies give fractions about 1 pp apart and D values 10-18 per cent
           # apart, so a key without it would serve one tracker's table for the other's
           # run -- the `require_all_inside` defect a third time.
           "source": str(source),
           # THE ACQUISITION OVERRIDES ARE INPUTS TOO. Both scale what lands in the
           # canonical table -- the pixel size scales every coordinate and D by its
           # square, the frame interval scales D linearly and sets `tau_s` -- so two
           # runs differing only in one produce DIFFERENT tables. A key without them
           # would serve one run's table to the other, which is the `require_all_inside`
           # defect a fourth time and the reason it is written out here rather than
           # remembered.
           "pixel_size_nm": pixel_size_nm,
           "frame_interval_ms": frame_interval_ms,
           # DROPPING A MOVIE CHANGES THE TABLE, so it changes the key. Without this a
           # skip_unsegmented run and a strict one would share an entry.
           "skip_unsegmented": bool(skip_unsegmented),
           # where tau came from, and the movies left out (`load_project`): each changes the table
           "frame_interval_from": str(frame_interval_from),
           "leave_out": {c: sorted(m) for c, m in sorted((leave_out or {}).items())},
           "canon_v": CANONICAL_CACHE_VERSION}
    # THE OUTLINE TOLERANCE WIDENS THE CELL FILTER, so it is an input like
    # `require_all_inside`. Keyed ONLY when set: at 0 the filter is the legacy one, exactly,
    # and leaving the field out keeps every cache entry already written valid.
    if outline_tolerance_nm:
        obj["outline_tol_nm"] = float(outline_tolerance_nm)
    # A FOLDER DESIGN DECIDES THE CONDITIONS AND CAN DECLARE THE REPLICATES (`folderdesign`): keyed ONLY when there
    # is one, so every entry written without a design stays valid. The files alone would not notice: declaring
    # replicate folders changes no file and every movie's replicate.
    if design is not None:
        from .folderdesign import fingerprint

        obj["design"] = fingerprint(design)
    blob = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def _grouping_code_fingerprint() -> str:
    """A hash of the SOURCE of the functions that shape a cached canonical table.

    Why a fingerprint and not a version number. ``CANONICAL_CACHE_VERSION`` invalidates
    entries only if a human remembers to raise it. That step is exactly what failed:
    ``_refuse_non_contiguous_blocks`` was added without a bump, so entries written
    before the guard stayed reachable and one of them served days [1, 4, 5]. A
    fingerprint of the code removes the human step, because editing the guard changes
    the hash whether or not anyone remembers anything.

    ``project.py`` already carried the lesson in capitals: A cache keyed on inputs IS
    keyed on half the problem, the code is an input. The mechanism did not fail; the
    human step inside it did.
    """
    import inspect

    from . import grouping

    parts = []
    for name in ("assign_groups", "_refuse_non_contiguous_blocks", "day_from_series",
                 "day_map_from_series_blocks"):
        fn = getattr(grouping, name, None)
        if fn is not None:
            try:
                parts.append(inspect.getsource(fn))
            except OSError:                            # pragma: no cover - frozen build
                parts.append(name)
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()[:16]


def _validate_cached_days(canonical, day_maps) -> None:
    """Run the day guard against a CACHED table before it is returned.

    A guard protects only the paths that call it, and a cache read is a path. The
    default series-to-day rule is refused inside ``assign_groups``; a cache read does
    not call ``assign_groups``, so nothing checked the days on the way out. This runs
    the same check on the stored table.

    Only conditions loaded WITHOUT an explicit day map are checked. An explicit mapping
    is a stated authority and may legitimately number days any way it likes.
    """
    from .grouping import _refuse_non_contiguous_blocks

    if "day" not in canonical.columns or "condition" not in canonical.columns:
        return
    supplied = set(day_maps or {})
    for condition, part in canonical.groupby("condition", observed=True):
        if str(condition) in supplied:
            continue
        movie_column = "movie" if "movie" in part.columns else None
        if movie_column is None:
            continue
        day_by_movie = (part.groupby(movie_column, observed=True)["day"]
                        .first().astype(int).to_dict())
        _refuse_non_contiguous_blocks(day_by_movie, 10)


def _load_cached_project(cache_file, available, *, day_maps=None):
    """Rebuild a Project from a canonical-cache pickle, or None on any problem.

    A corrupt / unreadable entry is treated as a miss (rebuild), never a crash. A stale
    entry is also a miss: one whose stored cache version or grouping fingerprint does
    not match this code. But a cached table whose DAYS the guard would reject is NOT a
    miss, it RAISES, because silently rebuilding it would hide that the entry existed.
    """
    if not cache_file.exists():
        return None
    try:
        payload = pickle.loads(cache_file.read_bytes())
        canonical = payload["canonical"]
        params = payload["params"]
        meshes = payload["meshes"]
        sources = payload["sources"]
        day_mappings = payload.get("day_mappings")
        stored_version = payload.get("cache_version")
        stored_fingerprint = payload.get("grouping_fingerprint")
    except Exception:                                  # pragma: no cover - defensive
        return None

    # An entry written before provenance existed cannot say which code wrote it, so it
    # is treated as stale rather than trusted.
    if stored_version != CANONICAL_CACHE_VERSION:
        return None
    if stored_fingerprint != _grouping_code_fingerprint():
        return None

    _validate_cached_days(canonical, day_maps)

    canonical.attrs["params"] = params
    if day_mappings:
        canonical.attrs["day_mappings"] = day_mappings
    return Project(
        canonical=canonical, meshes=meshes, params=params,
        conditions=list(available),
        roots={name: Path(p) for name, p in available.items()},
        movie_source=_rebased_sources(sources, available),
        day_maps=payload.get("day_maps") or {},
        # FROM THE STORED PARAMS. A cache read is a path, and this one rebuilt a Project
        # without the loader's sigma, so clearing an override on a cached project fell
        # back to theoretical and moved the floor by 5.7x. The cache stores the LOADER's
        # answer, so those params are exactly what reset should return to.
        # PARAMS CAN BE None HERE. An entry written before acquisition parameters were
        # stored, or a planted one, carries none, and assuming otherwise turned a cache
        # read that used to work into an AttributeError. The guard tests caught it.
        loaded_sigma=((params.localization_sigma_source,
                       params.localization_precision_nm) if params is not None
                      else ("theoretical", None)))


def _rebased_sources(sources: dict, available: dict) -> dict:
    """Each movie's folder under its condition's root AS OPENED NOW, not as it was when the entry was written.

    The cache sits beside the data (`.smtracker_warm`) and its key is built from paths RELATIVE to each condition,
    so it travels with the drive -- but the entry stored every movie's folder ABSOLUTE. Found 20 Oct 2026: one user's
    drive is L: on one machine and E: on another; the entry written on L: was a hit on E:, and every view that opens
    a movie's own files (the recorded frames, the track viewer, QC, triage) looked on L:, which does not exist here:
    "no recorded stack for series001 in D:\\data\\project\\...". A movie's folder IS its condition's root
    (``sources[qualified_movie(name, original)] = (condition_root, original)`` in `load_project`), so it is put back
    under ``available[name]``. A key that matches no loaded condition is kept as stored."""
    out = {}
    for key, (old_root, original) in (sources or {}).items():
        for name, root in available.items():
            if qualified_movie(name, str(original)) == key:
                out[key] = (Path(root), str(original))
                break
        else:
            out[key] = (Path(old_root), str(original))
    return out


def _store_cached_project(cache_file, project, day_provenance):
    """Persist the parsed project for instant re-launch. Best-effort, never fatal."""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(pickle.dumps({
            "canonical": project.canonical, "params": project.params,
            "meshes": project.meshes, "sources": project.movie_source,
            "day_mappings": day_provenance or None,
            # PERSISTED because a cache READ is a path, and a path that drops this
            # hands the next caller the derived-from-canonical map with its hole in it.
            "day_maps": project.day_maps or None,
            # PROVENANCE INSIDE THE ENTRY. Before this, an entry could not say which
            # code wrote it, so auditing the cache meant unpickling every payload and
            # inferring from the day column. Both fields are checked on read.
            "cache_version": CANONICAL_CACHE_VERSION,
            "grouping_fingerprint": _grouping_code_fingerprint(),
            "written_at": _dt.datetime.now().isoformat(timespec="seconds")}))
    except Exception:                                  # pragma: no cover - defensive
        pass


class ProjectError(RuntimeError):
    """Raised when a multi-condition project cannot be assembled honestly."""


def qualified_movie(condition: str, movie: str) -> str:
    """``"<condition>_<movie>"``: the movie's name inside a merged project."""
    return f"{condition}_{movie}"


@dataclass
class Project:
    """A loaded multi-condition project."""

    canonical: pd.DataFrame
    meshes: dict
    params: AcquisitionParams
    conditions: list[str]
    roots: dict[str, Path]
    #: qualified movie -> (condition root, original movie name), for image lookup
    movie_source: dict[str, tuple[Path, str]] = field(default_factory=dict)
    #: The sigma the LOADER settled on, kept so clearing a user override returns to it.
    #:
    #: WHY IT IS STORED. Resetting used to mean "go to theoretical", which was the same
    #: thing while theoretical was the only automatic source. It is not any more: a
    #: u-track project loads with the tracker's own per-localisation sigma, so forcing
    #: theoretical on reset would hand back a value 5.7x larger than the one the project
    #: opened with and silently move every floor. It cannot be recomputed on demand
    #: either, because `dx`/`dy` do not survive into the canonical table.
    loaded_sigma: tuple = ("theoretical", None)
    #: condition -> {ORIGINAL movie name: day}, EXACTLY the mapping handed to the loader.
    #:
    #: WHY THIS IS STORED RATHER THAN READ BACK OFF `canonical`. The canonical table is
    #: POST-CELL-FILTER, so a movie with no segmented cells (series009 in the reference
    #: data) has no rows in it at all. A day map derived from that table therefore has a
    #: HOLE, and the raw loader (which reads every movie, filtered or not) refuses a
    #: mapping that does not cover them all. So the derived map is not merely less
    #: convenient than the real one, it is WRONG on exactly the projects where an
    #: unsegmented movie exists.
    #:
    #: A condition maps to None when no explicit mapping was used, which is a real answer
    #: ("the loader's own rule applied") and not a missing one.
    day_maps: dict[str, dict | None] = field(default_factory=dict)
    #: THE FOLDER DESIGN'S FACTORS (`folderdesign`, 24 Oct 2026): ``{condition: {factor: value}}`` and the factor
    #: names in level order, when the project folder holds a design; empty otherwise. What the charts group by.
    factors: dict = field(default_factory=dict)
    factor_names: tuple = ()
    #: REPLICATES DECLARED AS PAIRS on the group board (5 Nov 2026, "Pair replicates"): ``{condition: {replicate in
    #: the ``day`` column: tag}}``, a replicate of another condition with the same tag its partner. Empty unless the
    #: board's box was ticked; set by the doors (`with_replicate_pairs`, the board's Open workspace), read by the
    #: significance brackets' paired test only. Never derived from replicate numbers (Rule 0).
    replicate_pairs: dict = field(default_factory=dict)
    #: condition -> projected table. Projecting a condition costs seconds, and the panel
    #: redraws whenever a control moves, so it is computed once per condition and kept.
    _projected: dict = field(default_factory=dict, repr=False, compare=False)

    def condition(self, name: str) -> pd.DataFrame:
        """One condition's canonical table, with ``attrs`` carried over."""
        if name not in self.conditions:
            raise ProjectError(
                f"no condition {name!r}; have {', '.join(self.conditions)}"
            )
        out = self.canonical[self.canonical["condition"] == name].copy()
        out.attrs.update(self.canonical.attrs)
        return out

    def projected(self, name: str) -> pd.DataFrame:
        """One condition's table WITH cell coordinates (``L_um``, ``D_um``, ``l_norm``,
        ``d_norm``).

        Why this lives on the project rather than in the analysis that wants it. Cell
        coordinates cannot be derived from the canonical table alone - projecting needs
        the outlines - and the project is the only thing holding both. An analysis that
        tried to derive them would have to re-read the meshes off disk, and a panel that
        did it inline would be untested code deciding what the fit sees.

        Rows whose cell has no usable centreline are dropped, and that loss is real: a
        cell with no `mesh` ribs has no axis to project onto. `cellmesh.assignment_funnel`
        reports the same class of loss for the cell filter.
        """
        from .cellcoords import project_tracks

        if name not in self._projected:
            table = self.condition(name)
            out = project_tracks(table, self.meshes, require_centerline=False)
            out = out[out["l_norm"].notna()].copy()
            out.attrs.update(table.attrs)
            out.attrs["projected_from"] = name
            out.attrs["rows_without_centerline"] = int(len(table) - len(out))
            self._projected[name] = out
        return self._projected[name]

    def summary(self) -> str:
        lines = [
            f"{len(self.conditions)} conditions, "
            f"{self.canonical['uid'].nunique()} tracks, "
            f"{len(self.canonical)} localizations "
            f"(pixel {self.params.pixel_size_nm:g} nm, "
            f"τ {self.params.frame_interval_ms:g} ms)"
        ]
        for name in self.conditions:
            part = self.condition(name)
            lines.append(
                f"  {name:10} {part['uid'].nunique():5d} tracks | "
                f"{len(part):6d} localizations | "
                f"{part['cell'].nunique():3d} cells | "
                f"{part['movie'].nunique():3d} movies | "
                f"{part['day'].nunique():2d} replicate(s)"
            )
        return "\n".join(lines)


def with_localization_sigma_nm(project: "Project",
                               sigma_nm: float | None,
                               provenance: str = "configured") -> "Project":
    """A copy of ``project`` whose localization σ is overridden (item D, one source).

    ``sigma_nm`` is a supplied σ in nm, stamped with ``provenance``, ``"configured"``
    (a lab constant, the default here) or ``"measured"`` (a stationary reference).
    ``None`` clears the override and returns to the σ the LOADER settled on, which is
    the tracker-reported value on a u-track project and the theoretical one where no
    tracker records an uncertainty. It is not a synonym for "theoretical". The value and its provenance always move
    together; there is no way to set one without the other. Rebuilds the frozen
    :class:`AcquisitionParams` and re-stamps it on a shallow copy of the canonical
    table so every downstream reader (the floor, the D-correction, the σ stamp,
    the σ-sensitivity panel) sees the new value, and, because σ is provenance key
    material, any cached view recomputes rather than serving a stale σ.
    """
    from dataclasses import replace

    if sigma_nm is not None:
        if provenance not in ("configured", "measured"):
            raise ValueError(
                "an overridden σ must be 'configured' (a fixed value) or 'measured' "
                f"(a stationary reference), got {provenance!r}; 'theoretical' is what "
                "you get by passing sigma_nm=None")
        new_params = replace(project.params, localization_precision_nm=float(sigma_nm),
                             localization_sigma_source=provenance)
    else:
        source, value = getattr(project, "loaded_sigma", ("theoretical", None))
        new_params = replace(project.params, localization_precision_nm=value,
                             localization_sigma_source=source)
    canon = project.canonical.copy(deep=False)
    canon.attrs = dict(project.canonical.attrs)
    canon.attrs["params"] = new_params
    return replace(project, canonical=canon, params=new_params)


def with_track_cap(project: "Project", max_steps) -> "Project":
    """A copy of ``project`` without its tracks longer than ``max_steps`` steps
    (`canonical.cap_track_length`), every condition at once -- one table feeds every
    analysis, so a cap QC alone saw would make QC disagree with the fits. Apply it to the
    UNCAPPED project (the caller keeps that one to undo the cap); ``None`` returns
    ``project``. Projections are not carried over: they were made from the uncapped table."""
    from dataclasses import replace

    from .canonical import cap_track_length

    if max_steps is None:
        return project
    return replace(project, canonical=cap_track_length(project.canonical, max_steps),
                   _projected={})


def with_exclusions(project: "Project", items) -> "Project":
    """A copy of ``project`` without the excluded movies, cells and tracks (`exclusions.apply_exclusions`, every
    condition at once, recorded on ``canonical.attrs``) -- the workspace's Exclusions dialog (16 Oct 2026). ``items``
    are plain ``{"kind", "key", "reason"}`` dicts; none returns ``project``. Apply it to the project WITHOUT
    exclusions (the window keeps that one, capped as the user set it), so a change replaces the old set rather than
    compounding it. Projections are not carried over: they were made from the whole table."""
    from dataclasses import replace

    from .exclusions import apply_exclusions
    from .exclusions_view import exclusion_set

    items = list(items or ())
    if not items:
        return project
    return replace(project, canonical=apply_exclusions(project.canonical, exclusion_set(items)), _projected={})


def regroup(project: "Project", day_maps: dict[str, dict[str, int]]) -> "Project":
    """A copy of ``project`` with ``day`` relabelled from a FRESH per-condition
    mapping, applied to the ALREADY LOADED table rather than re-read from disk.

    Cheap on purpose. ``day`` is a plain label column -- `grouping.assign_groups`
    fills it once at load time, and nothing about the raw read, the unit
    conversion, the track-length filter or the cell filter depends on its VALUE
    -- so moving a series between groups after loading is a `.map()` over an
    existing column, not a reason to re-open a single `.mat` file. This is what
    lets the onboarding group board's rearrangement stay free while it is being
    dragged: nothing here runs until this function is actually called, and only
    the caller decides when that is (the Confirm screen's "Open workspace").

    ``day_maps`` keys are the ORIGINAL (unqualified) movie name, exactly the
    shape ``day_map.json`` already uses -- this relabels the SAME column the
    loader filled, rather than inventing a second one beside it.

    A series absent from a condition's mapping is the user's own "no group
    yet" / "held out" choice (`model.Project.is_included`): its rows are
    DROPPED from that condition rather than given some placeholder day, since a
    series outside every group is -- by the Confirm screen's own definition --
    not part of the fit. A condition left with no rows at all is dropped from
    `conditions` too, so the tabs handed to the browser match what is actually
    there.
    """
    from dataclasses import replace

    canonical = project.canonical
    keep = pd.Series(True, index=canonical.index)
    new_day = canonical["day"].copy()
    updated_maps = dict(project.day_maps)
    for condition, mapping in day_maps.items():
        cond_mask = (canonical["condition"] == condition).to_numpy()
        if not cond_mask.any():
            continue
        qualified = {qualified_movie(condition, m): d for m, d in mapping.items()}
        movie_col = canonical.loc[cond_mask, "movie"]
        new_day.loc[cond_mask] = movie_col.map(qualified)
        keep.loc[cond_mask] = movie_col.isin(qualified.keys())
        updated_maps[condition] = dict(mapping)

    relabelled = canonical.loc[keep].copy()
    relabelled.attrs = dict(canonical.attrs)
    relabelled["day"] = new_day.loc[keep].astype("int64")

    present = set(relabelled["condition"].unique()) if len(relabelled) else set()
    conditions = [c for c in project.conditions if c in present]
    return replace(project, canonical=relabelled, day_maps=updated_maps,
                   conditions=conditions)


def without_held_out(project: "Project", day_maps) -> "Project":
    """``project`` without the series the group board HELD OUT or left with "No group", as remembered in each
    condition's ``day_map.json`` (`daymap.DayMapping.included`; tester D11, ruled 4 Nov 2026): what the board's Open
    workspace leaves, for every door that opens the analysis without the board -- Open session, and the classic
    `--browse` and `open_gui` doors (ruled the same day: "one rule everywhere"). ``project`` itself when no mapping
    holds any. The loader reads every movie (the board must be able to show a held-out one again), so
    the hold-outs are applied here, by `regroup`, the board's own relabelling."""
    held = {c: m.included() for c, m in (day_maps or {}).items()
            if getattr(m, "held_out", None) or getattr(m, "no_group", None)}
    return regroup(project, held) if held else project


def with_replicate_pairs(project: "Project", day_maps) -> "Project":
    """``project`` carrying the pairs declared on the group board, as remembered in each condition's ``day_map.json``
    (`daymap.DayMapping.included_pairs`: keyed by the replicate numbers the fit reads, after `without_held_out`'s
    renumbering). Conditions with none are left out; ``project`` itself, its pairs emptied, when none has any."""
    from dataclasses import replace

    pairs = {c: m.included_pairs() for c, m in (day_maps or {}).items()
             if c in project.conditions and getattr(m, "pairs", None)}
    pairs = {c: p for c, p in pairs.items() if p}
    return replace(project, replicate_pairs=pairs)


#: What a MOVIE looks like, directly inside a folder: no `**`, no depth.
#:
#: This is the one thing about the layout that IS fixed: u-track writes
#: `<movie>/TrackingPackage/tracks/*.mat` and `tracking_gui` writes
#: `<condition>/trackmate/*.xml`. Everything above that is meaningless in this
#: project ("never hardcode a depth or a parent name") which is why the condition
#: is found by asking whether a CHILD is a movie, rather than by counting levels.
MOVIE_TAILS = ("TrackingPackage/tracks/*.mat", "*.xml")


def _is_movie_folder(path: Path, globs) -> bool:
    """True if `path` IS one movie's output rather than a folder of movies."""
    if UTRACK_GLOB in globs and any(path.glob("TrackingPackage/tracks/*.mat")):
        return True
    # A TRACK TABLE, like TrackMate, has no per-movie folder: the folder of tables is
    # what sits inside a condition.
    if TABLE_GLOB in globs and path.name == TABLE_FOLDER and any(
            p.suffix.lower() in TABLE_SUFFIXES for p in path.iterdir() if p.is_file()):
        return True
    # TrackMate has no per-movie folder: the movie is the XML. So the thing that
    # sits inside a condition is the `trackmate/` folder itself.
    return (TRACKMATE_GLOB in globs and path.name == "trackmate"
            and any(path.glob("*.xml")))


#: What a CONDITION folder holds besides its movies, written by the lab's own steps (preparation, tracking,
#: segmentation, the day picker). A folder holding none of these and no movie, but folders that do, is a GROUP of
#: conditions -- one protein's media, say -- and its folders are the conditions (asked 15 Oct 2026, one user's
#: `protein/<condition>`).
CONDITION_MARKERS = ("cell_meshes", "bleaching", "raw", "phase", "sum", "trackmate", TABLE_FOLDER,
                     "day_map.json", "crops.json", "origin.csv")


def _has_condition_markers(folder: Path) -> bool:
    from .cellmesh import mesh_folder

    return any((folder / name).exists() for name in CONDITION_MARKERS) or mesh_folder(folder).is_dir()


def _movie_names(folder: Path, globs) -> list:
    """The movies under ``folder`` by name, for the trackers in ``globs`` (`dayfolders.movie_names`)."""
    from .dayfolders import movie_names

    return movie_names(folder, utrack=UTRACK_GLOB in globs, trackmate=TRACKMATE_GLOB in globs,
                       table=TABLE_GLOB in globs)


def _group_members(folder: Path, globs, has_results) -> list:
    """The conditions inside ``folder`` when it is a GROUP of conditions, else ``[]``.

    A group holds no movie and no condition marker of its own; every folder in it with results is condition-like
    (markers of its own, or movies directly inside). A lone result folder WITHOUT markers is not a group: that is a
    bare condition's `tif/` (``<condition>/tif/<movie>/TrackingPackage``), and taking it for a group would name a
    condition `tif`.

    AND ITS MOVIE NAMES REPEAT ACROSS ITS FOLDERS -- the one thing that tells a group from a condition whose
    replicate days are sub-folders. Both look the same inside (`condition/1`, `/2`, `/3` each hold
    outlines and `tif/<movie>` exactly as one user's `protein/condition` does), but a condition's movies
    are numbered on across its days (series011.., series021..) -- it could not load as one otherwise -- while
    separate conditions each start again (series101 in every one). Without this test the groups rule took every
    such condition apart into day folders named 1, 2, 3 and refused the project (protein A, protein C, protein B and protein D on the lab's data drive,
    found 15 Oct 2026 by the optimiser comparison). With unique names the folder stays ONE condition, as before."""
    if _has_condition_markers(folder):
        return []
    from .dayfolders import replicate_folders

    if replicate_folders(folder) is not None:
        return []                       # ONE condition whose days are sub-folders (`dayfolders`), not a group
    kids = [c for c in sorted(folder.iterdir())
            if c.is_dir() and c.name != "backups" and has_results(c)]
    if not kids or any(_is_movie_folder(c, globs) for c in kids):
        return []

    def unwrapped(c: Path) -> Path:
        # THROUGH FOLDERS THAT ONLY WRAP ONE: `comparison/strain A/20240101/strain A-0.5 xylose` is the
        # condition strain A, a date folder between (20 Oct 2026)
        for _ in range(4):
            if _has_condition_markers(c):
                break
            inner = [g for g in c.iterdir() if g.is_dir() and g.name != "backups" and has_results(g)]
            if len(inner) != 1 or _is_movie_folder(inner[0], globs):
                break
            c = inner[0]
        return c

    def condition_like(c: Path) -> bool:
        c = unwrapped(c)
        return _has_condition_markers(c) or any(
            _is_movie_folder(g, globs) for g in c.iterdir() if g.is_dir()) or replicate_folders(c) is not None

    if not all(condition_like(c) for c in kids):
        return []
    if len(kids) == 1 and not _has_condition_markers(kids[0]):
        return []
    seen, repeated = set(), False
    for kid in kids:
        names = set(_movie_names(kid, globs))
        if seen & names:
            repeated = True
            break
        seen |= names
    return kids if repeated else []


def discover_conditions(root: str | Path, *, source: str = "auto", design=None) -> dict[str, Path]:
    """Condition folders under `root`, or `root` itself when it IS one condition.

    A FOLDER DESIGN DECIDES WHEN THERE IS ONE (`folderdesign`, 24 Oct 2026): ``design`` None reads the project's own
    (`folderdesign.DESIGN_FILE` in ``root``); a `folderdesign.FolderDesign` is used as given; ``False`` ignores any.
    The conditions are then the design's -- one per combination of factor values, named by them -- and a design
    that does not fit the folders is refused by name (`folderdesign.apply`), never loaded another way. Without a
    design every rule below applies, unchanged.

    ``source="auto"`` accepts EITHER tracker's layout, because the folder that holds
    a TrackMate batch is as much a condition as one holding u-track output.
    Restricting this to u-track is what made the launcher's Tracking stage a dead
    end: a project tracked here could be found by nothing that loads projects.

    A folder that is itself one condition is one condition. Pointing at
    `…/protein LB` (movies directly inside, no condition sub-folders) used to
    raise and send the user up a level, where the siblings are different experiments
    rather than conditions of this one. Asked for: "if no subfolder is found and
    instead tracking data folder -> open just these as one condition".

    The test is whether a child is a movie, not how deep the file sits. The old
    rule called a child a condition when it held results anywhere below it, and the
    glob is recursive, so in a single-condition folder every MOVIE matched and each
    came back as its own condition. No error, just a project with one condition per
    movie. An attempted fix that derived the condition at a fixed depth above each
    result file was worse and was caught by the reference project: it returned `tif`
    for `<condition>/tif/<movie>/TrackingPackage/…`, because the folders above a
    movie carry no meaning here and a depth cannot be assumed. Only the movie TAIL
    is fixed, so that is what is matched.
    """
    root = Path(root)
    if not root.is_dir():
        raise ProjectError(f"project folder not found: {root}")
    in_force = design_in_force(root, design)
    if in_force is not None:
        return _design_result(root, in_force).available()
    if source == "auto":
        globs = (UTRACK_GLOB, TRACKMATE_GLOB, TABLE_GLOB)
    elif source == "utrack":
        globs = (UTRACK_GLOB,)
    elif source == "trackmate":
        globs = (TRACKMATE_GLOB,)
    elif source == "table":
        globs = (TABLE_GLOB,)
    else:
        raise ProjectError(
            f"source={source!r} is not one of {('auto',) + PROJECT_SOURCES}")

    def has_results(folder: Path) -> bool:
        for pattern in globs:
            for hit in folder.glob(pattern):
                parts = hit.relative_to(folder).parts
                if "backups" in parts or any(q.startswith("._") for q in parts):
                    continue
                if pattern == TABLE_GLOB and hit.suffix.lower() not in TABLE_SUFFIXES:
                    continue
                return True
        return False

    # A CONDITION WHOSE REPLICATE DAYS ARE SUB-FOLDERS (`dayfolders`) is one condition, opened on its own too:
    # `LB` holding `LB-1`, `LB-2`, `LB-3` came back as three conditions, one per day (20 Oct 2026).
    from .dayfolders import replicate_folders

    if replicate_folders(root) is not None:
        return {root.name: root}
    children = [c for c in sorted(root.iterdir())
                if c.is_dir() and c.name != "backups" and has_results(c)]
    # A CONDITION OPENED ON ITS OWN, ITS MOVIES IN tif/: outlines (or another marker) here and one bare folder of
    # movies is the lab's own condition layout, never a project whose one condition is called `tif` -- which is
    # what opening the reference project or a day folder like `condition/rep1` gave (found 16 Oct 2026). The
    # same reading `_group_members` makes one level down.
    if (len(children) == 1 and _has_condition_markers(root) and not _has_condition_markers(children[0])
            and not _is_movie_folder(children[0], globs)):
        return {root.name: root}
    # ONE MOVIE AMONG THE CHILDREN MEANS THIS FOLDER IS THE CONDITION. A comparison
    # root holds condition folders, and a condition folder holds movies; the two
    # cannot be told apart by counting, only by what the children ARE.
    if children and not any(_is_movie_folder(c, globs) for c in children):
        # A GROUP OF CONDITIONS (one protein's media) opens as its conditions, under their own names.
        groups = {child: _group_members(child, globs, has_results) for child in children}
        expanded = [m for child in children for m in (groups[child] or [child])]
        names = [m.name for m in expanded]
        if len(names) != len(set(names)):
            # TWO GROUPS HOLDING CONDITIONS OF ONE NAME (`21.03/exp/free` and `21.03/stat/free`, one user's growth
            # phases; asked 20 Oct 2026) are named by their group: `exp free`, `stat free`. The fallback below took
            # `exp` and `stat` for the conditions, which then held every movie name twice and could only be
            # refused; day folders, which it was written for, are recognised before this (`dayfolders`).
            named = {}
            for child in children:
                for member in groups[child] or [child]:
                    named[f"{child.name} {member.name}" if groups[child] else member.name] = member
            if len(named) == len(expanded):
                return named
            # ...UNLESS TWO WOULD SHARE A NAME, which a group's conditions never do and a condition's replicate
            # days as sub-folders always do (D:/data/project/condition/1, /2, /3 and M/1, /2, /3, whose movie
            # names also repeat across days). Then the folder opens exactly as before the groups rule, its
            # children as conditions, and the load says what it always said; nothing that opened before is
            # refused now, nothing refused before is refused differently (15 Oct 2026).
            expanded = children
        return {m.name: m for m in expanded}
    if has_results(root):
        return {root.name: root}

    wanted = " or ".join(repr(g) for g in globs)
    raise ProjectError(
        f"neither {root} nor any folder under it contains {wanted}. A single "
        "condition holds its movies directly; a project with several conditions holds one "
        "folder per condition, each a u-track project or a folder of "
        "TrackMate XMLs."
    )

def design_in_force(root, design=None):
    """The `folderdesign.FolderDesign` a load of ``root`` uses: ``design`` itself, the project's own file when
    ``design`` is None, or None (``design`` False, or no file). A broken design file raises `ProjectError`."""
    from .folderdesign import DesignError, FolderDesign, read

    if design is False:
        return None
    if isinstance(design, FolderDesign):
        return design
    if design is not None:
        raise ProjectError(f"design must be a FolderDesign, False or None, got {type(design).__name__}")
    try:
        return read(root)
    except DesignError as exc:
        raise ProjectError(str(exc)) from None


def _design_result(root, design):
    """`folderdesign.apply`, refused by name when the design does not fit."""
    from .folderdesign import DESIGN_FILE, apply

    result = apply(design, root)
    if not result.ok:
        why = "; ".join(result.problems) or "it finds no condition"
        raise ProjectError(f"the folder design ({design.describe()}; {DESIGN_FILE} in {root}) does not fit the "
                           f"folders: {why}. Change the design, or delete the file to load the folder without one.")
    return result


def tracker_options(root: str | Path) -> dict[str, dict[str, int]]:
    """``{condition: {"utrack": n, "trackmate": n}}``: which trackers left files in each
    condition under ``root``, counted, so a window can ASK where a condition was tracked
    twice instead of failing. A listing only; nothing is read.

    Reported: a condition holding both u-track and TrackMate output stopped the analysis
    door with an error. `detect_project_source` is right to refuse a silent choice; the
    answer is a question, and this is what the question is asked from.
    """
    return {name: sources_present(path)
            for name, path in discover_conditions(root, source="auto").items()}


#: Where the tracking stage records the calibration it tracked with.
CROPS_FILENAME = "crops.json"


def _trackmate_acquisition_params(condition_root, *, localization_precision_nm=None,
                                  pixel_size_nm=None):
    """Acquisition parameters for a TrackMate project, READ and never guessed.

    A TrackMate project has no ``movieData.mat``, so the u-track route to pixel size and
    frame interval does not exist. It does not need to be invented: the PREPARATION
    stage already writes both into ``crops.json`` beside the ranges, and
    ``movieprep.save_crops`` says why: "ranges alone would let a run be reproduced at
    the wrong pixel size or frame interval, which is exactly the pair this project keeps
    having to correct".

    So the order is: an explicitly supplied ``pixel_size_nm`` wins, otherwise
    ``crops.json``, otherwise REFUSE. There is deliberately no fallback: a wrong pixel
    size rescales every D by its square (the .nd2's 160 nm against the true 106 costs a
    factor 2.28) and a wrong frame interval scales it linearly, and neither shows up as
    an error anywhere downstream: the numbers simply come out wrong and plausible.

    The search is recursive because the folder above a movie carries no meaning here
    either; ``crops.json`` may sit at the condition root or one level down beside the
    prepared stacks.
    """
    from .canonical import AcquisitionParams

    found = sorted(Path(condition_root).rglob(CROPS_FILENAME))
    payload = None
    if found:
        from .movieprep import load_crops

        payload = load_crops(found[0])

    pixel = pixel_size_nm
    pixel_from = "the pixel size given to the loader (pixel_size_nm=)" if pixel is not None else None
    if pixel is None and payload is not None:
        pixel = payload.get("pixel_size_nm")
        pixel_from = str(found[0]) if pixel is not None else None
    interval_s = payload.get("frame_interval_s") if payload is not None else None
    interval_from = str(found[0]) if interval_s is not None else None
    # A FOLDER OF TRACK TABLES FROM ANOTHER TOOL may have no movies, so no preparation run
    # and no crops.json: its `track_tables/format.json` may declare the two numbers instead
    # (`tracktable.folder_format`). Read, like crops.json, and never defaulted.
    from .tracktable import FORMAT_FILE, TABLE_FOLDER, folder_format

    declarations = [(folder / FORMAT_FILE, folder_format(folder))
                    for folder in sorted(Path(condition_root).rglob(TABLE_FOLDER)) if folder.is_dir()]
    for path, declared in declarations:
        if pixel is None and declared.get("pixel_size_nm") is not None:
            pixel, pixel_from = float(declared["pixel_size_nm"]), str(path)
        if interval_s is None and declared.get("frame_interval_ms") is not None:
            interval_s, interval_from = float(declared["frame_interval_ms"]) / 1000.0, str(path)
    # A DECLARATION THAT LOSES IS SAID (asked 26 Nov 2026, "agree": the docs site found that a format.json
    # disagreeing with the saved calibration was ignored without a word). The order above stays -- the loader's
    # argument, then crops.json, then format.json -- but a format.json naming another number is a person who
    # believes the tracks were taken at it, and every D depends on which one is right.
    for path, declared in declarations:
        for key, used, used_from, unit, scale in (
                ("pixel_size_nm", pixel, pixel_from, "nm", 1.0),
                ("frame_interval_ms", interval_s, interval_from, "ms", 1000.0)):
            if used is None or declared.get(key) is None or str(path) == used_from:
                continue
            said, kept = float(declared[key]), float(used) * scale
            if abs(said - kept) > 1e-3 * kept:
                warnings.warn(f"{path} gives {key} {said:g} {unit}, but {kept:g} {unit} from {used_from} is "
                              f"used. Every D depends on it: correct the one that is wrong.",
                              RuntimeWarning, stacklevel=3)

    missing = [n for n, v in (("pixel size", pixel), ("frame interval", interval_s))
               if v is None]
    if missing:
        raise ProjectError(
            f"{condition_root}: these tracks (TrackMate, or a track table) do not record "
            f"their acquisition parameters ({' and '.join(missing)} unknown). Give them in "
            f"the {CROPS_FILENAME} written by Calibration & save, in "
            f"track_tables/format.json (\"pixel_size_nm\", \"frame_interval_ms\"), or as pixel_size_nm=. "
            f"SMTracker uses no defaults, because a wrong pixel size changes every D by its square."
        )

    return AcquisitionParams(
        pixel_size_nm=float(pixel),
        frame_interval_ms=float(interval_s) * 1000.0,
        localization_precision_nm=localization_precision_nm,
    )


#: Where a u-track movie's frame interval comes from (`load_project(frame_interval_from=...)`). ``"record"``: what
#: u-track recorded -- the default, so every script and the manuscript load as before. ``"camera"``: the raw
#: acquisition's own timing where it sits beside the movie (`rawmovie.frame_interval`: the `.nd2` time loop, the
#: average spacing of the `.stk` timestamps), the record elsewhere; asked by the analysis APP (`APP_FRAME_INTERVAL`).
#: Asked 20 Oct 2026: "in the end the camera frame interval is the correct one, if they are recorded in the
#: metadata, why not use them?" -- u-track records what it was TOLD: one user's movies say 20 ms (the exposure) where
#: the camera took a frame every 25.0-25.8 ms, protein A's 20.0 where the `.nd2` loop says 20.47.
FRAME_INTERVAL_SOURCES = ("record", "camera")
APP_FRAME_INTERVAL = "camera"


def camera_files(condition_root) -> dict:
    """``{movie: its raw acquisition}`` for the u-track movies under a condition: ``raw/<movie>.nd2|.stk`` in the
    movie's own condition folder (a day folder's, for one in day folders), else the file loose there. Keyed by the
    name the condition knows the movie by (`dayfolders.namer`: ``series001 (19.03.2025)`` in day folders reusing
    names)."""
    from .dayfolders import namer

    root = Path(condition_root)
    known_as = namer(root)
    out = {}
    for data in sorted(root.glob("**/movieData.mat")):
        if "backups" in data.parts:
            continue
        movie_dir = data.parent
        movie = movie_dir.name
        home = movie_dir.parent.parent if movie_dir.parent.name.lower().startswith("tif") else movie_dir.parent
        for candidate in [home / "raw" / f"{movie}{suffix}" for suffix in (".nd2", ".stk")] + [
                home / f"{movie}{suffix}" for suffix in (".nd2", ".stk")]:
            if candidate.is_file():
                out[known_as(movie_dir)] = candidate
                break
    return out


#: Camera intervals closer than this (relative) are ONE interval: the clock's jitter, not two acquisitions.
#: Each `.nd2` reports its own loop average, so the movies of one session read 20.2981 and 20.2986 ms; kept
#: apart they made a "mixed" project that the app then "rescaled" by a factor of 1.0000, with a warning each
#: time (reported 20 Oct 2026). 0.1 % is ten times the largest such spread seen and a fortieth of the smallest
#: real one (one user's 25.0 against 25.8 ms).
CAMERA_SAME_INTERVAL = 0.001


def _settle_camera(camera_by_condition: dict) -> dict:
    """Every condition's camera intervals with those within `CAMERA_SAME_INTERVAL` of each other, across the whole
    project, set to their cluster's median: ``{condition: {movie: (ms, source, file)}}``, same shape."""
    import numpy as np

    entries = sorted(((v[0], c, m) for c, camera in camera_by_condition.items() for m, v in camera.items()))
    settled, cluster = {}, []

    def close(group):
        median = float(np.median([ms for ms, _c, _m in group]))
        for _ms, c, m in group:
            settled[(c, m)] = median

    for entry in entries:
        if cluster and entry[0] / cluster[0][0] - 1.0 > CAMERA_SAME_INTERVAL:
            close(cluster)
            cluster = []
        cluster.append(entry)
    if cluster:
        close(cluster)
    return {c: {m: (settled[(c, m)], v[1], v[2]) for m, v in camera.items()}
            for c, camera in camera_by_condition.items()}


def _recorded_intervals(condition_root, movies) -> dict:
    """``{movie: ms}`` as u-track RECORDED them, for the movies named (None where it recorded none)."""
    from .dayfolders import namer
    from .utrack_meta import MetadataError, read_movie_metadata

    known_as = namer(condition_root)
    out = {}
    for data in sorted(Path(condition_root).glob("**/movieData.mat")):
        if "backups" in data.parts or known_as(data.parent) not in movies:
            continue
        try:
            meta = read_movie_metadata(data, frame_interval_ms=1.0)          # never refuses for the interval
        except MetadataError:
            continue
        rec = meta.recorded_frame_interval_s
        out[known_as(data.parent)] = None if rec is None else 1000.0 * float(rec)
    return out


def camera_frame_intervals(condition_root) -> dict:
    """``{movie: (ms, source, file)}``: the frame interval each movie's CAMERA measured (`camera_files`,
    `rawmovie.frame_interval`), for those whose file records timing. A file that cannot be read is named in a
    warning and its movie keeps u-track's interval."""
    from .rawmovie import RawMovieError, frame_interval

    out = {}
    for movie, raw in camera_files(condition_root).items():
        try:
            ms, source = frame_interval(raw)
        except RawMovieError as exc:
            warnings.warn(f"{raw}: its frame interval could not be read ({exc}); {movie} keeps the interval "
                          "u-track recorded.", RuntimeWarning, stacklevel=2)
            continue
        if ms is not None and ms > 0:
            out[movie] = (float(ms), str(source), raw)
    return out


def camera_exposures(condition_root) -> dict:
    """``{movie: (exposure_ms, interval_ms, file)}``: how long each movie's camera was open a frame and its measured
    frame interval (`camera_files`, `rawmovie.exposure_and_interval`), for the movies whose file records an exposure
    (the interval None where it records no timing). A file that cannot be read is named in a warning and left out."""
    from .rawmovie import RawMovieError, exposure_and_interval

    out = {}
    for movie, raw in camera_files(condition_root).items():
        try:
            exposure, interval = exposure_and_interval(raw)
        except RawMovieError as exc:
            warnings.warn(f"{raw}: its exposure could not be read ({exc}); {movie} is left out of it.",
                          RuntimeWarning, stacklevel=2)
            continue
        if exposure is not None and exposure > 0:
            out[movie] = (float(exposure), None if not interval or interval <= 0 else float(interval), raw)
    return out


def _utrack_acquisition_params(condition_root, *, localization_precision_nm=None,
                               pixel_size_nm=None, frame_interval_ms=None, camera=None, leave_out=()):
    """Acquisition parameters for a u-track project, with explicit values allowed.

    Read first, always. `acquisition_params_from_project` is asked before anything else
    and its answer wins wherever an override was not supplied, so this cannot become a
    quiet way to run a project at the wrong scale.

    The overrides exist because a MovieData can be incomplete. Measured on
    `protein E-mNG Malate`: `timeInterval_` is missing from every movie there while its
    sibling `protein E-mNG Glucose` has one, so the reader refuses, correctly, since a frame
    interval scales every D linearly and it will not guess. The value is not unknown,
    though; it is in the acquisition itself (`.stk`, 25.0 ms), which is a BETTER source
    than the MovieData either way: u-track records the nominal setting, and the two
    disagree in every project checked here (one protein's 20.0 recorded against 20.4732 measured,
    protein E/Glucose 25.15 against 25.00).

    `pixel_size_nm` already existed for the TrackMate path and is accepted here for the
    same reason and with the same precedence. Fractions and maps are scale-invariant, so
    what these move is D and only D.
    """
    from .utrack_meta import MetadataError

    read = None
    try:
        read = acquisition_params_from_project(
            condition_root, localization_precision_nm=localization_precision_nm,
            frame_intervals_ms={m: v[0] for m, v in (camera or {}).items()} or None, leave_out=leave_out)
    except MetadataError:
        if pixel_size_nm is None or frame_interval_ms is None:
            raise
    if read is not None and pixel_size_nm is None and frame_interval_ms is None:
        return read
    return AcquisitionParams(
        pixel_size_nm=float(pixel_size_nm if pixel_size_nm is not None
                            else read.pixel_size_nm),
        frame_interval_ms=float(frame_interval_ms if frame_interval_ms is not None
                                else read.frame_interval_ms),
        localization_precision_nm=localization_precision_nm,
    )


def acquisition_record(params: AcquisitionParams) -> dict:
    """One condition's acquisition as RECORDED, in plain numbers (None where the tracker recorded none):
    ``{"pixel_size_nm", "frame_interval_ms", "search_radius_px", "psf_sigma_px"}``. ``frame_interval_ms`` is the
    condition's stated interval; per-movie intervals travel in the table's ``tau_s``. Kept per condition on
    ``canonical.attrs["acquisition_by_condition"]`` for the screens; nothing computes from it."""
    def number(value):
        return None if value is None else float(value)

    return {"pixel_size_nm": number(getattr(params, "pixel_size_nm", None)),
            "frame_interval_ms": number(getattr(params, "frame_interval_ms", None)),
            "search_radius_px": number(getattr(params, "search_radius_px", None)),
            "psf_sigma_px": number(getattr(params, "psf_sigma_px", None))}


def pixel_mix_text(mixed: dict, used: float) -> str:
    """What a pixel-size mix within `canonical.PIXEL_SIZE_WARN_REL` means, in words: each condition's recorded size and
    how far its D reads off when converted at ``used`` (D scales with the pixel size squared)."""
    parts = []
    for name, nm in mixed.items():
        off = 100.0 * ((float(used) / float(nm)) ** 2 - 1.0)
        parts.append(f"{name} {nm:.4g} nm" + ("" if abs(off) < 0.005 else f" (D {off:+.1f} %)"))
    return (f"pixel sizes differ by up to {100.0 * (max(mixed.values()) / min(mixed.values()) - 1.0):.2f} %: "
            + ", ".join(parts) + f"; every coordinate converted at {float(used):.4g} nm, fractions unchanged")


def radius_mix_text(radii: dict) -> str:
    """What a project linked at several search radii means, in words, or "" when its conditions were linked alike:
    the load's warning, QC's "Is the comparison fair?" caption and the Confirm page's line read it (tester D10, ruled
    4 Nov 2026). ``radii`` is ``{condition: radius_px or None}`` (`AcquisitionParams.search_radius_px_by_condition`);
    each condition is named where there are few, else each radius with how many conditions have it.

    Said only where two RECORDED radii differ, as the Confirm page did before: a track table or a TrackMate condition
    records none, and a declared truth beside u-track's 8 px (free mNG's projects) is not a project linked at two
    radii. Such a condition still never borrows another's radius (its ceiling is unknown, `search_radius_px_for`), and
    where the sentence is said it is named, "none recorded"."""
    from .canonical import distinct_radii, radius_words

    if len(distinct_radii(r for r in radii.values() if r is not None)) <= 1:
        return ""
    if len(radii) <= 4:
        listed = ", ".join(f"{name} {radius_words(r)}" for name, r in radii.items())
    else:
        by_value: dict = {}
        for name, r in radii.items():
            by_value.setdefault(radius_words(r), []).append(name)
        listed = "; ".join(f"{value} for {' and '.join(names)}" if len(names) <= 2
                           else f"{value} for {len(names)} conditions"
                           for value, names in sorted(by_value.items(), key=lambda kv: -len(kv[1])))
    return (f"Not one search radius: {listed}. The radius caps the fastest D a track can show, so a comparison "
            "across them also compares linking. Each condition's detection ceiling and crowding checks use its own "
            "radius; the fitted D and fractions do not depend on it.")


def _say_search_radii(params) -> None:
    """The load's WARNING when the conditions were linked at different search radii, on a fresh read and a cached
    one alike (a cache hit runs no merge): every condition's radius named (`radius_mix_text`)."""
    text = radius_mix_text(getattr(params, "search_radius_px_by_condition", None) or {})
    if text:
        warnings.warn(text + ".", RuntimeWarning, stacklevel=3)


def _with_linking_radius(params: AcquisitionParams, by_movie, movies, condition: str) -> AcquisitionParams:
    """A TrackMate condition's parameters carrying its LINKING DISTANCE as its search radius (ruled 4 Nov 2026: "read
    TrackMate's linking distance from its XML as that condition's search radius"; before, a TrackMate condition had
    none, and in a project beside u-track it borrowed the first condition's). ``by_movie`` is
    ``{movie: pixels or None}`` from `grouping.load_trackmate_project` (`trackmate.linking_radius_px`), ``movies`` the
    movies read. Movies linked at different distances give the LARGEST, the fastest step any of them can show, with a
    warning naming each; an XML recording none adds nothing, and none recorded leaves the radius unknown, never
    defaulted. A track table is no TrackMate run and is never given one."""
    from .canonical import distinct_radii

    recorded = {m: float(r) for m, r in (by_movie or {}).items() if m in movies and r is not None}
    if not recorded:
        return params
    radius = max(recorded.values())
    if len(distinct_radii(recorded.values())) > 1:
        listed = ", ".join(f"{m} {r:g} px" for m, r in sorted(recorded.items()))
        warnings.warn(f"{condition}: its TrackMate movies were linked at different distances ({listed}). Its search "
                      f"radius is the largest, {radius:g} px, the fastest step any of them can show.",
                      RuntimeWarning, stacklevel=3)
    return replace(params, search_radius_px=radius)


def _check_params_agree(per_condition: dict[str, AcquisitionParams], mixed: dict | None = None,
                        movies: dict | None = None) -> AcquisitionParams:
    """Merge the conditions' acquisition parameters, or refuse if they cannot be.

    Pixel size stays fatal to mix beyond 1 %: it rescales every coordinate, and there is no
    per-movie route through the geometry (cell outlines, r_norm, the shell), a
    project with two pixel sizes is two projects. Within 1 % (`canonical.PIXEL_SIZE_WARN_REL`,
    ruled 26 Oct 2026) the conditions load at the FIRST one's pixel size with a warning, ``mixed``
    is filled with each condition's size, and QC says how far each D reads off (`pixel_mix_text`);
    rounding (`canonical.same_setting`) is silent.

    Frame interval does NOT, since τ now travels per movie. Two conditions imaged at
    different rates are a STRATUM, not an impossibility: each movie's D is computed
    with its own τ, and ``AcquisitionParams.tau_s`` refuses to hand out a scalar, so
    any code path that cannot handle the mixture fails loudly at the point of use
    rather than silently averaging two different measurements.

    What the caller still has to weigh, and what ``tau_strata()`` is for: at 20.3 vs
    30 ms the static-error offset σ²/τ differs by ~0.014 µm²/s, the detection ceiling
    r²/(4τ) by ~48 %, and motion blur differs. So a comparison ACROSS τ strata is
    confounded by acquisition even when the arithmetic is right: it is reported as
    strata, never merged into one number.
    """
    from .canonical import PIXEL_SIZE_WARN_REL, same_setting

    reference_name, reference = next(iter(per_condition.items()))
    sizes = {name: float(params.pixel_size_nm) for name, params in per_condition.items()}
    for name, params in per_condition.items():
        a, b = reference.pixel_size_nm, params.pixel_size_nm
        # ONE SETTING TO A PART IN A MILLION (`canonical.same_setting`): u-track wrote 106 nm as 106.000003 in some
        # projects, and an exact comparison refused a folder that held both. Within 1 % they load with a warning
        # (`canonical.PIXEL_SIZE_WARN_REL`, ruled 26 Oct 2026), further apart they are two projects.
        if same_setting(a, b):
            continue
        if abs(float(b) / float(a) - 1.0) > PIXEL_SIZE_WARN_REL:
            raise ProjectError(
                f"pixel_size_nm differs between conditions: {reference_name} = {a}, "
                f"{name} = {b}, more than {PIXEL_SIZE_WARN_REL:.0%} apart. Every coordinate scales with it, "
                "so these cannot be analysed as one project.")

    if any(not same_setting(reference.pixel_size_nm, nm) for nm in sizes.values()):
        text = pixel_mix_text(sizes, reference.pixel_size_nm)
        warnings.warn(text[:1].upper() + text[1:] + ". QC's 'Is the comparison fair?' repeats this beside the numbers.",
                      RuntimeWarning, stacklevel=2)
        if mixed is not None:
            mixed.update(sizes)

    # A CONDITION WITH ONE STATED RATE (TrackMate, a track table: `crops.json` or `format.json`) GETS IT PER MOVIE
    # whenever the project carries rates per movie (any u-track condition) or its stated rates differ (30 Oct 2026).
    # Without its movies in the map, a u-track condition at 25.0-25.9 ms beside a track table made the project
    # mixed and `tau_s_for` refused every table movie at canonicalisation ("no frame interval recorded for movie");
    # and two TrackMate conditions at 20 and 30 ms left the map empty, so every movie read the FIRST condition's
    # tau, the second's D 1.5-fold off with only a warning. ``movies`` is ``{condition: [its movies]}``, the movies
    # read (`load_project`); a project of single-rate conditions at one rate keeps no map, as before.
    movies = movies or {}
    stated = {round(float(p.frame_interval_ms), 6) for p in per_condition.values()
              if not p.frame_interval_ms_by_movie}
    per_movie = (any(p.frame_interval_ms_by_movie for p in per_condition.values()) or len(stated) > 1)
    merged: dict[str, float] = {}
    for name, params in per_condition.items():
        by_movie = params.frame_interval_ms_by_movie or {}
        if not by_movie and per_movie and movies.get(name):
            by_movie = {movie: float(params.frame_interval_ms) for movie in movies[name]}
        for movie, ms in by_movie.items():
            merged[qualified_movie(name, movie)] = float(ms)
        if not by_movie:                       # condition with a single stated rate
            merged.setdefault(f"__condition__{name}", float(params.frame_interval_ms))
    rates = {round(v, 6) for v in merged.values()} if merged else set()
    if len(rates) > 1:
        # Not an error, but the analyst must know, because it changes the ceiling
        # and the blur, not just the arithmetic.
        strata = ", ".join(f"{r:g} ms" for r in sorted(rates))
        warnings.warn(
            f"conditions were imaged at different frame intervals ({strata}). Every D uses its "
            "own movie's interval, but the detection ceiling and motion blur differ between "
            "the groups, so a comparison across them also compares acquisitions. Report them "
            "as separate groups.",
            RuntimeWarning, stacklevel=2)
    # THE SEARCH RADIUS PER CONDITION where they were linked at different radii (tester D10, ruled 4 Nov 2026): the
    # merge kept the FIRST condition's for all, so free mNG's 8 px conditions drew their apparent-D ceiling, QC's
    # crowding and the SDA limits at 5 px. `AcquisitionParams` keeps a map only where the radii differ (None, "none
    # recorded", is a value of its own: a condition never borrows another's), so one radius reads as before. The
    # load's warning is `_say_search_radii`, said on a cache hit too.
    radii = {name: params.search_radius_px for name, params in per_condition.items()}
    return replace(reference,
                   frame_interval_ms_by_movie={k: v for k, v in merged.items()
                                               if not k.startswith("__condition__")}
                   or None,
                   search_radius_px_by_condition=radii if len(radii) > 1 else None)


def load_project(
    root: str | Path,
    *,
    conditions=None,
    localization_precision_nm: float | None = None,
    series_per_day: int = 10,
    min_localizations: int | None = None,
    allow_unsegmented: bool = True,
    day_maps=None,
    use_canonical_cache: bool = True,
    require_all_inside: bool = True,
    source="auto",
    pixel_size_nm: float | None = None,
    frame_interval_ms: float | None = None,
    skip_unsegmented: bool = False,
    on_condition=None,
    on_movie=None,
    on_stage=None,
    mixed_tau: str = "refuse",
    outline_tolerance_nm: float = 0.0,
    frame_interval_from: str = "record",
    leave_out: dict | None = None,
    max_tau_spread: float | None = None,
    design=None,
) -> Project:
    """Load every condition folder under ``root`` into one canonical table.

    ``design`` (`discover_conditions`): the project's folder design, read from its file when None, ignored when
    False. Its factors are on ``Project.factors`` / ``factor_names``; it is part of the cache key.

    ``frame_interval_from`` (`FRAME_INTERVAL_SOURCES`): ``"record"`` (the default) takes each u-track movie's frame
    interval as u-track recorded it; ``"camera"`` takes it from the raw acquisition beside the movie where it
    records timing (`camera_frame_intervals`), as the analysis app asks (`APP_FRAME_INTERVAL`). An explicit
    ``frame_interval_ms`` beats both. What was used is on ``canonical.attrs["frame_interval_camera"]``, and a
    camera interval differing from u-track's by more than 0.5 % is named in a warning.

    ``leave_out`` ``{condition: {movie: reason}}`` leaves those movies out before anything of theirs is read --
    the answer to a :class:`utrack_meta.MissingAcquisitionError`, which names every movie, in every condition,
    tracked without a pixel size or a frame interval (``.by_condition``), so a window can ask once. Recorded on
    ``canonical.attrs["left_out_movies"]``.

    ``outline_tolerance_nm`` widens the cell filter: a track is kept when every point lies
    inside its cell's outline OR within this distance of it (`cellmesh.assign_cells`,
    ``tolerance_px``, converted with the project's pixel size). ``0`` (the default) is the
    legacy filter, exactly, so the reference and the manuscript load as before; the
    analysis app reads the Options screen's "Outline tolerance" (`options.outline_tolerance_nm`).
    Recorded on ``canonical.attrs["outline_tolerance_nm"]`` with the tracks it added.

    ``mixed_tau`` is what happens when the movies were recorded at frame intervals a few
    per cent apart (`rescale_mixed_tau`): ``"refuse"`` (the default) keeps each its own, so
    a pooled fit refuses as before -- every script, the manuscript's included, loads exactly
    as it did; ``"rescale"`` puts them on one tau, which the analysis APP asks for
    (`APP_MIXED_TAU`, the experimenter's ruling of 13 Oct 2026). Applied AFTER the cache,
    which keeps the tables as read. ``max_tau_spread`` is the largest spread put on one tau
    (`MAX_TAU_RESCALE` when None); the app reads it from the Options screen
    (`options.max_tau_spread`).

    ``on_stage`` (optional) is called ``on_stage(text)`` at the few steps AFTER
    the files are read -- assigning cells, canonicalising, writing the cache.
    Those have no count to report and, on the reference project, they are half
    the wall clock: reading thirty movies took 8 s and the rest took another 8,
    with nothing to say for it. A step that is named is not a hang.

    ``on_movie`` (optional) is called ``on_movie(condition, movie, done, total)``
    after each file is read, and it is the one that matters on a project with a
    single condition: ``on_condition`` fires once there and says "1 of 1" for
    the whole load. It is passed down to the tracker's own reader, which is the
    loop the time is actually in.

    ``on_condition`` (optional) is called ``on_condition(name, done, total)`` after
    each condition is read, so a window can say what it is working on. It exists
    because the alternative is worse: a UI that wanted per-condition progress would
    have to loop the conditions ITSELF, calling this once each, and that is a
    different load -- separate cache keys, and no cross-condition acquisition-parameter
    check, which is the thing that catches two conditions imaged at different pixel
    sizes. Progress is a property of this loop, so the hook belongs in it.

    NOTHING IS CALLED ON A CACHE HIT, and that is correct rather than an omission:
    a cached project returns whole, in one step, with no work to report.

    ``day_maps`` (optional) is ``{condition: DayMapping}``, a CONFIRMED
    replicate-day mapping from ``smtracker.daymap`` (resolution + confirmation
    happen in the UI, so this only applies what it is given). When a condition has
    one, its per-movie ``day`` comes from the mapping instead of the
    ``series_per_day`` rule, and the mapping's provenance record is stored on
    ``canonical.attrs["day_mappings"]`` so a figure records which mapping produced
    it. Conditions without a mapping keep the series-based default.

    ``source`` is ``"auto"``, ``"utrack"``, ``"trackmate"``, or a mapping
    ``{condition: "utrack" | "trackmate"}`` -- the CHOICE made where a condition was
    tracked by both (the analysis door asks; `tracker_options` lists the options).
    Conditions missing from the mapping are detected as with ``"auto"``. A mapping may
    mix trackers across conditions: that is a stated choice, not a silent one, and the
    tool each condition was read from is kept on ``canonical.attrs["trackers"]``.
    ``"auto"`` still refuses a mix it was not told about.
    """
    root = Path(root)
    if frame_interval_from not in FRAME_INTERVAL_SOURCES:
        raise ProjectError(f"frame_interval_from must be one of {FRAME_INTERVAL_SOURCES}, "
                           f"got {frame_interval_from!r}")
    leave_out = {str(c): {str(m): str(r) for m, r in dict(ms).items()} for c, ms in (leave_out or {}).items()}
    outline_tolerance_nm = float(outline_tolerance_nm or 0.0)
    if not math.isfinite(outline_tolerance_nm) or outline_tolerance_nm < 0:
        raise ProjectError(
            f"the outline tolerance must be 0 nm or more, got {outline_tolerance_nm!r}")
    chosen = None
    if isinstance(source, dict):
        chosen = {str(k): str(v) for k, v in source.items()}
        wrong = {k: v for k, v in chosen.items() if v not in PROJECT_SOURCES}
        if wrong:
            raise ProjectError(
                f"tracker choice {wrong} is not one of {PROJECT_SOURCES}")
    in_force = design_in_force(root, design)
    available = discover_conditions(root, source="auto" if chosen is not None
                                    else source, design=in_force if in_force is not None else False)
    if conditions is not None:
        missing = [c for c in conditions if c not in available]
        if missing:
            raise ProjectError(
                f"conditions not found under {root}: {missing}. "
                f"Available: {', '.join(available)}"
            )
        available = {c: available[c] for c in conditions}

    # WHICH TRACKER, RESOLVED ONCE AND BEFORE THE CACHE KEY. `detect_project_source`
    # looks at the folder rather than guessing, and REFUSES a folder holding both
    # layouts: that is two datasets with one name, and they disagree by about 1 pp in
    # the fractions and 10-18 per cent in D. Resolving per condition rather than once
    # for the project would let a comparison mix trackers silently, which is the same
    # failure one level up.
    if chosen is not None:
        unknown = sorted(set(chosen) - set(available))
        if unknown:
            raise ProjectError(
                f"a tracker was chosen for conditions not under {root}: {unknown}")
        per_source = {name: chosen.get(name) or detect_project_source(path)
                      for name, path in available.items()}
    elif source not in ("auto",) + PROJECT_SOURCES:
        raise ProjectError(
            f"source={source!r} is not one of {('auto',) + PROJECT_SOURCES}")
    elif source == "auto":
        detected = {name: detect_project_source(path) for name, path in available.items()}
        distinct = sorted(set(detected.values()))
        if len(distinct) > 1:
            listed = ", ".join(f"{c}: {s}" for c, s in sorted(detected.items()))
            raise ProjectError(
                f"the conditions under {root} were not all tracked by the same tool "
                f"({listed}); the trackers differ by about 1 pp in the slow fraction and "
                f"10-18 per cent in D. Load them separately, or pass source= to force one.")
        per_source = dict.fromkeys(available, distinct[0])
    else:
        per_source = dict.fromkeys(available, source)
    # ONE TRACKER KEEPS ITS OLD CACHE KEY; a mix is keyed by every condition's choice,
    # so u-track for A and TrackMate for B never meets a cache entry made the other way.
    distinct_sources = sorted(set(per_source.values()))
    resolved_source = (distinct_sources[0] if len(distinct_sources) == 1 else
                       "per-condition:" + ",".join(f"{c}={s}" for c, s
                                                   in sorted(per_source.items())))

    # Canonical cache (perf ITEM 2): re-parsing every .mat and rebuilding the
    # canonical table on each launch is pure waste, measured ~12.6 s. A launch
    # whose input files and load options are unchanged reads the parsed project
    # back in <1 s. Keyed on the FILES, so an edit or a different mapping rebuilds.
    cache_file = None
    if use_canonical_cache:
        key = _canonical_cache_key(
            available, day_maps=day_maps,
            localization_precision_nm=localization_precision_nm,
            min_localizations=min_localizations, series_per_day=series_per_day,
            allow_unsegmented=allow_unsegmented,
            require_all_inside=require_all_inside, source=resolved_source,
            pixel_size_nm=pixel_size_nm, frame_interval_ms=frame_interval_ms,
            skip_unsegmented=skip_unsegmented, outline_tolerance_nm=outline_tolerance_nm,
            frame_interval_from=frame_interval_from, leave_out=leave_out, design=in_force)
        cache_file = Path(root) / ".smtracker_warm" / "canonical" / f"{key}.pkl"
        cached = _load_cached_project(cache_file, available, day_maps=day_maps)
        if cached is not None:
            _say_search_radii(cached.params)
            # a cached project is read on a path that skips the acquisition read, so a format.json that
            # disagrees with the calibration in use is said again here (the read is two small json files)
            for name, condition_root in available.items():
                if per_source[name] in ("trackmate", "table"):
                    _trackmate_acquisition_params(condition_root, pixel_size_nm=pixel_size_nm)
            return _with_factors(rescale_mixed_tau(cached, mode=mixed_tau, max_spread=_tau_spread(max_tau_spread)),
                                 root, in_force)

    # EVERY u-track CONDITION'S ACQUISITION FIRST, so a movie tracked without a pixel size or frame interval is
    # found in all of them before any tracks are read, and named together (`MissingAcquisitionError.by_condition`)
    from .utrack_meta import MissingAcquisitionError

    utrack_params, camera_used, missing_by_condition = {}, {}, {}
    camera_measured = {}
    for name, condition_root in available.items():
        if per_source[name] in ("trackmate", "table"):
            continue
        camera = (camera_frame_intervals(condition_root)
                  if frame_interval_from == "camera" and frame_interval_ms is None else {})
        camera_measured[name] = {m: v for m, v in camera.items() if m not in leave_out.get(name, {})}
    settled = _settle_camera(camera_measured)
    for name, condition_root in available.items():
        if per_source[name] in ("trackmate", "table"):
            continue
        camera = settled[name]
        try:
            utrack_params[name] = _utrack_acquisition_params(
                condition_root, localization_precision_nm=localization_precision_nm,
                pixel_size_nm=pixel_size_nm, frame_interval_ms=frame_interval_ms,
                camera=camera, leave_out=tuple(leave_out.get(name, {})))
        except MissingAcquisitionError as exc:
            missing_by_condition[name] = exc.missing
            continue
        camera_used[name] = camera
    if missing_by_condition:
        listed = "; ".join(f"{c}: " + ", ".join(f"{m} (no {f.rstrip('_')})" for m, f, _p in rows)
                           for c, rows in missing_by_condition.items())
        error = MissingAcquisitionError(
            f"tracked without a value that every D depends on: {listed}. Set it in u-track, or leave "
            "these movies out.", [row for rows in missing_by_condition.values() for row in rows])
        error.by_condition = missing_by_condition
        raise error

    raws, merged_meshes, per_params, sources = [], {}, {}, {}
    #: condition -> the movies read: a condition with one stated rate gets it per movie (`_check_params_agree`)
    read_movies: dict[str, list] = {}
    #: qualified movie -> tracks the TRACKER produced, before any filter here.
    #: Recorded while the raw table still exists, because nothing downstream can
    #: recover it: the canonical table is post-filter and has no row at all for a
    #: track the outline or length filter dropped. "kept / found" on the confirm
    #: screen is this number and `uid.nunique()`; without it the screen can only
    #: report half the pair.
    #: movies the tracker produced NO tracks for -> their condition. Recorded so the
    #: load report can name them; an empty movie contributes to no statistic, so a
    #: condition that lost several has a smaller effective n than its folder listing.
    empty_movies: dict[str, str] = {}
    #: the same, per condition and by the movie's own name: QC's "contributes nothing" reads it
    #: (`qc_view.contributes_nothing`), and the canonical table has no row to find them by.
    empty_by_condition: dict[str, list] = {}
    day_provenance = {}
    used_day_maps: dict[str, dict | None] = {}
    origin_shifts: dict = {}
    alignments: dict = {}
    for done, (name, condition_root) in enumerate(available.items(), start=1):
        if per_source[name] in ("trackmate", "table"):
            # NEITHER RECORDS ITS ACQUISITION: both read it from the preparation stage's
            # crops.json, or from an explicit pixel size (never guessed)
            per_params[name] = _trackmate_acquisition_params(
                condition_root, localization_precision_nm=localization_precision_nm,
                pixel_size_nm=pixel_size_nm,
            )
        else:
            per_params[name] = utrack_params[name]
        meshes = read_project_cell_meshes(
            condition_root, allow_unsegmented=allow_unsegmented
        )
        # A confirmed day mapping (if any) overrides the series-based default.
        day_arg = None
        if day_maps and name in day_maps and day_maps[name] is not None:
            mapping = day_maps[name]
            day_arg = getattr(mapping, "movie_to_day", mapping)     # DayMapping or dict
            if not getattr(mapping, "ok", True):
                raise ProjectError(
                    f"replicate mapping for {name!r} is not usable: "
                    f"{'; '.join(mapping.errors)}")
            if hasattr(mapping, "provenance_dict"):
                day_provenance[name] = mapping.provenance_dict()
        # RECORDED BEFORE THE LOAD, so it is the argument actually used rather than
        # anything reconstructed afterwards. See Project.day_maps for why the canonical
        # table cannot answer this.
        used_day_maps[name] = dict(day_arg) if day_arg else None
        if per_source[name] == "trackmate":
            raw = load_trackmate_project(
                condition_root, condition=name, day=day_arg,
                series_per_day=series_per_day, pixel_size_nm=pixel_size_nm,
            )
        elif per_source[name] == "table":
            raw = load_table_project(
                condition_root, condition=name, day=day_arg,
                series_per_day=series_per_day,
                pixel_size_nm=per_params[name].pixel_size_nm,
            )
        else:
            raw = load_utrack_project(
                condition_root, condition=name, day=day_arg,
                series_per_day=series_per_day,
                on_movie=(None if on_movie is None else
                          lambda movie, i, n, c=name: on_movie(c, movie, i, n)),
            )
        if leave_out.get(name):
            raw = raw.loc[~raw["movie"].astype(str).isin(set(leave_out[name]))].copy()
        if raw.empty:
            raise ProjectError(f"condition {name!r} produced no tracks")
        if per_source[name] == "trackmate":
            # TRACKMATE'S LINKING DISTANCE IS THE CONDITION'S SEARCH RADIUS (ruled 4 Nov 2026), read from its XMLs
            per_params[name] = _with_linking_radius(per_params[name], raw.attrs.get("linking_radius_px_by_movie"),
                                                    set(raw["movie"].astype(str)), name)
        # OUTLINES UNDER THE TRACKS' MOVIE NAMES (23 Oct 2026): ``001.mat`` for ``series001_YFPLaser`` when it is the
        # one outline with that series number (`cellmesh.meshes_by_movie_name`)
        meshes = meshes_by_movie_name(meshes, pd.unique(raw["movie"].astype(str)))
        # THE PIXEL ORIGIN (22 Oct 2026): outlines that record theirs are moved onto the tracker's, and the
        # alignment of tracks and outlines is MEASURED before any filter, for QC (`cellmesh.outline_alignment`)
        meshes, shifts = to_tracker_origin(meshes, per_source[name])
        if shifts:
            origin_shifts[name] = shifts
        aligned = outline_alignment(raw, meshes)
        if aligned is not None:
            alignments[name] = dict(aligned, tracker=per_source[name])
        # movies the tracker produced NOTHING for: carried per condition so the load
        # report can say which, rather than the count silently disagreeing with the
        # folder listing
        for movie in raw.attrs.get("empty_movies", []):
            empty_movies[qualified_movie(name, movie)] = name
            empty_by_condition.setdefault(name, []).append(str(movie))

        # Qualify the movie name. uid and cell are derived from it downstream,
        # so this is the only rename needed to make the merge unambiguous.
        originals = raw["movie"].unique()
        read_movies[name] = [str(m) for m in originals]
        raw = raw.copy()
        raw["movie"] = raw["movie"].map(lambda m: qualified_movie(name, m))
        if on_condition is not None:
            # NEVER FATAL. A window that raises in its progress handler -- because it
            # closed while the thread was mid-load, which is the ordinary case -- must
            # not take the load down with it.
            try:
                on_condition(name, done, len(available))
            except Exception:                          # noqa: BLE001 - progress only
                pass
        for original in originals:
            sources[qualified_movie(name, original)] = (condition_root, str(original))

        for movie, cells in meshes.items():
            merged_meshes[qualified_movie(name, movie)] = [
                replace(cell, movie=qualified_movie(name, movie)) for cell in cells
            ]
        raws.append(raw)

    pixel_mixed: dict = {}
    params = _check_params_agree(per_params, pixel_mixed, movies=read_movies)
    # EACH CONDITION'S OWN RECORD, before the merge keeps only the first condition's PSF: what the Confirm page shows
    # per condition (`onboarding_view.condition_acquisition`). DISPLAY ONLY: nothing computes from it (the search
    # radius the analysis reads is carried on ``params`` per condition where they differ, D10).
    acquisition_by_condition = {name: acquisition_record(p) for name, p in per_params.items()}
    raw = pd.concat(raws, ignore_index=True)
    raw.attrs["empty_movies"] = sorted(empty_movies)

    # A MOVIE WITH NO OUTLINES IS DROPPED BY NAME, NEVER SILENTLY. `assign_cells`
    # refuses to mix segmented and unsegmented movies and is right to: the cell filter
    # is what makes a track's position meaningful, so an unsegmented movie would
    # contribute tracks no filter ever saw. Dropping them is a DATA DECISION, so they
    # are named on the project rather than merely excluded, `spatialsqd` already made
    # the same choice for the same reason, and this is that option on the loader.
    dropped_unsegmented: list[str] = []
    if skip_unsegmented:
        have = set(merged_meshes)
        present = sorted(set(raw["movie"].astype(str)))
        dropped_unsegmented = [m for m in present
                               if m not in have and m.split(":")[-1] not in have]
        if dropped_unsegmented:
            raw = raw.loc[~raw["movie"].astype(str)
                          .isin(dropped_unsegmented)].copy()
            if raw.empty:
                raise ProjectError(
                    "every movie lacks cell outlines: "
                    f"{', '.join(dropped_unsegmented)}")

    # SIGMA FROM THE DATA WHERE THE TRACKER RECORDS IT. u-track fits every spot and
    # writes the fit's own coordinate uncertainty; the theoretical fallback is
    # `psf_sigma_px * pixel_size_nm`, which is the WIDTH OF THE SPOT rather than the
    # precision of its centre. Measured over 26 groups and 219 582 localisations the
    # two differ by 5.7x (17.3 nm against 98.8), and sigma decides whether a slow-state
    # D is a number or a bound, so the wrong quantity there is not cosmetic.
    #
    # Only when the caller supplied nothing: an explicit sigma always wins, and its own
    # provenance travels with it. A tracker that records no uncertainty (TrackMate)
    # raises, and the theoretical value stands and is stamped as theoretical, because a
    # silent fallback would put an optical estimate behind a "tracker_reported" label.
    # The loader's answer, whichever branch settles it, so clearing a user override
    # later returns to what the project actually opened with.
    loaded_sigma = ((params.localization_sigma_source,
                     params.localization_precision_nm)
                    if localization_precision_nm is not None
                    else ("theoretical", None))
    if localization_precision_nm is None:
        from .ingest import IngestError, reported_sigma

        try:
            measured = reported_sigma(raw, pixel_size_nm=params.pixel_size_nm)
        except IngestError:
            pass
        else:
            params = replace(params,
                             localization_precision_nm=measured.median_nm,
                             localization_sigma_source="tracker_reported")
            loaded_sigma = ("tracker_reported", measured.median_nm)
            per_params = {k: replace(v,
                                     localization_precision_nm=measured.median_nm,
                                     localization_sigma_source="tracker_reported")
                          for k, v in per_params.items()}

    def say(text: str) -> None:
        if on_stage is not None:
            try:
                on_stage(text)
            except Exception:                          # noqa: BLE001 - progress
                pass

    kwargs = {} if min_localizations is None else {"min_localizations": min_localizations}
    say("assigning tracks to cells")
    assigned = assign_cells(raw, merged_meshes,
                            require_all_inside=require_all_inside,
                            tolerance_px=outline_tolerance_nm / float(params.pixel_size_nm))
    say("building the canonical table")
    canonical = to_canonical(assigned, params, **kwargs)
    canonical.attrs["outline_tolerance_nm"] = outline_tolerance_nm
    canonical.attrs["acquisition_by_condition"] = acquisition_by_condition
    if pixel_mixed:
        # {condition: its recorded pixel size} when they differ within 1 % (`_check_params_agree`): QC says so
        canonical.attrs["pixel_size_mixed"] = dict(pixel_mixed)
    canonical.attrs["empty_movies_by_condition"] = {c: sorted(v) for c, v in empty_by_condition.items()}
    if dropped_unsegmented:
        canonical.attrs["dropped_unsegmented"] = dropped_unsegmented
    canonical.attrs["outline_alignment"] = alignments
    if origin_shifts:
        canonical.attrs["outline_origin_shift"] = origin_shifts
    if day_provenance:
        # Which day mapping produced this project, for provenance / export.
        canonical.attrs["day_mappings"] = day_provenance
    # WHICH TRACKER EACH CONDITION WAS READ FROM, always: a chosen mix must stay visible.
    canonical.attrs["trackers"] = dict(per_source)
    if leave_out:
        canonical.attrs["left_out_movies"] = {qualified_movie(c, m): r for c, ms in leave_out.items()
                                              for m, r in ms.items()}
    # VALUES A PERSON STATED where u-track's record was empty (`utrack_meta.stated_values`), each with its file
    stated = {}
    for name, condition_root in available.items():
        if per_source[name] in ("trackmate", "table"):
            continue
        from .dayfolders import namer
        from .utrack_meta import STATED_FILE, MetadataError, read_movie_metadata

        known_as = namer(condition_root)
        for record in sorted(Path(condition_root).glob(f"**/{STATED_FILE}")):
            for data in sorted(record.parent.glob("**/movieData.mat")):
                if "backups" in data.parts:
                    continue
                try:                                   # USED only where the record was empty: ask the reader
                    used = read_movie_metadata(data, frame_interval_ms=1.0).stated
                except MetadataError:
                    continue
                if used:
                    stated[qualified_movie(name, known_as(data.parent))] = used
    if stated:
        canonical.attrs["acquisition_stated"] = stated
    # WHERE EACH MOVIE'S FRAME INTERVAL CAME FROM, and what u-track had said, wherever the camera was used
    canonical.attrs["frame_interval_from"] = frame_interval_from
    camera_record, differing = {}, []
    for name, camera in camera_used.items():
        recorded = _recorded_intervals(available[name], camera)
        for movie, (ms, source, raw_file) in camera.items():
            was = recorded.get(movie)
            camera_record[qualified_movie(name, movie)] = {
                "camera_ms": ms, "measured_ms": camera_measured[name][movie][0], "recorded_ms": was, "source": source,
                "file": str(raw_file)}
            if was is None or abs(ms / was - 1.0) > 0.005:
                differing.append((name, movie, ms, was))
    if camera_record:
        canonical.attrs["frame_interval_camera"] = camera_record
    if differing:
        shown = ", ".join(f"{c} {m} {ms:.2f} ms" + (f" (u-track: {was:g})" if was else " (u-track: none)")
                          for c, m, ms, was in differing[:6])
        warnings.warn(
            f"frame interval from the camera for {len(differing)} movie(s) where u-track recorded another: "
            f"{shown}{' ...' if len(differing) > 6 else ''}. Every D depends on it; the fractions do not change.",
            RuntimeWarning, stacklevel=2)
    # ONE SOURCE OF TRUTH FOR THE PRE-FILTER COUNT, and it is a VIEW rather than a
    # second computation. This used to be counted here, per condition, off the raw table;
    # `assign_cells` counts the same tracks a few lines later because it must (it is the
    # only place that also knows what it DROPPED, and the dropped tracks leave no rows
    # behind for anyone to count afterwards). Two loops over one quantity is exactly the
    # duplication hard rule 5 forbids, and the two would drift the first time either
    # changed what "a track" means.
    #
    # THE SHAPE IS KEPT because `ui.onboarding.adapt` reads it: `{qualified movie: found}`.
    # Its own fallback -- a movie absent from this dict is scored `found = kept` -- stays
    # correct, since a movie the tracker produced nothing for reaches neither counter.
    by_movie = canonical.attrs.get("cell_filter_by_movie")
    if by_movie is None:
        # LOUD, because the alternative is the exact defect both bumps above were for: an
        # empty dict here sends every reader down its found = kept fallback and reports a
        # 100 per cent keep rate on a project that has not been measured.
        raise ProjectError(
            "the canonical table carries no per-movie cell-filter counts, so the "
            "pre-filter track count cannot be formed and a keep rate would read 100 per cent.")
    canonical.attrs["tracks_found"] = {m: int(v["found"]) for m, v in by_movie.items()}
    project = Project(
        canonical=canonical,
        meshes=merged_meshes,
        params=params,
        conditions=list(available),
        roots={name: path for name, path in available.items()},
        movie_source=sources,
        day_maps=used_day_maps,
        loaded_sigma=loaded_sigma,
    )
    if cache_file is not None:
        say("writing the project cache")
        _store_cached_project(cache_file, project, day_provenance)
    _say_search_radii(params)
    return _with_factors(rescale_mixed_tau(project, mode=mixed_tau, max_spread=_tau_spread(max_tau_spread)),
                         root, in_force)


def _with_factors(project: Project, root, design) -> Project:
    """``project`` carrying the design's factors for its conditions (none without a design). Set after the cache,
    never stored in it: renaming a factor changes no table."""
    if design is None:
        return project
    factors = _design_result(root, design).factors()
    return replace(project, factors={c: factors[c] for c in project.conditions if c in factors},
                   factor_names=tuple(design.factor_names))


def _tau_spread(value: float | None) -> float:
    """``max_tau_spread`` as given, or the standard; refused outside [0, 0.5]."""
    spread = MAX_TAU_RESCALE if value is None else float(value)
    if not 0.0 <= spread <= 0.5:
        raise ProjectError(f"max_tau_spread is a fraction from 0 to 0.5 (0.10 = 10 %), got {value}")
    return spread


#: `rescale_mixed_tau`'s modes.
MIXED_TAU_MODES = ("rescale", "refuse")
#: What the analysis app's doors pass (`app`, `open_gui`, the onboarding scan): ONE tau for
#: a small spread, as ruled (13 Oct 2026, "Rescale to one tau"). Scripts keep the loader's
#: default, "refuse", so no manuscript table moves without someone choosing it.
#: The largest spread of frame intervals (max / min - 1) that is put on one tau. Above it
#: the movies stay STRATA: at 20.3 vs 30 ms the detection ceiling differs by ~48 % and the
#: motion blur differs, which no rescaling of displacements can undo. 10 %, RULED 22 Oct 2026
#: (the experimenter: "lets change it to 10", and an option "if people wanna decrease or increase it",
#: `options.max_tau_spread`): protein B's MSD project holds 20.467-20.481 ms and, in Glu + Mal,
#: 21.741 ms -- a 6.2 % spread the 5 % of 13 Oct left unmerged, so QC refused the project.
MAX_TAU_RESCALE = 0.10
APP_MIXED_TAU = "rescale"


def rescale_mixed_tau(project: "Project", *, mode: str = "rescale",
                      max_spread: float = MAX_TAU_RESCALE) -> "Project":
    """The project on ONE frame interval, when its movies were recorded a few per cent apart.

    RULED BY THE EXPERIMENTER (13 Oct 2026). One user's protein I and protein I MM-fructose have days
    1-3 at 50.475 ms and day 4 at 51.74 ms per frame, protein I LB day 3 at 51.75 ms, and whole
    conditions sit at one or the other. Steps at different intervals are draws from
    different distributions even at the same D (tau is inside every likelihood), so the
    pooled and day-shared fits refused them, and every section reading one tau for the
    project refused the project. Of the four answers offered -- tau per day inside the
    fit, each day alone, drop the odd day, rescale -- the experimenter chose: "Rescale to
    one tau". It overrides the rule "never merge displacements across tau" for these
    small spreads, explicitly, and only for them.

    WHAT IS DONE. The reference is the interval carrying the most localisations. Every
    track recorded at another interval tau_i has its positions scaled about its own first
    localisation by sqrt(tau_ref / tau_i): every displacement, at every lag, is then
    multiplied by exactly that factor, every squared displacement by tau_ref / tau_i, so a
    molecule's D reads the same at either interval (E[r^2] = 4 D tau). Its times and tau
    become the reference's. Positions move by at most ~1.2 % of a track's own extent
    (2.5 % tau spread), a few nanometres: the cell assignment was made before, and the
    maps' bins are tens of times larger. What is NOT exact: the static localisation term
    (4 sigma^2) is scaled too, by the same ~2.5 %, a negligible shift here; and the
    detection ceiling and motion blur of the rescaled movies stay their own.

    RECORDED: ``canonical.attrs["tau_rescaled"]`` = ``{movie: {"frame_interval_ms",
    "rescaled_to_ms", "r2_factor"}}``, and a warning names the movies at load. A spread
    over ``max_spread``, a single interval, or ``mode="refuse"`` returns the project
    unchanged."""
    import numpy as np

    if mode not in MIXED_TAU_MODES:
        raise ValueError(f"mixed_tau must be one of {MIXED_TAU_MODES}, got {mode!r}")
    can = project.canonical
    if mode == "refuse" or "tau_s" not in can.columns or can.empty:
        return project
    tau = can["tau_s"].to_numpy(float)
    rates = np.unique(np.round(tau, 9))
    if len(rates) < 2:
        return project
    if rates.max() / rates.min() - 1.0 > max_spread:
        return project
    rounded = np.round(tau, 9)
    values, counts = np.unique(rounded, return_counts=True)
    ref = float(values[np.argmax(counts)])
    off = np.abs(rounded - ref) > 1e-12

    out = can.copy()
    out.attrs.update(can.attrs)
    factor = ref / tau[off]                                   # tau_ref / tau_i, per row
    scale = np.sqrt(factor)
    rows = out.loc[off, ["uid", "frame", "x_um", "y_um"]]
    first = rows.loc[rows.groupby("uid")["frame"].idxmin()].set_index("uid")
    x0 = rows["uid"].map(first["x_um"]).to_numpy(float)
    y0 = rows["uid"].map(first["y_um"]).to_numpy(float)
    out.loc[off, "x_um"] = x0 + (rows["x_um"].to_numpy(float) - x0) * scale
    out.loc[off, "y_um"] = y0 + (rows["y_um"].to_numpy(float) - y0) * scale
    if "t_s" in out.columns:
        out.loc[off, "t_s"] = out.loc[off, "t_s"].to_numpy(float) * factor
    out.loc[off, "tau_s"] = ref

    record = {}
    # the canonical table's movie names are already qualified ("<condition>_<movie>")
    moved = can.loc[off, ["movie", "tau_s"]].drop_duplicates("movie")
    for movie, tau_i in moved.itertuples(index=False):
        record[str(movie)] = {
            "frame_interval_ms": round(1000.0 * float(tau_i), 6),
            "rescaled_to_ms": round(1000.0 * ref, 6),
            "r2_factor": float(ref / float(tau_i))}
    out.attrs["tau_rescaled"] = record
    by_movie = project.params.frame_interval_ms_by_movie or {}
    params = replace(project.params, frame_interval_ms=1000.0 * ref,
                     frame_interval_ms_by_movie={m: 1000.0 * ref for m in by_movie} or None)
    # THE TABLE CARRIES ITS OWN COPY of the acquisition (`canonical.attrs["params"]`, what
    # QC and every reader of a bare table use): it gets the one tau too. Missed until
    # 14 Oct 2026, when QC on one user's project still saw the mixed intervals and refused.
    if "params" in out.attrs:
        out.attrs["params"] = params
    intervals = sorted({v["frame_interval_ms"] for v in record.values()})
    warnings.warn(
        f"frame intervals {', '.join(f'{v:g}' for v in intervals)} ms rescaled to "
        f"{1000.0 * ref:g} ms for {len(record)} movie(s) (squared displacements x "
        f"{min(v['r2_factor'] for v in record.values()):.4f}-"
        f"{max(v['r2_factor'] for v in record.values()):.4f}). One frame interval is used for "
        f"spreads up to {max_spread:.0%}.",
        RuntimeWarning, stacklevel=2)
    return replace(project, canonical=out, params=params, _projected={})


#: Where a movie's tracked length was read, in the order they are asked.
FRAME_SOURCES = ("tracking run", "movieData.mat", "crops.json")


def _recorded_ranges(path) -> dict:
    """``{movie: (first, last)}`` (1-based, inclusive) from a run record's ``crops`` (TrackMate's ``run.json``,
    u-track's ``utrack_run.json``); empty where the file is absent or unreadable, a bad entry skipped."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        crops = json.loads(path.read_text(encoding="utf-8")).get("crops") or {}
    except ValueError:
        return {}
    out = {}
    for movie, span in crops.items():
        try:
            lo, hi = int(span[0]), int(span[1])
        except (TypeError, ValueError, IndexError):
            continue
        if hi >= lo >= 1:
            out[str(movie)] = (lo, hi)
    return out


def movie_first_frames(project: "Project", condition: str) -> dict:
    """``{original movie: (first frame, source)}``: the acquisition frame each movie's TRACKED range starts on
    (1-based), from the record the tracker left -- TrackMate's ``trackmate/run.json``, u-track's
    ``tif/utrack_run.json`` -- then the preparation's ``crops.json`` (editable after a run, hence last). The
    induction clock needs it: a movie's time since an addition is its recording start (`rawmovie.acquisition_time`)
    plus the frames cut before the range (23 Oct 2026). A movie no record covers is ABSENT; nothing is invented."""
    root = Path(project.roots[condition])
    out: dict = {}
    for record, source in ((root / "trackmate" / "run.json", "tracking run"),
                           (root / "tif" / "utrack_run.json", "u-track run")):
        for movie, (lo, _hi) in _recorded_ranges(record).items():
            out.setdefault(movie, (lo, source))
    from .movieprep import MoviePrepError, find_crops, load_crops

    found = find_crops(root)
    if found is not None:
        try:
            crops = load_crops(found)["crops"]
        except MoviePrepError:
            crops = {}
        for movie, (lo, _hi) in crops.items():
            out.setdefault(str(movie), (int(lo), "crops.json"))
    return out


def movie_frames(project: "Project", condition: str) -> dict:
    """``{original movie: (frames, source)}``: how many frames each movie of
    ``condition`` was TRACKED over, read from the record the tracker left.

    WHY READ IT. The canonical table holds only the tracks that survived, so a movie's
    length read off it is the span from its first kept track to its last. On dense data
    that is 96-99 per cent of the movie (`qc.detection_density` measured it), and on
    SPARSE data it is not a length at all: on one user's d2.1 series001 one track spans 11 of
    the 1001 frames tracked, so a cell there would be credited 11 frames of watching and
    its density read ninety times too high. Sample size per unit TIME needs the time.

    The records, in order: the TrackMate run's own ``trackmate/run.json`` (the ranges
    actually tracked, written by `trackmate_run.record_run`), u-track's
    ``movieData.mat`` (``nFrames_``, the cropped stack it imported), then the
    preparation stage's ``crops.json`` (which can have been edited after a run, hence
    last). A movie no record covers is ABSENT from the result -- the caller decides
    what to do and says so; nothing is invented here.
    """
    root = Path(project.roots[condition])
    out: dict = {}
    for movie, (lo, hi) in _recorded_ranges(root / "trackmate" / "run.json").items():
        out[movie] = (hi - lo + 1, "tracking run")
    try:
        from .utrack_meta import MetadataError, read_project_metadata

        metas = read_project_metadata(root)
    except (MetadataError, OSError, ValueError):
        metas = []
    for meta in metas:
        if meta.n_frames and str(meta.movie) not in out:
            out[str(meta.movie)] = (int(meta.n_frames), "movieData.mat")
    from .movieprep import MoviePrepError, find_crops, load_crops

    found = find_crops(root)
    if found is not None:
        try:
            crops = load_crops(found)["crops"]
        except MoviePrepError:
            crops = {}
        for movie, (lo, hi) in crops.items():
            out.setdefault(str(movie), (hi - lo + 1, "crops.json"))
    return out
