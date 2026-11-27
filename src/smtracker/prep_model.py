"""The preparation stage's state and rules, with no Qt in them.

WHY THIS MODULE EXISTS. The image-preparation + tracking window (`tracking_gui` with its
`croppanel`) grew to about 6 700 lines of plain Qt, and much of its arithmetic -- which
ranges a common cut touches, what the overview rows count, which condition a result
belongs to -- lived inside the widget classes. It is being rebuilt in the workspace's
style (`smtracker.ui`), and two windows cannot each carry their own copy of a rule. So the
rules move HERE first, the old window is rewired to call them, and its tests -- which
already pin the recorded defects -- prove nothing changed. The new window then calls the
same functions. Text for the screen lives in `prep_view`; this module returns data.

INVARIANTS
- Imports no Qt, ever (`tests/test_headless_spine.py` runs it with Qt blocked).
- Pure where it can be: a function takes the ranges and lengths it needs and returns new
  ones; nothing here holds a window's state behind the caller's back.
- A range is ``(first, last)``, 1-based and inclusive, as in `crops.json`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

__all__ = ["CutReport", "ranges_with_read_movies", "common_cut", "saved_crops",
           "OutlineState", "outline_states", "ConditionCounts", "resolve_counts",
           "curves_from_cache", "refloor", "write_curves", "write_projections",
           "save_condition", "CALIBRATION_KEYS", "PrepError", "ConditionState",
           "open_condition", "open_session", "read_movie_curves", "file_curves",
           "propose", "propose_all", "set_crop", "apply_common_cut", "apply_common_edges", "auto_flat_tolerance", "apply_flat_tolerance", "AUTO_FLAT_LADDER", "AUTO_FLAT_SHARE", "save",
           "REQUIRED_CALIBRATION", "set_calibration", "copy_calibration",
           "fill_from_acquisition", "probe_acquisition", "loose_acquisitions",
           "seed_tracking", "set_tracking", "copy_tracking", "measurable",
           "record_noise", "rethreshold", "thresholds_for", "tracking_gaps", "summarise_noise",
           "TRACKS_FOLDER", "SECONDS_PER_MOVIE", "MAX_AT_ONCE", "default_at_once", "minutes_at_once",
           "track_files", "RunInputs", "run_inputs",
           "track_condition", "STACKS_FOLDER", "stack_files", "export_inputs",
           "export_movie", "RADIUS_CHECK_MOVIES", "RADIUS_CHECK_MULTIPLE", "radius_check",
           "pilot_steps", "camera_pixel_size", "TAIL_BLOCK_FRAMES", "tail_purity",
           "UTRACK_SECONDS_PER_FRAME", "set_utrack", "utrack_psf_preview",
           "utrack_earlier", "UTrackInputs", "utrack_inputs", "utrack_background_masks",
           "track_condition_utrack", "OWN_INTERVAL_ABOVE", "movie_intervals",
           "NEXT_STEPS", "next_step"]


@dataclass(frozen=True)
class CutReport:
    """What a common start or end frame did to one condition's ranges.

    ``crops`` is the full new set (untouched ranges included), ``cut`` the movies whose
    range was set, ``shorter`` the movies whose END landed at their own last frame because
    they are shorter than the common end (or than their start plus ``length``), and
    ``skipped`` ``(movie, first, last)`` for every movie left alone because the cut would
    have emptied its range. ``proposed``: read movies that had no range yet and were given
    the start rule's own before the cut (`apply_common_cut`); ``unfloored``: those the rule
    found no flat stretch in, which start at frame 1 and are named.
    """

    crops: dict
    start: int | None = None
    end: int | None = None
    cut: tuple = ()
    shorter: tuple = ()
    skipped: tuple = field(default_factory=tuple)
    #: "Track N frames": every movie ends ``length`` frames after its own start.
    length: int | None = None
    proposed: tuple = ()
    unfloored: tuple = ()


def ranges_with_read_movies(crops: dict, n_frames: dict) -> dict:
    """Every movie's range, INCLUDING one read but never given a range (``1..n``).

    "For all movies" means the movies on the list, not only the ones somebody has already
    touched: a movie read but not yet proposed for would otherwise keep its whole length
    through a press that says it cut everything. ``n_frames`` maps a movie to its length,
    for the movies that have been read.
    """
    ranges = {stem: (int(a), int(b)) for stem, (a, b) in dict(crops).items()}
    for stem, n in dict(n_frames).items():
        if stem not in ranges and n:
            ranges[stem] = (1, int(n))
    return ranges


def common_cut(crops: dict, n_frames: dict, *, start: int | None = None,
               end: int | None = None, length: int | None = None) -> CutReport:
    """One common START or END frame for every movie of a condition, or one LENGTH from
    each movie's own start; the other edge kept.

    Asked for as "set frame X where all tails are cut from all movies", then the same for
    the start, then (23 Oct 2026) "track the first x frames after the cut range begins":
    ``length`` frames from each start, so a movie starting at 400 ends at 1399 for 1000.
    An end never runs past what a movie holds: its length when it has been read, else the
    end it already had (which was set inside it). A range the cut would empty is left alone
    and named, never inverted. Exactly one of ``start``, ``end`` and ``length``.
    """
    if sum(edge is not None for edge in (start, end, length)) != 1:
        raise ValueError("common_cut takes exactly one of start=, end= and length=")
    if length is not None and int(length) < 2:
        raise ValueError(f"a movie needs at least 2 frames to track, got length={length}")
    # A SKIPPED MOVIE KEEPS EXACTLY WHAT IT HAD, so an unranged one stays unranged: only
    # the movies the cut actually sets are written into the result.
    new = {stem: (int(a), int(b)) for stem, (a, b) in dict(crops).items()}
    cut, shorter, skipped = [], [], []
    for stem, (first, last) in ranges_with_read_movies(crops, n_frames).items():
        wanted = None
        if end is not None or length is not None:
            wanted = int(end) if end is not None else first + int(length) - 1
            n = n_frames.get(stem)
            target = (first, min(wanted, int(n)) if n else min(wanted, last))
        else:
            target = (int(start), last)
        if target[1] <= target[0]:
            skipped.append((stem, first, last))
            continue
        if wanted is not None and target[1] < wanted:
            shorter.append(stem)
        new[stem] = target
        cut.append(stem)
    return CutReport(crops=new, start=start, end=end, cut=tuple(cut),
                     shorter=tuple(shorter), skipped=tuple(skipped), length=length)


def saved_crops(folder) -> dict:
    """A condition's ranges as recorded in its `crops.json`, or ``{}`` if none is saved.

    For a condition never opened in this session: its ranges exist only on disk, and a
    common cut has to reach them too. A file that cannot be read counts as no ranges, the
    same answer the window gave before.
    """
    from .movieprep import MoviePrepError, find_crops, load_crops
    from .organise import OrganiseError, acquisition_folder

    try:
        acq, _why = acquisition_folder(Path(folder))
        saved = find_crops(acq, Path(folder))
        return dict(load_crops(saved)["crops"]) if saved else {}
    except (MoviePrepError, OrganiseError, OSError):
        return {}


# ---------------------------------------------------------------- the condition overview
@dataclass(frozen=True)
class OutlineState:
    """Outlines for one condition, from `cellmesh.outline_coverage` (a listing only)."""

    n_movies: int
    n_with_outlines: int
    has_folder: bool

    @property
    def complete(self) -> bool:
        return (self.has_folder and self.n_movies > 0
                and self.n_with_outlines >= self.n_movies)


def outline_states(folders: dict, groups: dict) -> dict:
    """``{condition: OutlineState}``. Reads directories, never movies or outlines.

    ``groups`` is the ONE listing (`movieprep.discover_across_conditions`) the overview
    shares: a second discovery is a second opinion about what a condition contains. A
    condition whose listing fails is left out, as before -- the offer is advisory.
    """
    from .cellmesh import outline_coverage
    from .rawmovie import movie_name

    out = {}
    for name, folder in dict(folders).items():
        try:
            movies = [movie_name(m) for m in groups.get(name, ())]
            cover = outline_coverage(folder, movies)
        except Exception:                                 # noqa: BLE001 - advisory
            continue
        out[name] = OutlineState(n_movies=len(cover.movies),
                                 n_with_outlines=int(cover.n_with_outlines),
                                 has_folder=cover.folder is not None)
    return out


@dataclass(frozen=True)
class ConditionCounts:
    """What one overview row counts: movies, curves read, ranges, movies with no floor."""

    n_movies: int
    n_curves: int
    n_crops: int
    n_bad: int = 0

    @property
    def done(self) -> int:
        """The bar's value. A movie with no curve is FINISHED WITH, not pending: a sparse
        movie whose crop floor never formed is not a broken one."""
        return min(self.n_curves + self.n_bad, self.n_movies)


def resolve_counts(row, *, live=None, remembered=None,
                   session_crops=None) -> ConditionCounts:
    """The counts one row shows: from disk (``row``, a `movieprep.ConditionProgress`),
    overridden by what the session knows.

    ``live`` is the panel's ``(movies, curves, crops, errors)`` for the condition ON
    SCREEN -- curves and crops reach disk only when saved, so the folder being worked on
    would otherwise sit at zero. It is CLAMPED: a stale pass once read "13/10".
    ``remembered`` is ``(movies, curves[, errors])`` for a condition read earlier this
    session, resolved before the bar is drawn so the note and the bar cannot disagree.
    """
    n_movies, n_curves, n_crops, n_bad = row.n_movies, row.n_curves, row.n_crops, 0
    if live is not None:
        if live[0]:
            n_movies = live[0]
            n_curves, n_crops = min(live[1], n_movies), min(live[2], n_movies)
            n_bad = live[3] if len(live) > 3 else 0
    elif remembered is not None:
        n_movies, n_curves = remembered[0], remembered[1]
        n_bad = remembered[2] if len(remembered) > 2 else 0
        n_crops = len(session_crops or {})
    return ConditionCounts(int(n_movies), int(n_curves), int(n_crops), int(n_bad))


# ---------------------------------------------------------------- curves: cache and floor
def curves_from_cache(movies, bleach_dir, *, flat_tolerance: float | None = None):
    """Rebuild curves from `bleaching/*.csv`. Returns ``(loaded, missing)``.

    ``loaded`` is ``{movie: MovieCurves}``; ``missing`` is ``[(movie, [what is absent])]``.
    ONE SOURCE OF TRUTH for the cache: the panel loading the folder on screen and the walk
    loading a condition nobody is looking at both read it, and a second copy would let one
    re-read a condition the other would load instantly -- 7.7 s a movie for nothing.
    Curves saved before the movie name existed sit under the FILE's stem and are found.
    """
    import pandas as pd

    from .movieprep import FLAT_TOLERANCE, MovieCurves, crop_floor
    from .rawmovie import movie_name

    tolerance = FLAT_TOLERANCE if flat_tolerance is None else float(flat_tolerance)
    bleach_dir = Path(bleach_dir)
    loaded, missing = {}, []
    for path in movies:
        path = Path(path)
        stem = movie_name(path)
        files = {label: bleach_dir / f"{stem}_{label}.csv" for label in ("full", "density")}
        if stem != path.stem and not files["full"].is_file():
            files = {label: bleach_dir / f"{path.stem}_{label}.csv"
                     for label in ("full", "density")}
        absent = [k for k, f in files.items() if not f.is_file()]
        if absent:
            missing.append((stem, absent))
            continue
        try:
            full = pd.read_csv(files["full"])
            density = pd.read_csv(files["density"])
            # THE FLOOR IS RECOMPUTED FROM THE SAVED DENSITY, not re-measured from the
            # movie: `crop_floor` takes the density it would otherwise compute, which is
            # the expensive half.
            floor = crop_floor(None, density=density,
                               means=full["mean_counts"].to_numpy(float),
                               flat_tolerance=tolerance)
        except Exception as exc:                       # noqa: BLE001 - named, then read
            missing.append((stem, [f"unreadable ({type(exc).__name__})"]))
            continue
        loaded[stem] = MovieCurves(movie=stem, n_frames=int(full["frame"].max()),
                                   bleaching=full, density=density, floor=floor)
    return loaded, missing


def refloor(curves, flat_tolerance: float):
    """``curves`` with its crop floor at ``flat_tolerance``; the same object if it already
    is, or if the floor cannot be rebuilt (an absent floor would read as "no defensible
    start", which is a different claim). Reads no movie: two arrays per call."""
    from .movieprep import MovieCurves, crop_floor

    held = getattr(getattr(curves, "floor", None), "flat_tolerance", None)
    if held is None or abs(held - float(flat_tolerance)) < 1e-12:
        return curves
    try:
        floor = crop_floor(None, density=curves.density,
                           means=curves.bleaching["mean_counts"].to_numpy(float),
                           flat_tolerance=float(flat_tolerance))
    except Exception:                                  # noqa: BLE001 - keep what it has
        return curves
    return MovieCurves(movie=curves.movie, n_frames=curves.n_frames,
                       bleaching=curves.bleaching, density=curves.density, floor=floor)


#: The tolerances `auto_flat_tolerance` tries, loosest first: 2 % is the Nikon's standard,
#: the rest the steps measured on one user's MetaMorph movies (`movieprep.OFFSET_WARN_SHARE`).
AUTO_FLAT_LADDER = (0.02, 0.01, 0.0075, 0.005, 0.0025)
#: The flat band's share of the movies' SIGNAL `auto_flat_tolerance` allows. Set 23 Oct 2026 against the
#: experimenter's eye ("for these movies i need 0.5% strictest to have nearly no altitude in the curve"):
#: on one user's MetaMorph conditions 15 % gives 0.5 % for the brighter and 0.25 % for the dimmer,
#: on the Nikon 2 % (band 5 %); the 25 % above which the band "reads camera offset" (`OFFSET_WARN_SHARE`) left
#: the brighter conditions at 1 %.
AUTO_FLAT_SHARE = 0.15


def auto_flat_tolerance(curves_list, *, ladder=AUTO_FLAT_LADDER, target_share: float | None = None) -> tuple:
    """``(tolerance or None, words)``: the loosest tolerance at which a condition's flat band
    is at most ``target_share`` (`AUTO_FLAT_SHARE`) of its movies' SIGNAL (median over movies).

    "Flat within x %" is a share of the RAW level, camera offset included. Measured 23 Oct
    2026: on the Nikon (dark ~507 under ~300 counts) 2 % is 5 % of the signal; on one user's
    MetaMorph movies (dark ~2062 under 31-135 counts) it is 32-132 %, so any stretch reads
    flat and the start lands at frame 50-130 -- he set 0.5 % by eye ("nearly no altitude in
    the curve"), where the band is 8-33 %. Until now the band's share was only WARNED about
    above 25 % (`movieprep.OFFSET_WARN_SHARE`); here a share chooses. Needs the dark
    frames before the laser (`movieprep.leading_dark_level`): a condition without them is
    left at the box's value, said so, since its offset cannot be read.
    """
    from statistics import median

    from .movieprep import leading_dark_level

    target = AUTO_FLAT_SHARE if target_share is None else float(target_share)
    held = [c for c in curves_list if c is not None and getattr(c, "bleaching", None) is not None]
    if not held:
        return None, "no movie read yet"
    dark = [leading_dark_level(c.bleaching["mean_counts"].to_numpy(float)) for c in held]
    if not any(d is not None for d in dark):
        return None, ("no dark frames before the laser in these movies, so the camera offset "
                      "cannot be read and the box's value stands")
    last = None
    for tol in ladder:
        shares = [getattr(refloor(c, tol).floor, "flat_band_share", None) for c in held]
        shares = [s for s in shares if s is not None]
        if not shares:
            continue
        last = (tol, median(shares), len(shares))
        if last[1] <= target:
            return tol, (f"set to {100 * tol:g} %: the flat band is {100 * last[1]:.0f} % of these "
                         f"movies' signal (median of {last[2]}), within the {100 * target:.0f} % allowed")
    if last is None:
        return None, "no flat stretch at any tolerance tried, so the box's value stands"
    return last[0], (f"set to {100 * last[0]:g} %, the strictest tried: the flat band is still "
                     f"{100 * last[1]:.0f} % of these movies' signal")


def apply_flat_tolerance(state: ConditionState, tolerance: float, *, typed: bool = False,
                         note: str = "") -> list:
    """Read one condition's floors at ``tolerance`` and re-propose the movies whose range is
    still exactly what a proposal gave (an edited range is kept). Returns those re-proposed."""
    state.flat_tolerance = float(tolerance)
    state.flat_typed = state.flat_typed or typed
    state.flat_note = note
    for stem, curves in list(state.curves.items()):
        state.curves[stem] = refloor(curves, state.flat_tolerance)
    again = []
    for stem, got in list(state.proposals.items()):
        if state.crops.get(stem) == (int(got["start"]), int(got["end"])) and got.get("start_rule"):
            try:
                propose(state, stem, got["start_rule"])
                again.append(stem)
            except PrepError:
                pass
    return again


# ---------------------------------------------------------------- saving a condition
def write_curves(curves: dict, crops: dict, out_root, frame_interval_s, *,
                 on_progress=None) -> str:
    """The full, cropped and density curves as CSV, plus the picture, per movie with a
    range. Returns the note.

    THE DENSITY GOES OUT TOO: without it a reopened folder can redraw the bleaching panel
    and not the spot-count one, and `crop_floor` cannot be rebuilt, so the folder would
    have to be read again. A failure here must not lose `crops.json`, so each movie is
    attempted on its own and the misses are reported, not raised.
    """
    bleach_dir = Path(out_root) / "bleaching"
    written, failed = [], []
    for stem, movie_curves in curves.items():
        crop = crops.get(stem)
        if crop is None:
            continue
        try:
            bleach_dir.mkdir(parents=True, exist_ok=True)
            full = movie_curves.bleaching
            cropped = full[(full["frame"] >= crop[0]) & (full["frame"] <= crop[1])]
            for label, table in (("full", full), ("cropped", cropped),
                                 ("density", movie_curves.density)):
                table.to_csv(bleach_dir / f"{stem}_{label}.csv", index=False)
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.figure import Figure

            from .plots import draw_bleaching_pair

            figure = Figure(figsize=(7.0, 4.4), dpi=130, tight_layout=True)
            draw_bleaching_pair(figure, full, cropped, movie=stem, crop=crop,
                                frame_interval_s=frame_interval_s)
            figure.savefig(bleach_dir / f"{stem}.png")
            written.append(stem)
        except Exception as exc:                    # noqa: BLE001 - named, not raised
            failed.append(f"{stem} ({type(exc).__name__})")
        if on_progress is not None:
            on_progress(len(written) + len(failed), len(curves), "bleaching curves")
    note = f"curves written for {len(written)} movie(s)"
    if failed:
        note += f"; no curve for {len(failed)}: " + ", ".join(failed[:3])
    if on_progress is not None:
        on_progress(len(curves), len(curves), note)
    return note


def write_projections(movies, crops: dict, out_root, kind: str, *,
                      on_progress=None) -> str:
    """The `sum/` image per movie, from the frames inside each crop. Returns the note.

    The slow half of saving: it reads frames. The slice is `prepare_movie`'s (1-based
    inclusive) and the stretch is the macro's own `saturated`, so this image and the one
    written with the stacks are the same image. ANOTHER PROGRAM'S IMAGE IS KEPT
    (`movieprep.foreign_file`) -- the lab macro's projection, or the microscope's own.
    """
    import tifffile

    from .movieprep import (LEGACY_SATURATED, SOFTWARE_TAG, display_stretch,
                            foreign_file, projection)
    from .optional_deps import describe_missing, missing_for_movies
    from .rawmovie import movie_name, read_raw_movie

    sum_dir = Path(out_root) / "sum"
    by_name = {movie_name(p): Path(p) for p in movies}
    written, failed, kept = [], [], []
    for stem, crop in sorted(crops.items()):
        path = by_name.get(stem)
        if path is None:
            continue
        try:
            movie = read_raw_movie(path)
            block = movie._frames[crop[0] - 1:crop[1]]
            projected = projection(block, kind=kind)
            stretched, _lo, _hi = display_stretch(projected,
                                                  saturated=LEGACY_SATURATED["sum"])
            target = sum_dir / f"{stem}.tif"
            writer = foreign_file(target)
            if writer is not None:
                kept.append(f"{stem} ({writer})")
            else:
                # THE DIRECTORY IS MADE WHEN THERE IS SOMETHING TO PUT IN IT. Creating it
                # up front left an EMPTY `sum/` behind a Save that wrote nothing, and
                # `channels.resolve_channels` reads a folder whose NAME matches as a
                # resolved channel -- so it reported a successful `sum` with zero fields
                # and the classical segmenter, which reads `sum`, silently found nothing.
                sum_dir.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(target, stretched, software=SOFTWARE_TAG)
                written.append(stem)
        except Exception as exc:                    # noqa: BLE001 - named, not raised
            # THE MESSAGE, not just the type: "series001 (TypeError)" cost a round trip
            # to turn back into "saturated is a required argument".
            failed.append(f"{stem}: {type(exc).__name__}: {exc}")
        if on_progress is not None:
            on_progress(len(written) + len(failed), len(crops), f"{kind} projection")
    # NOTHING WRITTEN, FOR A REASON AN INSTALL FIXES, IS A REFUSAL -- not a note.
    # Measured on D:/data/project: 13 of 13 movies raised `RawMovieError: reading
    # .nd2 needs the nd2 package`, each one caught and appended here, and the note read
    # "0 sd projection(s) written" beside an empty `sum/`. Rule 3 says fail loud, and
    # this is the shape it was written for: every movie failing the SAME way is one
    # fact about the INSTALL, not many facts about the movies.
    if failed and not written and not kept:
        missing = missing_for_movies(by_name.values())
        if missing:
            raise PrepError(
                f"no {kind} projection could be written: " + describe_missing(missing))
    note = f"{len(written)} {kind} projection(s) written"
    if kept:
        note += (f"; {len(kept)} kept as they were, written by another program: "
                 + ", ".join(kept[:3]))
    if failed:
        note += f"; none for {len(failed)}: " + ", ".join(failed[:3])
    if on_progress is not None:
        on_progress(len(crops), len(crops), note)
    return note


def save_condition(out_root, crops: dict, *, pixel_size_nm, frame_interval_s,
                   curves: dict | None = None, movies=(), label=None, emission_nm=None,
                   numerical_aperture=None, projection: str | None = None,
                   on_progress=None) -> Path:
    """Write a condition's `crops.json`, and its curves and projections beside it.

    `crops.json` is the single record of a crop and goes first, so a failing curve or
    projection cannot cost it. ``curves`` writes the bleaching record (``None`` skips it),
    ``projection`` the `sum/` images (``None`` skips them). Returns the path written.
    """
    from .movieprep import CROPS_FILENAME, save_crops

    path = Path(out_root) / "bleaching" / CROPS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    save_crops(path, crops, pixel_size_nm=pixel_size_nm,
               frame_interval_s=frame_interval_s, label=label,
               emission_nm=emission_nm, numerical_aperture=numerical_aperture)
    if curves is not None:
        write_curves(curves, crops, out_root, frame_interval_s, on_progress=on_progress)
    if projection is not None:
        write_projections(movies, crops, out_root, projection, on_progress=on_progress)
    return path


# ---------------------------------------------------------------- the session
#: The calibration keys a condition carries, in `movieprep.CALIBRATION_FIELDS`' order.
CALIBRATION_KEYS = ("pixel_size_nm", "frame_interval_s", "label", "emission_nm",
                    "numerical_aperture")


class PrepError(RuntimeError):
    """A preparation request the session refuses, by name."""


@dataclass
class ConditionState:
    """One condition of a preparation session: its movies, curves, ranges, calibration.

    THE SESSION IS THE RECORD UNTIL SAVE. Ranges and calibration live here as they are
    edited; `crops.json` is written only by `save` (the single record of what is tracked),
    and ``saved`` remembers what it last held so an unsaved edit can be named.
    """

    name: str
    folder: Path
    acquisitions: Path | None
    note: str
    movies: list
    crops: dict = field(default_factory=dict)
    saved: dict = field(default_factory=dict)
    #: The calibration as `crops.json` last held it, so an edit can be named unsaved.
    saved_calibration: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    curves: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    proposals: dict = field(default_factory=dict)
    #: The emission was filled from a fluorophore's catalogue peak (not typed, not read from
    #: the acquisition), so choosing another fluorophore replaces it (`set_calibration`).
    emission_from_label: bool = False
    #: The flat tolerance THIS condition's floors are read at (a fraction; None = the
    #: session's box), chosen by `auto_flat_tolerance` once its movies are read unless
    #: somebody typed one (``flat_typed``); ``flat_note`` says why (23 Oct 2026: "for
    #: these movies i need 0.5% strictest", the conditions of a session differ).
    flat_tolerance: float | None = None
    flat_typed: bool = False
    flat_note: str = ""
    #: The common cuts pressed this session, ``{"start": f, "end": f}`` or ``{"length": n}``
    #: (an end and a length replace each other): what a movie read LATER is given too
    #: (`apply_common_edges`), as the panel's note says.
    common: dict = field(default_factory=dict)
    #: Detection and linking: ``{threshold, radius_px, linking_max_distance_px,
    #: max_frame_gap}`` (`trackmate_run.TRACKING_FIELDS`). Empty until `seed_tracking`.
    tracking: dict = field(default_factory=dict)
    #: Which tracking fields somebody TYPED, so a derived value never replaces them.
    typed: set = field(default_factory=set)
    #: The last noise measurement (`trackmate_run.summarise_thresholds`) or None.
    noise: dict | None = None
    #: Which movies have cell outlines (`OutlineState`), counted when the folder opened.
    outlines: "OutlineState | None" = None
    #: Track files an EARLIER run left in ``trackmate/`` (counted when the folder opened,
    #: and after every run), so a second run cannot silently land on top of the first.
    earlier_run: int = 0
    #: Stacks already in ``tif/`` (counted when the folder opened, and after an export):
    #: two crops in one folder cannot be told apart afterwards.
    stacks: int = 0
    #: What the acquisition says about the camera (``camera_pixel_um``,
    #: ``objective_magnification``), for the pixel-size helper; never the pixel size.
    camera: dict = field(default_factory=dict)
    #: The folder's layout (`organise.folder_layout`: organised, loose files,
    #: half-organised, nothing found), listed when the folder opened; None before.
    layout: object = None

    #: The movie names, in `movies`' order, worked out ONCE (`rawmovie.movie_names`):
    #: they are asked for on every redraw, and naming a movie lists its folder.
    movie_names: list = field(default_factory=list)

    @property
    def names(self) -> list:
        if len(self.movie_names) != len(self.movies):
            from .rawmovie import movie_name

            self.movie_names = [movie_name(p) for p in self.movies]
        return self.movie_names

    def path_of(self, stem: str):
        return dict(zip(self.names, self.movies)).get(stem)

    def n_frames(self) -> dict:
        return {stem: int(c.n_frames) for stem, c in self.curves.items()
                if getattr(c, "n_frames", None)}

    def counts(self) -> ConditionCounts:
        names = set(self.names)
        return ConditionCounts(n_movies=len(names),
                               n_curves=len(set(self.curves) & names),
                               n_crops=len(set(self.crops) & names),
                               n_bad=len(set(self.errors) & names))

    @property
    def unsaved(self) -> bool:
        """Ranges or calibration edited since the last save."""
        return (dict(self.crops) != dict(self.saved)
                or (bool(self.saved) and dict(self.calibration) != dict(self.saved_calibration)))

    @property
    def unread(self) -> list:
        """Movies with neither curves nor an error: still to be read."""
        return [p for p, stem in zip(self.movies, self.names)
                if stem not in self.curves and stem not in self.errors]

    def unread_named(self) -> list:
        """``[(path, movie name)]`` for the movies still to be read."""
        return [(p, stem) for p, stem in zip(self.movies, self.names)
                if stem not in self.curves and stem not in self.errors]


def open_condition(name: str, folder) -> ConditionState:
    """One condition, from directory listings and its `crops.json`. Opens no movie.

    The acquisitions are found where `organise.acquisition_folder` says (``raw/`` once
    filed, the folder itself before), and listed by `rawmovie.discover_raw_movies`, which
    leaves the phase snapshot beside an acquisition out. A saved crop set is read under
    MOVIE names (`load_crops` does that), and its calibration seeds the condition's.
    """
    from .movieprep import MoviePrepError, find_crops, load_crops
    from .organise import OrganiseError, acquisition_folder
    from .rawmovie import RawMovieError, discover_raw_movies

    folder = Path(folder)
    try:
        acq, note = acquisition_folder(folder)
    except OrganiseError as exc:
        acq, note = None, str(exc)
    movies = []
    if acq is not None and acq.is_dir():
        try:
            movies, _companions = discover_raw_movies(acq)
        except RawMovieError as exc:
            note = f"{note}; {exc}"
    calibration = {key: None for key in CALIBRATION_KEYS}
    crops = {}
    try:
        found = find_crops(acq, folder) if acq is not None else find_crops(folder)
        if found is not None:
            payload = load_crops(found)
            crops = {k: (int(a), int(b)) for k, (a, b) in payload["crops"].items()}
            for key in CALIBRATION_KEYS:
                if payload.get(key) is not None:
                    calibration[key] = payload[key]
    except MoviePrepError as exc:
        note = f"{note}; the saved crop set could not be read: {exc}"
    # NAMED ONCE, HERE, on the thread that opens the folder: naming a movie lists its
    # folder, and the window asks for the names on every redraw.
    from .rawmovie import movie_names as _movie_names

    names = _movie_names([Path(p).stem for p in movies], folder) if movies else {}
    named = [names.get(Path(p).stem, Path(p).stem) for p in movies]
    state = ConditionState(name=name, folder=folder, acquisitions=acq, note=note,
                           movies=list(movies), crops=dict(crops), saved=dict(crops),
                           calibration=calibration, saved_calibration=dict(calibration),
                           movie_names=named)
    state.outlines = outline_states({name: folder}, {name: list(movies)}).get(name)
    state.earlier_run = len(track_files(folder))
    state.stacks = len(stack_files(folder))
    from .organise import folder_layout

    # LISTINGS ONLY, on the thread that opens the folder: which of the four layouts the
    # folder is in, so a half-filed condition is said before anything is read from it.
    state.layout = folder_layout(folder)
    return state


def open_session(folder) -> dict:
    """``{condition: ConditionState}`` for a folder holding conditions, or ONE condition.

    `organise.discover_project_folders` decides which: a folder that is itself a project
    returns only itself, a parent returns its condition folders. The folder's own name
    labels a single condition, as the layout rules say (the folders above carry no
    meaning). Raises `PrepError` naming the folder when nothing in it can be prepared.
    """
    from .organise import OrganiseError, discover_project_folders

    folder = Path(folder)
    # CONDITION FOLDERS WIN OVER LOOSE FILES IN THE PARENT. One user's d2 folder holds its
    # three conditions AND two stray acquisitions of its own; "a parent that is itself a
    # project returns only itself" then opened those two as the whole session and hid
    # the three real conditions. So children are asked for first, and only a folder with
    # no condition below it is taken as the condition itself.
    try:
        found = discover_project_folders(folder, include_self=False)
    except OrganiseError:
        found = []
    if not found:
        try:
            found = discover_project_folders(folder)
        except OrganiseError as exc:
            raise PrepError(str(exc)) from exc
    return {state.folder.name: open_condition(state.folder.name, state.folder)
            for state in found}


def loose_acquisitions(folder) -> list:
    """Raw acquisitions sitting directly in ``folder`` when it ALSO holds conditions --
    named so the session can say they were not opened."""
    from .rawmovie import ACQUISITION_SUFFIXES

    try:
        return sorted(p.name for p in Path(folder).iterdir()
                      if p.is_file() and p.suffix.lower() in ACQUISITION_SUFFIXES)
    except OSError:
        return []


def read_movie_curves(path, flat_tolerance: float | None = None):
    """One movie's curves at the tolerance given. MODULE-LEVEL so a process pool can
    run it (`smtracker.ui` may not define a pool callable): about 2-8 s a movie, all of
    it reading frames."""
    from .movieprep import movie_curves
    from .rawmovie import movie_name, read_raw_movie

    curves = movie_curves(read_raw_movie(path), name=movie_name(path))
    return curves if flat_tolerance is None else refloor(curves, flat_tolerance)


def file_curves(state: ConditionState, stem: str, curves, error: str = "", *,
                propose_rule: str | None = None, cut: dict | None = None) -> None:
    """File one movie's read under its condition, and propose its range if asked.

    A movie that already has a range keeps it: a proposal never overwrites what is
    there, so a range somebody set is not replaced by a movie arriving late.
    """
    if error:
        state.errors[stem] = error
        state.curves.pop(stem, None)
        return
    state.errors.pop(stem, None)
    state.curves[stem] = curves
    if propose_rule and stem not in state.crops:
        try:
            propose(state, stem, propose_rule, **(cut or {}))
        except PrepError:
            pass
        # a common cut pressed before this movie was read applies to it too
        apply_common_edges(state, stem)


def propose(state: ConditionState, stem: str, start_rule: str, **cut) -> dict:
    """`movieprep.propose_crop` for one read movie, written into the session."""
    from .movieprep import MoviePrepError, propose_crop

    curves = state.curves.get(stem)
    if curves is None:
        raise PrepError(f"{stem} has not been read yet")
    try:
        got = propose_crop(curves, start_rule=start_rule, **cut)
    except MoviePrepError as exc:
        raise PrepError(str(exc)) from exc
    state.crops[stem] = (int(got["start"]), int(got["end"]))
    state.proposals[stem] = got
    return got


def propose_all(state: ConditionState, start_rule: str, **cut) -> tuple:
    """Propose for every read movie. Returns ``(proposed, failed)``: names, and
    ``(name, reason)`` for the movies the rule found no floor in."""
    done, failed = [], []
    for stem in sorted(state.curves):
        try:
            propose(state, stem, start_rule, **cut)
            done.append(stem)
        except PrepError as exc:
            failed.append((stem, str(exc)))
    return done, failed


def set_crop(state: ConditionState, stem: str, start: int, end: int) -> tuple:
    """Set one range by hand, refusing one that is empty or outside the movie."""
    start, end = int(start), int(end)
    n = state.n_frames().get(stem)
    if start < 1 or end <= start:
        raise PrepError(f"{stem}: {start}-{end} is not a range (start before end, from 1)")
    if n is not None and end > n:
        raise PrepError(f"{stem}: ends at {end}, past its last frame {n}")
    state.crops[stem] = (start, end)
    return state.crops[stem]


def apply_common_cut(state: ConditionState, *, start_rule: str | None = None,
                     cut: dict | None = None, **edge) -> CutReport:
    """`common_cut` on one condition, written into the session and remembered for the
    movies read later.

    A READ MOVIE WITH NO RANGE YET IS GIVEN THE START RULE'S OWN FIRST (``start_rule``,
    `propose`): reported 23 Oct 2026, "when i cut all tails, the start jumps to frame 1
    instead of staying at x % flat which i choose". A movie read from the cache, or read
    before "Propose for all" was pressed, had curves and no range, and `common_cut` took it
    as ``1..n``. One the rule finds no flat stretch in keeps ``1..n`` and is named
    (``unfloored``), never silently started at 1.
    """
    proposed, unfloored = [], []
    if start_rule:
        for stem in sorted(state.curves):
            if stem in state.crops:
                continue
            try:
                propose(state, stem, start_rule, **(cut or {}))
                proposed.append(stem)
            except PrepError:
                unfloored.append(stem)
    report = common_cut(state.crops, state.n_frames(), **edge)
    state.crops = dict(report.crops)
    if "end" in edge:
        state.common.pop("length", None)
    if "length" in edge:
        state.common.pop("end", None)
    state.common.update({k: int(v) for k, v in edge.items() if v is not None})
    return replace(report, proposed=tuple(proposed), unfloored=tuple(unfloored))


def apply_common_edges(state: ConditionState, stem: str) -> None:
    """Give one movie, read after a common cut, the condition's common edges (start first,
    then the end or the length from that start). A cut that would empty its range leaves
    it as it was, as `common_cut` does."""
    if stem not in state.crops or not state.common:
        return
    for key in ("start", "end", "length"):
        if key in state.common:
            one = common_cut({stem: state.crops[stem]}, {stem: state.n_frames().get(stem)}
                             if state.n_frames().get(stem) else {}, **{key: state.common[key]})
            state.crops[stem] = one.crops[stem]


def save(state: ConditionState, *, projection: str | None = None,
         write_curves_too: bool = True, on_progress=None) -> Path:
    """Write the condition's `crops.json` (ranges + calibration), curves and projections.

    REFUSED BY NAME without a pixel size and a frame interval: neither has a default
    anywhere in this project, every D scales with them, and `load_crops` would refuse the
    file it wrote.
    """
    gaps = [key for key in ("pixel_size_nm", "frame_interval_s")
            if state.calibration.get(key) is None]
    if gaps:
        raise PrepError(f"{state.name}: set the {words_list(gaps)} before saving")
    if not state.crops:
        raise PrepError(f"{state.name}: no ranges to save yet")
    cal = state.calibration
    path = save_condition(
        state.folder, state.crops, pixel_size_nm=cal["pixel_size_nm"],
        frame_interval_s=cal["frame_interval_s"],
        curves=state.curves if write_curves_too else None, movies=state.movies,
        label=cal.get("label"), emission_nm=cal.get("emission_nm"),
        numerical_aperture=cal.get("numerical_aperture"), projection=projection,
        on_progress=on_progress)
    state.saved = dict(state.crops)
    state.saved_calibration = dict(state.calibration)
    return path


# ---------------------------------------------------------------- calibration edits
#: Fields every D scales with. Neither has a default anywhere in this project, so an
#: empty cell for one of them is refused rather than taken as "delete".
REQUIRED_CALIBRATION = ("pixel_size_nm", "frame_interval_s")

#: A field as a person reads it, never its code name (tester B12/B13, 3 Nov 2026: "needs pixelize", "set radius_px",
#: "pixel size nm must be positive"). The calibration words are the Calibration grid's, the tracking words the
#: Detect & link grid's.
FIELD_WORDS = {"pixel_size_nm": "pixel size", "frame_interval_s": "frame interval", "label": "fluorophore",
               "emission_nm": "emission wavelength", "numerical_aperture": "NA",
               "threshold": "detection threshold", "radius_px": "spot radius",
               "linking_max_distance_px": "linking radius", "max_frame_gap": "frame gap"}


#: Typed calibration outside these is WARNED about, never refused (asked by the tester, 3 Nov 2026: 1060 nm for 106
#: was saved silently, every D a hundred times too large). Pixel size in nm (a 100x objective on the lab's cameras
#: gives 106), frame interval in ms.
PLAUSIBLE_PIXEL_NM = (30.0, 500.0)
PLAUSIBLE_FRAME_MS = (1.0, 500.0)
#: A crop shorter than this many frames gets a note, never a refusal (asked by the tester the same night: a proposal
#: of 575-600, 26 frames, and a typed 10..12 were accepted without a word).
SHORT_CROP_FRAMES = 100


def plausibility_note(name: str, key: str, value: float) -> str:
    """A note when a typed pixel size or frame interval (``value`` in nm or ms, as typed) lies outside
    `PLAUSIBLE_PIXEL_NM` / `PLAUSIBLE_FRAME_MS`, else "". The value is kept: an unusual camera is possible."""
    bounds = {"pixel_size_nm": (PLAUSIBLE_PIXEL_NM, "nm", "its square"),
              "frame_interval_s": (PLAUSIBLE_FRAME_MS, "ms", "it")}.get(key)
    if bounds is None:
        return ""
    (low, high), unit, scale = bounds
    if low <= value <= high:
        return ""
    return (f"{name}: a {field_words(key)} of {value:g} {unit} is outside the usual {low:g} to {high:g} {unit}. "
            f"It is kept as typed; check it, because every D scales with {scale}.")


def short_crop_note(crops: dict) -> str:
    """``{stem: (start, end)}``: a note naming the ranges shorter than `SHORT_CROP_FRAMES` (inclusive count), else
    "". Never a refusal: a short movie may be all there is."""
    short = [(stem, end - start + 1) for stem, (start, end) in sorted(crops.items())
             if end - start + 1 < SHORT_CROP_FRAMES]
    if not short:
        return ""
    named = ", ".join(f"{stem} ({n} frames)" for stem, n in short[:3]) + (" ..." if len(short) > 3 else "")
    return (f"Shorter than {SHORT_CROP_FRAMES} frames: {named}. Kept; a short range gives few tracks, so check "
            "it is the stretch you meant.")


def field_words(key: str) -> str:
    """``key`` in words (`FIELD_WORDS`), else with its unit suffix dropped and spaces for underscores."""
    if key in FIELD_WORDS:
        return FIELD_WORDS[key]
    for unit in ("_nm", "_px", "_ms", "_s"):
        if key.endswith(unit):
            key = key[: -len(unit)]
            break
    return key.replace("_", " ")


def words_list(keys) -> str:
    """``["a", "b", "c"]`` as "a, b and c", each in words (`field_words`)."""
    said = [field_words(k) for k in keys]
    return said[0] if len(said) == 1 else ", ".join(said[:-1]) + " and " + said[-1]

#: Short names a fluorophore is typed under, to the catalogue's (`optics.LABEL_PRESETS`); matched without case.
LABEL_ALIASES = {"mng": "mNeonGreen", "mneon": "mNeonGreen", "gfp": "GFP / mGFP", "mgfp": "GFP / mGFP",
                 "jf549": "HaloTag JF549"}


def set_calibration(state: ConditionState, key: str, text: str) -> str:
    """Set one calibration field from what was typed. Returns a note, ``""`` when silent.

    The frame interval is typed in MILLISECONDS and kept in seconds, as `crops.json`
    records it. AN EMPTY CELL IS NOT A DELETION for a required field: the classic window
    once wrote ``None`` over a pixel size that way and left a file `load_crops` refuses.
    A fluorophore named after a known label fills an EMPTY emission from the catalogue,
    and says so; it never replaces a recorded one (509 nm read from an `.nd2` against
    mNeonGreen's catalogue 517 is a measurement against a lookup).
    """
    from .optics import preset_names
    from .optics import preset as label_preset

    if key not in CALIBRATION_KEYS:
        raise PrepError(f"unknown calibration field {key!r}")
    text = (text or "").strip()
    if not text:
        if key in REQUIRED_CALIBRATION:
            raise PrepError(f"{state.name}: the {field_words(key)} cannot be "
                            f"empty; it has no default, and every D scales with it")
        state.calibration[key] = None
        return ""
    if key == "label":
        # A KNOWN LABEL BY ANY SPELLING (23 Oct 2026: "mNG emission not coming when i click on
        # mneongreen"): matched without case, "mNG" too, and kept under the catalogue's name.
        names = {name.casefold(): name for name in preset_names()}
        names.update({alias: name for alias, name in LABEL_ALIASES.items() if name in names.values()})
        known = names.get(text.casefold())
        state.calibration["label"] = known or text
        if known is None:
            return ""
        emission = float(label_preset(known).emission_nm)
        recorded = state.calibration.get("emission_nm")
        # an emission the PREVIOUS label filled follows the new one; a recorded one stays
        if recorded is None or state.emission_from_label:
            state.calibration["emission_nm"] = emission
            state.emission_from_label = True
            return f"{state.name}: emission {emission:g} nm filled from {known}'s catalogue peak"
        if abs(float(recorded) - emission) > 0.5:
            return (f"{state.name}: emission kept at the recorded {float(recorded):g} nm; {known}'s "
                    f"catalogue peak is {emission:g} nm (type it in the Emission box to use it)")
        return ""
    try:
        value = float(text.replace(",", "."))
    except ValueError as exc:
        raise PrepError(f"{state.name}: {text!r} is not a number") from exc
    if value <= 0:
        raise PrepError(f"{state.name}: the {field_words(key)} must be more than 0")
    state.calibration[key] = value / 1000.0 if key == "frame_interval_s" else value
    if key == "emission_nm":
        state.emission_from_label = False                    # typed: a label never replaces it
    return plausibility_note(state.name, key, value)


def copy_calibration(states: dict, source: str) -> list:
    """`movieprep.replicate_calibration` across the session: the fields ``source`` HAS,
    onto every other condition. Returns the names that changed."""
    from .movieprep import MoviePrepError, replicate_calibration

    try:
        merged = replicate_calibration({n: dict(s.calibration) for n, s in states.items()},
                                       source)
    except MoviePrepError as exc:
        raise PrepError(str(exc)) from exc
    changed = []
    for name, values in merged.items():
        if values != states[name].calibration:
            states[name].calibration = dict(values)
            changed.append(name)
    return changed


def probe_acquisition(movie, recorded_interval_s=None) -> dict:
    """What one acquisition says about its calibration, and whether a RECORDED frame
    interval agrees with it (`rawmovie.check_recorded_interval`): a wrong interval is
    invisible downstream -- every D scales as 1/tau and nothing looks wrong. Found once
    on real data as 20 ms recorded against 51.7477 measured. Module-level for a pool."""
    from .movieprep import detect_calibration
    from .rawmovie import check_recorded_interval, read_raw_movie

    check = (None if recorded_interval_s is None
             else check_recorded_interval(recorded_interval_s, movie))
    # THE CAMERA, FOR THE PIXEL-SIZE HELPER: the sensor pixel and the objective's bare
    # magnification, so the user supplies only the term the file drops (the tube lens).
    meta = read_raw_movie(movie).meta
    camera = {key: float(getattr(meta, key)) for key in
              ("camera_pixel_um", "objective_magnification") if getattr(meta, key, None)}
    return {"measured": detect_calibration(movie), "check": check, "camera": camera}


def fill_from_acquisition(state: ConditionState, measured: dict) -> list:
    """Fill the calibration fields nobody recorded from what the ACQUISITION measured
    (`movieprep.detect_calibration`: the frame interval and NA, never the pixel size).
    A recorded value is never overwritten. Returns the keys filled."""
    filled = []
    for key, value in (measured or {}).items():
        if key in CALIBRATION_KEYS and state.calibration.get(key) is None and value:
            state.calibration[key] = float(value)
            filled.append(key)
    return filled


# ---------------------------------------------------------------- detection and linking
def seed_tracking(state: ConditionState, base: dict | None = None) -> list:
    """Fill the tracking fields nobody set: the linking radius and frame gap from ``base``
    -- the user's TrackMate standard (`options`), else the validated
    `trackmate_run.ADAPTIVE_BASE` -- and the SPOT RADIUS DERIVED FROM THIS CONDITION'S
    OWN OPTICS (`trackmate_run.spot_radius_per_condition`) -- 2.80 px for mNeonGreen at
    NA 1.49, 3.50 for a red dye. Where the optics are incomplete the radius stays EMPTY:
    a flat 3.0 printed beside derived values reads as a measurement (the classic grid did
    exactly that). A TYPED value is never replaced. The threshold is never seeded: it is
    measured per condition. Returns the fields filled.

    AN UNTYPED VALUE FOLLOWS THE STANDARD: every other write to these two fields marks
    them typed (`set_tracking`, `copy_tracking`), so an untyped one only ever held the
    standard, and a new standard moves it. A typed one stays.
    """
    from .trackmate_run import ADAPTIVE_BASE, spot_radius_per_condition

    base = {**ADAPTIVE_BASE, **(base or {})}
    filled = []
    for key in ("linking_max_distance_px", "max_frame_gap"):
        if key not in state.typed and state.tracking.get(key) != base[key]:
            state.tracking[key] = base[key]
            filled.append(key)
    # EACH SPOT'S PRECISION after TrackMate (6 Nov 2026, `fit_spot_precision`): the Options switch, on unless set off;
    # a switch, not a number shown in a box, so not counted among the fields filled
    if "spot_precision" not in state.typed:
        state.tracking["spot_precision"] = bool(base.get("spot_precision", True))
    if "radius_px" not in state.typed:
        derived = spot_radius_per_condition({state.name: state.calibration}).get(state.name)
        if derived is None:
            # The optics were cleared after a radius was derived from them: a radius
            # left standing would read as derived from optics that are no longer there.
            state.tracking.pop("radius_px", None)
        elif derived != state.tracking.get("radius_px"):
            state.tracking["radius_px"] = float(derived)
            filled.append("radius_px")
    return filled


def set_tracking(state: ConditionState, key: str, text: str,
                 base: dict | None = None) -> str:
    """Set one tracking field from what was typed. Returns a note, ``""`` when silent.

    A TYPED THRESHOLD PINS ONE NUMBER FOR EVERY MOVIE of the condition and drops the
    per-movie table -- otherwise the grid would show one number while the run used
    another. Measuring again returns to per-movie. Empty clears a typed value back to
    the derived or measured one.
    """
    from .trackmate_run import TRACKING_FIELDS

    keys = [k for k, _label, _unit in TRACKING_FIELDS]
    if key not in keys:
        raise PrepError(f"unknown tracking field {key!r}")
    text = (text or "").strip()
    if not text:
        state.typed.discard(key)
        if key == "threshold":
            state.tracking["threshold"] = (None if state.noise is None
                                           else float(state.noise["threshold"]))
        else:
            state.tracking.pop(key, None)
            seed_tracking(state, base)
        return ""
    try:
        value = float(text.replace(",", "."))
    except ValueError as exc:
        raise PrepError(f"{state.name}: {text!r} is not a number") from exc
    if key == "max_frame_gap":
        if value < 0 or value != int(value):
            raise PrepError(f"{state.name}: the frame gap is a whole number of frames, "
                            f"0 or more")
        value = int(value)
    elif value <= 0:
        raise PrepError(f"{state.name}: the {field_words(key)} must be more than 0")
    state.tracking[key] = value
    state.typed.add(key)
    return (f"{state.name}: threshold {value:g} typed: one threshold for every movie"
            if key == "threshold" else "")


def copy_tracking(states: dict, source: str) -> list:
    """The source's linking radius and frame gap onto every other condition.

    NOT THE THRESHOLD (measured from each condition's own noise) AND NOT THE SPOT RADIUS
    (derived from each condition's own optics): copying either would report one
    condition's measurement over another's data. Returns the names that changed.
    """
    src = states[source].tracking
    changed = []
    for name, state in states.items():
        if name == source:
            continue
        moved = False
        for key in ("linking_max_distance_px", "max_frame_gap"):
            if src.get(key) is not None and state.tracking.get(key) != src[key]:
                state.tracking[key] = src[key]
                state.typed.add(key)
                moved = True
        if moved:
            changed.append(name)
    return changed


def measurable(state: ConditionState) -> list:
    """``[(path, movie, (first, last))]`` when EVERY movie has a range, else ``[]``.

    The threshold is measured INSIDE each crop, and a threshold over some of the movies
    is a different quantity from one over the set, with nothing on screen to say which.
    Measuring is not running, so the ranges on screen count (they need not be saved).
    """
    rows = [(p, stem, state.crops.get(stem)) for p, stem in zip(state.movies, state.names)]
    if not rows or any(crop is None for _p, _s, crop in rows):
        return []
    return [(p, stem, tuple(crop)) for p, stem, crop in rows]


def record_noise(state: ConditionState, summary: dict) -> None:
    """File a noise measurement: the median threshold, unless one was typed."""
    state.noise = dict(summary)
    if "threshold" not in state.typed:
        state.tracking["threshold"] = float(summary["threshold"])


def thresholds_for(state: ConditionState) -> dict | None:
    """``{movie: threshold}`` -- each movie detected against its OWN noise -- or None
    when one threshold applies to all (typed, or never measured)."""
    if "threshold" in state.typed or not state.noise:
        return None
    return {row["movie"]: float(row["threshold"])
            for row in state.noise.get("per_movie", ()) if row.get("ok")}


def tracking_gaps(state: ConditionState) -> list:
    """The tracking fields still missing (`trackmate_run.missing_tracking_settings`)."""
    from .trackmate_run import missing_tracking_settings

    return missing_tracking_settings({state.name: state.tracking}).get(state.name, [])


def summarise_noise(rows, *, radius_px: float, k: float | None = None) -> dict:
    """`trackmate_run.summarise_thresholds` over rows measured one movie at a time
    (`trackmate_run.movie_threshold_row`), so the parallel measurement is summarised with
    exactly `batch_threshold`'s arithmetic. ``k`` is the multiplier the rows were measured
    at (the TrackMate standard), recorded on the summary. Raises `PrepError` when no movie
    measured."""
    from .trackmate_run import TrackMateRunError, UTRACK_MATCHED_K, summarise_thresholds

    try:
        return summarise_thresholds(rows, radius_px=radius_px,
                                    k=UTRACK_MATCHED_K if k is None else float(k))
    except TrackMateRunError as exc:
        raise PrepError(str(exc)) from exc


def rethreshold(state: ConditionState, k: float) -> bool:
    """A measured condition's thresholds at another multiplier ``k``, without measuring
    again: each movie's threshold IS ``k x`` its LoG noise (`adaptive_threshold`), so the
    noise already measured gives it exactly. A typed threshold stays (`record_noise`).
    Returns whether anything changed."""
    noise = state.noise
    if not noise or float(noise.get("k", 0.0)) == float(k):
        return False
    rows = [dict(row, threshold=float(k) * float(row["sigma_log"])) if row.get("ok")
            else dict(row) for row in noise.get("per_movie", ())]
    record_noise(state, summarise_noise(rows, radius_px=float(noise["radius_px"]), k=k))
    return True


# ---------------------------------------------------------------- tracking
#: Where a condition's TrackMate run writes its XMLs and `run.json`, and where
#: `discover_trackmate_project` reads them back.
TRACKS_FOLDER = "trackmate"

#: About what one movie costs to track, shown BEFORE a run. The recorded figure for this
#: project's movies is ~30 s each (the classic window used the same), and a run of nine
#: conditions is hours; a window that simply goes busy teaches people to kill it.
SECONDS_PER_MOVIE = 30

#: MOVIES AT ONCE (asked 23 Oct 2026: "can we maybe add a parallel cpu running for tracking in utrack and
#: trackmate ... a mode in the tool"). Each movie is its own process -- a MATLAB session holding a share of the
#: condition's movies, or a Fiji per movie -- so N run side by side. MEASURED on one user's protein F run (i5-9300H, 4
#: cores / 8 threads, 16 GB): one u-track session used 2.0 logical cores and the machine sat at 27 %; a movie
#: took ~200 s. So the standard is half the logical cores less one for the window and the disk, at most
#: `MAX_AT_ONCE` (each session also holds a MATLAB, ~0.5-1.5 GB).
MAX_AT_ONCE = 4


def default_at_once(logical_cores: int | None = None) -> int:
    """Movies tracked at once by default on this machine: ``logical cores // 2 - 1``, 1 to `MAX_AT_ONCE`."""
    import os

    cores = int(logical_cores or os.cpu_count() or 2)
    return max(1, min(MAX_AT_ONCE, cores // 2 - 1))


def minutes_at_once(minutes: float, movies: int, at_once: int) -> float:
    """A run's estimate with ``at_once`` movies side by side: the work shared over the lanes it can fill."""
    return float(minutes) / max(1, min(int(at_once or 1), int(movies or 1)))


def track_files(folder) -> list:
    """The ``*.xml`` an earlier run left in ``<condition>/trackmate/``. A listing only."""
    out = Path(folder) / TRACKS_FOLDER
    return sorted(out.glob("*.xml")) if out.is_dir() else []


@dataclass(frozen=True)
class RunInputs:
    """Everything one condition's run is given, resolved FROM THE CONDITION.

    ``crops`` and the calibration are what ``crops.json`` holds -- the single record of
    what was tracked -- never ranges that exist only on screen. ``per_movie`` is
    ``{movie: threshold}`` (each movie against its own noise) or None for one value."""

    condition: str
    movies: tuple
    crops: dict
    settings: object
    output_dir: Path
    per_movie: dict | None
    pixel_size_nm: float | None
    frame_interval_s: float | None
    #: EACH SPOT'S PRECISION after TrackMate (6 Nov 2026, `spotfit`; asked: "is it possible in the tracking part for
    #: trackmate to make this an option to do the precision etc?"): every tracked spot refitted with a 2D Gaussian of
    #: this PSF width (pixels, from the saved optics, `optics.psf_sigma_nm`), its precision written beside the XML.
    #: None where the optics are incomplete: then not fitted, and the run says so.
    psf_sigma_px: float | None = None
    spot_precision: bool = True

    @property
    def minutes(self) -> float:
        return len(self.movies) * SECONDS_PER_MOVIE / 60.0


def run_inputs(state: ConditionState) -> RunInputs:
    """What ``state``'s run would be given, or `PrepError` saying why it cannot start.

    REFUSED BY NAME, in the order a person fixes them: no acquisitions; edits not saved
    (the run reads `crops.json`, so an unsaved range would be tracked as the OLD one with
    nothing on screen saying so); nothing saved; a movie with no saved range (tracking a
    raw acquisition whole includes the dense early frames the crop exists to remove);
    a tracking setting missing. Nothing here touches the disk.
    """
    from .trackmate_run import TrackMateRunError, TrackMateSettings

    if not state.movies:
        raise PrepError(f"{state.name}: no acquisitions")
    if state.unsaved:
        raise PrepError(f"{state.name}: ranges or calibration edited since the last save. "
                        f"Press Save on Calibration & save first; crops.json records what is tracked.")
    if not state.saved:
        raise PrepError(f"{state.name}: no saved crop ranges; set them on Movies & crop "
                        f"and press Save")
    absent = [stem for stem in state.names if stem not in state.saved]
    if absent:
        raise PrepError(f"{state.name}: {len(absent)} movie(s) have no saved range: "
                        + ", ".join(absent[:4]) + (" ..." if len(absent) > 4 else ""))
    gaps = tracking_gaps(state)
    if "threshold" in gaps:
        raise PrepError(f"{state.name}: no detection threshold; press Measure on "
                        f"Detect & link, or type one")
    if gaps:
        raise PrepError(f"{state.name}: set the {words_list(gaps)} on Detect & link")
    values = state.tracking
    link = float(values["linking_max_distance_px"])
    try:
        settings = TrackMateSettings(
            radius_px=float(values["radius_px"]), threshold=float(values["threshold"]),
            max_frame_gap=int(values["max_frame_gap"]),
            # BOTH linking distances move together: u-track has one search radius, so
            # letting TrackMate's two drift apart creates a policy with no equivalent.
            linking_max_distance_px=link, gap_closing_max_distance_px=link)
    except (TrackMateRunError, TypeError, ValueError) as exc:
        raise PrepError(f"{state.name}: the settings do not make a run ({exc})") from exc
    calibration = state.saved_calibration
    return RunInputs(
        condition=state.name, movies=tuple(state.movies),
        crops={stem: tuple(state.saved[stem]) for stem in state.names},
        settings=settings, output_dir=Path(state.folder) / TRACKS_FOLDER,
        per_movie=thresholds_for(state),
        pixel_size_nm=calibration.get("pixel_size_nm"),
        frame_interval_s=calibration.get("frame_interval_s"),
        psf_sigma_px=psf_sigma_px_of(calibration),
        spot_precision=bool(values.get("spot_precision", True)))


def psf_sigma_px_of(calibration: dict) -> float | None:
    """The PSF width in PIXELS from a condition's saved optics (emission, NA, pixel size), or None where one is
    missing: never a default (`optics.psf_sigma_nm`)."""
    from .optics import OpticsError, psf_sigma_nm

    try:
        emission = float(calibration.get("emission_nm") or 0)
        aperture = float(calibration.get("numerical_aperture") or 0)
        pixel = float(calibration.get("pixel_size_nm") or 0)
        if pixel <= 0:
            return None
        return psf_sigma_nm(emission_nm=emission, numerical_aperture=aperture) / pixel
    except (OpticsError, TypeError, ValueError):
        return None


def track_condition(inputs: RunInputs, *, install, on_progress=None, stop=None,
                    at_once: int = 1) -> dict:
    """Track one condition with TrackMate and say what came out, as PLAIN data.

    The run record is written BEFORE the run (its inputs matter most when a run dies half
    way) and again with the outcome. The result is READ BACK from the XMLs rather than
    trusted: the batch table counts what TrackMate said, and only the files give the
    USABLE track count and the step distribution (`trackmate_run.summarise_run`).

    ``on_progress(index, total, row)`` is `run_batch`'s, with ``row`` a plain dict;
    ``stop()`` is asked before each movie. Runs on a THREAD: the work is a Fiji
    subprocess per movie, so the GIL is free while it waits.
    """
    from .trackmate_run import (
        STOPPED_ERROR,
        TrackMateRunError,
        batch_qc,
        describe_batch,
        record_run,
        run_batch,
        summarise_run,
        track_counts,
    )

    def record(outcome=None):
        return record_run(
            inputs.output_dir, settings=inputs.settings, movies=inputs.movies,
            crops=inputs.crops, pixel_size_nm=inputs.pixel_size_nm,
            frame_interval_s=inputs.frame_interval_s, source="raw",
            threshold_provenance=("adaptive_per_movie" if inputs.per_movie
                                  else "one_for_the_batch"),
            per_movie_thresholds=inputs.per_movie, outcome=outcome, at_once=at_once)

    path = record()
    progress = None
    if on_progress is not None:
        def progress(index, total, row):
            on_progress(index, total, {k: row[k] for k in
                                       ("movie", "n_spots", "n_tracks", "threshold",
                                        "seconds", "ok", "error")})
    table = run_batch(list(inputs.movies), inputs.output_dir, settings=inputs.settings,
                      install=install, crops=inputs.crops, thresholds=inputs.per_movie,
                      on_progress=progress, stop=stop, at_once=at_once)
    stopped = int((table["error"] == STOPPED_ERROR).sum())
    failed = int((~table["ok"]).sum()) - stopped
    flagged = 0
    try:
        qc = batch_qc(table)
        verdict = describe_batch(qc)
        flagged = int(qc["flagged"].sum()) if "flagged" in qc else 0
    except TrackMateRunError as exc:
        # NO MOVIE TRACKED -- every one failed, or the run was stopped before the first.
        # Said, not raised: the record is still written, and the rows name each reason.
        verdict = str(exc)
    if stopped:
        verdict = (f"Stopped: {stopped} movie(s) not run; this condition is not fully "
                   f"tracked.\n{verdict}")
    precision = fit_spot_precision(inputs, table)
    if precision["words"]:
        verdict = f"{verdict}\n{precision['words']}"
    raw = _read_back(table)
    summary = summarise_run(raw, inputs.settings, table=table)
    tracks, usable = track_counts(raw)
    outcome = {"movies": int(len(table)), "failed": failed, "stopped": stopped,
               "flagged": flagged, "tracks": tracks, "usable_tracks": usable,
               "spot_precision": {k: precision[k] for k in ("fitted_movies", "failed_movies")}}
    record(outcome)
    return {"condition": inputs.condition, "record": str(path), "outcome": outcome,
            "verdict": verdict, "summary": summary,
            "track_files": len(track_files(inputs.output_dir.parent))}


def fit_spot_precision(inputs: RunInputs, table) -> dict:
    """Each spot of this run's XMLs refitted for its precision (`spotfit.fit_xml`), beside its XML, when the option is
    on. NEVER fails the run: a movie whose fit failed is counted and named, its XML left without precision (the
    analysis then refuses per-spot precision for its condition, by name); skipped, and said, without the optics."""
    out = {"fitted_movies": 0, "failed_movies": 0, "words": ""}
    if not inputs.spot_precision:
        return out
    if not inputs.psf_sigma_px:
        out["words"] = ("Spot precision was not fitted: the optics (emission, NA, pixel size) are incomplete, so the PSF "
                        "width is unknown. Save them on Calibration & save and track again.")
        return out
    from .spotfit import fit_xml

    failures = []
    condition_dir = Path(inputs.output_dir).parent
    for row in table.itertuples():
        if not (row.ok and row.xml):
            continue
        xml = Path(row.xml)
        try:
            fit_xml(xml, movie_root=condition_dir, movie=xml.stem, psf_sigma_px=float(inputs.psf_sigma_px))
            out["fitted_movies"] += 1
        except Exception as exc:                           # noqa: BLE001 - counted and named, never fatal
            failures.append(f"{xml.stem} ({type(exc).__name__}: {exc})")
    out["failed_movies"] = len(failures)
    out["words"] = (f"Spot precision fitted for {out['fitted_movies']} movie(s)"
                    + (f"; not for {len(failures)}: " + "; ".join(failures[:3]) if failures else "") + ".")
    return out


def _read_back(table):
    """The localisations of every XML this run WROTE, under their movie names, or None.
    Only this run's files: an XML from an earlier run in the folder is not this result."""
    import pandas as pd

    from .rawmovie import movie_name
    from .trackmate import read_trackmate_xml

    parts = []
    for row in table.itertuples():
        if row.ok and row.xml:
            parts.append(read_trackmate_xml(row.xml).assign(movie=movie_name(Path(row.xml))))
    return pd.concat(parts, ignore_index=True) if parts else None


# ---------------------------------------------------------------- stacks for u-track
#: Where an exported cropped stack goes: ``tif/<movie>.tif``, what the lab's Fiji macro
#: wrote and what u-track imports.
STACKS_FOLDER = "tif"


def stack_files(folder) -> list:
    """The flat ``tif/*.tif`` stacks already in a condition. A listing only; u-track's
    own exploded ``tif/<movie>/`` folders are not stacks and are not counted."""
    out = Path(folder) / STACKS_FOLDER
    return sorted(p for p in out.glob("*.tif") if p.is_file()) if out.is_dir() else []


def export_inputs(state: ConditionState) -> dict:
    """What exporting ``state``'s cropped stacks needs, or `PrepError` saying why not.

    THE SAVED RECORD, as for a tracking run: the stack's frames are the range in
    `crops.json`, and its calibration is the one written there -- both are stamped into
    the TIFF, and a stack cut from a range that exists only on screen would disagree
    with the record of what was cut. ``{"movies": [(path, name, (first, last))],
    "out_root", "pixel_size_nm", "frame_interval_s"}``."""
    if not state.movies:
        raise PrepError(f"{state.name}: no acquisitions")
    if state.unsaved:
        raise PrepError(f"{state.name}: edited since the last save. Press Save first; the stack is cut from the "
                        f"saved range.")
    absent = [stem for stem in state.names if stem not in state.saved]
    if not state.saved or absent:
        raise PrepError(f"{state.name}: {len(absent) or len(state.names)} movie(s) have no "
                        f"saved range")
    calibration = state.saved_calibration
    missing = [label for key, label in (("pixel_size_nm", "pixel size"),
                                        ("frame_interval_s", "frame interval"))
               if not calibration.get(key)]
    if missing:
        raise PrepError(f"{state.name}: no {' or '.join(missing)} saved; both are written "
                        f"into the stack, and neither has a default")
    return {"movies": [(p, stem, tuple(state.saved[stem]))
                       for p, stem in zip(state.movies, state.names)],
            "out_root": Path(state.folder),
            "pixel_size_nm": float(calibration["pixel_size_nm"]),
            "frame_interval_s": float(calibration["frame_interval_s"])}


def export_movie(path, out_root, *, name: str, start: int, end: int, pixel_size_nm: float,
                 frame_interval_s: float, projection: str = "sd") -> dict:
    """Cut ONE movie into ``tif/<name>.tif`` (`movieprep.prepare_movie`) and say how it
    went as a plain row; a failure is a row with its error, never an absence -- the rule
    `prepare_batch` keeps. Module-level, so the process pool runs one movie per worker."""
    from .movieprep import PrepConfig, prepare_movie

    row = {"movie": name, "ok": False, "error": "", "n_frames_kept": None,
           "crop_start": int(start), "crop_end": int(end)}
    try:
        config = PrepConfig(crop_start=int(start), crop_end=int(end),
                            pixel_size_nm=float(pixel_size_nm),
                            frame_interval_s=float(frame_interval_s),
                            projection=projection or "sd")
        got = prepare_movie(path, out_root, config, name=name)
        row.update(ok=True, n_frames_kept=int(got["n_frames_kept"]))
    except Exception as exc:                           # noqa: BLE001 - kept and named
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


# ---------------------------------------------------------------- the linking radius
#: How many movies the wider-radius check tracks, at each of two radii: two is what
#: keeps it near two minutes, and the twelve-group table rested on six.
RADIUS_CHECK_MOVIES = 2
#: How much wider the second run goes: wide enough to REVEAL what the narrow radius hid
#: (8 px against 5 shows 17.5 % on a lab project), not "wide enough to be uncensored",
#: which no affordable pilot is.
RADIUS_CHECK_MULTIPLE = 1.6


def radius_check(inputs: RunInputs, *, install, on_progress=None, stop=None) -> dict:
    """Track the same couple of movies at the radius in use AND at 1.6x it, everything
    else identical, and count what appears beyond the narrow radius
    (`trackmate_run.widening_yield`).

    TWO RUNS, NOT ONE AGAINST THE TRACKS ALREADY THERE: those may come from another
    detector, and the statistic would report the detector difference as truncation.
    Written to ``trackmate_radius_check/`` so it can never be mistaken for a real run.
    A DISCLOSURE, NOT AN INSTRUCTION TO WIDEN -- fractions were validated at 5 px."""
    from dataclasses import replace

    from .trackmate_run import read_batch, run_batch, widening_yield

    if not inputs.frame_interval_s or not inputs.pixel_size_nm:
        raise PrepError(f"{inputs.condition}: the check needs the frame interval and pixel "
                        f"size saved: a radius means a different D at a different frame "
                        f"interval, and neither has a default")
    narrow = float(inputs.settings.linking_max_distance_px)
    wide = round(narrow * RADIUS_CHECK_MULTIPLE, 1)
    movies = list(inputs.movies)[:RADIUS_CHECK_MOVIES]
    out = inputs.output_dir.parent / "trackmate_radius_check"
    total = 2 * len(movies)
    tables = []
    for n, radius in enumerate((narrow, wide)):
        settings = replace(inputs.settings, linking_max_distance_px=radius,
                           gap_closing_max_distance_px=radius)

        def progress(index, _total, row, done=n * len(movies), radius=radius):
            if on_progress is not None:
                on_progress(done + index, total, {**{k: row[k] for k in
                                                     ("movie", "n_tracks", "ok", "error")},
                                                  "radius": radius})

        table = run_batch(movies, out / f"link{radius:g}", settings=settings,
                          install=install, crops=inputs.crops, thresholds=inputs.per_movie,
                          on_progress=progress, stop=stop)
        tables.append(read_batch(table))
    got = widening_yield(tables[0], tables[1], narrow_radius_px=narrow, wide_radius_px=wide,
                         tau_s=float(inputs.frame_interval_s),
                         pixel_size_nm=float(inputs.pixel_size_nm))
    return {"condition": inputs.condition, "checked_movies": [Path(m).name for m in movies],
            **_plain(got)}


def _plain(value):
    """numpy scalars and tuples to plain Python, for a payload that crosses to Qt."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


def pilot_steps(folder, radius_px: float) -> dict:
    """The linking radius READ OFF tracks already in ``folder``: the step lengths (the
    floor) and how often another detection sits within one radius (the ceiling).

    D is not needed -- D is FITTED from these very steps, so they are one level less
    derived. u-track output is preferred (its search cap is recorded); TrackMate's
    otherwise, at the radius ITS run.json says it linked with -- the classic window
    reported truncation against whatever the box held, which is not the radius that
    produced the steps. Raises `PrepError` with the way out when nothing is tracked."""
    import json

    import pandas as pd

    from .trackmate_run import TrackMateRunError, linking_ambiguity, step_length_summary

    folder = Path(folder)
    parts, used, source = [], None, None
    mats = sorted(folder.glob("tif/*/TrackingPackage/tracks/*.mat"))
    if mats:
        from .ingest import read_utrack_mat

        failed = 0
        for path in mats[:8]:
            try:
                parts.append(read_utrack_mat(path).assign(movie=path.parent.parent.parent.name))
            except Exception:                          # noqa: BLE001 - counted and said
                failed += 1
        if parts:
            source = f"u-track output, {len(parts)} movie(s)" + (
                f" ({failed} unreadable)" if failed else "")
            try:
                from .utrack_meta import read_project_metadata

                recorded = [m.max_search_radius_px for m in read_project_metadata(folder)
                            if m.max_search_radius_px]
                used = max(recorded) if recorded else None
            except Exception as exc:                   # noqa: BLE001 - said, not hidden
                source += f" (no recorded search radius: {type(exc).__name__})"
    if not parts:
        xmls = track_files(folder)
        if xmls:
            from .rawmovie import movie_name
            from .trackmate import read_trackmate_xml

            for path in xmls[:8]:
                parts.append(read_trackmate_xml(path).assign(movie=movie_name(path)))
            source = f"TrackMate output, {len(parts)} movie(s)"
            record = folder / TRACKS_FOLDER / "run.json"
            if record.is_file():
                try:
                    used = float(json.loads(record.read_text(encoding="utf-8"))
                                 ["settings"]["linking_max_distance_px"])
                except (OSError, ValueError, KeyError, TypeError):
                    used = None
    if not parts:
        raise PrepError(f"no tracked result in {folder.name} to measure from. Track a "
                        f"movie or two at a generous radius first, because a pilot's "
                        f"steps are cut at the radius that produced them")
    table = pd.concat(parts, ignore_index=True)
    try:
        steps = step_length_summary(table, radius_used_px=used)
        crowd = linking_ambiguity(table, radius_px=float(radius_px))
    except TrackMateRunError as exc:
        raise PrepError(f"could not measure: {exc}") from exc
    return {"source": source, "radius_used_px": used, "steps": _plain(steps),
            "crowd": _plain(crowd)}


def camera_pixel_size(camera_pixel_um, magnification, tube_lens) -> float:
    """The pixel size from the camera and the optics, or `PrepError`. The TUBE LENS is
    the term a .nd2 drops: 16 um / 100x = 160 nm, against 16 um / 150x = 106.7 nm."""
    from .optics import OpticsError, pixel_size_nm

    try:
        return float(pixel_size_nm(camera_pixel_um=float(camera_pixel_um),
                                   magnification=float(magnification),
                                   tube_lens=float(tube_lens)))
    except (OpticsError, TypeError, ValueError) as exc:
        raise PrepError(f"no pixel size from these: {exc}") from exc



# ---------------------------------------------------------------- where the signal runs out
#: The width of a block on the tail-purity picture, in frames.
TAIL_BLOCK_FRAMES = 200


def tail_purity(folder, crops: dict, *, block: int = TAIL_BLOCK_FRAMES) -> dict:
    """Localisations per area INSIDE the cell outlines against OUTSIDE them, block by
    block of ABSOLUTE frames, from the tracks already in ``<folder>/trackmate/``.

    Asked: "does it make sense to cut out the tails of the movies? a user has like 3000
    frames where there is like no signal, higher background contamination chance?" The
    answer is this ratio: real molecules bleach away while background detections do not,
    so the outside rate as a share of the in-cell rate RISES along a movie. Measured on
    One user's d1.1 (11 movies): 0.5 % in the first 250 frames of its crop, 3.5 % in the last
    300, which held 2 % of her in-cell localisations. The outside rate is a LOWER bound on
    the in-cell false rate -- cells autofluoresce -- so read the share as a floor.

    A track file's frames count from its movie's crop start, so each is placed on the
    acquisition's own frame axis with the saved range (`crops`). Movies without outlines,
    a brightfield snapshot or a saved range are skipped, each with its reason
    (``skipped_why``, and ``skipped_note`` in words). Raises `PrepError` with the reasons
    when there is nothing to measure. Module-level, for the pool.

    The snapshot (for the field's size) is found by the channel rule
    (`segmentation.channels`, ruled 18 Oct 2026): in ``phase/`` OR beside the movies. This
    looked only in ``phase/``, so on a loose folder EVERY movie was skipped and the user was
    told they lacked outlines or a range (handoff 18 Oct 2026, place 2)."""
    import numpy as np
    import tifffile

    from .cellmesh import mesh_folder, points_in_polygon, read_cell_meshes
    from .rawmovie import movie_name
    from .segmentation.channels import resolve_channels
    from .trackmate import read_trackmate_xml

    folder = Path(folder)
    xmls = track_files(folder)
    if not xmls:
        raise PrepError(f"{folder.name}: no tracks in trackmate/ to measure from. Track "
                        f"first; the in-cell rate needs detections and outlines.")
    channels = resolve_channels(folder, ("phase",))
    sums: dict = {}
    used, why = [], {}
    for xml in xmls:
        stem = movie_name(xml)
        mat = mesh_folder(folder) / f"{stem}.mat"
        phase = channels.image_path("phase", stem) if "phase" in channels.resolved else None
        crop = crops.get(stem)
        if not mat.is_file():
            why[stem] = TAIL_SKIP_NO_OUTLINES
            continue
        if phase is None:
            why[stem] = TAIL_SKIP_NO_SNAPSHOT
            continue
        if crop is None:
            why[stem] = TAIL_SKIP_NO_RANGE
            continue
        meshes = [m for m in read_cell_meshes(mat, on_empty="empty") if m.polygon is not None]
        if not meshes:
            why[stem] = TAIL_SKIP_NO_CELLS
            continue
        h, w = tifffile.TiffFile(phase).pages[0].shape[:2]
        gx, gy = np.meshgrid(np.arange(1, w + 1, dtype=float), np.arange(1, h + 1, dtype=float))
        gx, gy = gx.ravel(), gy.ravel()
        table = read_trackmate_xml(xml)
        px, py = table["x"].to_numpy(float) + 1.0, table["y"].to_numpy(float) + 1.0
        inside_px = np.zeros(gx.size, bool)
        inside = np.zeros(len(table), bool)
        for mesh in meshes:
            polygon = np.asarray(mesh.polygon, dtype=float)
            inside_px |= points_in_polygon(gx, gy, polygon)
            inside |= points_in_polygon(px, py, polygon)
        a_in, a_out = int(inside_px.sum()), int((~inside_px).sum())
        first, last = int(crop[0]), int(crop[1])
        # A track file counts frames from 1 at its crop start (`trackmate.FRAME_ORIGIN_
        # OFFSET` normalises TrackMate's 0), so frame f is acquisition frame first + f - 1;
        # subtracting the file's EARLIEST detection would assume one on the first frame.
        frames = first + (table["frame"].to_numpy() - 1)
        for b0 in range(first - (first - 1) % block, last + 1, block):
            lo, hi = max(b0, first), min(b0 + block - 1, last)
            if hi < lo:
                continue
            sel = (frames >= lo) & (frames <= hi)
            span = hi - lo + 1
            got = sums.setdefault(b0, [0, 0, 0.0, 0.0])
            got[0] += int((sel & inside).sum())
            got[1] += int((sel & ~inside).sum())
            got[2] += a_in * span
            got[3] += a_out * span
        used.append(stem)
    if not used:
        raise PrepError(f"{folder.name}: no movie has tracks, outlines, a snapshot and a saved "
                        f"range together: {tail_skip_note(why)}")
    rows = []
    total_in = sum(v[0] for v in sums.values())
    for b0 in sorted(sums):
        n_in, n_out, area_in, area_out = sums[b0]
        d_in = 1e3 * n_in / area_in if area_in else 0.0
        d_out = 1e3 * n_out / area_out if area_out else 0.0
        rows.append({"first": int(b0), "last": int(b0 + block - 1), "in": d_in, "out": d_out,
                     "share": (d_out / d_in) if d_in else None,
                     "n_in": int(n_in), "n_out": int(n_out),
                     "of_all": (n_in / total_in) if total_in else 0.0})
    return {"rows": rows, "movies": used, "skipped": sorted(why), "skipped_why": why,
            "skipped_note": tail_skip_note(why), "block": int(block)}


#: Why `tail_purity` left a movie out, one reason each, checked in this order.
TAIL_SKIP_NO_OUTLINES = "no outlines"
TAIL_SKIP_NO_SNAPSHOT = "no brightfield snapshot (in phase/ or beside the movie)"
TAIL_SKIP_NO_RANGE = "no saved range"
TAIL_SKIP_NO_CELLS = "outlines with no cell"


def tail_skip_note(why: dict) -> str:
    """The skipped movies counted by reason, for the window: ``"2 skipped: 1 no outlines,
    1 no saved range"``; empty when none was."""
    if not why:
        return ""
    from collections import Counter

    counts = Counter(why.values())
    parts = [f"{counts[r]} {r}" for r in (TAIL_SKIP_NO_OUTLINES, TAIL_SKIP_NO_SNAPSHOT,
                                          TAIL_SKIP_NO_RANGE, TAIL_SKIP_NO_CELLS) if counts[r]]
    return f"{len(why)} skipped: " + ", ".join(parts)


# ---------------------------------------------------------------- u-track
#: What u-track costs per frame on this machine: 185 s for 1501 frames and 154 s for 1001
#: (two lab movies, 12 Oct 2026, R2016a), so about 0.13 s a frame,
#: plus ~10 s for MATLAB to start. Shown BEFORE a run, like TrackMate's 30 s a movie.
UTRACK_SECONDS_PER_FRAME = 0.13
UTRACK_STARTUP_SECONDS = 10.0


def set_utrack(settings, key: str, text: str):
    """A new `utrack_run.UTrackSettings` with ``key`` set from what was typed.

    Empty returns the field to the lab standard, except the three that are OFF when empty
    (the PSF sigma, which u-track then derives from the optics; the background test; the
    resolution limit). A pair is typed ``0.5, 0.01``; a switch ``1``/``0``. Raises
    `PrepError` naming the field when the value is not one u-track takes."""
    from dataclasses import fields, replace

    from .utrack_run import LAB_STANDARD, UTrackError

    kinds = {f.name: f for f in fields(settings)}
    if key not in kinds:
        raise PrepError(f"unknown u-track setting {key!r}")
    text = (text or "").strip()
    optional = {"psf_sigma_px", "background_alpha", "res_limit_px"}
    default = getattr(LAB_STANDARD, key)
    if not text:
        value = None if key in optional else default
    else:
        try:
            if isinstance(default, bool):
                value = text.lower() in ("1", "true", "on", "yes")
            elif isinstance(default, tuple):
                parts = [float(v) for v in text.replace(";", " ").replace(",", " ").split()]
                if len(parts) != 2:
                    raise ValueError(text)
                value = tuple(parts)
            elif isinstance(default, int) and key != "strategy_bd":
                number = float(text.replace(",", "."))
                if number != int(number):
                    raise ValueError(text)
                value = int(number)
            else:
                value = float(text.replace(",", "."))
        except ValueError as exc:
            raise PrepError(f"{key.replace('_', ' ')}: {text!r} is not a value u-track "
                            f"takes") from exc
    try:
        return replace(settings, **{key: value})
    except UTrackError as exc:
        raise PrepError(str(exc).replace("u-track settings: ", "")) from exc


def utrack_psf_preview(state) -> float | None:
    """The PSF sigma u-track will derive for ``state``, in pixels, or None when the NA,
    emission wavelength or pixel size is missing. u-track computes it itself
    (`Channel.psfSigma_`); this is `optics.psf_sigma_nm`, which reproduced u-track's own
    value to 1.0013x on 7 of 7 projects -- shown so the number is seen before a run."""
    from .optics import OpticsError, psf_sigma_nm

    cal = state.calibration
    try:
        sigma = psf_sigma_nm(emission_nm=cal.get("emission_nm"),
                             numerical_aperture=cal.get("numerical_aperture"))
    except OpticsError:
        return None
    pixel = cal.get("pixel_size_nm")
    return float(sigma) / float(pixel) if pixel else None


def utrack_earlier(state) -> int:
    """How many of ``state``'s movies already hold a u-track run (``tif/<movie>/
    movieData.mat``): a second run lands on the first, so the caller must have asked."""
    from .utrack_run import earlier_runs

    return len(earlier_runs(state.folder, state.names))


@dataclass(frozen=True)
class UTrackInputs:
    """One condition's u-track run, resolved FROM THE SAVED RECORD, as `RunInputs` is."""

    condition: str
    folder: Path
    movies: tuple
    names: dict
    crops: dict
    settings: object
    pixel_size_nm: float
    frame_interval_s: float
    numerical_aperture: float | None
    emission_nm: float | None
    earlier: int

    @property
    def frames(self) -> int:
        return sum(int(b) - int(a) + 1 for a, b in self.crops.values())

    @property
    def minutes(self) -> float:
        return (UTRACK_STARTUP_SECONDS + self.frames * UTRACK_SECONDS_PER_FRAME) / 60.0


def utrack_inputs(state, settings) -> UTrackInputs:
    """What ``state``'s u-track run would be given, or `PrepError` saying why it cannot
    start -- in the order a person fixes them, as `run_inputs` refuses a TrackMate run.

    Beyond TrackMate's refusals: the pixel size and frame interval are REQUIRED (they go
    into movieData.mat, and every D scales with them); u-track derives the PSF from the NA
    and emission wavelength, so without both a sigma must be typed; and the background
    test needs outlines for every movie. Nothing here touches the disk but a listing."""
    if not state.movies:
        raise PrepError(f"{state.name}: no acquisitions")
    if state.unsaved:
        raise PrepError(f"{state.name}: ranges or calibration edited since the last save. "
                        f"Press Save on Calibration & save first; crops.json records what is tracked.")
    if not state.saved:
        raise PrepError(f"{state.name}: no saved crop ranges; set them on Movies & crop "
                        f"and press Save")
    absent = [stem for stem in state.names if stem not in state.saved]
    if absent:
        raise PrepError(f"{state.name}: {len(absent)} movie(s) have no saved range: "
                        + ", ".join(absent[:4]) + (" ..." if len(absent) > 4 else ""))
    cal = state.saved_calibration
    missing = [label for key, label in (("pixel_size_nm", "pixel size"),
                                        ("frame_interval_s", "frame interval"))
               if not cal.get(key)]
    if missing:
        raise PrepError(f"{state.name}: no {' or '.join(missing)} saved; u-track writes "
                        f"both into movieData.mat, and neither has a default")
    if settings.psf_sigma_px is None and not (cal.get("numerical_aperture")
                                              and cal.get("emission_nm")):
        raise PrepError(f"{state.name}: u-track derives the PSF from the NA and emission "
                        f"wavelength; save both on Calibration & save, or type a PSF sigma")
    if settings.background_alpha is not None:
        outlines = state.outlines
        if outlines is None or not outlines.complete:
            have = 0 if outlines is None else outlines.n_with_outlines
            # SAY WHERE IT IS SWITCHED ON AND HOW TO TRACK WITHOUT IT (23 Oct 2026: "why cannot i
            # track without cell meshes?" -- the test was on in remembered settings, off in the lab's)
            raise PrepError(f"{state.name}: the cell-free background test is on (alpha "
                            f"{settings.background_alpha:g}, u-track card) and needs cell outlines for "
                            f"every movie ({have} of {len(state.movies)} have them). Segment them, "
                            f"or clear that field (off) to track without it")
    return UTrackInputs(
        condition=state.name, folder=Path(state.folder), movies=tuple(state.movies),
        names={Path(p).stem: n for p, n in zip(state.movies, state.names)},
        crops={stem: tuple(state.saved[stem]) for stem in state.names},
        settings=settings, pixel_size_nm=float(cal["pixel_size_nm"]),
        frame_interval_s=float(cal["frame_interval_s"]),
        numerical_aperture=cal.get("numerical_aperture") or None,
        emission_nm=cal.get("emission_nm") or None, earlier=utrack_earlier(state))


def utrack_background_masks(inputs: UTrackInputs) -> dict:
    """``{movie: cell mask}`` for the background test: every outline of the movie,
    dilated by 2 px (a cell's own halo is not background), on the acquisition's grid.
    Raises `PrepError` naming a movie with no usable outline."""
    from .autofluorescence import AutofluorescenceError, cell_masks
    from .cellmesh import CellMeshError, mesh_folder, read_cell_meshes
    from .rawmovie import read_raw_movie

    out = {}
    for path in inputs.movies:
        name = inputs.names[Path(path).stem]
        mesh_file = mesh_folder(inputs.folder) / f"{name}.mat"
        try:
            meshes = read_cell_meshes(mesh_file, movie=name, on_empty="empty")
            meta = read_raw_movie(path).meta
            _cells, outside = cell_masks(meshes, (int(meta.height), int(meta.width)))
        except (CellMeshError, AutofluorescenceError) as exc:
            raise PrepError(f"{inputs.condition}: {name}: no usable outline for the "
                            f"background test ({exc})") from exc
        out[name] = ~outside
    return out


#: A movie whose MEASURED frame interval differs from its condition's by more than this
#: (relative) is tracked at its own: one user's two settings, 50.48 and 51.75 ms, differ by
#: 2.5 %, while movies of one setting jitter by 0.02 %. Below it the condition's value
#: stands (an error under 1 % on that movie's D), so a condition of one setting keeps
#: ONE tau and `AcquisitionParams.tau_s` keeps answering.
OWN_INTERVAL_ABOVE = 0.01


def movie_intervals(movies, names: dict, condition_interval_s: float, *,
                    above: float = OWN_INTERVAL_ABOVE) -> dict:
    """``{movie name: seconds}`` for the movies whose acquisition MEASURES an interval
    more than ``above`` away from the condition's (`movieprep.detect_calibration`). A movie
    that records none keeps the condition's value."""
    from .movieprep import detect_calibration

    own = {}
    for path in movies:
        measured = (detect_calibration(path) or {}).get("frame_interval_s")
        if measured and abs(measured / condition_interval_s - 1.0) > above:
            own[names.get(Path(path).stem, Path(path).stem)] = float(measured)
    return own


def track_condition_utrack(inputs: UTrackInputs, *, matlab, utrack, replace: bool = False,
                           on_progress=None, stop=None, work_folder=None, at_once: int = 1) -> dict:
    """Track one condition with u-track and say what came out, in `track_condition`'s
    shape (``condition, record, outcome, verdict, summary, track_files``), so the Track
    page shows either tracker's run the same way.

    The tracks are READ BACK from what u-track wrote (`ingest.read_utrack_mat`), not
    trusted from MATLAB's count: u-track's count is compound tracks, the analysis keeps
    segments of at least five localisations. ``on_progress(index, total, row)`` per movie;
    ``stop()`` is asked while MATLAB runs and before each movie."""
    import pandas as pd

    from .ingest import EmptyTrackFileError, read_utrack_mat
    from .trackmate_run import track_counts
    from .utrack_run import STOPPED_ERROR, UTrackError, record_run, run_batch

    masks = (utrack_background_masks(inputs)
             if inputs.settings.background_alpha is not None else None)
    own = movie_intervals(inputs.movies, inputs.names, inputs.frame_interval_s)
    total = len(inputs.movies)
    seen = []

    def progress(row):
        seen.append(row["movie"])
        if on_progress is not None:
            on_progress(len(seen), total, {k: row.get(k) for k in
                                           ("movie", "ok", "error", "n_tracks", "n_frames",
                                            "seconds")})
    try:
        table = run_batch(list(inputs.movies), inputs.folder, settings=inputs.settings,
                          matlab=matlab, utrack=utrack, crops=inputs.crops,
                          pixel_size_nm=inputs.pixel_size_nm,
                          frame_interval_s=inputs.frame_interval_s,
                          numerical_aperture=inputs.numerical_aperture,
                          emission_nm=inputs.emission_nm, names=inputs.names,
                          background_masks=masks, replace=replace, on_progress=progress,
                          stop=stop, movie_intervals=own, work_folder=work_folder or None,
                          at_once=at_once)
    except UTrackError as exc:
        raise PrepError(f"{inputs.condition}: {exc}") from exc
    parts, empty = [], []
    for row in table.itertuples():
        if row.ok and row.mat:
            try:
                parts.append(read_utrack_mat(row.mat, movie=row.movie))
            except EmptyTrackFileError:
                # u-track RAN and linked nothing: data, not a failure (14 Oct 2026: one such
                # movie ended an overnight batch of 28 conditions here). Counted and named,
                # as the loader does (`grouping`'s ``empty_movies``), never silent.
                empty.append(str(row.movie))
    raw = pd.concat(parts, ignore_index=True) if parts else None
    tracks, usable = track_counts(raw)
    stopped = int((table["error"] == STOPPED_ERROR).sum())
    failed = int((~table["ok"]).sum()) - stopped
    outcome = {"movies": int(len(table)), "failed": failed, "stopped": stopped,
               "tracks": tracks, "usable_tracks": usable, "no_tracks": len(empty)}
    path = record_run(inputs.folder, settings=inputs.settings, matlab=matlab, utrack=utrack,
                      movies=list(inputs.names.values()), crops=inputs.crops,
                      pixel_size_nm=inputs.pixel_size_nm,
                      frame_interval_s=inputs.frame_interval_s,
                      numerical_aperture=inputs.numerical_aperture,
                      emission_nm=inputs.emission_nm, outcome=outcome,
                      movie_intervals=own, work_folder=work_folder or None, at_once=at_once)
    bad = table[~table["ok"]]
    verdict_lines = []
    if stopped:
        verdict_lines.append(f"Stopped: {stopped} movie(s) not run; this condition is not "
                             f"fully tracked.")
    verdict_lines += [f"Failed {r.movie}: {r.error}" for r in bad.itertuples()
                      if r.error != STOPPED_ERROR]
    verdict_lines.append(f"u-track tracked {int(table['ok'].sum())} of {len(table)} "
                         f"movie(s).")
    if empty:
        verdict_lines.append(
            f"No tracks in {len(empty)} movie(s): {', '.join(sorted(empty))}. u-track found "
            f"nothing to link there (check focus and the field); the analysis skips them.")
    if own:
        verdict_lines.append(
            f"{len(own)} movie(s) recorded at their own frame interval "
            f"({', '.join(sorted({f'{v * 1000:.2f}' for v in own.values()}))} ms against "
            f"the condition's {inputs.frame_interval_s * 1000:.2f}): "
            + ", ".join(sorted(own)[:6]) + (" ..." if len(own) > 6 else ""))
    summary = (f"{tracks:,} track segments, {usable:,} of at least 5 localisations (the "
               f"ones the analysis keeps); {inputs.settings.stamp()}")
    return {"condition": inputs.condition, "record": str(path), "outcome": outcome,
            "verdict": "\n".join(verdict_lines), "summary": summary,
            "track_files": utrack_earlier_folder(inputs)}


def utrack_earlier_folder(inputs: UTrackInputs) -> int:
    from .utrack_run import earlier_runs

    return len(earlier_runs(inputs.folder, list(inputs.names.values())))


# ---------------------------------------------------------------- the Next button
#: Each preparation step and where its Next button goes: ``(target, title)``, the target
#: None for the last (the button then closes the window back to the menu).
NEXT_STEPS = {"movies": ("calibration", "Calibration & save"),
              "calibration": ("detect", "Detect & link"),
              "detect": ("track", "Track"),
              "track": (None, "Done, back to the menu")}


def next_step(section: str, states: dict, *, tracker: str = "trackmate",
              utrack_settings=None, busy: str = "", finished=()) -> dict | None:
    """Whether ``section``'s work is done, for its Next button: ``{target, label, ready,
    why}``, or None for a page that has no Next button (Batch, which is not a step of one
    folder's work). Asked for (13 Oct 2026): "a next button ... when the current screen is
    ready and done". Only a listing is touched (the run checks read nothing else).

    - Movies & crop: every movie read (or refused) and every movie with a range;
    - Calibration & save: every condition saved, nothing edited since;
    - Detect & link: every condition would start a run with the chosen tracker;
    - Track: nothing running, and every condition tracked (this session or earlier).
    """
    if section not in NEXT_STEPS:
        return None
    target, title = NEXT_STEPS[section]
    step = {"target": target, "label": f"Next: {title}" if target else title}
    why = ""
    if not states:
        why = "open a folder first"
    elif section == "movies":
        unread = sum(len(s.unread) for s in states.values())
        no_range = sum(max(c.n_movies - c.n_crops, 0)
                       for c in (s.counts() for s in states.values()))
        if unread:
            why = f"{unread} movie(s) still being read"
        elif no_range:
            why = (f"{no_range} movie(s) have no range yet: propose them, or set them on "
                   f"the chart")
    elif section == "calibration":
        waiting = [n for n, s in states.items()
                   if not s.saved or s.unsaved or any(m not in s.saved for m in s.names)]
        if waiting:
            why = (f"{len(waiting)} condition(s) not saved yet: "
                   + ", ".join(waiting[:3]) + (" ..." if len(waiting) > 3 else ""))
    elif section == "detect":
        blocked = []
        for name, state in states.items():
            try:
                if tracker == "utrack":
                    utrack_inputs(state, utrack_settings)
                else:
                    run_inputs(state)
            except PrepError as exc:
                blocked.append(str(exc))
        if blocked:
            why = f"{len(blocked)} condition(s) cannot be tracked yet: {blocked[0]}"
    elif section == "track":
        if busy:
            why = busy
        else:
            undone = [n for n, s in states.items() if n not in set(finished)
                      and not (utrack_earlier(s) if tracker == "utrack" else s.earlier_run)]
            if undone:
                why = (f"{len(undone)} condition(s) not tracked yet: "
                       + ", ".join(undone[:3]) + (" ..." if len(undone) > 3 else ""))
    step.update(ready=not why, why=why)
    return step

