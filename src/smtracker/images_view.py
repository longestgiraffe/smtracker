"""THE IMAGES & TRACKS SECTION's headless side: one movie's picture, its cell outlines
and its tracks, as plain data.

Asked for: "the start screen in matlab had like something where u can cycle through the
images, track view etc." -- v2 opened on a movie's phase image with its outlines and tracks
and stepped through the movies. This is that, in the workspace, plus a cell to zoom to and
an Export for a representative-cell panel (`figure_export`, kind "cell_image").

PLAIN PAYLOADS, so `ui.workspace.images_page` paints with no numpy: the image is 8-bit grey
BYTES (row-major, `width` x `height`), every outline and track a list of ``[x, y]`` points.
COORDINATES ARE PIXELS, 0-BASED, x right and y down -- the canonical table and the outlines
are 1-based MATLAB pixels, and `viewer.PIXEL_ORIGIN_OFFSET` is subtracted here, once, as the
napari bridge does.

NOTHING IS DRAWN ON A BLANK CANVAS (this project's rule): a movie whose image is missing
comes back with ``image`` None and the reason, and the page shows the reason instead of
tracks floating over nothing.

Colours are decided here, not in Qt: apparent D (log, the condition's own 2-98 % range, so
one colour means one D in every movie of the condition), each track its own hue, or one
colour. Apparent D is `qc.track_summary`'s per-track mean squared step / 4 tau -- an
APPARENT value averaged over any switching inside the track.

THE SAMPLE SIZE (the section's Overview). Asked for, looking at one user's d2: "most cells here
have no tracks or less? ... we also need a list of how many cells, replicates, tracks per
cell (per 1000 frames or something)". `inventory` counts every OUTLINED cell of every
condition -- a cell that yielded no track included, which the canonical table cannot do
because such a cell has no row in it -- and watches each for its movie's TRACKED frames
(`project.movie_frames`, the tracker's own record); `qc.sample_size` does the arithmetic,
per condition, per replicate day and per movie. The side card lists a movie's cells, or a
cell's tracks, as table rows (`cell_rows`, `track_rows`).
"""
from __future__ import annotations

import math
from pathlib import Path

from .projectcache import ProjectCache

__all__ = ["BACKGROUNDS", "COLOUR_MODES", "COLOUR_LABELS", "ImagesError",
           "condition_movies", "movie_payload", "cell_box", "cell_summary", "track_ink",
           "scale_bar_um", "movie_position", "cell_position", "inventory",
           "INVENTORY_COLUMNS", "inventory_rows", "inventory_caption", "cell_rows",
           "track_rows", "CELL_COLUMNS", "TRACK_COLUMNS", "nearest_track", "movie_line",
           "inventory_records", "write_inventory_csv", "first_movie_of_day", "LEVELS",
           "INVENTORY_PRINT_COLUMNS", "inventory_table",
           "LEVEL_LABELS", "ADJUST_RANGE", "display_lut", "grey_table", "display_words",
           "project_uid", "SPEEDS", "SPEED_LABELS", "TRAIL_FRAMES", "TICK_MS",
           "frame_view", "time_text", "track_start", "frame_block", "tracks_at",
           "BACKGROUND_TIPS", "BACKGROUND_SHORT", "movie_root", "LUTS", "LUT_LABELS",
           "lut_colours", "display_table", "MovieDisplay", "movie_levels", "movie_display_words",
           "MOVIE_FILTER_LABELS", "MOVIE_FILTER_TIPS", "SMOOTH_PX"]

#: THE MOVIE (asked 14 Oct 2026: "the test map screen from the UI designer is playing the actual movie
#: together with the track. implement in images&tracks like this as well next to phase and
#: sum"): the recorded frames under the tracks, each frame on its own frame. Where they come
#: from and how they are checked is `trackclip` (the stack the tracker was given, u-track's
#: frames, or the raw acquisition with the range tracked); at rest the first frame tracked.
BACKGROUNDS = ("phase", "sum", "movie")
BACKGROUND_LABELS = {"phase": "Phase", "sum": "Fluorescence sum", "movie": "Movie"}
#: The segmented control's words (the options grid must fit a 1280 px Windows window at
#: 150 %); the hover says each in full.
BACKGROUND_SHORT = {"phase": "Phase", "sum": "Sum", "movie": "Movie"}
BACKGROUND_TIPS = {"phase": "The phase-contrast image",
                   "sum": "The fluorescence sum image",
                   "movie": "The recorded frames: press play to watch the molecules under "
                            "their tracks"}
#: THE EXPERIMENTER'S OWN CHANNELS (17 Oct 2026: "the different fluor channels can be added and named and folder
#: direction can be given for them"): a channel is ``{"name", "folder"}``, shown as one more background keyed
#: ``ch:<name>``. A RELATIVE folder is looked up beside each movie, as ``phase/`` and ``sum/`` are (each condition
#: and each day folder its own); an ABSOLUTE one is used as it is. A movie's image is ``<movie>.tif`` / ``.tiff``
#: there, or a snapshot of that field (``series001.nd2.tif``, `segmentation.channels.snapshot_field`). An image
#: whose size is not the movie's (its phase or sum image) is REFUSED, never drawn under tracks it does not match.
CHANNEL_PREFIX = "ch:"


def channel_key(name: str) -> str:
    return CHANNEL_PREFIX + str(name)


def channel_name(key: str) -> str | None:
    """The channel a background key names, or None for a built-in background."""
    return key[len(CHANNEL_PREFIX):] if str(key).startswith(CHANNEL_PREFIX) else None


def background_keys(channels=None) -> tuple:
    """Every background on offer: the built-in three, then each channel."""
    return BACKGROUNDS + tuple(channel_key(n) for n in (channels or {}))


def background_label(key: str) -> str:
    """A background's name in words: "Phase", "Fluorescence sum", "Movie", or the channel's own."""
    return BACKGROUND_LABELS.get(key) or channel_name(key) or str(key)


def background_short(key: str) -> str:
    return BACKGROUND_SHORT.get(key) or channel_name(key) or str(key)


def background_tip(key: str, channels=None) -> str:
    name = channel_name(key)
    if name is None:
        return BACKGROUND_TIPS[key]
    folder = (channels or {}).get(name, "")
    return f"The {name} image, from {folder}" + ("" if Path(folder).is_absolute() else " beside each movie")


def channel_image_path(root, movie: str, folder: str):
    """The channel image of ``movie``: ``<folder>/<movie>.tif`` (or ``.tiff``, or a snapshot of that field), the
    folder beside the movie when relative. None when there is none."""
    from .dayfolders import on_disk
    from .segmentation.channels import snapshot_field

    root, movie = on_disk(root, movie)
    base = Path(folder) if Path(folder).is_absolute() else Path(root) / folder
    if not base.is_dir():
        return None
    for suffix in (".tif", ".tiff"):
        candidate = base / f"{movie}{suffix}"
        if candidate.is_file():
            return candidate
    for candidate in sorted(base.iterdir()):
        if candidate.suffix.lower() in (".tif", ".tiff") and snapshot_field(candidate) == movie:
            return candidate
    return None


def channels_problem(channels) -> str | None:
    """What is wrong with a list of ``{"name", "folder"}``, in words, or None."""
    names = []
    for i, ch in enumerate(channels or [], start=1):
        name, folder = str(ch.get("name", "")).strip(), str(ch.get("folder", "")).strip()
        if not name:
            return f"channel {i} has no name"
        if not folder:
            return f"{name} has no folder"
        if name.lower() in {v.lower() for v in (*BACKGROUND_LABELS.values(), *BACKGROUND_SHORT.values())}:
            return f"{name!r} is a built-in background; give the channel another name"
        if name.lower() in names:
            return f"two channels are called {name!r}"
        names.append(name.lower())
    return None


def channel_folder_for(picked, roots) -> str:
    """How a folder picked in the dialog is kept: RELATIVE to the condition folder it lies in (so every condition
    looks in its own folder of that name, as phase/ and sum/), else the absolute path."""
    picked = Path(picked).resolve()
    for root in roots:
        try:
            return picked.relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            continue
    return str(picked)


def channel_map(channels) -> dict:
    """``{name: folder}`` from a list of ``{"name", "folder"}``, in order."""
    return {str(c["name"]).strip(): str(c["folder"]).strip() for c in channels or []}


COLOUR_MODES = ("appd", "track", "plain")
#: Short, so the options grid fits a 1280 px Windows window (150 % scaling); the hover says
#: what each means (`COLOUR_TIPS`).
COLOUR_LABELS = {"appd": "Apparent D", "track": "Per track", "plain": "Single"}
COLOUR_TIPS = {"appd": "Each track coloured by its apparent D, slow blue to fast red",
               "track": "Every track its own colour",
               "plain": "Every track in one colour"}
#: Slow to fast: blue, yellow, red -- reads on a grey image, unlike a ramp through grey.
APPD_RAMP = ("#2f6bff", "#22c3e6", "#ffd23f", "#ff7a2f", "#e8203a")
#: The napari viewer's own inks: outlines yellow, a single track magenta.
OUTLINE_INK = "#ffe14d"
PLAIN_INK = "#ff3fb4"
#: Padding around a cell's outline when it is zoomed to, in pixels.
CELL_PAD_PX = 4

#: Per project OBJECT, never per id (`projectcache`: an id is reused once a project is freed).
_CACHE = ProjectCache()


class ImagesError(ValueError):
    """A condition or movie that cannot be shown, named."""


def _hex(rgb) -> str:
    return "#" + "".join(f"{round(max(0, min(255, c))):02x}" for c in rgb)


def _rgb(value: str) -> tuple:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def ramp_colour(f: float, ramp=APPD_RAMP) -> str:
    """The ramp's colour at ``f`` in [0, 1], linear between its stops."""
    f = min(1.0, max(0.0, float(f)))
    x = f * (len(ramp) - 1)
    i = min(len(ramp) - 2, int(x))
    a, b = _rgb(ramp[i]), _rgb(ramp[i + 1])
    t = x - i
    return _hex(tuple(a[k] + (b[k] - a[k]) * t for k in range(3)))


def hue_colour(index: int) -> str:
    """Track ``index``'s own colour: golden-angle hues, bright enough for a grey image."""
    import colorsys

    h = (index * 0.618033988749895) % 1.0
    return _hex(tuple(255 * c for c in colorsys.hsv_to_rgb(h, 0.75, 1.0)))


def track_ink(mode: str, *, appd, index: int, appd_range) -> str:
    """The colour one track is drawn in under ``mode`` (one rule for the page and print)."""
    if mode not in COLOUR_MODES:
        raise ImagesError(f"colour must be one of {COLOUR_MODES}, got {mode!r}")
    if mode == "plain":
        return PLAIN_INK
    if mode == "track":
        return hue_colour(index)
    lo, hi = appd_range
    if appd is None or not appd > 0 or not hi > lo:
        return "#bdbdbd"
    f = (math.log10(appd) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    return ramp_colour(f)


# -- one condition, read once -------------------------------------------------------- #
def _condition(project, condition: str) -> dict:
    """The condition's tracks and outlines under ORIGINAL movie names, its per-track
    apparent D, the D range the colours span, and where each movie's images are."""
    key = (condition, "condition")
    if (project, key) in _CACHE:
        return _CACHE.get(project, key)
    import numpy as np

    from .qc import track_summary
    from .trackviewer import _original_names

    if condition not in list(project.conditions):
        raise ImagesError(f"no condition {condition!r} in this project")
    canonical, meshes = _original_names(project, condition)
    if canonical.empty:
        raise ImagesError(f"condition {condition!r} has no tracks")
    summary = track_summary(canonical)
    appd = {str(u): (float(d) if np.isfinite(d) else None)
            for u, d in zip(summary["uid"], summary["appd_um2_per_s"])}
    finite = np.array([d for d in appd.values() if d is not None and d > 0])
    appd_range = ((float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))
                  if finite.size >= 2 else (0.01, 1.0))
    # Each movie's folder from THIS condition's own movies: two conditions can both hold a
    # "series001", so a map over every source would hand one condition the other's image.
    from pathlib import Path

    source = getattr(project, "movie_source", None) or {}
    roots = {}
    for qualified in project.condition(condition)["movie"].astype(str).unique():
        if qualified in source:
            root, original = source[qualified]
            roots[str(original)] = root
    fallback = project.roots[condition]
    # Every movie the TRACKER produced output for, kept tracks or not: a movie whose tracks
    # the cell filter removed entirely has no row in the table but was tracked all the same.
    here = Path(fallback).resolve()
    tracked = {str(original) for root, original in source.values()
               if Path(root).resolve() == here}
    movies = sorted(set(canonical["movie"].astype(str)) | set(meshes))
    out = {"canonical": canonical, "meshes": meshes, "appd": appd, "appd_range": appd_range,
           "movies": movies, "roots": {m: roots.get(m, fallback) for m in movies},
           "params": canonical.attrs["params"], "tracked": tracked}
    return _CACHE.put(project, key, out)


def forget(project=None) -> None:
    """Drop the cached conditions (all, or one project's) -- after a re-load."""
    _CACHE.forget(project)


def condition_movies(project, condition: str) -> list:
    """The movies a condition holds, by their own names, in order."""
    return list(_condition(project, condition)["movies"])


def _image_bytes(root, movie: str, kind: str, channels=None) -> tuple:
    """``(width, height, bytes, error)``: the image as 8-bit grey, its 0.5-99.5 % range
    stretched -- or the reason there is none."""
    import numpy as np

    from .viewer import ViewerError, background_image

    name = channel_name(kind)
    try:
        if name is None:
            image = np.asarray(background_image(root, movie, kind=kind), dtype=float)
        else:
            image = _channel_image(root, movie, name, (channels or {}).get(name))
    except ViewerError as exc:
        return 0, 0, b"", str(exc)
    while image.ndim > 2:                 # a stack: its first plane, as the viewer shows
        image = image[0]
    lo, hi = np.percentile(image, [0.5, 99.5])
    scaled = np.clip((image - lo) / ((hi - lo) or 1.0), 0.0, 1.0)
    grey = np.ascontiguousarray((scaled * 255.0 + 0.5).astype(np.uint8))
    return int(grey.shape[1]), int(grey.shape[0]), grey.tobytes(), None


def _channel_image(root, movie: str, name: str, folder):
    """A channel's image of ``movie`` as floats, refused when missing or not the movie's size."""
    import numpy as np

    from .viewer import ViewerError, background_image

    if not folder:
        raise ViewerError(f"no channel called {name!r}")
    path = channel_image_path(root, movie, folder)
    if path is None:
        raise ViewerError(f"no {name} image of {movie} in {folder}")
    import tifffile

    image = np.asarray(tifffile.imread(path), dtype=float)
    plane = image
    while plane.ndim > 2:
        plane = plane[0]
    for kind in ("phase", "sum"):
        try:
            reference = np.asarray(background_image(root, movie, kind=kind))
        except ViewerError:
            continue
        while reference.ndim > 2:
            reference = reference[0]
        if reference.shape != plane.shape:
            raise ViewerError(
                f"the {name} image of {movie} is {plane.shape[1]} x {plane.shape[0]} px but its {kind} image is "
                f"{reference.shape[1]} x {reference.shape[0]}, so it is not the same field and the tracks would be "
                "drawn in the wrong place")
        break
    return image


def _available(root, movie: str, channels=None) -> list:
    from pathlib import Path

    from .dayfolders import on_disk
    from .trackclip import has_frames

    mine = [channel_key(n) for n, folder in (channels or {}).items()
            if channel_image_path(root, movie, folder) is not None]
    root, movie = on_disk(root, movie)          # a day folder's movie where it is

    # ASKED OF THE SAME RESOLVER THE IMAGE COMES FROM. Stat-ing
    # `root / k / f"{movie}.tif"` reported no Phase background at all wherever the
    # snapshots lie loose in the condition folder, and missed a filed
    # `phase/series001.nd2.tif` too -- so the window opened on Phase (its default) and
    # drew the "no image" text over a blank canvas. `viewer._background_path` knows both
    # layouts and both namings; asking anything else here would be a second opinion about
    # which backgrounds exist, and the two would disagree on exactly these folders.
    from .viewer import _background_path

    out = [k for k in BACKGROUNDS
           if k != "movie" and _background_path(root, movie, k) is not None]
    if has_frames(root, movie):
        out.append("movie")
    return out + mine


def movie_root(project, condition: str, movie: str):
    """The folder a movie of ``condition`` was read from (its own: a condition may hold
    movies from several folders)."""
    return _condition(project, condition)["roots"][movie]


def _spots(tracks) -> list:
    """Every localisation drawn, as ``(frame, x, y)`` in 0-based pixels."""
    return [(f, x, y) for t in tracks for (x, y), f in zip(t["points"], t["frames"])]


def _recorded(project, condition: str, movie: str, tracks):
    """The movie's recorded frames (`trackclip.open_movie`, checked against its own
    localisations) and one display stretch for all of them, read once per movie."""
    key = (condition, "frames", movie)
    if (project, key) in _CACHE:
        return _CACHE.get(project, key)
    from .trackclip import movie_stretch, open_movie

    data = _condition(project, condition)
    recorded = _movie_frames(project, condition).get(movie)
    frames = open_movie(data["roots"][movie], movie,
                        expected_frames=int(recorded[0]) if recorded else None,
                        spots=_spots(tracks))
    return _CACHE.put(project, key, (frames, movie_stretch(frames)))


def _recorded_frames(project, condition: str, movie: str) -> tuple:
    """``(frames, stretch)`` of the movie's recorded frames, read once (`_recorded`)."""
    key = (condition, "frames", movie)
    if (project, key) not in _CACHE:
        payload = movie_payload(project, condition, movie, background="movie")
        if payload["image"] is None:
            raise ImagesError(payload["image_error"])
    return _CACHE.get(project, key)


def frame_block(project, condition: str, movie: str, lo: int, hi: int, display=None) -> dict:
    """Frames ``lo``..``hi`` of the movie (the tracker's numbering) as 8-bit grey under
    the movie's one stretch -- or, with ``display`` (`MovieDisplay`), filtered and mapped
    from its black and white in raw counts: ``{first, last, width, height, frames: [bytes,
    ...]}``. The Movie background's playback reads these a block at a time on a worker."""
    from .trackclip import grey_frames

    frames, stretch = _recorded_frames(project, condition, movie)
    if display is not None:
        stretch = tuple(movie_levels(project, condition, movie, display)["auto"])
    lo, hi = max(1, int(lo)), min(int(hi), frames.n_frames)
    return {"first": lo, "last": hi, "width": frames.width, "height": frames.height,
            "frames": grey_frames(frames, lo, hi, stretch, display) if hi >= lo else []}


# -- the Movie background's display: black and white in raw counts, and two filters --------- #
def _movie_display():
    from .trackclip import FrameDisplay

    return FrameDisplay


#: `trackclip.FrameDisplay`: how the Movie background shows its frames (display only).
def MovieDisplay(**kwargs):                                    # noqa: N802 - a constructor
    """A `trackclip.FrameDisplay` (black / white in raw counts, None = Auto; per_frame;
    smooth_px; background "none" | "median" | "rolling")."""
    return _movie_display()(**kwargs)


MOVIE_FILTER_LABELS = {"none": "None", "median": "Running median", "rolling": "Rolling"}
MOVIE_FILTER_TIPS = {
    "none": "The frames as recorded",
    "median": "Subtract each pixel's running median over the neighbouring frames. Static signal (cell autofluorescence, "
              "haze, uneven illumination) is removed; moving and blinking spots stay",
    "rolling": "Subtract, per frame, the frame's background at a scale larger than a spot (Fiji's rolling-ball "
               "subtraction, with a flat square for the ball)"}
#: The smoothing a spot is seen at: the PSF's sigma, ~1 px on this microscope (0.932 px, `optics`).
SMOOTH_PX = 1.0


def movie_levels(project, condition: str, movie: str, display=None) -> dict:
    """What the black and white sliders run over for this movie as ``display`` filters it
    (`trackclip.frame_levels`: ``low``, ``high``, ``auto``, ``counts``), read once per
    movie and filter setting."""
    from .trackclip import frame_levels

    display = display or MovieDisplay()
    frames, _stretch = _recorded_frames(project, condition, movie)
    key = (condition, "levels", movie, display.filters, float(display.saturated_pct))
    if (project, key) in _CACHE:
        return _CACHE.get(project, key)
    return _CACHE.put(project, key, frame_levels(frames, display))


def movie_display_words(display, levels: dict | None = None, lut: str = "grey") -> str:
    """What the Movie background's display is, in words: the side card's tooltip and the
    figure legend."""
    if display.black is None and display.white is None:
        rng = (f"Auto per frame ({display.saturated_pct:g} % saturated)" if display.per_frame
               else f"Auto ({display.saturated_pct:g} % saturated)")
        if levels and not display.per_frame:
            rng += f", black {levels['auto'][0]:.0f} and white {levels['auto'][1]:.0f} counts"
    else:
        auto = (levels or {}).get("auto") or [None, None]
        black = display.black if display.black is not None else auto[0]
        white = display.white if display.white is not None else auto[1]
        rng = ("black " + (f"{black:.0f}" if black is not None else "Auto") + " and white "
               + (f"{white:.0f}" if white is not None else "Auto") + " counts")
    parts = [f"linear {_LUT_WORDS[lut]}, raw camera counts, {rng}"]
    if display.background == "median":
        parts.append(f"each pixel's running median over {display.median_frames} frames subtracted")
    elif display.background == "rolling":
        parts.append(f"each frame's background subtracted (grey opening, {2 * display.rolling_px + 1} px square)")
    if display.smooth_px > 0:
        parts.append(f"smoothed with a Gaussian of {display.smooth_px:g} px")
    return "; ".join(parts) + "; applied to the whole image, display only"


def movie_payload(project, condition: str, movie: str | None = None, *,
                  background: str = "phase", colour: str = "appd", channels: dict | None = None) -> dict:
    """Everything the page draws for one movie, as plain data (see the module head). ``channels``: the
    experimenter's own, ``{name: folder}`` (`CHANNEL_PREFIX`)."""
    from .viewer import PIXEL_ORIGIN_OFFSET

    channels = dict(channels or {})
    if background not in background_keys(channels):
        raise ImagesError(f"background must be one of {background_keys(channels)}, got {background!r}")
    if colour not in COLOUR_MODES:
        raise ImagesError(f"colour must be one of {COLOUR_MODES}, got {colour!r}")
    data = _condition(project, condition)
    movies = data["movies"]
    if not movies:
        raise ImagesError(f"condition {condition!r} has no movies")
    if movie is None:
        movie = movies[0]
    if movie not in movies:
        raise ImagesError(f"no movie {movie!r} in condition {condition!r}")
    root = data["roots"][movie]
    available = _available(root, movie, channels)
    if background != "movie":
        width, height, pixels, error = _image_bytes(root, movie, background, channels)
    params = data["params"]
    per_px = float(params.um_per_pixel)

    outlines = []
    for mesh in data["meshes"].get(movie, []):
        points = [[float(x) - PIXEL_ORIGIN_OFFSET, float(y) - PIXEL_ORIGIN_OFFSET]
                  for x, y in mesh.polygon]
        outlines.append({"cell": f"{movie}:{mesh.cell_id}", "label": f"cell {mesh.cell_id}",
                         "points": points, "n_tracks": 0, "box": _box(points)})
    by_cell = {o["cell"]: o for o in outlines}

    from .canonical import without_attrs

    # WITHOUT the table's .attrs: pandas deep-copies them into every per-track slice below,
    # which was 0.64 of 0.68 s a movie on protein A/Malate -- and the gallery reads every movie.
    table = without_attrs(data["canonical"])
    table = table[table["movie"].astype(str) == movie].sort_values(["uid", "frame"],
                                                                   kind="stable")
    tracks = []
    in_cells = 0
    for index, (uid, part) in enumerate(table.groupby("uid", sort=False, observed=True)):
        cell = part["cell"].iloc[0] if "cell" in part.columns else None
        cell = None if cell is None or cell != cell else str(cell)
        appd = data["appd"].get(str(uid))
        xs = (part["x_um"].to_numpy() / per_px - PIXEL_ORIGIN_OFFSET).tolist()
        ys = (part["y_um"].to_numpy() / per_px - PIXEL_ORIGIN_OFFSET).tolist()
        tracks.append({"uid": str(uid), "cell": cell,
                       "number": str(uid).rsplit(":", 1)[-1],
                       "points": [[float(x), float(y)] for x, y in zip(xs, ys)],
                       "frames": [int(f) for f in part["frame"]],
                       "appd": appd,
                       "colour": track_ink(colour, appd=appd, index=index,
                                           appd_range=data["appd_range"])})
        if cell is not None:
            in_cells += 1
            if cell in by_cell:
                by_cell[cell]["n_tracks"] += 1
    for outline in outlines:
        mine = [t for t in tracks if t["cell"] == outline["cell"]]
        outline["n_localizations"] = sum(len(t["points"]) for t in mine)
        outline["median_appd"] = _median([t["appd"] for t in mine if t["appd"] is not None])
    time_axis = _time_axis(project, condition, movie, table, params)
    source = None
    if background == "movie":
        from .trackclip import TrackClipError, grey_frames

        try:
            frames, stretch = _recorded(project, condition, movie, tracks)
            first = max(1, int(time_axis["first"]))
            width, height = frames.width, frames.height
            pixels, error = grey_frames(frames, first, first, stretch)[0], None
            source = frames.words
        except (TrackClipError, OSError, ValueError) as exc:
            width, height, pixels, error = 0, 0, b"", str(exc)
    lo, hi = data["appd_range"]
    legend = {"appd": f"apparent D, {lo:.3g} to {hi:.3g} µm²/s (log; this condition's "
                      "2-98 %), slow blue to fast red",
              "track": "each track its own colour",
              "plain": "every track one colour"}[colour]
    return {
        "condition": condition, "movie": movie, "movies": list(movies),
        "index": movies.index(movie), "background": background,
        "background_label": background_label(background),
        "backgrounds": available, "colour_mode": colour,
        "image": ({"width": width, "height": height, "pixels": pixels}
                  if error is None else None),
        "image_error": error,
        "outlines": outlines, "tracks": tracks,
        "appd_range": [float(lo), float(hi)], "ramp": list(APPD_RAMP),
        "outline_ink": OUTLINE_INK, "legend": legend,
        "pixel_size_nm": float(params.pixel_size_nm), "um_per_pixel": per_px,
        "counts": {"tracks": len(tracks), "in_cells": in_cells, "cells": len(outlines)},
        "frames": _movie_frames(project, condition).get(movie),
        "time": time_axis,
        "frames_source": source,
        "caption": (f"{movie} · {len(outlines)} cells · {len(tracks)} tracks, "
                    f"{in_cells} inside a cell outline · "
                    + (f"{background_label(background)} image" if background != "movie"
                       else f"the recorded movie, from {source}" if source
                       else "the recorded movie")
                    + f" · tracks: {legend}"),
    }


# -- brightness and contrast: how the 8-bit image is SHOWN, never what it holds ---------- #
#: Both sliders run over this, 0 being the automatic 0.5-99.5 % stretch.
ADJUST_RANGE = (-100, 100)
#: Contrast at either end multiplies or divides the slope by this.
CONTRAST_SPAN = 4.0


def display_lut(brightness: int = 0, contrast: int = 0) -> list:
    """The grey each of the image's 256 levels is shown at: a LINEAR map about mid-grey
    (contrast sets its slope, x1/4 to x4; brightness shifts it, up to half the range either
    way), clipped to 0-255. One table serves the page (a colour table on the image, so a
    slider costs nothing) and the exported figure, so what is printed is what was seen.

    Linear on purpose: a journal allows a brightness/contrast change applied to the whole
    image and stated, which is what this is; a gamma curve is a different claim."""
    lo, hi = ADJUST_RANGE
    if not (lo <= brightness <= hi and lo <= contrast <= hi):
        raise ImagesError(f"brightness and contrast run from {lo} to {hi}")
    slope = CONTRAST_SPAN ** (contrast / float(hi))
    shift = 127.5 * brightness / float(hi)
    return [int(round(min(255.0, max(0.0, (v - 127.5) * slope + 127.5 + shift))))
            for v in range(256)]


#: THE COLOUR TABLES (asked 14 Oct 2026: "a contrast and brightness picker (and color picker
#: maybe, LUT) would be cool for the movie player"). Display only, like brightness and
#: contrast: each of the 256 levels the sliders give is SHOWN in a colour, the levels
#: themselves untouched. Single hues as a fluorophore is shown (black to the colour); Hot and
#: Inferno, perceptual ramps that separate dim levels a grey table runs together.
LUTS = ("grey", "green", "magenta", "cyan", "hot", "inferno")
LUT_LABELS = {"grey": "Grey", "green": "Green", "magenta": "Magenta", "cyan": "Cyan",
              "hot": "Hot", "inferno": "Inferno"}
_LUT_WORDS = {"grey": "grey", "green": "black to green", "magenta": "black to magenta",
              "cyan": "black to cyan", "hot": "the Hot colour table",
              "inferno": "the Inferno colour table"}


def lut_colours(lut: str = "grey") -> list:
    """The ``(r, g, b)`` each of the 256 display levels is drawn in."""
    if lut not in LUTS:
        raise ImagesError(f"LUT must be one of {LUTS}, got {lut!r}")
    if lut == "grey":
        return [(v, v, v) for v in range(256)]
    if lut in ("green", "magenta", "cyan"):
        mask = {"green": (0, 1, 0), "magenta": (1, 0, 1), "cyan": (0, 1, 1)}[lut]
        return [(v * mask[0], v * mask[1], v * mask[2]) for v in range(256)]
    import numpy as np
    from matplotlib import colormaps

    rgba = colormaps[lut](np.linspace(0.0, 1.0, 256))
    return [tuple(int(round(255 * c)) for c in px[:3]) for px in rgba]


def display_table(brightness: int = 0, contrast: int = 0, lut: str = "grey") -> list:
    """`display_lut` through the colour table ``lut``, as opaque ARGB words: the form a Qt
    colour table takes, so the sliders and the LUT cost nothing on screen."""
    colours = lut_colours(lut)
    out = []
    for level in display_lut(brightness, contrast):
        r, g, b = colours[level]
        out.append(0xFF000000 | (r << 16) | (g << 8) | b)
    return out


def grey_table(brightness: int = 0, contrast: int = 0) -> list:
    """`display_lut` as opaque grey ARGB words, the form a Qt colour table takes."""
    return display_table(brightness, contrast, "grey")


def display_words(brightness: int = 0, contrast: int = 0, lut: str = "grey") -> str:
    """What the image's display is, in words, for the side card and the figure legend."""
    base = f"linear {_LUT_WORDS[lut]}, 0.5-99.5 % of the image's range"
    if not brightness and not contrast:
        return base
    parts = []
    if brightness:
        parts.append(f"brightness {brightness:+d}")
    if contrast:
        parts.append(f"contrast {contrast:+d} (x{CONTRAST_SPAN ** (contrast / 100.0):.2f})")
    return f"{base}, {', '.join(parts)}, applied to the whole image"


def scale_bar_um(width_um: float) -> float:
    """A round scale-bar length near an eighth of what is shown (page and print alike)."""
    for value in (0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        if value >= width_um / 8.0:
            return value
    return 100.0


def scale_bar_text(value_um: float) -> str:
    return f"{value_um:g} µm"


def ramp_ends(lo: float, hi: float) -> tuple:
    """The colour ramp's two end labels, by THE formatter of every colour bar's ends (`chart_text.scale_number`;
    ".3g" before)."""
    from .chart_text import scale_ends

    low, high = scale_ends(lo, hi)
    return low, f"{high} µm²/s"


def movie_position(payload: dict) -> str:
    """"series003 · 3 of 9"."""
    return (f"{payload['movie']} · {payload['index'] + 1} of {len(payload['movies'])}")


def movie_position_widest(payload: dict) -> str:
    """The longest `movie_position` over the condition's movies: the movie button keeps
    this width, so the arrows beside it never move from one movie to the next."""
    movies = payload["movies"]
    return max((f"{m} · {i + 1} of {len(movies)}" for i, m in enumerate(movies)),
               key=len, default="")


# -- the movie gallery: every movie of a condition as a picture --------------------------- #
def gallery_title(condition_label: str, n_movies: int) -> tuple:
    """The gallery's heading and the count beside it."""
    return f"Movies in {condition_label}", f"{n_movies} movie{'' if n_movies == 1 else 's'}"


def thumbnail_caption(counts: dict) -> str:
    """"5 cells · 187 tracks" under a gallery tile."""
    cells, tracks = counts["cells"], counts["tracks"]
    return (f"{cells} cell{'' if cells == 1 else 's'} · "
            f"{tracks} track{'' if tracks == 1 else 's'}")


def movie_thumbnail(project, condition: str, movie: str, *, background: str = "phase",
                    colour: str = "appd", channels: dict | None = None) -> dict:
    """What a gallery tile draws: `movie_payload`'s own image, outlines and tracks (one
    reading of the movie, not a second one), without the playback axis or the words."""
    payload = movie_payload(project, condition, movie, background=background, colour=colour, channels=channels)
    return {"movie": payload["movie"], "background": background, "colour_mode": colour,
            "image": payload["image"], "image_error": payload["image_error"],
            "outlines": [{"cell": o["cell"], "points": o["points"]}
                         for o in payload["outlines"]],
            "tracks": [{"uid": t["uid"], "cell": t["cell"], "points": t["points"],
                        "colour": t["colour"]} for t in payload["tracks"]],
            "outline_ink": payload["outline_ink"], "counts": payload["counts"],
            "caption": thumbnail_caption(payload["counts"])}


def cell_position(payload: dict, cell) -> str:
    """"cell 2 of 3", or "Whole field" when no cell is zoomed."""
    cells = [o["cell"] for o in payload["outlines"]]
    if cell is None or cell not in cells:
        return f"Whole field · {len(cells)} cell{'s' if len(cells) != 1 else ''}"
    return f"cell {cell.rsplit(':', 1)[-1]} · {cells.index(cell) + 1} of {len(cells)}"


def _box(points) -> list:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def cell_box(payload: dict, cell: str, pad: float = CELL_PAD_PX) -> list:
    """``[x0, y0, x1, y1]`` around ``cell``'s outline, padded and kept inside the image."""
    outline = next((o for o in payload["outlines"] if o["cell"] == cell), None)
    if outline is None:
        raise ImagesError(f"no cell {cell!r} in {payload['movie']}")
    x0, y0, x1, y1 = outline["box"]
    width = payload["image"]["width"] if payload.get("image") else x1 + pad
    height = payload["image"]["height"] if payload.get("image") else y1 + pad
    return [max(-0.5, x0 - pad), max(-0.5, y0 - pad),
            min(width - 0.5, x1 + pad), min(height - 0.5, y1 + pad)]


def cell_summary(payload: dict, cell: str) -> str:
    """One line about a cell: its tracks and their median apparent D."""
    tracks = [t for t in payload["tracks"] if t["cell"] == cell]
    median = _median([t["appd"] for t in tracks if t["appd"] is not None])
    text = f"cell {cell.rsplit(':', 1)[-1]} · {len(tracks)} tracks"
    return text + (f" · median apparent D {median:.3g} µm²/s" if median is not None else "")


def _median(values):
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def _plain(value):
    """A pandas/numpy scalar as a plain int, float or None (NaN is None)."""
    if value is None or isinstance(value, str):
        return value
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


# -- the sample size: outlined cells, replicate days, tracks per cell per unit time -------- #
DASH = "–"


def _movie_frames(project, condition: str) -> dict:
    """``{movie: (frames, source)}``, read once per condition (`project.movie_frames`)."""
    key = (condition, "frames")
    if (project, key) not in _CACHE:
        from .project import movie_frames

        _CACHE.put(project, key, movie_frames(project, condition))
    return _CACHE.get(project, key)


def _condition_cells(project, condition: str) -> tuple:
    """``(cells, notes)``: one row per OUTLINED cell of ``condition`` (`qc.sample_size`'s
    input) and what could not be counted, by name."""
    key = (condition, "cells")
    if (project, key) in _CACHE:
        return _CACHE.get(project, key)
    import pandas as pd

    from .canonical import CanonicalError, without_attrs

    data = _condition(project, condition)
    table = without_attrs(data["canonical"])
    params = data["params"]
    recorded = _movie_frames(project, condition)
    day_map = (getattr(project, "day_maps", None) or {}).get(condition) or {}
    per_cell = table.groupby("cell", observed=True).agg(
        n_tracks=("uid", "nunique"), n_localizations=("uid", "size"))
    per_movie = table.groupby("movie", observed=True).agg(
        lo=("frame", "min"), hi=("frame", "max"), tau=("tau_s", "median"),
        day=("day", "first"))
    taus = per_movie["tau"].dropna()
    rows = []
    notes = {"no_outlines": [], "untracked": [], "observed": [], "sources": {}}
    outlined = {m for m, cells in data["meshes"].items() if cells}
    notes["no_outlines"] = sorted(data["tracked"] - outlined)
    for movie in sorted(outlined):
        if movie in recorded:
            frames, source = recorded[movie]
        elif movie in per_movie.index:
            frames = int(per_movie.at[movie, "hi"] - per_movie.at[movie, "lo"] + 1)
            source = "observed span"
            notes["observed"].append(movie)
        else:
            # Outlined, but never tracked or tracked with no record of how long: its cells
            # cannot be put on a time axis, and counting them as watched-for-nothing would
            # be inventing a zero.
            notes["untracked"].append(movie)
            continue
        notes["sources"][source] = notes["sources"].get(source, 0) + 1
        if movie in per_movie.index:
            tau = float(per_movie.at[movie, "tau"])
            day = per_movie.at[movie, "day"]
        else:
            try:
                tau = params.tau_s_for(movie)
            except CanonicalError:
                tau = float(taus.median()) if len(taus) else float("nan")
            day = day_map.get(movie)
        for mesh in data["meshes"][movie]:
            cell = f"{movie}:{mesh.cell_id}"
            got = per_cell.loc[cell] if cell in per_cell.index else None
            rows.append({"condition": condition,
                         "day": None if day is None or day != day else int(day),
                         "movie": movie, "cell": cell,
                         "n_tracks": int(got["n_tracks"]) if got is not None else 0,
                         "n_localizations": (int(got["n_localizations"])
                                             if got is not None else 0),
                         "frames": int(frames), "tau_s": tau, "frames_source": source})
    if not rows:
        raise ImagesError(f"condition {condition!r} has no outlined cell in a tracked movie")
    out = (pd.DataFrame(rows), notes)
    return _CACHE.put(project, key, out)


_SIZE_KEYS = ("day", "movie", "n_movies", "n_cells", "n_empty_cells", "n_tracks",
              "n_localizations", "tracks_per_cell_median", "tracks_per_cell_mean",
              "largest_cell_share", "frames_min", "frames_max", "cell_minutes",
              "tracks_per_cell_per_1000_frames", "tracks_per_cell_per_minute")


def _size_row(row) -> dict:
    return {k: _plain(row[k]) for k in _SIZE_KEYS if k in row.index}


def inventory(project, conditions=None) -> dict:
    """The sample size of every condition, per condition, per replicate day and per movie,
    as plain data. A condition that cannot be counted is kept and named (``error``)."""
    from .qc import QCError, flag_low_yield, sample_size

    conditions = list(conditions if conditions is not None else project.conditions)
    entries = []
    for condition in conditions:
        try:
            cells, notes = _condition_cells(project, condition)
            whole = sample_size(cells, by=("condition",)).iloc[0]
            days = sample_size(cells, by=("condition", "day"))
            movies = sample_size(cells, by=("condition", "movie"))
        except (ImagesError, QCError) as exc:
            entries.append({"condition": condition, "error": str(exc)})
            continue
        row = _size_row(whole)
        judged = flag_low_yield(days)
        day_rows = [dict(_size_row(r), low_yield=bool(r["low_yield"]))
                    for _, r in judged.iterrows()]
        notes = dict(notes, yield_rule=judged.attrs.get("note", ""))
        row["n_days"] = sum(1 for d in day_rows if d.get("day") is not None)
        day_of = {str(m): _plain(d) for m, d in
                  cells.groupby("movie", observed=True)["day"].first().items()}
        movie_rows = {}
        for _, r in movies.iterrows():
            movie_rows[str(r["movie"])] = dict(_size_row(r), day=day_of.get(str(r["movie"])))
        entries.append({"condition": condition, "error": None, "row": row,
                        "days": day_rows, "movies": movie_rows, "notes": notes})
    return {"conditions": entries}


_RECORD_KEYS = ("n_movies", "n_cells", "n_empty_cells", "n_tracks", "n_localizations",
                "tracks_per_cell_median", "tracks_per_cell_mean", "largest_cell_share",
                "frames_min", "frames_max", "cell_minutes",
                "tracks_per_cell_per_1000_frames", "tracks_per_cell_per_minute")


def inventory_records(payload: dict) -> list:
    """Every level of the inventory as flat records -- condition, then each replicate day,
    then each movie -- for a table file."""
    out = []
    for entry in payload["conditions"]:
        if entry.get("error"):
            out.append({"level": "condition", "condition": entry["condition"], "day": None,
                        "movie": None, "error": entry["error"]})
            continue
        out.append({"level": "condition", "condition": entry["condition"], "day": None,
                    "movie": None, **{k: entry["row"].get(k) for k in _RECORD_KEYS}})
        for day in entry["days"]:
            out.append({"level": "day", "condition": entry["condition"],
                        "day": day.get("day"), "movie": None,
                        **{k: day.get(k) for k in _RECORD_KEYS}})
        for movie, row in sorted(entry["movies"].items()):
            out.append({"level": "movie", "condition": entry["condition"],
                        "day": row.get("day"), "movie": movie,
                        **{k: row.get(k) for k in _RECORD_KEYS}})
    return out


def write_inventory_csv(payload: dict, path) -> "Path":
    """The inventory as a CSV, one row per condition, replicate day and movie."""
    import csv
    from pathlib import Path

    path = Path(path)
    records = inventory_records(payload)
    columns = ["level", "condition", "day", "movie", *_RECORD_KEYS, "error"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({k: ("" if record.get(k) is None else record.get(k))
                             for k in columns})
    return path


def first_movie_of_day(payload: dict, condition: str, day):
    """The first movie (by name) of ``condition``'s replicate ``day``, or None."""
    for entry in payload["conditions"]:
        if entry["condition"] == condition and not entry.get("error"):
            movies = sorted(m for m, r in entry["movies"].items() if r.get("day") == day)
            return movies[0] if movies else None
    return None


#: Short headers, so the table fits a laptop window: "Tracks/cell" is the median cell, and
#: "/1000 frames" and "/min" continue it (tracks per cell per 1000 frames, per minute) --
#: said in full in the caption.
INVENTORY_COLUMNS = ("Condition", "Replicates", "Movies", "Frames", "Cells", "No tracks",
                     "Tracks", "Tracks/cell", "/1000 frames", "/min")


def _number(value, digits: int = 1, *, strip: bool = False) -> str:
    """A value at ``digits`` decimals (a rate keeps them, so a column lines up); ``strip``
    drops trailing zeros (a median of whole tracks reads 2, not 2.0)."""
    if value is None:
        return DASH
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".") if strip and "." in text else text


def _frames_text(row) -> str:
    lo, hi = row.get("frames_min"), row.get("frames_max")
    if lo is None:
        return DASH
    return f"{lo:,}" if lo == hi else f"{lo:,}–{hi:,}"


def _size_cells(row, first: str, days: str, movies: str | None = None) -> list:
    empty, cells = row.get("n_empty_cells") or 0, row.get("n_cells") or 0
    share = f" ({100.0 * empty / cells:.0f} %)" if cells else ""
    return [first, days, str(row.get("n_movies", DASH)) if movies is None else movies,
            _frames_text(row), str(cells), f"{empty}{share}", f"{row.get('n_tracks', 0):,}",
            _number(row.get("tracks_per_cell_median"), strip=True),
            _number(row.get("tracks_per_cell_per_1000_frames")),
            _number(row.get("tracks_per_cell_per_minute"))]


#: The Overview's two levels under each condition.
LEVELS = ("day", "movie")
LEVEL_LABELS = {"day": "By replicate", "movie": "By movie"}


def inventory_rows(payload: dict, level: str = "day") -> list:
    """The Overview table: each condition, then beneath it its replicate days (``level``
    "day") or its movies ("movie"). Every row is ``{"kind", "condition", "day", "movie",
    "cells"}``; ``cells[0]`` is the name the row shows (the page draws a condition's as its
    coloured name). Under one day the day rows would repeat the condition's, so they are
    left out; movies always show."""
    if level not in LEVELS:
        raise ImagesError(f"level must be one of {LEVELS}, got {level!r}")
    out = []
    for entry in payload["conditions"]:
        condition = entry["condition"]
        if entry.get("error"):
            out.append({"kind": "error", "condition": condition, "day": None, "movie": None,
                        "cells": [condition] + [DASH] * 8 + ["not counted"]})
            continue
        row = entry["row"]
        out.append({"kind": "condition", "condition": condition, "day": None, "movie": None,
                    "cells": _size_cells(row, condition, str(row["n_days"]))})
        if level == "movie":
            for movie, mrow in sorted(entry["movies"].items()):
                day = mrow.get("day")
                out.append({"kind": "movie", "condition": condition, "day": day,
                            "movie": movie,
                            "cells": _size_cells(mrow, movie,
                                                 DASH if day is None else f"replicate {day}", "")})
            continue
        days = entry["days"]
        if len(days) > 1 or (days and days[0].get("day") is None):
            for day in days:
                name = (f"replicate {day['day']}" if day.get("day") is not None
                        else "replicate not known")
                if day.get("low_yield"):
                    # the QC flag (`qc.flag_low_yield`): said on the row, not only coloured
                    name += " · low yield"
                out.append({"kind": "day", "condition": condition, "day": day.get("day"),
                            "movie": None, "cells": _size_cells(day, name, "")})
    return out


#: The Sample size table in print: the screen's short headings spelled out.
INVENTORY_PRINT_COLUMNS = ("Condition", "Replicates", "Movies", "Frames", "Cells",
                           "Cells without tracks", "Tracks", "Tracks per cell (median)",
                           "Tracks per cell per 1000 frames", "Tracks per cell per minute")


def inventory_table(payload: dict, level: str = "day", labels: dict | None = None) -> dict:
    """The Sample size table as the "table" figure kind draws it: the screen's rows, a
    condition by its shown name, its days or movies indented under it, the headings
    spelled out and the caption without its screen hints."""
    labels = labels or {}
    rows = []
    for item in inventory_rows(payload, level):
        cells = [str(c) for c in item["cells"]]
        if item["kind"] in ("day", "movie"):
            cells[0] = "\u2003" + cells[0]
        else:
            cells[0] = str(labels.get(cells[0], cells[0]))
        rows.append(cells)
    return {"title": "Sample size", "columns": list(INVENTORY_PRINT_COLUMNS), "rows": rows,
            "caption": inventory_caption(payload, printed=True),
            "file_name": "sample_size" if level == "day" else "sample_size_by_movie"}


def inventory_caption(payload: dict, *, printed: bool = False) -> str:
    """What a row means, where the frames came from, and every movie not counted, named.
    ``printed``: the words under the exported table (no screen hints, full headings)."""
    sources: dict = {}
    named = []
    for entry in payload["conditions"]:
        if entry.get("error"):
            named.append(f"{entry['condition']}: {entry['error']}")
            continue
        notes = entry["notes"]
        for source, n in notes["sources"].items():
            sources[source] = sources.get(source, 0) + n
        if notes["no_outlines"]:
            named.append(f"{entry['condition']}: tracked with no cell outlines, so its "
                         f"tracks were left out ({', '.join(notes['no_outlines'])})")
        if notes["untracked"]:
            named.append(f"{entry['condition']}: outlined with no tracking record, not "
                         f"counted ({', '.join(notes['untracked'])})")
        if notes["observed"]:
            named.append(f"{entry['condition']}: no record of how many frames were "
                         "tracked, so the span of the tracks is used instead, which is a lower bound "
                         f"({', '.join(notes['observed'])})")
    read = ", ".join(f"{source}, {n} movie{'s' if n != 1 else ''}"
                     for source, n in sorted(sources.items()))
    rates = ("Tracks per cell is the median cell; per 1000 frames and per minute, all "
             "tracks are pooled over every cell's watch time." if printed else
             "Tracks/cell is the median cell; /1000 frames and /min are tracks per cell per "
             "1000 frames and per minute, with all tracks pooled over every cell's watch time.")
    text = ("Cells are the outlined cells, including cells with no track; each track counts "
            "once, in its cell. Each cell is watched for its movie's tracked "
            f"frames (read from: {read or 'nothing'}). {rates} These are densities of "
            "detected tracks, not concentrations. A replicate marked “low yield” holds "
            "fewer tracks per cell per 1000 frames than its condition's median replicate "
            "by more than 3.5 × MAD."
            + ("" if printed else " Click a row to see its movies."))
    if named:
        text += " " + " ".join(f"{n}." for n in named)
    return text


# -- the side card: a movie's cells, a cell's tracks ------------------------------------ #
CELL_COLUMNS = ("Cell", "Tracks", "Locs", "Median D")
TRACK_COLUMNS = ("Track", "Frames", "Locs", "D")


def cell_rows(payload: dict) -> tuple:
    """``(rows, keys)``: one row per outlined cell of the movie, a cell with no track
    included (its Median D a dash). Keys are the cell ids."""
    rows, keys = [], []
    for outline in payload["outlines"]:
        median = outline.get("median_appd")
        rows.append([outline["label"], str(outline["n_tracks"]),
                     str(outline.get("n_localizations", 0)),
                     f"{median:.3g}" if median is not None else DASH])
        keys.append(outline["cell"])
    return rows, keys


def track_rows(payload: dict, cell) -> tuple:
    """``(rows, keys)``: the tracks of ``cell`` (every track when None) in the order they
    start, each with its number, its frames, its localisations and its apparent D (µm²/s).
    Keys are the uids."""
    tracks = [t for t in payload["tracks"] if cell is None or t["cell"] == cell]
    tracks.sort(key=lambda t: (t["frames"][0], t["uid"]))
    rows = [[t.get("number", t["uid"].rsplit(":", 1)[-1]),
             f"{t['frames'][0]}–{t['frames'][-1]}", str(len(t["points"])),
             f"{t['appd']:.3g}" if t["appd"] is not None else DASH] for t in tracks]
    return rows, [t["uid"] for t in tracks]


def movie_line(payload: dict) -> str:
    """"3 cells · 93 tracks · 31 per cell per 1000 frames · 1,001 frames tracked"."""
    cells, tracks = payload["counts"]["cells"], payload["counts"]["tracks"]
    if not cells:
        return ("no cell outlines, so the cell filter kept none of this movie's tracks"
                if not tracks else f"no cell outlines · {tracks} tracks")
    text = (f"{cells} cell{'s' if cells != 1 else ''} · {tracks} track"
            f"{'s' if tracks != 1 else ''}")
    outside = tracks - payload["counts"]["in_cells"]
    if outside:
        text += f" ({outside} outside every outline)"
    frames = payload.get("frames")
    if frames and cells:
        rate = 1000.0 * payload["counts"]["in_cells"] / (cells * frames[0])
        text += (f" · {_number(rate)} tracks per cell per 1000 frames · {frames[0]:,} frames "
                 "tracked")
    return text


# -- playback: the movie over time ------------------------------------------------------- #
#: The speeds offered, as multiples of the acquisition's own rate.
SPEEDS = (0.25, 1.0, 4.0)
#: The bundled Inter carries both glyphs (U+00BC, U+00D7), checked against its cmap.
SPEED_LABELS = ("\u00bc\u00d7", "1\u00d7", "4\u00d7")
#: One hover per speed, against the acquisition's own frame rate.
SPEED_TIPS = ("A quarter of the acquisition's frame rate", "The acquisition's own frame rate",
              "Four times the acquisition's frame rate")
#: How many frames of a track stay drawn behind its molecule: `playback.DEFAULT_TRAIL_FRAMES`
#: (a test holds the two equal; this module does not import numpy to read it).
TRAIL_FRAMES = 20
#: How often the page advances while playing, in ms (about 33 redraws a second).
TICK_MS = 30
#: Frames shown before a picked track starts, so its molecule is seen arriving.
LEAD_FRAMES = 5


def _time_axis(project, condition: str, movie: str, table, params) -> dict:
    """The movie's frames and its frame interval: the frames TRACKED (the tracker's own
    record, `project.movie_frames`), else the span of its tracks."""
    tau_ms = None
    if len(table) and "tau_s" in table.columns:
        tau_ms = float(table["tau_s"].median()) * 1000.0
    if tau_ms is None or not tau_ms > 0:
        try:
            tau_ms = float(params.tau_s_for(movie)) * 1000.0
        except Exception:                                  # noqa: BLE001 - mixed project
            tau_ms = float(params.frame_interval_ms)
    recorded = _movie_frames(project, condition).get(movie)
    if recorded:
        first, last = 1, int(recorded[0])
    elif len(table):
        first, last = int(table["frame"].min()), int(table["frame"].max())
    else:
        first, last = 1, 1
    return {"first": first, "last": max(first, last), "frame_interval_ms": tau_ms}


def frame_view(payload: dict, frame: int, *, cell=None, trail: int = TRAIL_FRAMES) -> dict:
    """What playback draws at ``frame``: each track's path over the last ``trail`` frames
    up to and including it, and whether its molecule is ON this frame (a dot). Only
    ``cell``'s tracks when a cell is zoomed. A track that ended inside the window keeps
    its trail until the window passes it, as `playback.TrackPlayer` draws it."""
    from bisect import bisect_right

    frame = int(frame)
    shown, on = [], 0
    for track in payload["tracks"]:
        if cell is not None and track["cell"] != cell:
            continue
        frames = track["frames"]
        if frames[0] > frame or frames[-1] <= frame - trail:
            continue
        lo, hi = bisect_right(frames, frame - trail), bisect_right(frames, frame)
        if hi <= lo:
            continue
        here = frames[hi - 1] == frame
        on += here
        shown.append({"uid": track["uid"], "points": track["points"][lo:hi],
                      "colour": track["colour"], "on_frame": here})
    return {"frame": frame, "trails": shown, "on_frame": on}


def tracks_at(payload: dict, frame: int, *, cell=None, trail: int = TRAIL_FRAMES) -> list:
    """The tracks AS PLAYBACK DRAWS THEM at ``frame``: each cut to its last ``trail``
    frames up to and including it (`frame_view`'s rule), as track dicts the figure
    exporter draws -- so an Export of a paused movie prints what is on screen."""
    from bisect import bisect_right

    frame = int(frame)
    out = []
    for track in payload["tracks"]:
        if cell is not None and track["cell"] != cell:
            continue
        frames = track["frames"]
        if frames[0] > frame or frames[-1] <= frame - trail:
            continue
        lo, hi = bisect_right(frames, frame - trail), bisect_right(frames, frame)
        if hi <= lo:
            continue
        out.append(dict(track, points=track["points"][lo:hi], frames=frames[lo:hi]))
    return out


def time_text(payload: dict, frame=None, view=None) -> str:
    """The time bar's readout: every frame at rest, else the frame, the time and how many
    molecules are on it."""
    axis = payload["time"]
    if frame is None:
        return (f"all {axis['last'] - axis['first'] + 1:,} frames · press play to watch "
                f"them in order")
    seconds = (int(frame) - axis["first"]) * axis["frame_interval_ms"] / 1000.0
    text = f"frame {int(frame):,} of {axis['last']:,} · {seconds:.2f} s"
    if view is not None:
        n = view["on_frame"]
        text += f" · {n} molecule{'s' if n != 1 else ''}"
    return text


def track_start(payload: dict, uid) -> int:
    """Where playing a picked track begins: a few frames before its first, so its molecule
    is seen arriving (the movie's first frame for an unknown track)."""
    first = payload["time"]["first"]
    track = next((t for t in payload["tracks"] if t["uid"] == uid), None)
    if track is None:
        return first
    return max(first, int(track["frames"][0]) - LEAD_FRAMES)


def project_uid(project, condition: str, uid: str) -> str:
    """The PROJECT's uid for a track this page names by its original movie
    (``series033:12`` -> ``<condition>_series033:12``), so the viewers that read the
    project's own tables (`trackviewer.track_target`) find it. The qualified movie is
    looked up where the project records it (`movie_source`, by this condition's own
    root), never rebuilt from the name alone; a movie the project does not record is
    qualified the loader's way (`project.qualified_movie`)."""
    from pathlib import Path

    from .project import qualified_movie

    movie, _, number = str(uid).rpartition(":")
    root = Path(project.roots[condition]).resolve()
    for qualified, (source_root, original) in (getattr(project, "movie_source", None)
                                               or {}).items():
        if str(original) == movie and Path(source_root).resolve() == root:
            return f"{qualified}:{number}"
    return f"{qualified_movie(condition, movie)}:{number}"


def nearest_track(payload: dict, x: float, y: float, *, cell=None,
                  max_distance: float = 1.5):
    """The uid of the track drawn nearest ``(x, y)`` (pixels), within ``max_distance`` --
    or None, so a click on nothing selects nothing. Only ``cell``'s tracks when given."""
    best, best_d = None, float(max_distance)
    for track in payload["tracks"]:
        if cell is not None and track["cell"] != cell:
            continue
        points = track["points"]
        pairs = list(zip(points, points[1:])) or [(points[0], points[0])]
        for (ax, ay), (bx, by) in pairs:
            dx, dy = bx - ax, by - ay
            span = dx * dx + dy * dy
            f = 0.0 if span == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy)
                                                   / span))
            d = ((ax + f * dx - x) ** 2 + (ay + f * dy - y) ** 2) ** 0.5
            if d < best_d:
                best, best_d = track["uid"], d
    return best
