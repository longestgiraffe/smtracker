"""A movie's RECORDED FRAMES, numbered the tracker's way: for the Spatial Map test page's
player (a clip around one track) and the Images & tracks section's Movie background (the
whole field, played under its tracks). Headless.

THREE PLACES THE FRAMES CAN BE, tried in this order (asked 14 Oct 2026: "work with raw
movie instead of tif as well, if done with trackmate for example?"):

1. ``<root>/tif/<movie>.tif`` -- the single multi-page TIFF the preparation writes and
   TrackMate is GIVEN (`movieprep`'s module docstring): page f - 1 is the tracker's
   frame f. Measured on the HiWi Test project (Glucose, series001-003): the mean grey
   value around a localisation peaks on page f - 1 and falls off symmetrically either
   side (`docs/claude/spatial_maps.md`).
2. ``<root>/tif/<movie>/<movie>/<movie>_c1_tNNNN.tif`` -- the one-file-per-frame folder
   u-track READ (`utrack_run.write_frames`; what Bio-Formats' "Crop ALL" wrote before
   it): file t f is frame f. A u-track project has no single stack, so before this the
   player found nothing on it (one user's 1500-frame movies, 14 Oct 2026).
3. The RAW acquisition (``.nd2`` / ``.stk``, in ``<root>/raw/`` or beside the outputs)
   with the range the run TRACKED: a run from the raw file cuts frames lo-hi itself
   (`trackmate_run`'s script), so the tracker's frame f is raw frame lo + f - 1. The
   range is read from the run's own record (``trackmate/run.json``, `record_run`), else
   the preparation's ``crops.json`` (editable after a run, hence second); a record that
   says the movie was tracked whole means lo = 1.

A source is USED only when it agrees with what the tracker recorded:
- its frame count equals the frames tracked (``expected_frames``, `project.movie_frames`),
  when that is known -- a stack of another length cannot number its frames the
  tracker's way;
- and, given ``spots`` (localisations: frame, x, y in 0-based pixels), the spots are
  BRIGHTEST on their own frame: the mean grey around them is read on frames f - 3 .. f + 3
  and the peak must sit at f (`check_alignment`). This is the UI designer's measurement made a guard:
  a crop range off by even one frame moves the peak, and tracks drawn on the wrong frames
  would look right. With fewer than `MIN_CHECK_SPOTS` spots it is not checked, and the
  source's words say so.
A source that disagrees is passed over WITH ITS REASON, the next one tried; when none
agrees, `TrackClipError` names every place looked and why each was refused.

Invariants:
- Only the asked frames are read: stacks are memory-mapped when uncompressed, else read
  page by page; ``.nd2`` through its lazy reader (`rawmovie`).
- One display stretch (0.5-99.5 %) per clip, or one per MOVIE for playback
  (`movie_stretch`, sampled across the frames tracked), so a frame's brightness is
  comparable with its neighbours' and a molecule's blinking is not stretched away.
- The results are plain (ints and bytes): the Qt side never sees numpy.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["TrackClipError", "MovieFrames", "stack_path", "utrack_frames_folder",
           "open_movie", "has_frames", "check_alignment", "movie_stretch", "grey_frames",
           "clip_frames", "MIN_CHECK_SPOTS", "CHECK_REACH", "forget", "FrameDisplay", "BACKGROUND_FILTERS",
           "auto_range", "filtered_frames", "frame_levels"]

#: Fewer spots than this and the alignment is not checked (the words say so).
MIN_CHECK_SPOTS = 12
#: How many spots the check reads at most, spread over the movie.
CHECK_SPOTS = 40
#: Frames either side of a spot's own that the check compares against.
CHECK_REACH = 3
#: The own frame passes when it is within this share of the contrast (brightest against the frames two or more away)
#: of the brightest frame: a near tie, not an offset (`check_alignment`, 6 Nov 2026).
NEAR_TIE_SHARE = 0.10


class TrackClipError(RuntimeError):
    """The recorded frames cannot be read, or cannot be trusted to line up."""


def stack_path(root, movie: str) -> Path:
    """Where a movie's prepared stack lives in its condition folder (a day folder's, for ``series001 (rep1)``)."""
    from .dayfolders import on_disk

    root, movie = on_disk(root, movie)
    return Path(root) / "tif" / f"{movie}.tif"


def utrack_frames_folder(root, movie: str) -> Path:
    """Where u-track's one-file-per-frame copy of a movie lives."""
    from .dayfolders import on_disk

    root, movie = on_disk(root, movie)
    return Path(root) / "tif" / movie / movie


# -- the three readers --------------------------------------------------------------- #
class _TiffStack:
    """A TIFF stack (or MetaMorph ``.stk``) as a lazy ``(n, h, w)``: read by OFFSET when
    its frames lie contiguous and uncompressed (one seek, one read), else page by page.
    NO HANDLE IS HELD between reads: a mapped or open file on Windows cannot be replaced,
    and the preparation rewrites ``tif/<movie>.tif`` while the analysis may be open."""

    def __init__(self, path: Path):
        import tifffile

        self.path = path
        with tifffile.TiffFile(str(path)) as handle:
            series = handle.series[0]
            shape = tuple(series.shape)
            self.dtype = series.dtype.newbyteorder(handle.byteorder)
            self.offset = series.dataoffset
            self.n_pages = len(handle.pages)
        if len(shape) == 2:
            shape = (1,) + shape
        if len(shape) != 3:
            raise TrackClipError(f"{path}: expected grey frames, got shape {shape}")
        self.shape = shape
        self.ndim = 3

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        import numpy as np
        import tifffile

        start, stop, _step = key.indices(self.shape[0])
        n, (h, w) = stop - start, self.shape[1:]
        if self.offset is not None:
            size = h * w * self.dtype.itemsize
            with open(self.path, "rb") as f:
                f.seek(self.offset + start * size)
                raw = f.read(n * size)
            if len(raw) == n * size:
                return np.frombuffer(raw, dtype=self.dtype).reshape(n, h, w)
        with tifffile.TiffFile(str(self.path)) as handle:
            if self.n_pages == self.shape[0]:
                return np.stack([np.asarray(handle.pages[k].asarray())
                                 for k in range(start, stop)])
            # one page holding every plane, compressed: there is no lazy way in
            return np.asarray(handle.series[0].asarray())[start:stop]


def _tiff_array(path: Path):
    return _TiffStack(path)


class _UtrackFrames:
    """u-track's folder of single frames as one lazy stack (file t f = frame f)."""

    def __init__(self, folder: Path, movie: str):
        files = sorted(folder.glob(f"{movie}_c1_t*.tif"),
                       key=lambda p: int(p.stem.rsplit("_t", 1)[1]))
        if not files:
            raise TrackClipError(f"no frames in {folder}")
        numbers = [int(p.stem.rsplit("_t", 1)[1]) for p in files]
        if numbers != list(range(1, len(files) + 1)):
            raise TrackClipError(f"{folder}: the frame files are not numbered 1..{len(files)}"
                                 " without a gap")
        self.files = files
        import tifffile

        first = tifffile.imread(str(files[0]))
        self.shape = (len(files),) + tuple(first.shape[-2:])
        self.ndim = 3

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        import numpy as np
        import tifffile

        start, stop, _step = key.indices(len(self.files))
        return np.stack([np.asarray(tifffile.imread(str(self.files[k])))
                         for k in range(start, stop)])


class _RawFrames:
    """A raw acquisition as one lazy stack: ``.nd2`` through `rawmovie`'s lazy reader,
    ``.stk`` / ``.tif`` mapped or read by page."""

    def __init__(self, path: Path):
        if path.suffix.lower() == ".nd2":
            from .rawmovie import RawMovieError, read_raw_movie

            try:
                self._data = read_raw_movie(path)._frames
            except RawMovieError as exc:
                raise TrackClipError(str(exc)) from exc
        else:
            self._data = _tiff_array(path)
        self.shape = tuple(self._data.shape)
        self.ndim = 3

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        import numpy as np

        return np.asarray(self._data[key])


@dataclass
class MovieFrames:
    """One movie's recorded frames, numbered the tracker's way.

    ``kind`` is "stack" | "utrack" | "raw"; tracker frame f is index ``first_index + f - 1``
    of ``path``'s frames; ``n_frames`` is how many frames from tracker frame 1 on it holds
    (the frames tracked when the tracker recorded them); ``words`` say where they came from
    and how far they were checked."""

    kind: str
    path: Path
    first_index: int
    n_frames: int
    width: int
    height: int
    words: str
    check: dict = field(default_factory=dict)
    _data: object = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def read(self, lo: int, hi: int):
        """Tracker frames ``lo``..``hi`` (1-based, inclusive) as a float32 ``(k, h, w)``."""
        import numpy as np

        lo, hi = int(lo), int(hi)
        if lo < 1 or hi < lo:
            raise TrackClipError(f"frames {lo}-{hi} are not a range of a movie")
        if hi > self.n_frames:
            raise TrackClipError(f"{self.path} has {self.n_frames:,} frames from the "
                                 f"tracker's first on; frames {lo}-{hi} run past its end.")
        a, b = self.first_index + lo - 1, self.first_index + hi
        with self._lock:                   # one file handle's reader at a time
            block = np.asarray(self._data[a:b])
        if block.ndim != 3:
            raise TrackClipError(f"{self.path}: expected grey frames, got shape "
                                 f"{block.shape[1:]}")
        return block.astype(np.float32)


# -- where the frames are, and whether they line up ---------------------------------- #
def _raw_candidates(root: Path, movie: str) -> list:
    from .rawmovie import ACQUISITION_SUFFIXES, movie_name

    out = []
    for folder in (root / "raw", root):
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if (path.is_file() and path.suffix.lower() in ACQUISITION_SUFFIXES
                    and (path.stem == movie or movie_name(path) == movie)):
                out.append(path)
    return out


def _tracked_range(root: Path, movie: str, raw: Path):
    """``(lo, hi or None, where)``: the 1-based range of ``raw`` that was tracked, from the
    run's record, else the preparation's crops; None when nothing records it."""
    run = root / "trackmate" / "run.json"
    if run.is_file():
        try:
            record = json.loads(run.read_text(encoding="utf-8"))
        except ValueError:
            record = {}
        given = {Path(str(m)).stem for m in (record.get("movies") or [])}
        crops = record.get("crops")
        if crops and movie in crops:
            lo, hi = (int(v) for v in crops[movie][:2])
            return lo, hi, "the tracking run's record (trackmate/run.json)"
        if crops is None and (raw.stem in given or movie in given) and any(
                Path(str(m)).suffix.lower() == raw.suffix.lower()
                for m in record.get("movies") or []):
            return 1, None, "the tracking run's record: tracked whole"
    from .movieprep import MoviePrepError, find_crops, load_crops

    found = find_crops(root)
    if found is not None:
        try:
            crops = load_crops(found)["crops"]
        except MoviePrepError:
            crops = {}
        if movie in crops:
            lo, hi = crops[movie]
            return int(lo), int(hi), f"the saved crop range ({found.name})"
    return None


def _candidates(root: Path, movie: str, expected):
    """``(kind, path, first_index, n_frames, words, data)`` per source found, in order,
    plus the reasons any was refused before the alignment check."""
    found, refused = [], []
    stack = stack_path(root, movie)
    if stack.is_file():
        try:
            data = _tiff_array(stack)
            n = len(data)
            if expected is not None and n != int(expected):
                refused.append(f"{stack} has {n:,} pages but the tracker recorded "
                               f"{int(expected):,} frames, so its pages cannot be matched "
                               "to the track's frames")
            else:
                found.append(("stack", stack, 0, n, "the stack the tracker was given "
                              f"(tif/{movie}.tif)", data))
        except Exception as exc:                           # noqa: BLE001 - named below
            refused.append(f"{stack}: {exc}")
    folder = utrack_frames_folder(root, movie)
    if folder.is_dir():
        try:
            data = _UtrackFrames(folder, movie)
            n = len(data)
            if expected is not None and n != int(expected):
                refused.append(f"{folder} holds {n:,} frames but the tracker recorded "
                               f"{int(expected):,}")
            else:
                found.append(("utrack", folder, 0, n, "the frames u-track read "
                              f"(tif/{movie}/{movie}/)", data))
        except TrackClipError as exc:
            refused.append(str(exc))
    for raw in _raw_candidates(root, movie):
        try:
            data = _RawFrames(raw)
        except Exception as exc:                           # noqa: BLE001 - named below
            refused.append(f"{raw}: {exc}")
            continue
        total = len(data)
        ranged = _tracked_range(root, movie, raw)
        if ranged is None:
            if expected is not None and total == int(expected):
                found.append(("raw", raw, 0, total, f"the raw acquisition ({raw.name}), "
                              "all of it: its length is the frames tracked", data))
            else:
                refused.append(f"{raw}: no record says which of its {total:,} frames were "
                               "tracked (no trackmate/run.json range, no crops.json entry)")
            continue
        lo, hi, where = ranged
        hi = total if hi is None else hi
        n = hi - lo + 1
        if lo < 1 or hi > total or n < 1:
            refused.append(f"{raw}: the range {lo}-{hi} from {where} lies outside its "
                           f"{total:,} frames")
        elif expected is not None and n != int(expected):
            refused.append(f"{raw}: the range {lo}-{hi} from {where} is {n:,} frames but "
                           f"the tracker recorded {int(expected):,}")
        else:
            found.append(("raw", raw, lo - 1, n, f"the raw acquisition ({raw.name}), "
                          f"frames {lo:,}-{hi:,} as {where} gives them", data))
    return found, refused


def check_alignment(frames: MovieFrames, spots, *, reach: int = CHECK_REACH,
                    n_spots: int = CHECK_SPOTS) -> dict:
    """Is each spot BRIGHTEST on its own frame? The mean grey of the 3 x 3 pixels around
    up to ``n_spots`` localisations (``(frame, x, y)``, 0-based pixels), read on their
    frame and ``reach`` frames either side.

    Returns ``{checked, n, profile: {shift: mean}, peak, words}``; ``peak`` 0 is a pass.
    Not checked (``checked`` False) with fewer than `MIN_CHECK_SPOTS` usable spots."""
    import numpy as np

    usable = []
    for f, x, y in spots:
        f, cx, cy = int(f), int(round(float(x))), int(round(float(y)))
        if (1 + reach <= f <= frames.n_frames - reach and 1 <= cx < frames.width - 1
                and 1 <= cy < frames.height - 1):
            usable.append((f, cx, cy))
    if len(usable) < MIN_CHECK_SPOTS:
        return {"checked": False, "n": len(usable), "profile": {}, "peak": None,
                "words": f"not checked ({len(usable)} spot(s) to check against, fewer "
                         f"than {MIN_CHECK_SPOTS})"}
    pick = np.unique(np.linspace(0, len(usable) - 1, min(n_spots, len(usable)))
                     .round().astype(int))
    sums = np.zeros(2 * reach + 1)
    for k in pick:
        f, cx, cy = usable[k]
        block = frames.read(f - reach, f + reach)
        sums += block[:, cy - 1:cy + 2, cx - 1:cx + 2].reshape(2 * reach + 1, -1).mean(axis=1)
    profile = sums / len(pick)
    shifts = list(range(-reach, reach + 1))
    peak = shifts[int(np.argmax(profile))]
    others = float(np.mean([v for s, v in zip(shifts, profile) if abs(s) >= 2]))
    # A NEAR TIE IS NO MISALIGNMENT (6 Nov 2026): a molecule that barely moves is as bright on the frame beside its own,
    # so where many are bound the own frame and its neighbour tie (SynR high, series029: 2317.0 against 2318.4, on a
    # contrast of 107 to the frames two or more away) and the argmax falls either way by chance. The own frame passes
    # within a tenth of that contrast of the brightest; a real offset of one frame leaves it about 80 below (series028
    # read one frame off).
    near_tie = peak != 0 and profile[reach] >= profile.max() - NEAR_TIE_SHARE * (profile.max() - others)
    if near_tie:
        peak = 0
    return {"checked": True, "n": int(len(pick)),
            "profile": {s: float(v) for s, v in zip(shifts, profile)}, "peak": peak,
            "words": (f"checked: {len(pick)} spots brightest on their own frame "
                      f"({profile[reach]:.0f} against {others:.0f} two or more frames "
                      "away" + (", a near tie with the frame beside it: molecules that barely move are as bright "
                                "there" if near_tie else "") + ")") if peak == 0 else
                     (f"spots are brightest {abs(peak)} frame(s) "
                      f"{'after' if peak > 0 else 'before'} their own")}


_OPEN: dict = {}
_OPEN_LOCK = threading.Lock()
_OPEN_MAX = 12


def forget() -> None:
    """Drop every opened movie (after files changed on disk)."""
    with _OPEN_LOCK:
        _OPEN.clear()


def _stamp(root: Path, movie: str) -> tuple:
    """What a cached opening depends on: the files' modification times."""
    paths = [stack_path(root, movie), utrack_frames_folder(root, movie),
             root / "trackmate" / "run.json", *_raw_candidates(root, movie)]
    return tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in paths)


def has_frames(root, movie: str) -> bool:
    """Whether any place for the recorded frames exists (cheap: no file is opened)."""
    from .dayfolders import on_disk

    root, movie = on_disk(root, movie)          # a day folder's movie where it is
    return (stack_path(root, movie).is_file() or utrack_frames_folder(root, movie).is_dir()
            or bool(_raw_candidates(root, movie)))


def open_movie(root, movie: str, *, expected_frames: int | None = None,
               spots=None) -> MovieFrames:
    """The movie's recorded frames from the first source that agrees with the tracker
    (see the module head), checked against ``spots`` when given. Cached per movie until
    its files change."""
    from .dayfolders import on_disk

    root, movie = on_disk(root, movie)          # a day folder's movie where it is
    key = (str(root.resolve()), movie, expected_frames, spots is not None)
    stamp = _stamp(root, movie)
    with _OPEN_LOCK:
        hit = _OPEN.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
    found, refused = _candidates(root, movie, expected_frames)
    chosen = None
    for kind, path, first, n, words, data in found:
        shape = tuple(data.shape)
        frames = MovieFrames(kind=kind, path=path, first_index=first, n_frames=n,
                             width=int(shape[2]), height=int(shape[1]), words=words,
                             _data=data)
        if spots is not None:
            try:
                check = check_alignment(frames, spots)
            except TrackClipError as exc:
                refused.append(f"{path}: {exc}")
                continue
            if check["checked"] and check["peak"] != 0:
                refused.append(f"{path}: {check['words']}, so these frames do not line up "
                               "with the tracks")
                continue
            frames.check = check
            frames.words = f"{words}; alignment {check['words']}"
        chosen = frames
        break
    if chosen is None:
        looked = [f"tif/{movie}.tif", f"tif/{movie}/{movie}/", f"raw/{movie}.nd2|.stk"]
        detail = "; ".join(refused) if refused else "none of them exists"
        raise TrackClipError(f"no recorded stack for {movie} in {root}: looked for "
                             f"{', '.join(looked)} -- {detail}.")
    with _OPEN_LOCK:
        if len(_OPEN) >= _OPEN_MAX:
            _OPEN.pop(next(iter(_OPEN)))
        _OPEN[key] = (stamp, chosen)
    return chosen


# -- grey bytes for the screen --------------------------------------------------------- #
def movie_stretch(frames: MovieFrames, *, low: float = 0.5, high: float = 99.5,
                  samples: int = 12) -> tuple:
    """One display stretch for the whole movie: the ``low``-``high`` percentiles of
    ``samples`` frames spread over the frames tracked."""
    import numpy as np

    picks = np.unique(np.linspace(1, frames.n_frames, min(samples, frames.n_frames))
                      .round().astype(int))
    values = np.concatenate([frames.read(int(f), int(f)).ravel() for f in picks])
    a, b = (float(v) for v in np.percentile(values, [low, high]))
    return a, (b if b > a else a + 1.0)


def grey_frames(frames: MovieFrames, lo: int, hi: int, stretch, display: "FrameDisplay | None" = None) -> list:
    """Tracker frames ``lo``..``hi`` as 8-bit grey bytes under one ``stretch`` (a, b) -- or, with ``display``,
    filtered as it says and mapped from ITS black and white (raw counts; ``stretch`` is then the auto range of the
    filtered movie, used where black or white is left to Auto, per frame when ``display.per_frame``)."""
    import numpy as np

    if display is None:
        a, b = stretch
        block = frames.read(lo, hi)
        grey = np.clip((block - a) * (255.0 / (b - a)), 0, 255).astype(np.uint8)
        return [np.ascontiguousarray(f).tobytes() for f in grey]
    block = filtered_frames(frames, lo, hi, display)
    out = []
    for frame in block:
        a, b = display.range_for(frame if display.per_frame else None, stretch)
        grey = np.clip((frame - a) * (255.0 / (b - a)), 0, 255).astype(np.uint8)
        out.append(np.ascontiguousarray(grey).tobytes())
    return out


# -- how the frames are SHOWN: Fiji's B&C on raw counts, and two display filters ----------- #
#: Asked 22 Oct 2026: "i dunno about the brightness and contrast, is there another setting we could add, i feel like
#: in fiji u can better control when a signal is visible or not". Fiji's B&C sets black and white in the image's own
#: counts; the player stretched each movie to 8 bits ONCE and its brightness and contrast only moved those 256
#: levels, so a narrow window ran out of levels and a dim spot went with them. Chosen: black and white in raw counts
#: (Auto = Fiji's 0.35 % saturated, Reset = the full range), a smoothing at the spot's size, and a background
#: subtraction. ALL DISPLAY ONLY: nothing here reaches a track, a fit or a number; the figure legend states it.
BACKGROUND_FILTERS = ("none", "median", "rolling")


@dataclass(frozen=True)
class FrameDisplay:
    """How the Movie background shows the recorded frames.

    ``black`` / ``white``: the counts shown black and white (after the filters); None = Auto, the filtered movie's
    ``saturated_pct`` range (half at each end, as Fiji's Auto), or each frame's own with ``per_frame`` (a movie
    that bleaches stays visible to its end). ``smooth_px``: a Gaussian of this sigma, the spot's size (0 = off),
    as the detector sees a frame. ``background``: "median" removes each pixel's running median over
    ``median_frames`` frames -- whatever does not change (cell autofluorescence, haze, the illumination's falloff)
    goes, a moving or blinking spot stays; "rolling" removes, per frame, the frame's grey opening over a square of
    side ``2 rolling_px + 1`` (a top-hat: Fiji's rolling-ball subtraction with a flat square for the ball)."""

    black: float | None = None
    white: float | None = None
    per_frame: bool = False
    smooth_px: float = 0.0
    background: str = "none"
    median_frames: int = 21
    rolling_px: int = 8
    saturated_pct: float = 0.35

    def __post_init__(self):
        if self.background not in BACKGROUND_FILTERS:
            raise TrackClipError(f"background must be one of {BACKGROUND_FILTERS}, got {self.background!r}")
        if self.smooth_px < 0 or self.median_frames < 3 or self.rolling_px < 1:
            raise TrackClipError("smoothing cannot be negative, a running median needs 3 frames and a rolling "
                                 "background 1 px")
        if not 0.0 <= self.saturated_pct < 50.0:
            raise TrackClipError(f"saturated share must be 0-50 %, got {self.saturated_pct}")
        if self.black is not None and self.white is not None and not self.white > self.black:
            raise TrackClipError(f"white ({self.white:g}) must lie above black ({self.black:g})")

    @property
    def filters(self) -> tuple:
        """What changes the counts themselves (not only how they are shown): the levels depend on it."""
        return (float(self.smooth_px), self.background,
                int(self.median_frames) if self.background == "median" else 0,
                int(self.rolling_px) if self.background == "rolling" else 0)

    def range_for(self, frame, movie_auto) -> tuple:
        """``(black, white)`` for one frame: the set values, else Auto -- ``frame``'s own (per frame) or the
        movie's."""
        a, b = self.black, self.white
        if a is None or b is None:
            auto = auto_range(frame, self.saturated_pct) if frame is not None else movie_auto
            if auto is None:
                raise TrackClipError("Auto needs the movie's range: pass its levels, or set black and white")
            a = auto[0] if a is None else a
            b = auto[1] if b is None else b
        a, b = float(a), float(b)
        return (a, b) if b > a else (a, a + 1.0)


def auto_range(values, saturated_pct: float = 0.35) -> tuple:
    """Fiji's Auto: the range leaving ``saturated_pct`` per cent of the values saturated, half at each end."""
    import numpy as np

    half = float(saturated_pct) / 2.0
    a, b = (float(v) for v in np.percentile(np.asarray(values, float), [half, 100.0 - half]))
    return a, (b if b > a else a + 1.0)


def filtered_frames(frames: MovieFrames, lo: int, hi: int, display: FrameDisplay):
    """Tracker frames ``lo``..``hi`` as float32 ``(k, h, w)`` after ``display``'s background subtraction and
    smoothing. A running median reads the frames around the block too (clamped to the movie), so a block's edge
    is filtered as its middle is."""
    import numpy as np
    from scipy import ndimage

    if display.background == "median":
        half = int(display.median_frames) // 2
        a, b = max(1, int(lo) - half), min(frames.n_frames, int(hi) + half)
        wide = frames.read(a, b)
        size = min(int(display.median_frames), wide.shape[0])
        background = ndimage.median_filter(wide, size=(size, 1, 1), mode="nearest")
        block = (wide - background)[int(lo) - a:int(lo) - a + int(hi) - int(lo) + 1]
    else:
        block = frames.read(lo, hi)
        if display.background == "rolling":
            side = 2 * int(display.rolling_px) + 1
            block = block - ndimage.grey_opening(block, size=(1, side, side))
    if display.smooth_px > 0:
        block = ndimage.gaussian_filter(block, sigma=(0.0, float(display.smooth_px), float(display.smooth_px)))
    return np.ascontiguousarray(block, dtype=np.float32)


def frame_levels(frames: MovieFrames, display: FrameDisplay | None = None, *, samples: int = 12,
                 bins: int = 128) -> dict:
    """What the black and white sliders run over, for the movie as ``display`` filters it: the counts' full range
    (``low``, ``high``: Reset), the Auto range, and a histogram of ``samples`` frames spread over the frames
    tracked (``counts`` in ``bins`` equal bins from ``low`` to ``high``). Plain numbers."""
    import numpy as np

    display = display or FrameDisplay()
    picks = np.unique(np.linspace(1, frames.n_frames, min(samples, frames.n_frames)).round().astype(int))
    values = np.concatenate([filtered_frames(frames, int(f), int(f), display).ravel() for f in picks])
    low, high = float(values.min()), float(values.max())
    if not high > low:
        high = low + 1.0
    counts, _edges = np.histogram(values, bins=int(bins), range=(low, high))
    auto = auto_range(values, display.saturated_pct)
    return {"low": low, "high": high, "auto": [float(auto[0]), float(auto[1])],
            "counts": [int(c) for c in counts], "samples": int(len(picks))}


def clip_frames(root, movie: str, lo: int, hi: int, *, expected_frames: int | None = None,
                spots=None, low: float = 0.5, high: float = 99.5) -> dict:
    """Frames `lo`..`hi` (the tracker's 1-based numbering, inclusive) as 8-bit grey, one
    stretch over the clip.

    Returns ``{width, height, first, last, frames: [bytes, ...], source}`` with one entry
    per frame from `first` (= lo) to `last` (= hi); ``source`` says where they came from."""
    import numpy as np

    frames = open_movie(root, movie, expected_frames=expected_frames, spots=spots)
    stack = frames.read(lo, hi)
    a, b = np.percentile(stack, [low, high])
    if not b > a:
        b = a + 1.0
    grey = np.clip((stack - a) * (255.0 / (b - a)), 0, 255).astype(np.uint8)
    return {"width": int(grey.shape[2]), "height": int(grey.shape[1]), "first": int(lo),
            "last": int(hi), "frames": [np.ascontiguousarray(f).tobytes() for f in grey],
            "source": frames.words}
