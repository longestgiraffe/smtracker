"""u-track, driven from SMTracker: find MATLAB and u-track, hand u-track a movie the way
it reads one, run detection and tracking headless, and say what came out.

WHY THIS EXISTS BESIDE TRACKMATE. u-track is this project's canonical tracker -- every
deposited number comes from it -- and a lab member who has MATLAB and u-track installed
should not have to click through u-track's own GUI movie by movie (2063 movies on the lab's data drive were
run that way). SMTracker shows the settings, writes the movie, generates a MATLAB function
and runs it headless; the result lands exactly where the lab's runs put theirs,
``<condition>/tif/<movie>/TrackingPackage/tracks/Channel_1_tracking_result.mat`` beside
``movieData.mat``, so the analysis reads it with no change at all.

AN EXTERNAL ENGINE, FOUND AND NEVER SHIPPED -- like Fiji. The standing rule ("external
engines are reimplemented, never depended on") is about the ANALYSIS: nothing downstream
needs MATLAB. Tracking with u-track is optional, needs the user's own MATLAB licence, and
the installer is unchanged by it. MATLAB is called as a subprocess, never linked.

WHAT WAS READ, NOT ASSUMED (12 Oct 2026):
* The source on this machine, `D:\\mat\\u-track-master\\software` (u-track 2.3.x, GitHub
  snapshot, with the lab's "+"/"Crop ALL" additions to the movie selector; `u-track-magic`
  is the same code). The call sequence below is the process API in that tree.
* Every one of the 2063 ``movieData.mat`` on the lab's data drive (`LAB_STANDARD`): ``SubResolutionProcess``
  then ``TrackingProcess``, never anything else; ``background`` empty in all 2063.

THE MOVIE IS A FOLDER OF SINGLE-FRAME TIFFS, u-track's own working copy. The lab imported
each ``tif/<movie>.tif`` through Bio-Formats and pressed "Crop ALL", which writes
``tif/<movie>/<movie>/<movie>_c1_t0001.tif`` ...; `write_frames` writes the same files
straight from the acquisition over the SAVED crop range (the slice `movieprep.prepare_movie`
cuts, so the pixels are the lab macro's). Three reasons for this route over handing u-track
the file: the background test exists ONLY for a TIFF series (it builds per-frame file
names), the layout is the one the analysis already reads, and no Bio-Formats is needed in
MATLAB -- under R2016a's Java 7 the jar u-track ships (6.2.1, Java 8) does not load.

THE GAP IS THE TIME WINDOW, AND u-track's OWN GUI COUNTS IT DIFFERENTLY. ``timeWindow`` is
the largest frame difference gap closing may join; ``timeWindow = 1`` joins only
consecutive frames, i.e. bridges NOTHING. u-track's GUI shows "maximum gap" as
``timeWindow - 1``. Every movie on the lab's data drive has ``timeWindow = 1``: no gap was ever closed (and
the reference project measured no gaps in its tracks). `UTrackSettings.frames_bridged` is the GUI's
number; the script writes ``timeWindow = frames_bridged + 1``.

THE BACKGROUND TEST IS A THRESHOLD, NOT A CORRECTION (`detectSubResFeatures2D_StandAlone`
lines 244-250, 429-441, 470-482). Given one cell-free image per frame, u-track takes the
mean and SD over all its pixels and keeps a local maximum only if it ALSO clears that
absolute background at ``alphaLocMaxAbs``. Nothing is subtracted and the fit is unchanged.
u-track zeroes are NaN, so `write_background_frames` writes each frame with the cells set
to 0: the background is every pixel outside the outlines, not a hand-drawn rectangle.
"""
from __future__ import annotations

import json
import os
import platform
import string
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .childproc import NO_WINDOW

__all__ = [
    "UTrackError",
    "MatlabInstall",
    "UTrackInstall",
    "UTrackSettings",
    "LAB_STANDARD",
    "MATLAB_ENV",
    "UTRACK_ENV",
    "find_matlab",
    "find_utrack",
    "resolve_utrack",
    "describe_absence",
    "matlab_command",
    "check_utrack",
    "matlab_path_utrack",
    "frames_folder",
    "frame_name",
    "write_frames",
    "write_background_frames",
    "UTrackMovie",
    "job_function",
    "parse_status",
    "stage_folder",
    "land_movie",
    "run_job",
    "run_batch",
    "record_run",
    "earlier_runs",
    "STOPPED_ERROR",
    "RUN_RECORD_FILENAME",
]


class UTrackError(RuntimeError):
    """A u-track run that cannot be set up or did not run honestly."""


#: Environment variables a user or a test can point at an install; checked first.
MATLAB_ENV = "SMTRACKER_MATLAB"
UTRACK_ENV = "SMTRACKER_UTRACK"

#: The executable inside a MATLAB release folder, per platform.
_MATLAB_EXE = {"Windows": "bin/matlab.exe", "Darwin": "bin/matlab", "Linux": "bin/matlab"}

#: The file that identifies a u-track ``software`` folder: the 2-D single-particle package.
_UTRACK_MARK = Path("@UTrackPackage") / "UTrackPackage.m"


# ------------------------------------------------------------------ finding MATLAB
@dataclass(frozen=True)
class MatlabInstall:
    """A located MATLAB release and how it was located."""

    executable: Path
    release: str                       # "R2016a" -- from the folder, never guessed
    origin: str                        # "environment" | "configured" | "search"
    other_candidates: tuple = ()

    def stamp(self) -> str:
        extra = (f"; {len(self.other_candidates)} other release(s) also found"
                 if self.other_candidates else "")
        return f"MATLAB {self.release} at {self.executable} (found by {self.origin}){extra}"


def _platform() -> str:
    name = platform.system()
    if name not in _MATLAB_EXE:
        raise UTrackError(f"no MATLAB layout is known for platform {name!r}")
    return name


def _drives() -> list:
    """The drive roots on Windows that exist; ``["/"]`` elsewhere."""
    if os.name != "nt":
        return [Path("/")]
    out = []
    for letter in string.ascii_uppercase:
        root = Path(f"{letter}:/")
        try:
            if root.exists():
                out.append(root)
        except OSError:
            continue
    return out


def _release_of(path: Path) -> str:
    """``R2016a`` from a release folder (``R2016a`` or ``MATLAB_R2014b`` or
    ``MATLAB_R2023b.app``); the folder name itself when it says none."""
    name = path.name.removesuffix(".app")
    for part in name.replace("-", "_").split("_"):
        if len(part) == 6 and part[0] == "R" and part[1:5].isdigit() and part[5] in "ab":
            return part
    return name


def _matlab_in(folder: Path, system: str) -> Path | None:
    candidate = folder / _MATLAB_EXE[system]
    return candidate if candidate.is_file() else None


def _release_folders(system: str) -> list:
    """Every folder that may be a MATLAB release: a bounded listing, never a walk."""
    out = []
    if system == "Windows":
        for drive in _drives():
            for base in (drive / "Program Files" / "MATLAB", drive / "MATLAB"):
                try:
                    out += sorted(p for p in base.iterdir() if p.is_dir())
                except OSError:
                    pass
            try:                                   # <drive>:/MATLAB_R2014b, <drive>:/Program Files/MATLAB_R...
                out += sorted(p for p in drive.iterdir()
                              if p.is_dir() and p.name.upper().startswith("MATLAB_R"))
            except OSError:
                pass
    elif system == "Darwin":
        for base in (Path("/Applications"), Path("~/Applications").expanduser()):
            try:
                out += sorted(p for p in base.iterdir() if p.name.startswith("MATLAB_R"))
            except OSError:
                pass
    else:
        for base in (Path("/usr/local/MATLAB"), Path("/opt/MATLAB")):
            try:
                out += sorted(p for p in base.iterdir() if p.is_dir())
            except OSError:
                pass
    return out


def _as_matlab(place: Path, system: str) -> Path | None:
    """``place`` may be the executable, a ``bin`` folder or a release folder."""
    place = Path(place).expanduser()
    if place.is_file() and place.name.lower().startswith("matlab"):
        return place
    if place.is_dir():
        hit = _matlab_in(place, system)
        if hit is None and place.name.lower() == "bin":
            hit = _matlab_in(place.parent, system)
        return hit
    return None


def find_matlab(*, extra_paths=(), system: str | None = None) -> MatlabInstall | None:
    """Look for MATLAB. Order: ``SMTRACKER_MATLAB``, then ``extra_paths`` (a remembered
    choice), then the usual release folders, NEWEST RELEASE FIRST. Every other release
    found is reported, so a machine with R2014b and R2016a does not silently use the
    wrong one. None when nothing is found (`describe_absence` says where it looked)."""
    system = system or _platform()
    ordered = []
    env = os.environ.get(MATLAB_ENV)
    if env:
        ordered.append(("environment", Path(env)))
    ordered += [("configured", Path(p)) for p in extra_paths]
    searched = [("search", p) for p in _release_folders(system)]
    searched.sort(key=lambda item: _release_of(item[1]), reverse=True)
    ordered += searched

    found, seen = [], set()
    for origin, place in ordered:
        exe = _as_matlab(place, system)
        if exe is None:
            continue
        key = os.path.normcase(os.path.abspath(exe))
        if key in seen:
            continue
        seen.add(key)
        root = exe.parent.parent
        found.append((origin, exe, _release_of(root)))
    if not found:
        return None
    origin, exe, release = found[0]
    return MatlabInstall(executable=exe, release=release, origin=origin,
                         other_candidates=tuple(str(e) for _, e, _ in found[1:]))


# ------------------------------------------------------------------ finding u-track
@dataclass(frozen=True)
class UTrackInstall:
    """A located u-track ``software`` folder (the one holding ``@UTrackPackage``)."""

    software: Path
    origin: str
    other_candidates: tuple = ()

    def stamp(self) -> str:
        extra = (f"; {len(self.other_candidates)} other copy/copies also found"
                 if self.other_candidates else "")
        return f"u-track at {self.software} (found by {self.origin}){extra}"


def resolve_utrack(path) -> Path | None:
    """The u-track ``software`` folder at ``path``, or None. ``path`` may be that folder,
    the u-track folder holding it, or a folder holding a ``u-track...`` folder."""
    path = Path(path).expanduser()
    if not path.is_dir():
        return None
    for folder in (path, path / "software"):
        if (folder / _UTRACK_MARK).is_file():
            return folder
    try:
        children = sorted(p for p in path.iterdir()
                          if p.is_dir() and p.name.lower().startswith("u-track"))
    except OSError:
        children = []
    for child in children:
        for folder in (child, child / "software"):
            if (folder / _UTRACK_MARK).is_file():
                return folder
    return None


def _utrack_places(system: str) -> list:
    places = [Path("~/Documents/MATLAB").expanduser(), Path("~").expanduser(),
              Path("~/Downloads").expanduser(), Path("~/Desktop").expanduser()]
    if system == "Windows":
        for drive in _drives():
            places += [drive / "mat", drive, drive / "MATLAB"]
    return places


def matlab_path_utrack(matlab: MatlabInstall | None) -> list:
    """The u-track ``software`` folders on MATLAB's own SAVED path: the release's
    ``toolbox/local/pathdef.m`` and a user ``Documents/MATLAB/pathdef.m``. This is the copy
    MATLAB itself uses when someone runs u-track by hand, so it is the one to prefer when
    a machine holds two (this one holds `u-track-master` and `u-track-magic`)."""
    import re

    files = [Path("~/Documents/MATLAB/pathdef.m").expanduser()]
    if matlab is not None:
        files.insert(0, Path(matlab.executable).parent.parent / "toolbox" / "local"
                     / "pathdef.m")
    out = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for entry in re.findall(r"'([^']+)'", text):
            for folder in entry.split(";"):
                folder = folder.strip()
                if folder and (Path(folder) / _UTRACK_MARK).is_file() and folder not in out:
                    out.append(folder)
    return out


def find_utrack(*, extra_paths=(), matlab: MatlabInstall | None = None,
                system: str | None = None) -> UTrackInstall | None:
    """Look for u-track. Order: ``SMTRACKER_UTRACK``, ``extra_paths``, the copy on
    MATLAB's own saved path (`matlab_path_utrack`), then the usual places (a
    ``u-track...`` folder in Documents/MATLAB, home, Downloads, Desktop, and
    ``<drive>:/mat`` and drive roots on Windows). The first hit wins and the others are
    reported. A listing of a handful of folders, never a walk."""
    system = system or _platform()
    ordered = []
    env = os.environ.get(UTRACK_ENV)
    if env:
        ordered.append(("environment", Path(env)))
    ordered += [("configured", Path(p)) for p in extra_paths]
    ordered += [("MATLAB's path", Path(p)) for p in matlab_path_utrack(matlab)]
    ordered += [("search", p) for p in _utrack_places(system)]
    found, seen = [], set()
    for origin, place in ordered:
        hit = resolve_utrack(place)
        if hit is None:
            continue
        key = os.path.normcase(os.path.abspath(hit))
        if key in seen:
            continue
        seen.add(key)
        found.append((origin, hit))
    if not found:
        return None
    origin, software = found[0]
    return UTrackInstall(software=software, origin=origin,
                         other_candidates=tuple(str(s) for _, s in found[1:]))


def describe_absence(*, matlab: MatlabInstall | None, utrack: UTrackInstall | None) -> str:
    """What to tell a user when MATLAB or u-track is missing."""
    lines = []
    if matlab is None:
        lines.append("No MATLAB was found (looked in <drive>:/Program Files/MATLAB, "
                     "<drive>:/MATLAB and <drive>:/MATLAB_R... on every drive). u-track "
                     "needs MATLAB with the Statistics, Optimization and Image Processing "
                     f"toolboxes. Choose matlab.exe, or set {MATLAB_ENV}.")
    if utrack is None:
        lines.append("No u-track was found (a folder holding software/@UTrackPackage, "
                     "looked for in Documents/MATLAB, home, Downloads, Desktop and "
                     f"<drive>:/mat). Choose the u-track folder, or set {UTRACK_ENV}.")
    lines.append("TrackMate (Fiji) is the other tracker; u-track is optional.")
    return "\n".join(lines)


def matlab_command(install: MatlabInstall, job_folder, function: str, log_path) -> list:
    """The command line that runs ``function`` (an .m file in ``job_folder``) headless.

    R2016a has no ``-batch`` (added in R2019a), so it is ``-r`` with the error caught and
    an explicit exit code; ``-wait`` makes the Windows launcher wait for MATLAB, and
    ``-logfile`` keeps the command window. The function, not a script, because R2016a
    allows no local functions in scripts."""
    folder = str(Path(job_folder)).replace("'", "''")
    call = (f"try, addpath('{folder}'); {function}; catch e, disp(getReport(e)); "
            f"exit(1); end; exit(0)")
    cmd = [str(install.executable), "-nosplash", "-nodesktop", "-minimize",
           "-logfile", str(log_path), "-r", call]
    if platform.system() == "Windows":
        cmd.insert(1, "-wait")
    return cmd


_HEALTH = r"""function {function}()
STATUS = {status};
U = {utrack};
p = genpath(U); parts = strsplit(p, pathsep);
parts = parts(cellfun(@isempty, strfind(parts, [filesep 'bioformats'])));
addpath(strjoin(parts, pathsep));
w = which('UTrackPackage');
tb = {{'statistics_toolbox', 'optimization_toolbox', 'image_toolbox'}};
ok = cellfun(@(x) license('test', x), tb);
line = sprintf('{marker} HEALTH|%s|%s|%d|%d|%d', version, w, ok(1), ok(2), ok(3));
fid = fopen(STATUS, 'a'); fprintf(fid, '%s\n', line); fclose(fid);
end
"""


def check_utrack(matlab: MatlabInstall, utrack: UTrackInstall, *,
                 timeout: float = 300.0) -> dict:
    """Start MATLAB headless, put this u-track on the path, and say what answered:
    ``{"version", "utrack", "statistics", "optimization", "image"}``. The readiness
    check that matters is the one that exercises the exact route a run takes; a missing
    toolbox is reported here rather than as a stack trace twenty minutes into a run."""
    import tempfile

    # IGNORE_CLEANUP_ERRORS: a MATLAB still running when Python exits (the window closed
    # while it started) holds its log open, and the cleanup then printed a PermissionError
    # traceback at exit (reported 14 Oct 2026). The folder is in the temp dir either way.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        folder = Path(folder)
        status = folder / "health.txt"
        function = "smtracker_utrack_health"
        (folder / f"{function}.m").write_text(_HEALTH.format(
            function=function, status=_m_str(status), utrack=_m_str(utrack.software),
            marker=OK_MARKER), encoding="utf-8")
        log = folder / "matlab.log"
        try:
            done = subprocess.run(matlab_command(matlab, folder, function, log),
                                  capture_output=True, text=True, timeout=timeout,
                                  check=False, creationflags=NO_WINDOW)
        except OSError as exc:
            raise UTrackError(f"{matlab.executable} could not be started: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise UTrackError(f"MATLAB did not answer within {timeout:g} s; it may be "
                              f"waiting on a licence or activation window; start it "
                              f"once by hand") from exc
        line = next((ln for ln in _read_status(status)
                     if ln.startswith(f"{OK_MARKER} HEALTH|")), None)
        if line is None:
            tail = "\n".join(_read_status(log)[-8:])
            raise UTrackError(f"MATLAB ran (exit code {done.returncode}) but did not report "
                              f"back. Last lines of its log:\n{tail}")
    _tag, release, which, stats, optim, images = line.split("|", 5)
    return {"version": release.strip(), "utrack": which.strip(),
            "statistics": stats.strip() == "1", "optimization": optim.strip() == "1",
            "image": images.strip() == "1"}


# ------------------------------------------------------------------ settings
@dataclass(frozen=True)
class UTrackSettings:
    """Every u-track setting SMTracker writes, with u-track's own names in the script.

    ``psf_sigma_px`` None: u-track works it out itself from the NA, emission wavelength
    and pixel size the movie is given (``Channel.psfSigma_``), as it did for the lab's
    runs (0.9317 px for mNeonGreen at NA 1.49 and 106 nm). ``frames_bridged`` is u-track
    GUI's "maximum gap"; the script writes ``timeWindow = frames_bridged + 1``.
    ``background_alpha`` None: the background test is off.
    """

    # detection: SubResolutionProcess / detectSubResFeatures2D_StandAlone
    alpha_loc_max: float = 0.025
    psf_sigma_px: float | None = None
    integ_window: int = 0
    do_mmf: bool = False
    alpha_r: float = 0.05
    alpha_a: float = 0.05
    alpha_d: float = 0.05
    alpha_f: float = 0.0
    num_sigma_iter: int = 0
    bit_depth: int = 16
    background_alpha: float | None = None
    # linking: costMatRandomDirectedSwitchingMotionLink
    min_search_radius_px: float = 0.0
    max_search_radius_px: float = 5.0
    brown_std_mult: float = 3.0
    use_local_density: bool = True
    nn_window: int = 1
    # gap closing: gapCloseParam + costMatRandomDirectedSwitchingMotionCloseGaps
    frames_bridged: int = 0
    merge_split: int = 0
    min_track_len: int = 5
    brown_scaling: tuple = (0.5, 0.01)
    gap_penalty: float = 1.5
    amp_ratio_limit: tuple = (0.5, 2.0)
    res_limit_px: float | None = None
    gap_exclude_ms: bool = False
    strategy_bd: float = 0.0

    def __post_init__(self):
        problems = []
        for name in ("alpha_loc_max", "alpha_r", "alpha_a", "alpha_d"):
            if not 0 < getattr(self, name) < 1:
                problems.append(f"{name} must be between 0 and 1")
        if not 0 <= self.alpha_f < 1:
            problems.append("alpha_f must be in [0, 1); 0 turns the final test off")
        if self.background_alpha is not None and not 0 < self.background_alpha < 1:
            problems.append("background_alpha must be between 0 and 1, or None (off)")
        if self.psf_sigma_px is not None and self.psf_sigma_px <= 0:
            problems.append("psf_sigma_px must be positive, or None (u-track derives it)")
        if self.bit_depth <= 0 or self.bit_depth % 2:
            problems.append("bit_depth must be a positive even number (u-track refuses odd)")
        if self.integ_window < 0 or self.num_sigma_iter < 0:
            problems.append("integ_window and num_sigma_iter are 0 or more")
        if not 0 <= self.min_search_radius_px <= self.max_search_radius_px:
            problems.append("the search radius must run 0 <= min <= max")
        if self.max_search_radius_px <= 0 or self.brown_std_mult <= 0:
            problems.append("max_search_radius_px and brown_std_mult must be positive")
        if self.frames_bridged < 0 or self.nn_window < 1 or self.min_track_len < 1:
            problems.append("frames_bridged >= 0, nn_window >= 1, min_track_len >= 1")
        if self.merge_split not in (0, 1, 2, 3):
            problems.append("merge_split is 0 none, 1 both, 2 merge only, 3 split only")
        if problems:
            raise UTrackError("u-track settings: " + "; ".join(problems))

    @property
    def time_window(self) -> int:
        """u-track's ``timeWindow``: the frames bridged plus one."""
        return int(self.frames_bridged) + 1

    def matches_lab(self) -> bool:
        return self == LAB_STANDARD

    def stamp(self) -> str:
        bg = ("off" if self.background_alpha is None
              else f"cell-free background, alpha {self.background_alpha:g}")
        psf = ("from the optics" if self.psf_sigma_px is None
               else f"{self.psf_sigma_px:g} px")
        label = " (standard settings)" if self.matches_lab() else ""
        return (f"u-track{label}: alphaLocMax {self.alpha_loc_max:g}, PSF sigma {psf}, "
                f"background test {bg}; search {self.min_search_radius_px:g}-"
                f"{self.max_search_radius_px:g} px x{self.brown_std_mult:g}, "
                f"{self.frames_bridged} frame(s) bridged, min track {self.min_track_len}, "
                f"merge/split {self.merge_split}")

    def as_record(self) -> dict:
        out = asdict(self)
        out["time_window"] = self.time_window
        return out


#: What the lab's 2063 u-track runs used, read out of every movieData.mat
#: (12 Oct 2026). Four replicate days in the MSD groups differ (protein B/Malate series021-030
#: alphaLocMax 0.01; protein C/Glu + Mal series011-020 minSearchRadius 2; protein C/LB series021-030
#: and protein B/LB series001-010 split only), and are not this preset.
LAB_STANDARD = UTrackSettings()


# ------------------------------------------------------------------ the movie u-track reads
def frames_folder(condition_folder, name: str) -> Path:
    """``<condition>/tif/<movie>/<movie>``: where "Crop ALL" put the frames."""
    return Path(condition_folder) / "tif" / name / name


def frame_name(name: str, index: int, n_frames: int) -> str:
    """``series001_c1_t0001.tif``: u-track's own name (``BioFormatsReader``), padded to at
    least four digits and to the frame count, so every name in a folder has one width
    (u-track builds names from the first file's padding)."""
    width = max(4, len(str(int(n_frames))))
    return f"{name}_c1_t{int(index):0{width}d}.tif"


def _crop_block(raw_path, start: int, end: int):
    """The frames ``start..end`` (1-based inclusive) of an acquisition: the slice
    `movieprep.prepare_movie` cuts, so the pixels are the lab macro's."""
    import numpy as np

    from .rawmovie import read_raw_movie

    movie = read_raw_movie(raw_path)
    n = int(movie.meta.n_frames)
    if not 1 <= int(start) <= int(end) <= n:
        raise UTrackError(f"{Path(raw_path).name}: crop {start}-{end} is outside its "
                          f"{n} frames")
    return np.asarray(movie._frames[int(start) - 1:int(end)])


def write_frames(raw_path, condition_folder, name: str, start: int, end: int) -> Path:
    """Write u-track's working copy of one movie and return the folder.

    One uncompressed 16-bit TIFF per frame, named as u-track's "Crop ALL" names them. A
    folder already holding frames is replaced only by this function's caller deciding so
    (`run_batch` refuses an earlier run unless asked); stale frames left beside new ones
    would be read as part of the movie, so the folder is cleared of ``*.tif`` first."""
    import tifffile

    block = _crop_block(raw_path, start, end)
    folder = frames_folder(condition_folder, name)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.tif"):
        old.unlink()
    n = block.shape[0]
    for i in range(n):
        tifffile.imwrite(folder / frame_name(name, i + 1, n), block[i])
    return folder


def write_background_frames(raw_path, folder, name: str, start: int, end: int,
                            cell_mask) -> tuple[Path, str]:
    """One cell-free image per frame for u-track's background test, and the file stem.

    ``cell_mask`` is True inside the cells (dilate it first: a cell's own halo is not
    background). Cell pixels are written as 0, which u-track reads as NaN, so the
    background is EVERY pixel outside the outlines. Returns ``(folder, filename_base)``
    as ``detectionParam.background`` wants them."""
    import numpy as np
    import tifffile

    block = _crop_block(raw_path, start, end)
    mask = np.asarray(cell_mask, dtype=bool)
    if mask.shape != block.shape[1:]:
        raise UTrackError(f"{name}: the cell mask is {mask.shape}, the frames "
                          f"{block.shape[1:]}")
    if mask.all():
        raise UTrackError(f"{name}: the cells cover the whole field; no background left")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.tif"):
        old.unlink()
    n = block.shape[0]
    base = f"bg_{name}_c1_t"
    for i in range(n):
        frame = block[i].copy()
        frame[mask] = 0
        tifffile.imwrite(folder / (f"bg_{frame_name(name, i + 1, n)}"), frame)
    return folder, base


# ------------------------------------------------------------------ the MATLAB function
@dataclass(frozen=True)
class UTrackMovie:
    """One movie of a job: its frames, where u-track writes, and its calibration."""

    name: str
    frames: Path
    output: Path
    pixel_size_nm: float
    frame_interval_s: float
    numerical_aperture: float | None = None
    emission_nm: float | None = None
    background_dir: Path | None = None
    background_base: str | None = None


def _m_str(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _m_num(value) -> str:
    if value is None:
        return "[]"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (tuple, list)):
        return "[" + " ".join(_m_num(v) for v in value) + "]"
    return repr(float(value)) if not float(value).is_integer() else str(int(value))


OK_MARKER = "SMTRACKER_UT"
STOPPED_ERROR = "not run: the batch was stopped before this movie"

_JOB = r"""function {function}()
% generated by smtracker.utrack_run -- headless u-track. Do not edit; regenerate.
STATUS = {status};
STOP = {stop};
U = {utrack};
p = genpath(U); parts = strsplit(p, pathsep);
% u-track's own Bio-Formats jar needs Java 8; the movies here are TIFF series and need none.
parts = parts(cellfun(@isempty, strfind(parts, [filesep 'bioformats'])));
addpath(strjoin(parts, pathsep));
say(STATUS, sprintf('{marker} BEGIN n=%d utrack=%s matlab=%s', {n}, U, version));
names = {names};
frames = {frames};
outs = {outs};
calib = {calib};
bgdirs = {bgdirs};
bgbases = {bgbases};
for i = 1:numel(names)
    if exist(STOP, 'file')
        say(STATUS, sprintf('{marker} STOPPED %s', names{{i}}));
        continue;
    end
    say(STATUS, sprintf('{marker} START %s', names{{i}}));
    t0 = tic;
    try
        [nTracks, nFrames] = run_one(frames{{i}}, outs{{i}}, calib(i, :), bgdirs{{i}}, bgbases{{i}});
        say(STATUS, sprintf('{marker} OK %s tracks=%d frames=%d seconds=%.1f', ...
            names{{i}}, nTracks, nFrames, toc(t0)));
    catch e
        msg = strrep(strrep(getReport(e, 'basic'), sprintf('\n'), ' | '), sprintf('\r'), '');
        say(STATUS, sprintf('{marker} FAIL %s %s', names{{i}}, msg));
    end
end
say(STATUS, '{marker} END');
end

function [nTracks, nFrames] = run_one(frameDir, outDir, c, bgDir, bgBase)
ch = Channel(frameDir);
if ~isnan(c(4)), ch.emissionWavelength_ = c(4); end
MD = MovieData(ch, outDir);
MD.setPath(outDir);
MD.setFilename('movieData.mat');
MD.pixelSize_ = c(1);
MD.timeInterval_ = c(2);
if ~isnan(c(3)), MD.numAperture_ = c(3); end
MD.camBitdepth_ = {bit_depth};
MD.sanityCheck();
nFrames = MD.nFrames_;
pkg = UTrackPackage(MD);
MD.addPackage(pkg);
pkg.createDefaultProcess(1);
pkg.createDefaultProcess(2);

det = pkg.getProcess(1);
dp = det.funParams_;
dp.ChannelIndex = 1;
dp.firstImageNum = 1;
dp.lastImageNum = nFrames;
d = struct();
d.psfSigma = {psf_sigma};
if isempty(d.psfSigma), d.psfSigma = dp.detectionParam.psfSigma; end
if isempty(d.psfSigma)
    error('smtracker:psf', 'no PSF sigma: give the NA, emission wavelength and pixel size, or type a sigma');
end
d.bitDepth = {bit_depth};
d.alphaLocMax = {alpha_loc_max};
d.integWindow = {integ_window};
d.doMMF = {do_mmf};
d.testAlpha = struct('alphaR', {alpha_r}, 'alphaA', {alpha_a}, 'alphaD', {alpha_d}, 'alphaF', {alpha_f});
d.numSigmaIter = {num_sigma_iter};
d.visual = 0;
if isempty(bgDir)
    d.background = [];
else
    d.background = struct('imageDir', bgDir, 'filenameBase', bgBase, 'alphaLocMaxAbs', {background_alpha});
end
dp.detectionParam = d;
det.setPara(dp);
det.run();

trk = pkg.getProcess(2);
tp = trk.funParams_;
tp.ChannelIndex = 1;
tp.DetProcessIndex = find(cellfun(@(x) isa(x, 'SubResolutionProcess'), MD.processes_));
tp.probDim = 2;
tp.verbose = 1;
tw = {time_window};
tp.gapCloseParam = struct('timeWindow', tw, 'mergeSplit', {merge_split}, ...
    'minTrackLen', {min_track_len}, 'diagnostics', 0);
link = tp.costMatrices(1).parameters;
link.linearMotion = 0;
link.minSearchRadius = {min_search_radius};
link.maxSearchRadius = {max_search_radius};
link.brownStdMult = {brown_std_mult};
link.useLocalDensity = {use_local_density};
link.nnWindow = {nn_window};
link.kalmanInitParam = [];
link.diagnostics = [];
tp.costMatrices(1).funcName = 'costMatRandomDirectedSwitchingMotionLink';
tp.costMatrices(1).parameters = link;
gap = tp.costMatrices(2).parameters;
gap.linearMotion = 0;
gap.minSearchRadius = {min_search_radius};
gap.maxSearchRadius = {max_search_radius};
gap.brownStdMult = {brown_std_mult} * ones(tw, 1);
gap.useLocalDensity = {use_local_density};
gap.nnWindow = {nn_window};
gap.brownScaling = {brown_scaling};
gap.timeReachConfB = tw;
gap.ampRatioLimit = {amp_ratio_limit};
gap.lenForClassify = 5;
gap.linStdMult = 3 * ones(tw, 1);
gap.linScaling = [1 0.01];
gap.timeReachConfL = tw;
gap.maxAngleVV = 30;
gap.gapPenalty = {gap_penalty};
gap.resLimit = {res_limit};
gap.gapExcludeMS = {gap_exclude_ms};
gap.strategyBD = {strategy_bd};
tp.costMatrices(2).funcName = 'costMatRandomDirectedSwitchingMotionCloseGaps';
tp.costMatrices(2).parameters = gap;
trk.setPara(tp);
trk.run();
s = load(trk.outFilePaths_{{1}}, 'tracksFinal');
nTracks = numel(s.tracksFinal);
end

function say(path, line)
fid = fopen(path, 'a');
if fid > 0
    fprintf(fid, '%s\n', line);
    fclose(fid);
end
fprintf('%s\n', line);
end
"""


def job_function(movies, settings: UTrackSettings, *, utrack: UTrackInstall, status,
                 stop, function: str = "smtracker_utrack_job") -> str:
    """The MATLAB function that runs ``movies`` one after another, as text.

    PURE: values in, text out, so it is tested headless. Each movie is a fresh MovieData,
    a fresh UTrackPackage, detection then tracking, with the FULL ``detectionParam`` and
    both cost matrices written out (u-track merges only top-level fields, and an inherited
    default would sit unseen beside the ones set). The figure-drawing diagnostics and the
    per-frame image viewer are switched off, so nothing waits on a window. A movie that
    fails is a FAIL line with u-track's own report; the STOP file is asked before each.
    """
    movies = list(movies)
    if not movies:
        raise UTrackError("no movies to track")
    if settings.background_alpha is not None and any(m.background_dir is None
                                                     for m in movies):
        raise UTrackError("the background test is on, but a movie has no background "
                          "frames (write_background_frames)")

    def cell(values):
        return "{" + ", ".join(values) + "}"

    calib_rows = []
    for m in movies:
        if not m.pixel_size_nm or not m.frame_interval_s:
            raise UTrackError(f"{m.name}: pixel size and frame interval are required "
                              f"(neither has a default)")
        calib_rows.append(" ".join([
            _m_num(m.pixel_size_nm), _m_num(m.frame_interval_s),
            "NaN" if m.numerical_aperture is None else _m_num(m.numerical_aperture),
            "NaN" if m.emission_nm is None else _m_num(m.emission_nm)]))
    s = settings
    return _JOB.format(
        function=function, marker=OK_MARKER, n=len(movies),
        status=_m_str(status), stop=_m_str(stop), utrack=_m_str(utrack.software),
        names=cell(_m_str(m.name) for m in movies),
        frames=cell(_m_str(m.frames) for m in movies),
        outs=cell(_m_str(m.output) for m in movies),
        calib="[" + "; ".join(calib_rows) + "]",
        bgdirs=cell(_m_str(str(m.background_dir) + os.sep) if m.background_dir else "''"
                    for m in movies),
        bgbases=cell(_m_str(m.background_base or "") for m in movies),
        bit_depth=_m_num(s.bit_depth), psf_sigma=_m_num(s.psf_sigma_px),
        alpha_loc_max=_m_num(s.alpha_loc_max), integ_window=_m_num(s.integ_window),
        do_mmf=_m_num(s.do_mmf), alpha_r=_m_num(s.alpha_r), alpha_a=_m_num(s.alpha_a),
        alpha_d=_m_num(s.alpha_d), alpha_f=_m_num(s.alpha_f),
        num_sigma_iter=_m_num(s.num_sigma_iter),
        background_alpha=_m_num(s.background_alpha),
        time_window=_m_num(s.time_window), merge_split=_m_num(s.merge_split),
        min_track_len=_m_num(s.min_track_len),
        min_search_radius=_m_num(s.min_search_radius_px),
        max_search_radius=_m_num(s.max_search_radius_px),
        brown_std_mult=_m_num(s.brown_std_mult),
        use_local_density=_m_num(s.use_local_density), nn_window=_m_num(s.nn_window),
        brown_scaling=_m_num(s.brown_scaling), amp_ratio_limit=_m_num(s.amp_ratio_limit),
        gap_penalty=_m_num(s.gap_penalty), res_limit=_m_num(s.res_limit_px),
        gap_exclude_ms=_m_num(s.gap_exclude_ms), strategy_bd=_m_num(s.strategy_bd))


def parse_status(lines) -> dict:
    """``{"begin": str|None, "rows": {movie: row}, "ended": bool}`` from the status file.

    A row is ``{"movie", "ok", "error", "n_tracks", "n_frames", "seconds", "started"}``.
    A movie that STARTED and has no outcome yet is in flight (or MATLAB died on it)."""
    begin, ended, rows = None, False, {}
    for line in lines:
        if not line.startswith(OK_MARKER + " "):
            continue
        rest = line[len(OK_MARKER) + 1:]
        kind, _, tail = rest.partition(" ")
        if kind == "BEGIN":
            begin = tail
        elif kind == "END":
            ended = True
        elif kind in ("START", "OK", "FAIL", "STOPPED"):
            movie, _, detail = tail.partition(" ")
            row = rows.setdefault(movie, {"movie": movie, "ok": False, "error": "",
                                          "n_tracks": None, "n_frames": None,
                                          "seconds": None, "started": False})
            if kind == "START":
                row["started"] = True
            elif kind == "OK":
                fields = dict(part.split("=", 1) for part in detail.split() if "=" in part)
                row.update(ok=True, error="",
                           n_tracks=int(fields.get("tracks", 0)),
                           n_frames=int(fields.get("frames", 0)),
                           seconds=float(fields.get("seconds", "nan")))
            elif kind == "FAIL":
                row.update(ok=False, error=detail.strip() or "failed")
            else:
                row.update(ok=False, error=STOPPED_ERROR)
    return {"begin": begin, "rows": rows, "ended": ended}


def _read_status(path) -> list:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def run_job(movies, settings: UTrackSettings, *, matlab: MatlabInstall,
            utrack: UTrackInstall, job_folder, on_progress=None, stop=None,
            poll_s: float = 1.0, timeout_s: float | None = None) -> dict:
    """Run one MATLAB session over ``movies`` and return `parse_status`'s result plus
    ``returncode`` and ``log``.

    ONE SESSION PER CONDITION: MATLAB takes tens of seconds to start, so starting it per
    movie would cost more than some movies. ``on_progress(row)`` is called for each movie
    as its outcome line appears; ``stop()`` is asked while waiting, and a True answer
    writes the STOP file, which the job reads before each movie -- so the movie in hand
    finishes and the rest are reported STOPPED, never killed half-written."""
    job_folder = Path(job_folder)
    job_folder.mkdir(parents=True, exist_ok=True)
    status = job_folder / "status.txt"
    stop_file = job_folder / "STOP"
    log = job_folder / "matlab.log"
    for stale in (status, stop_file):
        if stale.exists():
            stale.unlink()
    function = "smtracker_utrack_job"
    (job_folder / f"{function}.m").write_text(
        job_function(movies, settings, utrack=utrack, status=status, stop=stop_file,
                     function=function), encoding="utf-8")
    try:
        proc = subprocess.Popen(matlab_command(matlab, job_folder, function, log),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
    except OSError as exc:
        raise UTrackError(f"{matlab.executable} could not be started: {exc}") from exc

    reported: set = set()
    started = time.monotonic()
    while True:
        done = proc.poll() is not None
        parsed = parse_status(_read_status(status))
        for movie, row in parsed["rows"].items():
            final = row["ok"] or row["error"]
            if final and movie not in reported:
                reported.add(movie)
                if on_progress is not None:
                    on_progress(dict(row))
        if done:
            break
        if stop is not None and stop() and not stop_file.exists():
            stop_file.write_text("stop", encoding="utf-8")
        if timeout_s is not None and time.monotonic() - started > timeout_s:
            proc.kill()
            raise UTrackError(f"MATLAB did not finish within {timeout_s:g} s; see {log}")
        time.sleep(poll_s)
    parsed = parse_status(_read_status(status))
    parsed["returncode"] = proc.returncode
    parsed["log"] = str(log)
    return parsed


# ------------------------------------------------------------------ a condition
RUN_RECORD_FILENAME = "utrack_run.json"


def earlier_runs(condition_folder, names) -> list:
    """The movies of ``names`` that already hold a u-track run (``movieData.mat`` in
    ``tif/<movie>/``). A second run lands on the first, so the caller must have asked."""
    root = Path(condition_folder) / "tif"
    return [n for n in names if (root / n / "movieData.mat").is_file()]


def record_run(condition_folder, *, settings: UTrackSettings, matlab, utrack, movies,
               crops, pixel_size_nm, frame_interval_s, numerical_aperture=None,
               emission_nm=None, outcome=None, movie_intervals: dict | None = None,
               work_folder=None, at_once: int = 1) -> Path:
    """What the run was GIVEN, beside what it produced, in ``tif/utrack_run.json``.
    Written before the run (its inputs matter most when a run dies) and again after."""
    from datetime import datetime, timezone

    out = Path(condition_folder) / "tif"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "smtracker.utrack_run",
        "matlab": None if matlab is None else {"executable": str(matlab.executable),
                                               "release": matlab.release},
        "utrack": None if utrack is None else str(utrack.software),
        "settings": settings.as_record(),
        "stamp": settings.stamp(),
        "calibration": {"pixel_size_nm": pixel_size_nm, "frame_interval_s": frame_interval_s,
                        "numerical_aperture": numerical_aperture, "emission_nm": emission_nm},
        # The movies tracked at their OWN measured interval rather than the condition's
        # (`run_batch`'s ``movie_intervals``); each one's movieData.mat carries it.
        "movie_frame_interval_s": {str(k): float(v)
                                   for k, v in (movie_intervals or {}).items()},
        "movies": list(movies),
        "crops": {str(k): list(v) for k, v in (crops or {}).items()},
        "crops_note": "1-based inclusive frame ranges of the acquisition, one per movie; "
                      "the frames u-track read are tif/<movie>/<movie>/*.tif.",
        "outcome": outcome,
        # Where u-track worked before each movie's results were moved here (`run_batch`'s
        # ``work_folder``); None = in place. Its movieData.mat paths name that folder.
        "work_folder": None if work_folder is None else str(work_folder),
        # MATLAB sessions that shared the movies (`run_batch`'s ``at_once``); settings and results do not depend on it
        "movies_at_once": int(at_once or 1),
    }
    path = out / RUN_RECORD_FILENAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def stage_folder(work_folder, condition_folder) -> Path:
    """One condition's working folder on a fast disk: the condition's name and a short hash
    of its full path, so two conditions sharing a name never share a folder."""
    import hashlib

    condition_folder = Path(condition_folder)
    key = hashlib.sha1(str(condition_folder.resolve()).encode("utf-8")).hexdigest()[:8]
    return Path(work_folder) / f"{condition_folder.name}-{key}"


def land_movie(stage, condition_folder, name: str) -> Path:
    """Move one movie's u-track folder (frames, movieData.mat, TrackingPackage, backups)
    from the work folder into ``<condition>/tif/<movie>``, replacing what is there item by
    item (only a run asked to replace gets here), and remove the working copy. Returns the
    project folder."""
    import shutil

    src = Path(stage) / "tif" / name
    dst = Path(condition_folder) / "tif" / name
    dst.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        for item in list(src.iterdir()):
            target = dst / item.name
            if item.is_dir() and item.name == "backups" and target.is_dir():
                # MERGED, never replaced: the project's backups/ may hold an earlier run
                # kept on purpose (a re-track backs it up there first), and u-track's own
                # backups/ of this run must not wipe it
                for child in list(item.iterdir()):
                    into = target / child.name
                    if into.is_dir():
                        shutil.rmtree(into)
                    elif into.exists():
                        into.unlink()
                    shutil.move(str(child), str(into))
                continue
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            shutil.move(str(item), str(target))
        shutil.rmtree(src, ignore_errors=True)
    return dst


def run_batch(movies, condition_folder, *, settings: UTrackSettings, matlab: MatlabInstall,
              utrack: UTrackInstall, crops: dict, pixel_size_nm: float,
              frame_interval_s: float, numerical_aperture: float | None = None,
              emission_nm: float | None = None, names: dict | None = None,
              background_masks: dict | None = None, replace: bool = False,
              job_folder=None, on_progress=None, stop=None,
              movie_intervals: dict | None = None, work_folder=None, at_once: int = 1):
    """Track one condition's acquisitions with u-track and return a row per movie.

    ``movies`` are acquisition paths, ``crops`` ``{movie: (first, last)}`` (1-based,
    inclusive, the saved ranges), ``names`` ``{path stem: movie name}`` when they differ.
    Refused BEFORE anything is written: a movie with no range, and -- unless ``replace`` --
    a movie that already holds a u-track run. Frames are written per movie (u-track's
    working copy), then ONE MATLAB session tracks them all. A failure is a row with its
    error, a stopped movie a row with `STOPPED_ERROR`: never an absence.

    ``movie_intervals`` ``{movie: seconds}`` gives a movie its OWN frame interval in its
    movieData.mat instead of the condition's ``frame_interval_s``: a condition can hold
    days recorded at different intervals (one user's 50.48 and 51.75 ms), and the analysis
    carries tau per movie (`utrack_meta.acquisition_params_from_project`).

    ``work_folder`` (a FAST disk) is where the frames are written and u-track works; each
    movie's folder is moved into the project as soon as its outcome line appears
    (`land_movie`). MEASURED on one user's movies on a USB hard disk: three sessions reading
    thousands of small TIFFs from one spinning disk spent ~7 minutes a movie before
    detection started (38 s on the internal SSD), so the sessions waited on the disk and
    ran no faster together than one alone.

    ``at_once`` MATLAB sessions share the movies, dealt in turn (movie 1, 2, 3 to sessions 1, 2, 3, movie 4 to
    session 1 ...), each in ``utrack_job/lane<i>``; ``on_progress`` is called from whichever session finished a
    movie, one call at a time.
    """
    import pandas as pd

    condition_folder = Path(condition_folder)
    movie_intervals = dict(movie_intervals or {})
    named = [(Path(m), (names or {}).get(Path(m).stem, Path(m).stem)) for m in movies]
    missing = [n for _p, n in named if n not in crops]
    if missing:
        raise UTrackError(f"no saved crop range for {', '.join(missing[:4])}"
                          + (" ..." if len(missing) > 4 else ""))
    earlier = earlier_runs(condition_folder, [n for _p, n in named])
    if earlier and not replace:
        raise UTrackError(f"{len(earlier)} movie(s) already hold a u-track run "
                          f"({', '.join(earlier[:3])}{' ...' if len(earlier) > 3 else ''}); "
                          f"tick Replace to run again")
    if settings.background_alpha is not None and not background_masks:
        raise UTrackError("the background test needs the cell outlines of every movie")

    stage = (condition_folder if work_folder is None
             else stage_folder(work_folder, condition_folder))
    record_run(condition_folder, settings=settings, matlab=matlab, utrack=utrack,
               movies=[n for _p, n in named], crops=crops, pixel_size_nm=pixel_size_nm,
               frame_interval_s=frame_interval_s, numerical_aperture=numerical_aperture,
               emission_nm=emission_nm, movie_intervals=movie_intervals,
               work_folder=work_folder, at_once=at_once)
    jobs, rows = [], {}
    for path, name in named:
        if stop is not None and stop():
            rows[name] = {"movie": name, "ok": False, "error": STOPPED_ERROR}
            continue
        start, end = crops[name]
        try:
            frames = write_frames(path, stage, name, start, end)
            bg_dir = bg_base = None
            if settings.background_alpha is not None:
                mask = (background_masks or {}).get(name)
                if mask is None:
                    raise UTrackError("no cell outlines for the background test")
                bg_dir, bg_base = write_background_frames(
                    path, stage / "tif" / name / "background", name, start,
                    end, mask)
            jobs.append(UTrackMovie(
                name=name, frames=frames, output=stage / "tif" / name,
                pixel_size_nm=float(pixel_size_nm),
                frame_interval_s=float(movie_intervals.get(name, frame_interval_s)),
                numerical_aperture=numerical_aperture, emission_nm=emission_nm,
                background_dir=bg_dir, background_base=bg_base))
        except Exception as exc:                        # noqa: BLE001 - kept and named
            rows[name] = {"movie": name, "ok": False,
                          "error": f"frames not written: {type(exc).__name__}: {exc}"}
    if jobs:
        folder = Path(job_folder) if job_folder else condition_folder / "tif" / "utrack_job"
        landed: set = set()
        lock = threading.Lock()

        def progress(row):
            # a finished movie leaves the work folder at once, failed or not: what u-track
            # wrote belongs in the project, and the fast disk only holds what is in hand
            with lock:
                if stage != condition_folder and row.get("movie") in {j.name for j in jobs}:
                    land_movie(stage, condition_folder, row["movie"])
                    landed.add(row["movie"])
                if on_progress is not None:
                    on_progress(row)

        # MOVIES AT ONCE (23 Oct 2026, `prep_model.MAX_AT_ONCE`): the movies dealt in turn to ``at_once`` MATLAB
        # sessions side by side, each in its own job folder (``utrack_job/lane<i>``) with its own STATUS and STOP
        # file, so Stop still finishes every session's movie in hand. One lane is the run it always was.
        n_lanes = max(1, min(int(at_once or 1), len(jobs)))
        lanes = [jobs[i::n_lanes] for i in range(n_lanes)]
        folders = [folder] if len(lanes) == 1 else [folder / f"lane{i + 1}" for i in range(len(lanes))]
        results: dict = {}
        try:
            if len(lanes) == 1:
                lane_results = [run_job(jobs, settings, matlab=matlab, utrack=utrack, job_folder=folder,
                                        on_progress=progress, stop=stop)]
            else:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
                    futures = [pool.submit(run_job, lane, settings, matlab=matlab, utrack=utrack,
                                           job_folder=where, on_progress=progress, stop=stop)
                               for lane, where in zip(lanes, folders)]
                    errors = [f.exception() for f in futures]
                if any(errors):
                    raise next(e for e in errors if e is not None)
                lane_results = [f.result() for f in futures]
            for lane, got in zip(lanes, lane_results):
                results.update({job.name: got for job in lane})
        finally:
            if stage != condition_folder:
                import shutil

                for job in jobs:
                    if job.name not in landed:
                        # never reached (stopped) or MATLAB died: its working copy goes,
                        # and anything u-track wrote is moved in rather than lost
                        land_movie(stage, condition_folder, job.name)
                shutil.rmtree(stage, ignore_errors=True)
        for job in jobs:
            result = results[job.name]
            row = result["rows"].get(job.name)
            if row is None:
                died = ("MATLAB ended before this movie (exit code "
                        f"{result['returncode']}); see {result['log']}")
                row = {"movie": job.name, "ok": False, "error": died}
            elif not (row["ok"] or row["error"]):
                row = dict(row, error=("MATLAB stopped during this movie (exit code "
                                       f"{result['returncode']}); see {result['log']}"))
            rows[job.name] = row
    table = pd.DataFrame([dict({"movie": n, "ok": False, "error": "", "n_tracks": None,
                                "n_frames": None, "seconds": None}, **rows.get(n, {}))
                          for _p, n in named])
    table["mat"] = [str(condition_folder / "tif" / n / "TrackingPackage" / "tracks"
                        / "Channel_1_tracking_result.mat") if ok else ""
                    for n, ok in zip(table["movie"], table["ok"])]
    stopped = int((table["error"] == STOPPED_ERROR).sum())
    record_run(condition_folder, settings=settings, matlab=matlab, utrack=utrack,
               movies=[n for _p, n in named], crops=crops, pixel_size_nm=pixel_size_nm,
               frame_interval_s=frame_interval_s, numerical_aperture=numerical_aperture,
               emission_nm=emission_nm, movie_intervals=movie_intervals,
               work_folder=work_folder, at_once=at_once,
               outcome={"movies": int(len(table)), "tracked": int(table["ok"].sum()),
                        "failed": int((~table["ok"]).sum()) - stopped,
                        "stopped": stopped,
                        "tracks": int(pd.to_numeric(table["n_tracks"]).fillna(0).sum())})
    return table
