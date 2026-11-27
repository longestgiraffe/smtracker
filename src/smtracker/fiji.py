"""Finding the Fiji that runs TrackMate, without shipping or fetching it.

Why not bundle it. TrackMate is a Fiji/ImageJ plugin written in Java; there is no
standalone or pip-installable form, so "include TrackMate" necessarily means "include
Fiji". Shipping Fiji inside the installer is LEGAL, GPL's mere-aggregation provision
means two separate programs on one medium do not become one work, and SMTracker calls
Fiji as a SUBPROCESS rather than linking it, but redistributing GPLv3 components brings
obligations with it: corresponding source or a written offer, preserved notices, rights
passed downstream, and Fiji is a mixed-licence bundle so that is per component.

(That is a reading of how these licences work, not legal advice. An installer leaving the
group is worth a second opinion.)

The IN-PROCESS route is deliberately not taken. `pyimagej` loads Java classes into the
Python interpreter through JPype, which is far closer to linking than a subprocess is and
is legally murkier for exactly the reason above. Everything here shells out.

**THE DOWNLOAD BUTTON IS GONE, AND THE REASON IS A RENAMED FILE.** SMTracker used to fetch
`downloads.imagej.net/fiji/latest/fiji-win64.zip` behind a confirm. Fiji renamed its
archives (`fiji-stable-win64-jdk.zip`, `fiji-latest-win64-jdk.zip`, 520-700 MB), the old
address answers 404, and the button failed on every machine (reported 12 Oct 2026). Rather
than chase the next rename, the user downloads Fiji from `FIJI_WEBSITE` and points at it.

**TWO GENERATIONS OF LAUNCHER.** Fiji up to 2.14 starts through the ImageJ launcher
(`ImageJ-win64.exe`, `Contents/MacOS/ImageJ-macosx`, `ImageJ-linux64`). Current Fiji
starts through Jaunch (`fiji-windows-x64.exe`, `Fiji.app/Contents/MacOS/fiji-macos-arm64`
beside the jars on a Mac, `fiji-linux-x64`); the stable download still carries an
`ImageJ-win64.exe` beside it, a newer one may not, and the unpacked folder may be called
`Fiji` rather than `Fiji.app`. A search that knew only the first generation missed every
current install -- reported from Lucas's PC, whose Fiji starts from
`fiji-windows-x64.exe`. MEASURED on the Oct 2026 download: one folder, `Fiji`, holding
`fiji-windows-x64.exe` and no `ImageJ-win64.exe` at all. Both generations are searched,
the current one first.

**AND THE CURRENT LAUNCHER WANTS ANOTHER COMMAND LINE, measured on Fiji 2.16.0.**
`--headless --console --run script` exits 1 in about 6 s without running the script: that
launcher config turns `--headless` into ImageJ's `-batch`, which breaks `--run` (Fiji's own
note, fiji/fiji#416; the Oct 2026 config no longer does it). `--no-splash
-Djava.awt.headless=true --run script` runs the script headless and exits 0 on both, and
the old `ImageJ-win64.exe` line still works where that launcher exists. And NOTHING comes
back through a pipe: Jaunch attaches to the calling console instead. So the scripts here
also write their marker to a file (`status_lines`), which works with either launcher.

u-track remains canonical. This adds a second tracker beside it, exactly as
`smtracker.trackmate` (the reader) already assumes: the reference data is u-track and must
stay readable forever, and TrackMate output is never a substitute for it.
"""
from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .childproc import NO_WINDOW

__all__ = [
    "FijiError",
    "FijiInstall",
    "FIJI_WEBSITE",
    "EXECUTABLES",
    "find_fiji",
    "resolve_root",
    "fiji_command",
    "status_lines",
    "launcher_hint",
    "describe_absence",
    "check_fiji",
    "HEALTH_MARKER",
]


class FijiError(RuntimeError):
    """Raised when Fiji cannot be located or run honestly."""


#: Where a user gets Fiji. Shown, never contacted: SMTracker downloads nothing.
FIJI_WEBSITE = "https://imagej.net/software/fiji/downloads"

#: The launcher inside a Fiji folder, per platform, relative to the folder holding
#: `jars/` (or, for a current Mac Fiji picked by its bundle, relative to the bundle).
#: Current (Jaunch) launchers first, then the ImageJ launcher of Fiji <= 2.14; the first
#: that exists is used, this machine's CPU first (`_by_machine`).
EXECUTABLES = {
    "Windows": ("fiji-windows-x64.exe", "fiji-windows-arm64.exe",
                "ImageJ-win64.exe", "ImageJ-win32.exe"),
    "Darwin": ("Fiji.app/Contents/MacOS/fiji-macos-arm64",
               "Fiji.app/Contents/MacOS/fiji-macos-x64",
               "Contents/MacOS/fiji-macos-arm64", "Contents/MacOS/fiji-macos-x64",
               "Contents/MacOS/ImageJ-macosx"),
    "Linux": ("fiji-linux-x64", "fiji-linux-arm64", "ImageJ-linux64", "ImageJ-linux32"),
}

#: Where a Fiji usually sits, per platform. Searched in order; the FIRST hit wins and is
#: REPORTED, so a machine with two installs does not silently use the wrong one. Each
#: place is tried as `Fiji.app` (the usual name) and as `Fiji`.
_SEARCH = {
    "Windows": ("C:/Fiji.app", "C:/Fiji", "C:/Program Files/Fiji.app",
                "C:/Program Files/Fiji", "C:/Program Files (x86)/Fiji.app",
                "~/Fiji.app", "~/Fiji", "~/Desktop/Fiji.app", "~/Desktop/Fiji",
                "~/Downloads/Fiji.app", "~/Downloads/Fiji", "D:/Fiji.app", "D:/Fiji"),
    "Darwin": ("/Applications/Fiji.app", "/Applications/Fiji", "~/Applications/Fiji.app",
               "~/Applications/Fiji", "~/Fiji.app", "~/Fiji"),
    "Linux": ("/opt/Fiji.app", "/opt/Fiji", "/usr/local/Fiji.app", "~/Fiji.app", "~/Fiji",
              "~/fiji/Fiji.app"),
}

#: Folders whose children named "fiji..." are looked at too, and THEIR "fiji..." children:
#: an unpacked download often lands as `Downloads/fiji-stable-win64-jdk/Fiji.app`. A
#: listing of a handful of folders, never a recursive walk.
_NEAR = {
    "Windows": ("~", "~/Desktop", "~/Downloads", "~/Documents", "C:/", "D:/",
                "C:/Program Files"),
    "Darwin": ("~", "~/Desktop", "~/Downloads", "~/Applications", "/Applications"),
    "Linux": ("~", "~/Desktop", "~/Downloads", "/opt"),
}

#: Environment variable a user or a test can point at an install, checked before anything
#: is searched for. An explicit answer always beats a guess.
FIJI_ENV = "SMTRACKER_FIJI"


@dataclass(frozen=True)
class FijiInstall:
    """A located Fiji, and how it was located."""

    root: Path
    executable: Path
    origin: str                       # "environment" | "configured" | "search"
    has_trackmate: bool = False
    other_candidates: tuple = ()

    @property
    def jaunch(self) -> bool:
        """True for the current launcher (`fiji-windows-x64.exe` ...), False for the
        ImageJ launcher of Fiji <= 2.14 (`ImageJ-win64.exe` ...)."""
        return not self.executable.name.startswith("ImageJ-")

    def stamp(self) -> str:
        extra = (f"; {len(self.other_candidates)} other install(s) also found"
                 if self.other_candidates else "")
        return (f"Fiji at {self.root} (found by {self.origin}, TrackMate "
                f"{'present' if self.has_trackmate else 'not found'}){extra}")


def _platform() -> str:
    name = platform.system()
    if name not in EXECUTABLES:
        raise FijiError(
            f"no Fiji launcher is known for platform {name!r}; "
            f"known: {', '.join(sorted(EXECUTABLES))}")
    return name


def _by_machine(names) -> tuple:
    """The launchers for THIS CPU first: an arm64 launcher cannot start on an Intel
    machine. The sort is stable, so the generation order is kept within each CPU."""
    arm = platform.machine().lower() in ("arm64", "aarch64")
    return tuple(sorted(names, key=lambda n: ("arm64" in n) != arm))


def _executable_in(root: Path, system: str) -> Path | None:
    for name in _by_machine(EXECUTABLES[system]):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def launcher_hint(system: str | None = None) -> str:
    """What a Fiji folder looks like, for a message about a folder that is not one."""
    system = system or _platform()
    names = [Path(n).name for n in EXECUTABLES[system]]
    current = next(n for n in names if not n.startswith("ImageJ-"))
    older = next(n for n in names if n.startswith("ImageJ-"))
    return (f"a Fiji folder (usually Fiji.app, or Fiji) holding {current}, or {older} "
            f"in an older Fiji")


def _has_trackmate(root: Path) -> bool:
    """Is TrackMate actually in this Fiji?

    Checked rather than assumed: Fiji ships it by default, but a trimmed or very old
    install may not have it, and discovering that from a Java stack trace after a
    twenty-minute run is the worst possible moment.
    """
    plugins = root / "plugins"
    jars = root / "jars"
    for folder in (plugins, jars):
        if folder.is_dir() and any(folder.glob("**/TrackMate*.jar")):
            return True
    return False


def _fiji_children(folder: Path) -> list:
    try:
        return sorted(p for p in folder.iterdir()
                      if p.is_dir() and p.name.lower().startswith("fiji"))
    except OSError:
        return []


def resolve_root(path, *, system: str | None = None) -> tuple[Path, Path] | None:
    """The Fiji at ``path`` as ``(root, launcher)``, or None.

    ``path`` may be the Fiji folder itself, the folder a download unpacked into (one
    holding a ``Fiji.app`` / ``Fiji`` / ``fiji-...`` folder, two levels at most), or on a
    Mac the ``Fiji.app`` bundle of a current Fiji, whose jars sit in the folder above it.
    The root is the folder holding ``jars/``, which is where TrackMate is looked for.
    """
    system = system or _platform()
    path = Path(path).expanduser()
    if not path.is_dir():
        return None
    children = _fiji_children(path)
    for folder in (path, *children, *(g for c in children for g in _fiji_children(c))):
        launcher = _executable_in(folder, system)
        if launcher is None:
            continue
        root = folder
        if not (root / "jars").is_dir() and (root.parent / "jars").is_dir():
            root = root.parent                     # a current Mac Fiji's .app bundle
        return root, launcher
    return None


def _near_candidates(system: str) -> list:
    out = []
    for base in _NEAR.get(system, ()):
        out += _fiji_children(Path(base).expanduser())
    return out


def find_fiji(*, extra_paths=(), system: str | None = None) -> FijiInstall | None:
    """Look for an installed Fiji. It only looks: SMTracker downloads nothing.

    Order: the ``SMTRACKER_FIJI`` environment variable, then any ``extra_paths`` the
    caller supplies (a remembered choice from the UI), then the usual locations, then
    folders named ``fiji...`` beside them. Each place may be a Fiji folder or the folder a
    download unpacked into (`resolve_root`). Returns None when nothing is found; the
    reason is in :func:`describe_absence` rather than raised, because "not installed" is
    an ordinary state and not an error.
    """
    system = system or _platform()

    ordered: list[tuple[str, Path]] = []
    env = os.environ.get(FIJI_ENV)
    if env:
        ordered.append(("environment", Path(env).expanduser()))
    ordered += [("configured", Path(p).expanduser()) for p in extra_paths]
    ordered += [("search", Path(p).expanduser()) for p in _SEARCH.get(system, ())]
    ordered += [("search", p) for p in _near_candidates(system)]

    found: list[tuple[str, Path, Path]] = []
    seen: set = set()
    for origin, place in ordered:
        hit = resolve_root(place, system=system)
        if hit is None:
            continue
        root, executable = hit
        key = os.path.normcase(os.path.abspath(root))
        if key in seen:
            continue
        seen.add(key)
        found.append((origin, root, executable))

    if not found:
        return None
    origin, root, executable = found[0]
    return FijiInstall(root=root, executable=executable, origin=origin,
                       has_trackmate=_has_trackmate(root),
                       other_candidates=tuple(r for _, r, _ in found[1:]))


def fiji_command(install: FijiInstall, script) -> list:
    """The command line that runs ``script`` headless in this Fiji.

    The ImageJ launcher (Fiji <= 2.14, and still shipped beside the new one in 2.16)
    takes ``--ij2 --headless --console``. The current launcher (Jaunch) is given the JVM's
    own headless switch instead of ``--headless``, which some of its configs turn into
    ImageJ's ``-batch`` and so never run the script (measured: exit 1, no script run).
    ``-D...`` is passed to the JVM by every Jaunch config, and ``--no-splash`` keeps the
    splash screen off.
    """
    if install.jaunch:
        return [str(install.executable), "--no-splash", "-Djava.awt.headless=true",
                "--run", str(script)]
    return [str(install.executable), "--ij2", "--headless", "--console", "--run",
            str(script)]


def status_lines(path) -> list:
    """The marker lines a script wrote to ``path``, or ``[]``. The current launcher
    returns nothing through a pipe, so a file is the one channel both generations
    share."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def describe_absence(*, system: str | None = None) -> str:
    """What to tell a user when no Fiji was found: the places looked and the way out."""
    system = system or _platform()
    looked = ", ".join(str(Path(p).expanduser()) for p in _SEARCH.get(system, ()))
    return (
        "No Fiji installation was found. TrackMate needs Fiji.\n"
        f"  Looked in: {looked}, and in folders named Fiji... in your home, Desktop and "
        "Downloads.\n"
        f"  Get Fiji from {FIJI_WEBSITE}, unzip it, then choose it: "
        f"{launcher_hint(system)}.\n"
        f"  Or set {FIJI_ENV} to an existing Fiji folder.\n"
        "  u-track data is unaffected."
    )


#: Printed by the health-check script and looked for in the output. Fiji's startup is
#: NOISY -- the JVM warns about PermSize, SLF4J warns about multiple bindings, plugins
#: announce themselves -- so parsing "the first line" or "--version" returns whichever
#: component spoke first. A marker we chose cannot be confused with any of it.
HEALTH_MARKER = "SMTRACKER_FIJI_OK"

_HEALTH_SCRIPT = """
from ij import IJ
try:
    from fiji.plugin.trackmate import Model
    tm = "yes"
except Exception:
    tm = "no"
line = "{marker} version=" + str(IJ.getVersion()) + " trackmate=" + tm
print(line)
status = open({status!r}, "a")
status.write(line + "\\n")
status.close()
"""


def check_fiji(install: FijiInstall, *, timeout: float = 300.0) -> dict:
    """Run a trivial script and confirm Fiji answers, the real readiness check.

    WHY NOT `--version`. Fiji's startup output is noise all the way down: the JVM warns
    about removed options, SLF4J warns about multiple bindings, plugins announce
    themselves. Measured on a real install, `--version` returned "OpenJDK 64-Bit Server
    VM warning: ignoring option PermSize=128m" and then, after filtering that, "SLF4J:
    Class path contains multiple SLF4J bindings." Neither is a version.

    So this runs a script that prints a marker WE chose, and looks for that. It also
    exercises the exact path everything else depends on -- headless script execution --
    and reports whether TrackMate's classes actually import, which a jar on disk does
    not guarantee.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "smtracker_health.py"
        status = Path(folder) / "smtracker_health.status"
        script = (_HEALTH_SCRIPT.replace("{marker}", HEALTH_MARKER)
                  .replace("{status!r}", repr(str(status).replace("\\", "/"))))
        path.write_text(script, encoding="utf-8")
        try:
            done = subprocess.run(
                fiji_command(install, path),
                capture_output=True, text=True, timeout=timeout, check=False, creationflags=NO_WINDOW)
        except OSError as exc:
            raise FijiError(f"{install.executable} could not be started: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise FijiError(
                f"{install.executable} did not answer within {timeout:g} s. Headless "
                "Fiji sometimes blocks on a first-run update prompt: open it once by "
                "hand and let it finish updating") from exc
        lines = ((done.stdout or "").splitlines() + (done.stderr or "").splitlines()
                 + status_lines(status))

    line = next((ln for ln in lines if HEALTH_MARKER in ln), None)
    if line is None:
        tail = os.linesep.join(ln for ln in lines[-6:] if ln.strip())
        raise FijiError(
            f"{install.executable} ran but never printed {HEALTH_MARKER}. It may have "
            f"failed to start headless. Last output:{os.linesep}{tail}")
    fields = dict(part.split("=", 1) for part in line.split()
                  if "=" in part and not part.startswith(HEALTH_MARKER))
    return {"version": fields.get("version", "unknown"),
            "trackmate_importable": fields.get("trackmate") == "yes",
            "executable": str(install.executable)}
