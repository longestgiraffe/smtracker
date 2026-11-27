"""Freeze SMTracker into a standalone folder with PyInstaller.

    py packaging/build.py [--variant lite|normal] [--dist DIR] [--work DIR] [--onefile] [--clean]

TWO BUILDS (ruled 26 Nov 2026, "THE 3.0.0 RELEASE IS THE PAPER'S BUILD"):

- **lite** (the default; exactly what this script built before the variants): no torch, the classical
  segmenter only;
- **normal**: Lite plus CUDA torch and Cellpose 4. No model ships in either: Cellpose-SAM downloads itself at first
  use. Omnipose is NEVER bundled (its licence is non-commercial, which a GPL application cannot carry); it keeps
  working when SMTracker runs from source with omnipose installed.

A Normal build is several GB: build it with ``--dist`` and ``--work`` on a drive with room, from an environment that
has torch (a CUDA build), torchvision and cellpose >= 4 and NOT omnipose. The variant is recorded in the bundle's
build_info.json (so `smtracker --version` says it) and in the zip's name.

On Windows smtracker.exe is a WINDOWS app (no console; 27 Nov 2026) and smtracker-console.exe beside it is the same
file asking for a console, for the commands (`console_twin`). On a Mac the build is SMTracker.app (27 Nov 2026,
`finish_app`, `Layout`), its command line Contents/MacOS/smtracker, and `--installer` makes a .pkg (`make_pkg`).

Then smoke-test the result — this is the check that matters, because a build
that *completes* is not a build that *runs*:

    packaging/dist/smtracker/smtracker-console.exe --selftest

(`--selftest` puts a simulated set of declared D and shares through the SQD fit and the HMM and compares with the
numbers stored in `smtracker.selfcheck`, then builds and tears down a napari viewer. `verify` below runs it and
asks for the analysis line by name.)

Why a script rather than a hand-edited .spec
--------------------------------------------
The collect list below is the one source of truth for what the bundle needs, and
it is version-controlled next to the reason each entry exists. A generated .spec
drifts silently the moment someone rebuilds with different flags.

Notes on the entries
--------------------
``napari`` and ``vispy`` ship data files (icons, shaders, themes, .pyi stubs)
that PyInstaller's module analysis alone does not find. ``npe2``, ``app_model``,
``in_n_out``, ``psygnal``, ``magicgui`` and ``superqt`` are the plugin/command
machinery napari resolves at runtime, so their metadata must be present.

``napari_builtins`` is the easy one to miss: it is a *plugin*, discovered via an
entry point rather than imported, so a bundle without it starts fine and then
has no file readers at all. The first build here hit exactly that — the frozen
binary passed its selftest while printing "Cannot find module 'napari_builtins'".

The Qt backend is PySide6, and that is a LICENSING decision, not a taste one
-----------------------------------------------------------------------------
PyQt5 is GPL: bundling it into a standalone distribution makes the *whole
application* GPL. PySide6 is LGPL, which a shipped installer can carry without
imposing that on the rest of the code — and this tool is deliberately separable
from the trackers (u-track and TrackMate are GPLv3, but they run inside Fiji as
their own applications and are never linked here), so it is free to choose its
own license unless a Qt binding nails it down. PySide6 is also Qt6, which is
where napari is heading, and it clears napari 0.8's theme-detection warning.

``PyQt5`` is therefore **excluded explicitly**: if a stray copy is ever installed
in the build environment, ``qtpy`` would happily select it and the resulting
bundle would silently ship under the wrong license.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

#: Packages whose data files, metadata or entry points must be collected whole.
COLLECT_ALL = [
    "napari",
    "napari_builtins",      # plugin: discovered by entry point, not by import
    "napari_svg",
    "napari_console",
    "vispy",                # GL shaders live in data files
    "npe2",
    "app_model",
    "in_n_out",
    "psygnal",
    "magicgui",
    "superqt",
    "pyconify",             # icon sets, fetched through entry points
    "matplotlib",           # the inspector canvas (backend_qtagg)
]

#: Collected whole WHERE INSTALLED: newer napari keeps its logos and icons in a package of their own,
#: `napari_resources`, which it opens with importlib.resources, so the static scan misses its data. Found on the Mac
#: build (27 Nov 2026, napari 0.10): `--selftest` failed with "No module named 'napari_resources.resources'". An
#: environment without the package (an older napari) builds exactly as before.
OPTIONAL_COLLECT_ALL = ["napari_resources"]


def _present(packages) -> list:
    import importlib.util

    return [p for p in packages if importlib.util.find_spec(p) is not None]


#: Never bundle these. PyQt5 is GPL and would relicense the whole application
#: (see the module docstring); the others are Qt bindings we do not ship.
#:
#: **THE SEGMENTATION STACK IS EXCLUDED UNTIL IT IS REACHABLE, and this is a size
#: decision with a date on it.** `torch` alone is 524 MB in the development environment
#: against a 407 MB bundle, so installing it — which is required to develop the
#: segmentation stage — would nearly triple the installer for a feature no window opens
#: yet. `docs/segmentation_integration.md` decides that CPU torch DOES ship once
#: segmentation is wired in (measured: about +300-400 MB after PyInstaller drops the
#: static `.lib` files, not the naive 3.5 GB), so these lines come out then, deliberately
#: and with the size checked, rather than by a dependency arriving unnoticed.
#:
#: Excluding a package PyInstaller would otherwise pull in is also how a bundle stays
#: honest about what it contains: a `--selftest` that passes because torch happened to be
#: on the build machine is not a test of the shipped article.
SEGMENTATION_STACK = [
    "torch", "torchvision", "omnipose", "cellpose_omni", "cellpose",
    "numba", "llvmlite", "cv2", "mahotas", "sklearn", "imagecodecs",
    "torchvf", "ncolor", "edt", "mgen", "peakdetect", "fastremap",
]

#: THE TWO BUILDS (26 Nov 2026). Lite excludes the whole stack above, as it always did. Normal KEEPS what torch and
#: Cellpose 4 need (torch, torchvision, cellpose, OpenCV, imagecodecs, fastremap; fill_voids, segment_anything, natsort,
#: roifile and tqdm were never on the list) and still excludes Omnipose and everything only it needs (omnipose,
#: cellpose_omni, torchvf, ncolor, edt, mgen, peakdetect; the licence is the reason) and the packages nothing in either
#: build uses (numba, llvmlite, mahotas, sklearn). `cellpose.gui` is Cellpose's own PyQt window: PyQt is GPL-incompatible
#: with the LGPL Qt choice above, and nothing here opens it.
VARIANTS = ("lite", "normal")
DEFAULT_VARIANT = "lite"
NORMAL_KEEPS = ("torch", "torchvision", "cellpose", "cv2", "imagecodecs", "fastremap")
OMNIPOSE_STACK = ("omnipose", "cellpose_omni", "torchvf", "ncolor", "edt", "mgen", "peakdetect")
NORMAL_EXCLUDES = ("cellpose.gui", "cellpose.cli")

#: STANDARD-LIBRARY MODULES THE SEGMENTATION PACK NEEDS and nothing in the bundle imports, so PyInstaller leaves them
#: out (found 26 Nov 2026 by running the frozen app with a pack: torchvision's first line is ``from modulefinder import
#: Module``). The list is every stdlib module the pack's wheels import at module level on Windows that the bundle lacked
#: (scanned from the unpacked pack against the bundle's PYZ). Pack code is outside the bundle, so only a build can add them.
PACK_STDLIB = ("modulefinder", "struct", "html.parser")

#: THE OTHER HALF OF THE SAME FINDING: a package the bundle holds PARTLY (only the modules the program imports) and the
#: pack's code needs more of. The bundle's copy wins (that is the design), so a module it lacks is simply not found:
#: torchvision's ``from PIL import ImageEnhance`` failed against a bundled PIL without it. Whole packages for the small
#: ones; single modules where the package is big (found by scanning the unpacked pack's imports against the bundle's PYZ).
PACK_COLLECT = ("PIL",)
PACK_MODULES = ("PIL.ImageEnhance",)


def excludes(variant: str = DEFAULT_VARIANT) -> list:
    """The modules PyInstaller is told to leave out for ``variant``."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown build variant {variant!r}; choose one of {', '.join(VARIANTS)}")
    base = ["PyQt5", "PyQt6", "PySide2"]
    if variant == "lite":
        return base + SEGMENTATION_STACK
    return base + [m for m in SEGMENTATION_STACK if m not in NORMAL_KEEPS] + list(NORMAL_EXCLUDES)


EXCLUDE = excludes("lite")

#: What a Normal build collects beyond what PyInstaller finds by itself: Cellpose's own modules (the SAM network is
#: imported inside functions), and the metadata `cellpose.version` and `importlib.metadata` read at run time.
NORMAL_SUBMODULES = ("cellpose", "segment_anything", "fill_voids", "fastremap", "roifile")
NORMAL_METADATA = ("cellpose", "torch", "torchvision", "tqdm", "imagecodecs")


def _git(*args: str) -> str:
    """One git answer about this repository, or "" where git cannot give one."""
    try:
        got = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                             text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return got.stdout.strip() if got.returncode == 0 else ""


def build_info(variant: str = DEFAULT_VARIANT) -> dict:
    """WHICH CODE THIS BUNDLE IS, written into it (`smtracker.buildstamp.BUILD_INFO_NAME`).

    A bundle has no repository, so the commit has to be carried in; without it a build sent
    to another machine says only "frozen build". `dirty` records uncommitted edits to the
    code that ships (src/ and packaging/), because then the commit names code that is not
    quite what was built."""
    import tomllib

    try:
        version = tomllib.loads((REPO / "pyproject.toml").read_text("utf-8"))["project"]["version"]
    except (OSError, KeyError, ValueError):
        version = "?"
    return {
        "commit": _git("rev-parse", "--short", "HEAD"),
        "commit_date": _git("log", "-1", "--format=%cI"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no", "--",
                           "src", "packaging")),
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
        "version": version,
        "variant": variant,
    }


def build_command(*, variant: str = DEFAULT_VARIANT, dist=None, work=None, onefile: bool = False,
                  clean: bool = True, stamp_dir=None) -> list:
    """The PyInstaller command for ``variant`` (nothing is run). ``dist`` and ``work`` default to
    ``packaging/dist`` and ``packaging/build``; a Normal build is several GB, so they can be named."""
    excluded = excludes(variant)
    command = [
        sys.executable, "-m", "PyInstaller", "--noconfirm",
        "--name", "smtracker",
        "--distpath", str(Path(dist) if dist else HERE / "dist"),
        "--workpath", str(Path(work) if work else HERE / "build"),
        "--specpath", str(HERE),
        "--paths", str(REPO / "src"),
    ]
    # THE ICON, AND IT HAS TO GO IN TWICE. `--icon` stamps the EXECUTABLE, which is what
    # the shell and the taskbar shortcut show; `--add-data` ships the file itself, which
    # is what `appicon.apply` reads at runtime to set the WINDOW icon. Either one alone
    # leaves the other wrong, and the frozen build is exactly where a data file goes
    # missing, so both are explicit rather than assumed to be collected.
    from smtracker.appicon import icon_path

    # a Mac app's icon comes from the 512 px PNG (PyInstaller turns it into the .icns); Windows takes the .ico
    executable_icon = icon_path(".png" if sys.platform == "darwin" else ".ico")
    if executable_icon is not None:
        command += ["--icon", str(executable_icon)]
    resources = REPO / "src" / "smtracker" / "resources"
    if resources.is_dir():
        command += ["--add-data", f"{resources}{os.pathsep}smtracker/resources"]
    else:
        print(f"warning: no {resources}; the frozen build will have no window icon "
              f"(cosmetic, see smtracker.appicon)", file=sys.stderr)

    # THE MANUAL, which every window's Help opens (`smtracker.manual`): one source,
    # docs/MANUAL.md, shipped beside the package; `markdown_it` renders it, and is
    # imported lazily, so it is named here rather than left to the static scan.
    manual_md = REPO / "docs" / "MANUAL.md"
    command += ["--add-data", f"{manual_md}{os.pathsep}smtracker/resources/manual",
                "--hidden-import", "markdown_it", "--collect-submodules", "markdown_it",
                "--hidden-import", "mdurl"]

    # THE BUNDLED TYPEFACES, for the reason they are bundled at all: the interface
    # must not depend on what is installed on the machine, and a standalone
    # installer is exactly where that cannot be inspected. PyInstaller collects
    # .py files, not .ttf, so without this the frozen build falls silently back
    # to Segoe UI -- silently, because `typography._family()` walks a fallback
    # stack and is designed never to fail. NOT cosmetic, unlike the icon above:
    # every size, weight and letter-spacing on these screens was measured in
    # Inter. `--selftest` cannot see it, so `typography.report()` is the check.
    fonts = REPO / "src" / "smtracker" / "ui" / "theme" / "fonts"
    if fonts.is_dir():
        command += ["--add-data", f"{fonts}{os.pathsep}smtracker/ui/theme/fonts"]
    else:
        print(f"warning: no {fonts}; the frozen build will render in whatever "
              f"the machine has (see smtracker.ui.theme.typography)", file=sys.stderr)

    # THE BUILD NAMES ITS CODE: `build_info.json`, beside the `smtracker` package, where
    # `buildstamp.build_info` looks. Written to a temporary folder, never into src/.
    stamp_dir = Path(stamp_dir or tempfile.mkdtemp(prefix="smtracker-stamp-"))
    info = build_info(variant)
    (stamp_dir / "build_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    command += ["--add-data", f"{stamp_dir / 'build_info.json'}{os.pathsep}smtracker"]
    print(f"stamping the build: commit {info['commit'] or '?'}"
          + (" WITH uncommitted edits" if info["dirty"] else "") + f", built {info['built']}")

    if clean:
        command.append("--clean")
    if os.name == "nt" and not onefile:
        # A WINDOWS APP: double-clicked, it opens its window and no console (`console_twin` makes the console
        # program beside it). On a Mac --windowed makes the .app bundle, another layout (`mac_layout`, below).
        command.append("--windowed")
    if sys.platform == "darwin" and not onefile:
        # A MAC APP (27 Nov 2026): SMTracker.app, double-clicked from Applications with no Terminal window; the
        # commands are typed as SMTracker.app/Contents/MacOS/smtracker --selftest. `finish_app` names it and stamps it.
        command += ["--windowed", "--osx-bundle-identifier", MAC_BUNDLE_ID]
    if onefile:
        # Slower to start (it unpacks to a temp dir every launch) and harder to
        # debug; the installer will ship the folder build.
        command.append("--onefile")
    for package in COLLECT_ALL + _present(OPTIONAL_COLLECT_ALL):
        command += ["--collect-all", package]
    # Belt-and-braces: pull in EVERY smtracker submodule, so the comparison-first
    # views (comparison, compare_plots, comparison_controller, warmcache, fastmode,
    # uncertainty, trackviewer, …) ship even where they are imported lazily inside
    # a function and PyInstaller's static scan might miss one.
    command += ["--collect-submodules", "smtracker"]
    for module in PACK_STDLIB + PACK_MODULES:
        command += ["--hidden-import", module]
    for package in PACK_COLLECT:
        command += ["--collect-submodules", package]
    for module in excluded:
        command += ["--exclude-module", module]
    if variant == "normal":
        for package in NORMAL_SUBMODULES:
            command += ["--collect-submodules", package]
        for package in NORMAL_METADATA:
            command += ["--copy-metadata", package]
    command.append(str(HERE / "entry.py"))
    return command


def build(*, variant: str = DEFAULT_VARIANT, dist=None, work=None, onefile: bool = False,
          clean: bool = True) -> int:
    """Freeze ``variant`` and return PyInstaller's exit code."""
    if variant == "normal":
        _require_normal_environment()
    command = build_command(variant=variant, dist=dist, work=work, onefile=onefile, clean=clean)
    print(f"building the {variant} variant")
    print(" ".join(command))
    return subprocess.call(command)


#: THE TWO PROGRAMS ON WINDOWS (27 Nov 2026, asked of the 3.0.0 build: "console always open and no way to closeit?").
#: smtracker.exe is built --windowed: double-clicked, it opens its window and no console, and closing a console no
#: longer ends it. Its commands (--selftest, --report, --version, --segmentation-pack) print, and a Windows app has
#: nowhere to print, so smtracker-console.exe sits beside it: the SAME file with the PE header's subsystem set to
#: console, as python.exe beside pythonw.exe. Both read the one _internal folder, so the bundle grows by one small file.
CONSOLE_NAME = "smtracker-console.exe"
SUBSYSTEM_WINDOWS, SUBSYSTEM_CONSOLE = 2, 3


def _subsystem_field(data) -> int:
    """Where the PE optional header's Subsystem sits: 24 bytes past the PE signature, then 68 (PE32 and PE32+ alike)."""
    pe = int.from_bytes(data[0x3C:0x40], "little")
    if bytes(data[pe:pe + 4]) != b"PE\0\0":
        raise RuntimeError("not a Windows executable: no PE signature")
    return pe + 24 + 68


def subsystem(path) -> int:
    """The Windows subsystem an executable asks for: 2 a window, 3 a console."""
    data = Path(path).read_bytes()[:4096]
    field = _subsystem_field(data)
    return int.from_bytes(data[field:field + 2], "little")


def console_twin(dist_dir=None) -> Path:
    """Write smtracker-console.exe beside the Windows app: its bytes with the subsystem set to console. Refused when
    smtracker.exe is not a Windows app, so a console build is never copied as if it were one."""
    dist_dir = Path(dist_dir or (HERE / "dist" / "smtracker"))
    data = bytearray((dist_dir / "smtracker.exe").read_bytes())
    field = _subsystem_field(data)
    found = int.from_bytes(data[field:field + 2], "little")
    if found != SUBSYSTEM_WINDOWS:
        raise RuntimeError(f"smtracker.exe asks for subsystem {found}, not a window ({SUBSYSTEM_WINDOWS}): "
                           "it was not built --windowed")
    data[field:field + 2] = SUBSYSTEM_CONSOLE.to_bytes(2, "little")
    twin = dist_dir / CONSOLE_NAME
    twin.write_bytes(bytes(data))
    return twin


# ---------------------------------------------------------------------------------
# THE MAC APP (27 Nov 2026, asked: "we need to build an installer for mac").
# ---------------------------------------------------------------------------------
#
# On a Mac the folder build is `--windowed`, so PyInstaller makes an .app bundle, and the layout differs from Windows':
#
#   Windows: dist/smtracker/smtracker.exe, dist/smtracker/_internal/..., the licence files beside the program
#   macOS:   dist/SMTracker.app/Contents/MacOS/smtracker (the program, which is also the command line),
#            Contents/Frameworks/... (what _internal holds elsewhere; PyInstaller's sys._MEIPASS), and
#            Contents/Resources/ (data, linked from Frameworks, and the licence files, README and source zip)
#
# `layout` answers where each part is, decided by the bundle's NAME (an .app or a folder), so a Windows bundle is read
# exactly as before and the tests can build either kind on any machine. `finish_app` renames PyInstaller's
# smtracker.app to SMTracker.app, writes the name and version into Info.plist, and `sign_app` signs it again ad-hoc
# after the licence files went into Contents/Resources (an Apple-silicon Mac runs no code whose seal is broken).

#: The bundle identifier, fixed for good (as the Windows installer's AppId): an install of a later version replaces the
#: earlier one by it. Named after the public repository.
MAC_BUNDLE_ID = "io.github.longestgiraffe.smtracker"
MAC_APP = "SMTracker.app"


class Layout:
    """Where the parts of a bundle are: ``root`` is what ships (the folder, or the .app), ``binary`` the program,
    ``cli`` the program commands are typed with, ``internal`` PyInstaller's support folder, ``legal`` the folder the
    licence files, README and source go into."""

    def __init__(self, bundle):
        self.root = Path(bundle)
        self.mac = self.root.suffix == ".app"
        if self.mac:
            contents = self.root / "Contents"
            self.binary = contents / "MacOS" / "smtracker"
            self.cli = self.binary
            self.internal = contents / "Frameworks"
            self.legal = contents / "Resources"
        else:
            self.binary = self.root / ("smtracker.exe" if os.name == "nt" else "smtracker")
            self.cli = self.root / CONSOLE_NAME if os.name == "nt" else self.binary
            self.internal = self.root / "_internal"
            self.legal = self.root

    @property
    def info_path(self) -> Path:
        return self.internal / "smtracker" / "build_info.json"

    def info(self) -> dict:
        return json.loads(self.info_path.read_text("utf-8")) if self.info_path.is_file() else {}


def default_bundle(dist=None) -> Path:
    """The bundle a build makes in ``dist`` (default packaging/dist): SMTracker.app on a Mac, else the folder."""
    dist = Path(dist) if dist else HERE / "dist"
    return dist / (MAC_APP if sys.platform == "darwin" else "smtracker")


def layout(bundle=None) -> Layout:
    return Layout(bundle or default_bundle())


def finish_app(dist=None) -> Path:
    """PyInstaller's ``smtracker.app`` becomes ``SMTracker.app`` with its name and version in Info.plist; the plain
    folder PyInstaller also leaves (``dist/smtracker``, the same files once more) is removed, so the .app is the one
    thing a later step can pick up."""
    import plistlib

    dist = Path(dist) if dist else HERE / "dist"
    made, app = dist / "smtracker.app", dist / MAC_APP
    if not made.is_dir():
        raise RuntimeError(f"PyInstaller made no {made}")
    # through a third name: a Mac's disk ignores case, so smtracker.app and SMTracker.app are ONE path, and removing
    # "the old SMTracker.app" first would remove the app just made
    passing = dist / "SMTracker-renaming.app"
    if passing.exists():
        shutil.rmtree(passing)
    made.rename(passing)
    if app.exists():
        shutil.rmtree(app)
    passing.rename(app)
    if (dist / "smtracker").is_dir():
        shutil.rmtree(dist / "smtracker")
    lay = Layout(app)
    info = lay.info()
    plist_path = app / "Contents" / "Info.plist"
    plist = plistlib.loads(plist_path.read_bytes())
    version = info.get("version") or "0"
    plist.update({"CFBundleName": "SMTracker", "CFBundleDisplayName": "SMTracker",
                  "CFBundleShortVersionString": version, "CFBundleVersion": version,
                  "NSHighResolutionCapable": True,
                  "NSHumanReadableCopyright": "GNU General Public License v3 or later"})
    minimum = bundle_min_macos(app)
    if minimum:
        plist["LSMinimumSystemVersion"] = minimum
    plist_path.write_bytes(plistlib.dumps(plist))
    return app


def sign_app(app) -> None:
    """Sign the .app again ad-hoc (no identity: an Apple Developer ID is paid). Its nested libraries keep the
    signatures PyInstaller gave them; the outer seal covers Info.plist and Contents/Resources, which changed."""
    done = subprocess.run(["codesign", "--force", "--sign", "-", str(app)], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"codesign failed: {done.stderr.strip()}")


def _min_macos(path) -> tuple | None:
    """The lowest macOS a Mach-O file loads on (its `minos`, read with vtool), or None for anything else."""
    try:
        got = subprocess.run(["vtool", "-show-build", str(path)], capture_output=True, text=True, timeout=30,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in got.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "minos":
            try:
                return tuple(int(x) for x in parts[1].split("."))
            except ValueError:
                return None
    return None


def _mach_o_files(app):
    for path in Path(app).rglob("*"):
        if path.is_file() and not path.is_symlink():
            with open(path, "rb") as handle:
                magic = handle.read(4)
            if magic in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xce\xfa\xed\xfe"):
                yield path


def bundle_min_macos(app, *, report=None) -> str:
    """The oldest macOS the .app opens on: the NEWEST `minos` of any binary inside it (one library built for a newer
    macOS is enough to stop it; on 15 Oct 2026 a build from Homebrew's Python needed macOS 26). ``report``, a dict,
    receives every binary's minos."""
    newest = None
    for path in _mach_o_files(app):
        minos = _min_macos(path)
        if minos:
            if report is not None:
                report[path] = minos
            newest = minos if newest is None or minos > newest else newest
    if newest is None:
        return ""
    return ".".join(str(x) for x in (newest + (0,))[:max(2, len(newest))])


def bundle_arch(app) -> str:
    """The processor the .app's program is built for (``arm64``, ``x86_64``, or both joined by ``+``)."""
    done = subprocess.run(["lipo", "-archs", str(Layout(app).binary)], capture_output=True, text=True)
    archs = done.stdout.split()
    if done.returncode != 0 or not archs:
        raise RuntimeError(f"lipo could not read {Layout(app).binary}: {done.stderr.strip()}")
    return "+".join(sorted(archs))


def _require_normal_environment() -> None:
    """A Normal build is made from an environment that HAS what it ships: a CUDA torch, torchvision, cellpose >= 4,
    and no omnipose (a build machine that has it would put it in the bundle's path scan, and its licence is not
    ours to ship). Refused by name rather than built wrong."""
    import importlib.util

    problems = []
    for name in ("torch", "torchvision", "cellpose", "segment_anything", "fill_voids", "fastremap"):
        if importlib.util.find_spec(name) is None:
            problems.append(f"{name} is not installed in this Python")
    if not problems:
        import torch

        if getattr(torch.version, "cuda", None) is None:
            problems.append(f"torch {torch.__version__} is a CPU-only build; the Normal build ships CUDA torch")
        if importlib.util.find_spec("cellpose.vit") is None:
            problems.append("cellpose is older than 4 (no cellpose.vit)")
    for name in OMNIPOSE_STACK[:2]:
        if importlib.util.find_spec(name) is not None:
            problems.append(f"{name} is installed here; build the Normal variant from an environment without "
                            f"Omnipose, whose licence the application cannot carry")
    if problems:
        raise SystemExit("cannot build the Normal variant: " + "; ".join(problems))




# ---------------------------------------------------------------------------------
# WHAT A LAB RECEIVES BESIDE THE PROGRAM (16 Nov 2026, asked: "Can u do the licences").
# ---------------------------------------------------------------------------------
#
# SMTracker is GPL-3.0-or-later (README), so a copy given to another lab must carry the
# licence and its complete source, and the libraries frozen into it must carry theirs.
# Before this the folder held only smtracker.exe and _internal: the GPL text sat in the
# repository alone, and nothing named PySide6's LGPL. Written after every build, beside
# the program:
#
# - LICENSE.txt: the GPL text (`GPL-License.txt`);
# - THIRD-PARTY-NOTICES.txt: every distribution whose modules are IN the bundle, found by
#   reading the frozen executable's own module list (`bundled_distributions`), never the
#   build environment's package list (which holds pytest, ruff and the rest); for each its
#   version, its licence as its metadata states it and the licence texts its metadata
#   ships, then CPython's licence and the typefaces';
# - source/smtracker-<commit>-source.zip: `git archive` of the commit built, the paths
#   `SOURCE_PATHS` (everything the build needs);
# - README.txt: how to start it, and where the above are.
#
# Qt is LGPL-3.0 through PySide6, whose wheel ships only the commercial licence's text: the
# LGPL-3.0 text is kept here (`packaging/licenses/LGPL-3.0.txt`, the FSF's text). The LGPL
# also asks that a user can swap the Qt libraries; the folder build keeps them as separate
# DLLs in `_internal`, one more reason `--onefile` is not what ships.

LICENSE_NAME = "LICENSE.txt"
NOTICES_NAME = "THIRD-PARTY-NOTICES.txt"
README_NAME = "README.txt"
SOURCE_DIR = "source"
LICENSE_TEXTS = HERE / "licenses"

#: The paths of the source zip: the package, what builds it, and what it ships.
SOURCE_PATHS = ("src", "packaging/build.py", "packaging/entry.py", "packaging/make_icons.py", "packaging/smtracker.iss",
                "packaging/logo", "packaging/licenses", "pyproject.toml", "README.md",
                "GPL-License.txt", "docs/MANUAL.md")
#: THE PUBLIC REPOSITORY (github.com/longestgiraffe/smtracker, 27 Nov 2026) holds `SOURCE_PATHS` and these: the
#: issue template "Report on GitHub" leads to. Never this repository's history, notes, scripts or data.
PUBLIC_PATHS = SOURCE_PATHS + (".github",)

#: Licence texts kept here for distributions whose metadata ships none of the licence they
#: are used under (PySide6 and shiboken6 ship only the commercial one).
EXTRA_TEXTS = {"fastremap": ("LGPL-3.0.txt",), "pyside6": ("LGPL-3.0.txt",), "pyside6-essentials": ("LGPL-3.0.txt",),
               "pyside6-addons": ("LGPL-3.0.txt",), "shiboken6": ("LGPL-3.0.txt",)}

#: Words in the name of a file that holds a licence text inside a distribution's metadata.
_TEXT_NAMES = ("LICEN", "COPYING", "NOTICE")


def bundled_top_levels(dist_dir) -> set:
    """The top-level module names IN the bundle: the frozen executable's archive of pure
    modules, and the packages and extension modules beside it in `_internal`."""
    from PyInstaller.archive.readers import CArchiveReader

    lay = Layout(dist_dir)
    names = set()
    # the program holding the archive: smtracker.exe, or on a Mac the .app's Contents/MacOS/smtracker
    archive = CArchiveReader(str(lay.binary if lay.mac else Path(dist_dir) / "smtracker.exe"))
    for entry in archive.toc:
        if entry.upper().startswith("PYZ"):
            names |= {m.split(".")[0] for m in archive.open_embedded_archive(entry).toc}
    internal = lay.internal
    if internal.is_dir():
        for child in internal.iterdir():
            if child.is_dir() and not child.name.endswith((".dist-info", ".egg-info")):
                names.add(child.name)
            elif child.suffix == ".pyd" or (lay.mac and child.suffix == ".so"):
                names.add(child.name.split(".")[0])
    return names


def bundled_distributions(dist_dir) -> list:
    """The distributions whose modules are in the bundle, sorted by name, smtracker left out
    (its licence is LICENSE.txt). Standard-library modules map to none: they are CPython's."""
    import importlib.metadata as metadata

    owners = metadata.packages_distributions()
    found = {}
    for top in bundled_top_levels(dist_dir):
        for name in owners.get(top, ()):
            if name.lower() != "smtracker":
                found.setdefault(name.lower(), name)
    dists = []
    for key in sorted(found):
        try:
            dists.append(metadata.distribution(found[key]))
        except metadata.PackageNotFoundError:          # pragma: no cover - mapped above
            continue
    return dists


def licence_words(dist) -> str:
    """A distribution's licence as its metadata states it: the SPDX expression, a short
    License field, else its licence classifiers, else "not stated in its metadata"."""
    meta = dist.metadata
    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        return expression
    field = (meta.get("License") or "").strip()
    if field and len(field) <= 120 and "\n" not in field:
        return field
    classifiers = [c.split("::")[-1].strip() for c in (meta.get_all("Classifier") or [])
                   if c.startswith("License ::")]
    if classifiers:
        return "; ".join(classifiers)
    return "see its text below" if field else "not stated in its metadata"


def licence_texts(dist) -> list:
    """``[(title, text)]``: the licence files its metadata ships, a long License field
    (some put the whole text there), and any kept here for it (`EXTRA_TEXTS`)."""
    texts = []
    for item in dist.files or ():
        name = Path(str(item)).name.upper()
        # Qt's commercial licence ships in PySide6's metadata too; it is not the one used here
        if (".dist-info" in str(item) and any(word in name for word in _TEXT_NAMES)
                and "COMMERCIAL" not in name):
            try:
                text = Path(dist.locate_file(item)).read_text("utf-8", errors="replace")
            except OSError:
                continue
            texts.append((Path(str(item)).name, text.strip()))
    field = (dist.metadata.get("License") or "").strip()
    if not texts and ("\n" in field or len(field) > 120):
        texts.append(("License (from its metadata)", field))
    for file_name in EXTRA_TEXTS.get(dist.metadata["Name"].lower().replace("_", "-"), ()):
        texts.append((file_name, (LICENSE_TEXTS / file_name).read_text("utf-8").strip()))
    return texts


#: THE NVIDIA LIBRARIES INSIDE TORCH'S WHEELS (the Normal build). A CUDA wheel of torch carries NVIDIA's runtime
#: libraries as DLLs in ``torch/lib`` (they are not separate Python packages on Windows, so no distribution lists
#: them), and the wheel ships no NVIDIA licence text with them. Named here, with where their licences are published.
CUDA_LIBRARIES = (
    ("CUDA runtime", "cudart64_12.dll"), ("cuBLAS", "cublas64_12.dll, cublasLt64_12.dll"),
    ("cuFFT", "cufft64_11.dll, cufftw64_11.dll"), ("cuRAND", "curand64_10.dll"),
    ("cuSOLVER", "cusolver64_11.dll, cusolverMg64_11.dll"), ("cuSPARSE", "cusparse64_12.dll"),
    ("NVRTC", "nvrtc64_120_0.dll, nvrtc64_120_0.alt.dll, nvrtc-builtins64_126.dll, caffe2_nvrtc.dll"),
    ("nvJitLink", "nvJitLink_120_0.dll"), ("NVTX", "nvToolsExt64_1.dll"),
    ("CUPTI and Nsight profiling", "cupti64_2024.3.2.dll, nvperf_host.dll"),
    ("cuDNN 9", "cudnn64_9.dll, cudnn_adv64_9.dll, cudnn_cnn64_9.dll, cudnn_engines_precompiled64_9.dll, "
                "cudnn_engines_runtime_compiled64_9.dll, cudnn_graph64_9.dll, cudnn_heuristic64_9.dll, "
                "cudnn_ops64_9.dll, zlibwapi.dll"),
)


def cuda_notice() -> list:
    """The paragraph naming NVIDIA's libraries and where their licences are (written when torch is in the bundle)."""
    lines = ["NVIDIA CUDA runtime libraries inside torch",
             "-" * 42,
             "The CUDA build of torch carries these NVIDIA libraries as DLLs in _internal/torch/lib. They are",
             "NVIDIA's proprietary software, redistributed unchanged under NVIDIA's own licences, not under",
             "the BSD licence of torch: the CUDA libraries under the NVIDIA CUDA Toolkit End User License",
             "Agreement (https://docs.nvidia.com/cuda/eula/index.html, whose Attachment A lists the files that",
             "may be redistributed with an application), cuDNN under the NVIDIA cuDNN Software License Agreement",
             "(https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html). The torch wheel",
             "ships no NVIDIA licence text, so none can be copied here; the two addresses above are where they",
             "are published. GPL does not apply to them: they are separate libraries, not part of SMTracker.",
             "",
             f"{'Library':<28} Files"]
    lines += [f"{name:<28} {files}" for name, files in CUDA_LIBRARIES]
    lines += ["", "Intel OpenMP (libiomp5md.dll, libiompstubs5md.dll) and libuv (uv.dll) also sit in that folder;",
              "torch's metadata lists their licences (Apache-2.0 with LLVM exception, MIT) among the texts below.",
              ""]
    return lines


#: Distributions that are LGPL and ship as separate extension modules in `_internal`: the licence asks that a user be
#: able to replace them, which the folder layout allows.
LGPL_EXTENSIONS = ("fastremap", "fill-voids")


def notices_text(dists, info: dict, mac: bool = False) -> str:
    """THIRD-PARTY-NOTICES.txt: a list first (name, version, licence), then every text. ``mac``: the .app's words
    for where the libraries are (Contents/Frameworks, .so files) instead of Windows' (_internal, DLL and .pyd)."""
    rule = "=" * 79
    where, kind, ext = ("SMTracker.app/Contents/Frameworks", "library", ".so") if mac else ("_internal", "DLL", ".pyd")
    python = sys.version.split()[0]
    lines = [f"SMTracker {info.get('version', '?')} (commit {info.get('commit') or '?'}): "
             "third-party software in this folder",
             "",
             "SMTracker itself is licensed under the GNU General Public License, version 3 or",
             "later (LICENSE.txt); its source is in the folder 'source'. It contains the",
             "software below, each under its own licence. Qt is used under the GNU Lesser",
             f"General Public License v3 through PySide6; its libraries are the separate {kind}",
             f"files in {where}/PySide6 and may be replaced. Qt's own third-party components",
             "are listed at https://doc.qt.io/qt-6/licenses-used-in-qt.html.",
             "",
             f"{'Package':<34} {'Version':<14} Licence"]
    lines += [f"{d.metadata['Name']:<34} {d.version:<14} {licence_words(d)}" for d in dists]
    lines += [f"{'Python and its standard library':<34} {python:<14} PSF-2.0 (its text below)",
              f"{'Inter, Geist Mono, IBM Plex Mono':<34} {'':<14} SIL Open Font License 1.1",
              ""]
    names = {d.metadata["Name"].lower().replace("_", "-") for d in dists}
    if "torch" in names:
        lines += cuda_notice()
    lgpl = sorted(n for n in LGPL_EXTENSIONS if n in names)
    if lgpl:
        lines += ["LGPL libraries beside the program",
                  "-" * 33,
                  f"{' and '.join(lgpl)} are licensed under the GNU Lesser General Public License v3 (texts below). "
                  "They are separate",
                  f"extension modules ({ext} files) in {where}, not linked into SMTracker's own files, so they can be "
                  "replaced by",
                  "a modified build of the same library. Their source is published by their authors (fastremap: "
                  "github.com/seung-lab/fastremap;",
                  "fill-voids: github.com/seung-lab/fill_voids), and the source zip of this folder names them in "
                  "pyproject.toml.",
                  ""]
    missing = [d.metadata["Name"] for d in dists if not licence_texts(d)]
    if missing:
        lines += ["No licence text is shipped in the metadata of: " + ", ".join(missing)
                  + ". Their licence is as listed above.", ""]
    for dist in dists:
        # one distribution may ship the same text many times (torch: some 140 licence files of its own third-party
        # code): each text is written once, under every file name that carried it
        grouped = {}
        for title, text in licence_texts(dist):
            grouped.setdefault(text, []).append(title)
        for text, titles in grouped.items():
            shown = titles[0] if len(titles) == 1 else f"{titles[0]} (and {len(titles) - 1} more with this text)"
            lines += [rule, f"{dist.metadata['Name']} {dist.version}: {shown}", rule, "", text, ""]
        extra = [t for ts in grouped.values() if len(ts) > 1 for t in ts[1:]]
        if extra:
            lines += [f"Files of {dist.metadata['Name']} with a text given above: " + "; ".join(extra), ""]
    python_licence = Path(sys.base_prefix) / "LICENSE.txt"
    if python_licence.is_file():
        lines += [rule, f"Python {python}: LICENSE.txt", rule, "",
                  python_licence.read_text("utf-8", errors="replace").strip(), ""]
    fonts = REPO / "src" / "smtracker" / "ui" / "theme" / "fonts" / "LICENSE.txt"
    lines += [rule, "The typefaces (Inter, Geist Mono, IBM Plex Mono): LICENSE.txt", rule, "",
              fonts.read_text("utf-8", errors="replace").strip(), ""]
    return "\n".join(lines)


def readme_text(info: dict, mac: bool = False) -> str:
    """README.txt: how to start it, and where the licence, notices and source are. ``mac``: the Mac app's README
    (`mac_readme_text`)."""
    if mac:
        return mac_readme_text(info)
    variant = info.get("variant", DEFAULT_VARIANT)
    # ONE build ships (ruled 26 Nov 2026 evening): the shipped README names no variant and tells of the pack.
    which = ({"lite": ["Segmenting cells with Cellpose-SAM, and training your own model, need PyTorch and",
                       "Cellpose. They are not in this folder: Segmentation pack, in the outlines window,",
                       "downloads them once from pytorch.org and pypi.org into a folder you choose. Without",
                       "them the classical segmenter and Edit outlines work as before.", ""],
              "normal": ["This is a DEVELOPER build: it also segments cells with Cellpose-SAM, on an NVIDIA GPU",
                         "or the processor. The model file (about 1.2 GB) is not in this folder: it downloads",
                         "once, the first time you segment with it, after asking you.", ""]}).get(variant, [])
    named = "" if variant == "lite" else f" {variant.capitalize()}"
    return "\n".join([
        f"SMTracker {info.get('version', '?')}{named} (commit {info.get('commit') or '?'}, "
        f"built {str(info.get('built', '?'))[:10]})",
        "",
        *which,
        "Single-molecule tracking analysis: from raw movies to diffusive-state fractions,",
        "condition by condition.",
        "",
        "To start: unzip this folder to a short path on your own disk (for example",
        "C:\\SMTracker: the longest file inside needs about 140 of the 260 characters",
        "Windows allows), then run smtracker.exe. Nothing is installed: SMTracker keeps",
        "its settings in %APPDATA%\\SMTracker and its caches in %LOCALAPPDATA%\\smtracker,",
        "and writes its own files (outlines, groupings, caches) beside your data. The",
        "program is not signed, so Windows may first say \"Windows protected your PC\":",
        "choose More info, then Run anyway.",
        "",
        "Tracking needs Fiji with TrackMate or MATLAB with u-track, installed separately.",
        "The manual opens from Help in every window.",
        "",
        "smtracker.exe opens no console window. For the commands, use smtracker-console.exe",
        "in a terminal: to check the installation, run smtracker-console.exe --selftest and",
        "look for \"analysis self-test: OK\". When something went wrong, Report a problem in",
        "the menu (or smtracker-console.exe --report) writes the zip to send.",
        "",
        "Licence: GNU General Public License v3 or later (LICENSE.txt). The source of this",
        "build is in the folder 'source'. The other software inside, and its licences, are",
        "listed in THIRD-PARTY-NOTICES.txt.",
        ""])


def mac_readme_text(info: dict) -> str:
    """README.txt of the Mac app: the installer, the app's command line, the "Open Anyway" of an unsigned app, removing
    it, and where the licence, notices and source are."""
    version, commit = info.get("version", "?"), info.get("commit") or "?"
    return "\n".join([
        f"SMTracker {version} for macOS (commit {commit}, built {str(info.get('built', '?'))[:10]})",
        "",
        "Single-molecule tracking analysis: from raw movies to diffusive-state fractions,",
        "condition by condition.",
        "",
        "To install: open the installer (SMTracker-...-macos-....pkg) and follow it. It puts",
        "SMTracker in Applications (for all users, which asks for your password) or in the",
        "Applications folder of your home folder (for you only, no password). Under",
        "Customize it offers what else to get, every choice off: the segmentation pack",
        "(PyTorch and Cellpose, about 2.5 GB; also later, in the outlines window) and Fiji",
        "with TrackMate (opens its download page). From the zip instead: drag",
        "SMTracker.app into Applications.",
        "",
        "SMTracker keeps its settings in ~/.config/smtracker and its caches in ~/.cache,",
        "and writes its own files (outlines, groupings, caches) beside your data.",
        "",
        "THE APP IS NOT SIGNED by an Apple Developer ID, so macOS stops it the first time:",
        "\"SMTracker cannot be opened because Apple cannot check it for malicious software\"",
        "(the same for the installer). Open System Settings > Privacy & Security, scroll",
        "to Security and press Open Anyway beside SMTracker, then confirm. Or, in Terminal:",
        "    xattr -cr /Applications/SMTracker.app",
        "which removes the download mark macOS checks (use the folder you installed to).",
        "",
        "Tracking needs Fiji with TrackMate or MATLAB with u-track, installed separately.",
        "The manual opens from Help in every window.",
        "",
        "SMTracker opens no Terminal window. For the commands, type in Terminal:",
        "    /Applications/SMTracker.app/Contents/MacOS/smtracker --selftest",
        "and look for \"analysis self-test: OK\". When something went wrong, Report a problem",
        "in the menu (or the same program with --report) writes the zip to send.",
        "",
        "To remove SMTracker: drag SMTracker.app from Applications to the Trash. The settings,",
        "the caches and a downloaded segmentation pack stay; remove the pack first in the",
        "outlines window (Segmentation pack), or delete ~/Library/Application Support/",
        "SMTracker/packs. The installer's receipt goes with",
        f"    pkgutil --forget {MAC_BUNDLE_ID}",
        "(add --volume ~ for an install for you only; sudo for one for all users).",
        "",
        "Licence: GNU General Public License v3 or later (LICENSE.txt). The source of this",
        "build is in the folder 'source' (inside the app: SMTracker.app/Contents/Resources,",
        "Show Package Contents in the Finder). The other software inside, and its licences,",
        "are listed in THIRD-PARTY-NOTICES.txt.",
        ""])


def add_legal(dist_dir=None) -> list:
    """Write LICENSE.txt, THIRD-PARTY-NOTICES.txt, README.txt and the source zip beside the
    program; returns the paths written. Raises where the commit's source cannot be had: a
    copy without it must not go out."""
    lay = layout(dist_dir)
    dist_dir = lay.legal                 # the folder itself; in a Mac app, Contents/Resources
    info = lay.info() or build_info()
    commit = info.get("commit") or _git("rev-parse", "--short", "HEAD")
    if not commit:
        raise RuntimeError("no commit to take the source from: the copy would ship without its source")
    written = [dist_dir / LICENSE_NAME, dist_dir / NOTICES_NAME, dist_dir / README_NAME]
    shutil.copyfile(REPO / "GPL-License.txt", written[0])
    written[1].write_text(notices_text(bundled_distributions(lay.root), info, mac=lay.mac), encoding="utf-8")
    written[2].write_text(readme_text(info, mac=lay.mac), encoding="utf-8")
    source_dir = dist_dir / SOURCE_DIR
    source_dir.mkdir(exist_ok=True)
    for old in source_dir.glob("smtracker-*-source.zip"):
        old.unlink()
    source = source_dir / f"smtracker-{commit}-source.zip"
    # the paths the commit holds (a bundle built from an older commit lacks the newer ones); never without src/
    paths = [path for path in SOURCE_PATHS if _git("ls-tree", "--name-only", commit, "--", path)]
    if "src" not in paths:
        raise RuntimeError(f"commit {commit} has no src/: the copy would ship without its source")
    made = subprocess.run(["git", "-C", str(REPO), "archive", "--format=zip",
                           f"--prefix=smtracker-{commit}/", "-o", str(source), commit, "--",
                           *paths], capture_output=True, text=True, check=False)
    if made.returncode != 0 or not source.is_file():
        raise RuntimeError(f"git archive failed: {made.stderr.strip()}")
    written.append(source)
    for path in written:
        print(f"  wrote {path.relative_to(dist_dir)} ({path.stat().st_size / 1e3:.0f} kB)")
    return written


# ---------------------------------------------------------------------------------
# VERIFICATION. A build that COMPLETES is not a build that RUNS.
# ---------------------------------------------------------------------------------

#: What the frozen binary must satisfy before anyone is told it works. Every check
#: captures the process's OWN exit code.
#:
#: **NEVER THROUGH A PIPE.** `prog | tail` reports `tail`'s status, so a packaging check
#: written that way cannot fail — recorded in `CLAUDE.md` after a check "verified" a
#: binary that was exiting 127 and printing nothing. `subprocess.run` returns the child's
#: code directly, which is the whole reason the checks live here rather than in a shell
#: line someone types.


def _run(binary, args, timeout=600):
    """Run the frozen binary and return (code, output). No shell, no pipe."""
    import subprocess

    try:
        done = subprocess.run([str(binary), *args], capture_output=True, text=True,
                              timeout=timeout)
        return done.returncode, (done.stdout or "") + (done.stderr or "")
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout} s"
    except OSError as exc:
        return -1, f"{type(exc).__name__}: {exc}"


def _folder_size_mb(path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


def verify_mac_app(lay: Layout, check, macos_target=None) -> None:
    """The checks only a Mac app needs: its name and version in Info.plist, an intact ad-hoc seal (an Apple-silicon
    Mac refuses code whose signature is broken, and adding a file to Contents/Resources breaks it), the processor, and
    the oldest macOS it opens on, held to ``macos_target`` where one is given."""
    import plistlib

    plist_path = lay.root / "Contents" / "Info.plist"
    plist = plistlib.loads(plist_path.read_bytes()) if plist_path.is_file() else {}
    version = lay.info().get("version")
    check("Info.plist names SMTracker, its version and its fixed identifier",
          plist.get("CFBundleName") == "SMTracker" and plist.get("CFBundleIdentifier") == MAC_BUNDLE_ID
          and plist.get("CFBundleShortVersionString") == version and plist.get("CFBundleExecutable") == "smtracker",
          f"{ {k: plist.get(k) for k in ('CFBundleName', 'CFBundleIdentifier', 'CFBundleShortVersionString')} }")
    done = subprocess.run(["codesign", "--verify", "--deep", "--strict", str(lay.root)], capture_output=True,
                          text=True)
    check("the app's ad-hoc signature is intact (codesign --verify --deep --strict)", done.returncode == 0,
          done.stderr.strip()[-300:])
    try:
        print(f"  processor: {bundle_arch(lay.root)}")
    except RuntimeError as exc:
        check("lipo reads the program's processor", False, str(exc))
    report = {}
    minimum = bundle_min_macos(lay.root, report=report)
    print(f"  opens on macOS {minimum or '?'} and later (the newest minos of {len(report)} binaries)")
    check("Info.plist's LSMinimumSystemVersion is the bundle's own minimum",
          plist.get("LSMinimumSystemVersion") == minimum, f"plist {plist.get('LSMinimumSystemVersion')}, "
          f"binaries {minimum}")
    if macos_target:
        target = tuple(int(x) for x in str(macos_target).split("."))
        too_new = {p: v for p, v in report.items() if v > target}
        check(f"every binary loads on macOS {macos_target}", not too_new,
              f"{len(too_new)} need newer, e.g. " + ", ".join(
                  f"{p.relative_to(lay.root)} ({'.'.join(map(str, v))})" for p, v in sorted(too_new.items())[:3])
              if too_new else "")


def verify(dist_dir=None, project=None, condition=None, prep_folder=None, variant=None,
           segment_image=None, macos_target=None) -> int:
    """Check the frozen bundle. Returns 0 when every check passes.

    `project` and `condition` enable the `--dry-run` check, which is the one that
    exercises the real analysis stack inside the bundle — .mat reading, the MCOS
    decoder, cell meshes and TIFF. `--selftest` does NOT make that check: it runs the
    analysis on a SIMULATED set against stored numbers (`smtracker.selfcheck`) and builds
    and tears down a viewer. It reads no file, which is a different thing to prove.

    `variant` (read from the bundle's own build_info.json when not given) decides what must be PRESENT:
    Lite must hold no torch; Normal must hold torch and cellpose, import them frozen and say whether CUDA is
    available (reported, not required: a build machine may have no GPU). `segment_image` enables
    `--segment-check IMAGE` on a Normal build: one real image segmented with Cellpose-SAM through the frozen app.

    `prep_folder` enables `--selftest-prep`: the preparation window, hidden, reads every
    movie in that folder through its SPAWNED worker pool. A frozen app starts each worker
    by re-launching its own executable, so this is the only check here that a pool works
    inside the bundle -- and, given an .nd2 and an .stk, that both raw readers shipped.
    """
    lay = layout(dist_dir)
    dist_dir = lay.root
    binary = lay.binary
    # THE COMMANDS run through the console program on Windows (`console_twin`), as a lab member types them; the
    # Windows app itself is checked by the bare start, a command with nowhere to print, and its own worker pool.
    # In a Mac app the program is the command line too (Contents/MacOS/smtracker).
    cli = lay.cli
    failures = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    print(f"verifying {binary}")
    if not binary.exists():
        print("  FAIL  the binary does not exist")
        return 1
    if os.name == "nt":
        kinds = {p.name: (subsystem(p) if p.is_file() else None) for p in (binary, cli)}
        check("smtracker.exe opens no console (a Windows app) and smtracker-console.exe is its console twin",
              kinds == {binary.name: SUBSYSTEM_WINDOWS, cli.name: SUBSYSTEM_CONSOLE}
              and cli.stat().st_size == binary.stat().st_size, f"subsystems {kinds}")
        if not cli.is_file():
            return 1
    if lay.mac:
        verify_mac_app(lay, check, macos_target)
    stamp = lay.info_path
    stamped = json.loads(stamp.read_text("utf-8")).get("variant") if stamp.is_file() else None
    variant = variant or stamped or DEFAULT_VARIANT
    check("the bundle's own build_info names the variant verified", stamped == variant,
          f"bundle says {stamped!r}, verifying as {variant!r}")
    print(f"  variant: {variant}")

    size = _folder_size_mb(dist_dir)
    print(f"  bundle is {size:.0f} MB")

    internal = lay.internal
    bindings = sorted(p.name for p in internal.iterdir()
                      if p.is_dir() and p.name in {"PySide6", "PyQt5", "PyQt6",
                                                   "PySide2"}) if internal.is_dir() else []
    check("PySide6 is the Qt binding", bindings == ["PySide6"],
          f"found {bindings or 'none'}")

    # THE EXCLUSIONS ARE CHECKED, not assumed. A dependency that arrives unnoticed is
    # exactly how a 407 MB bundle becomes a 1 GB one between two commits.
    if internal.is_dir():
        present = {p.name for p in internal.iterdir()}
        if variant == "lite":
            leaked = sorted(m for m in ("torch", "cellpose_omni", "omnipose", "numba")
                            if m in present)
            check("the segmentation stack is not bundled", not leaked,
                  f"leaked: {leaked}" if leaked else "")
        else:
            # NORMAL: torch and cellpose ship; Omnipose never does (its licence), nor what only it needs
            leaked = sorted(m for m in (*OMNIPOSE_STACK, "numba", "llvmlite", "sklearn") if m in present)
            check("Omnipose and what only it needs are not bundled", not leaked,
                  f"leaked: {leaked}" if leaked else "")
            missing = sorted(m for m in ("torch", "torchvision", "cellpose", "segment_anything", "fill_voids",
                                         "fastremap") if m not in present)
            check("torch, torchvision and cellpose are bundled", not missing,
                  f"missing: {missing}" if missing else "")
            check("the NVIDIA CUDA libraries ship inside torch",
                  (internal / "torch" / "lib" / "cudart64_12.dll").is_file(),
                  "torch/lib/cudart64_12.dll")

    # THE TYPEFACES, which are the check nothing else here can stand in for. Every
    # size, weight and letter-spacing on the shadcn screens was measured in Inter,
    # and `typography._family()` walks a fallback stack rather than failing -- so a
    # bundle that dropped them renders in Segoe UI, passes every other check on this
    # page, and says nothing. PyInstaller collects .py and not .ttf; the fonts are
    # here only because `--add-data` puts them here, which is exactly the kind of
    # line that gets lost when a package is reorganised.
    #
    # The PATH is half the check. `typography.FONT_DIR` is built from the module's
    # own `__file__`, so the data has to land beside the frozen module rather than
    # merely somewhere in the bundle.
    if internal.is_dir():
        font_dir = internal / "smtracker" / "ui" / "theme" / "fonts"
        # A SET, not a sorted list, and that is not fussiness. It was written as
        # an ordered comparison once and failed on a bundle that was completely
        # correct: the faces were then `Geist[wght].ttf` and `GeistMono[wght].ttf`,
        # and `sorted()` puts the Mono FIRST, because the next character after the
        # common prefix is `M` (0x4D) against `[` (0x5B). That is the worst kind of
        # check -- it costs a rebuild to disbelieve. The names no longer collide
        # that way, which is exactly why the set stays: a check that happens to
        # pass because of how two filenames sort is not a check.
        faces = {p.name for p in font_dir.glob("*.ttf")} if font_dir.is_dir() else set()
        check("the bundled typefaces ship, beside their module",
              faces == {"Inter[opsz,wght].ttf", "GeistMono[wght].ttf",
                       "IBMPlexMono-Regular.ttf", "IBMPlexMono-Medium.ttf"},
              f"found {sorted(faces) or 'nothing'} in {font_dir}")

    # THE LICENCES, THE NOTICES AND THE SOURCE (`add_legal`): a copy given to another lab must carry them.
    legal = lay.legal                    # beside the program; in a Mac app, Contents/Resources
    licence = legal / LICENSE_NAME
    check("the GPL ships beside the program (LICENSE.txt)", licence.is_file()
          and "GNU GENERAL PUBLIC LICENSE" in licence.read_text("utf-8", errors="replace")[:200])
    notices = legal / NOTICES_NAME
    notice_text = notices.read_text("utf-8", errors="replace") if notices.is_file() else ""
    check("the third-party notices name Qt's licence and carry the LGPL text",
          "PySide6" in notice_text and "GNU LESSER GENERAL PUBLIC LICENSE" in notice_text,
          "" if notice_text else f"no {NOTICES_NAME}")
    sources = sorted((legal / SOURCE_DIR).glob("smtracker-*-source.zip"))
    names = []
    if len(sources) == 1:
        import zipfile

        with zipfile.ZipFile(sources[0]) as archive:
            names = archive.namelist()
    check("the source of this build ships (source/smtracker-<commit>-source.zip)",
          any(n.endswith("src/smtracker/__init__.py") for n in names)
          and any(n.endswith("packaging/build.py") for n in names),
          f"found {[s.name for s in sources] or 'nothing'}")
    check("README.txt says how to start it", (legal / README_NAME).is_file())

    code, out = _run(cli, ["--selftest"])
    check("--selftest exits 0", code == 0, f"exit {code}")
    # THE BUNDLE COMPUTES WHAT THE SOURCE COMPUTES (`smtracker.selfcheck`): a simulated set of declared D and shares
    # through the frozen SQD fit and HMM, against the numbers stored with the code. Asked for by its own line, so an
    # exit code that is 0 for another reason cannot stand in for it.
    selftest_ok = "analysis self-test: OK" in out
    check("--selftest: the frozen analysis gives the stored numbers", selftest_ok,
          "" if selftest_ok else "no 'analysis self-test: OK' line")
    if code != 0 or "analysis self-test: OK" not in out:
        print("    " + "\n    ".join(out.strip().splitlines()[-12:]))

    code, out = _run(cli, ["--formats"])
    check("--formats exits 0", code == 0, f"exit {code}")

    # THE SEGMENTATION ENGINES, asked of the frozen app itself (`smtracker.segmentation.check`).
    code, out = _run(cli, ["--segment-check"])
    print("    " + "\n    ".join((out or "").strip().splitlines()[-9:]))
    said = [line.strip() for line in (out or "").splitlines()]
    engines = next((l for l in said if l.startswith("segmentation engines:")), "")
    torch_line = next((l for l in said if l.startswith("torch:")), "")
    if variant == "lite":
        check("--segment-check: the Lite build has the classical segmenter and no torch",
              code == 0 and engines == "segmentation engines: classical"
              and torch_line == "torch: not part of this build (the classical segmenter only)", f"exit {code}")
    else:
        text = out or ""
        check("--segment-check: torch imports inside the Normal bundle",
              code == 0 and torch_line.startswith("torch: ") and "not part of" not in torch_line
              and "will not import" not in torch_line, torch_line or f"exit {code}")
        check("--segment-check: cellpose 4 imports inside the Normal bundle and Cellpose is an engine",
              engines == "segmentation engines: classical, cellpose", engines)
        print(f"  INFO  CUDA: {next((l.strip() for l in text.splitlines() if l.startswith('cuda:')), 'not said')}"
              " (reported, not required)")
        if segment_image:
            code, out = _run(cli, ["--segment-check", str(segment_image)], timeout=1800)
            print("    " + "\n    ".join((out or "").strip().splitlines()[-4:]))
            check("--segment-check IMAGE: Cellpose-SAM segments a real image in the frozen app",
                  code == 0 and "cellpose)" in (out or ""), f"exit {code}")
        else:
            print("  SKIP  --segment-check IMAGE (pass --segment-image: a phase .tif; downloads Cellpose-SAM "
                  "if it is not yet on this machine)")

    # THE REPORT A LAB MEMBER SENDS CAN BE WRITTEN BY THE BUNDLE (`smtracker.crashlog`): a frozen app that cannot
    # pack its own log is found here, not on the day something went wrong.
    import zipfile

    with tempfile.TemporaryDirectory() as scratch:
        code, out = _run(cli, ["--report", scratch])
        made = sorted(Path(scratch).glob("*.zip"))
        names = set()
        if made:
            with zipfile.ZipFile(made[0]) as bundle:           # closed before the folder is removed (Windows)
                names = set(bundle.namelist())
        check("--report writes the report zip", code == 0 and "about.txt" in names,
              f"exit {code}; zip holds {sorted(names) or 'nothing'}")

    # THE BUNDLE NAMES ITS CODE, AND IT IS THIS REPOSITORY'S CODE. A bundle verified after
    # the repository moved on would otherwise pass while describing an older commit.
    code, out = _run(cli, ["--version"])
    head = _git("rev-parse", "--short", "HEAD")
    check("--version names the commit it was built from",
          code == 0 and bool(head) and head in (out or ""),
          f"exit {code}; repository at {head or '?'}; bundle says: "
          f"{(out or '').strip().splitlines()[0] if (out or '').strip() else 'nothing'}")

    # BARE INVOCATION MUST NOT DIE. `smtracker` with no arguments is how a lab member
    # starts the shipped installer, and it once aborted at exit 127 having printed
    # nothing, because a QDialog was built before any QApplication existed. It opens a
    # window, so it cannot be run to completion here — but a 127 or an instant non-zero
    # is the failure, and a timeout means it got as far as showing something.
    code, out = _run(binary, [], timeout=25)
    check("bare invocation does not abort", code is None or code == 0,
          "opened a window (timed out, as expected)" if code is None else f"exit {code}")
    if binary != cli:
        # A WINDOWS APP STARTS WITH NO stdout OR stderr (`packaging/entry.py` gives it nowhere to write instead):
        # a command that prints must still end 0, so a library writing to a missing stream cannot end it.
        code, out = _run(binary, ["--version"])
        check("the Windows app runs a command with no console", code == 0, f"exit {code}")

    if project and condition:
        code, out = _run(cli, ["--root", str(project), "--condition", condition,
                                  "--dry-run"], timeout=900)
        check("--dry-run over a real project exits 0", code == 0, f"exit {code}")
        if code != 0:
            print("    " + "\n    ".join(out.strip().splitlines()[-8:]))
    else:
        print("  SKIP  --dry-run (pass --project and --condition to enable it); "
              "this is the check that exercises .mat, MCOS, meshes and TIFF frozen")

    if prep_folder:
        code, out = _run(cli, ["--selftest-prep", str(prep_folder)], timeout=900)
        check("--selftest-prep reads every movie through the worker pool", code == 0,
              f"exit {code}")
        report = [line for line in out.strip().splitlines()
                  if line.startswith(("selftest-prep", "    "))]
        print("    " + "\n    ".join(report or out.strip().splitlines()[-6:]))
        if binary != cli:
            # the pool a double-clicked app starts is made of the WINDOWS app, not of the console twin
            code, out = _run(binary, ["--selftest-prep", str(prep_folder)], timeout=900)
            check("the Windows app's own worker pool reads every movie (--selftest-prep through smtracker.exe)",
                  code == 0, f"exit {code}")
    else:
        print("  SKIP  --selftest-prep (pass --prep-folder: conditions holding raw "
              "movies and no saved curves); the check that the worker pool runs frozen")

    # THE SHIPPED MANUAL IS THE WEBSITE'S (27 Nov 2026): the copy inside the bundle against a fresh copy made from
    # the website's checkout (`manual_from_site.differs_from`); skipped, in words, where there is no checkout
    site = next((Path(r) for r in (os.environ.get("SMTRACKER_SITE"), "D:/smtracker-site")
                 if r and (Path(r) / "content" / "docs" / "meta.json").is_file()), None)
    bundled = internal / "smtracker" / "resources" / "manual" / "MANUAL.md"
    if site is None:
        print("  SKIP  the shipped manual is the website's (no website checkout: set SMTRACKER_SITE to the "
              "smtracker-site folder)")
    elif not bundled.is_file():
        check("the shipped manual is the website's", False, f"no manual in the bundle at {bundled}")
    else:
        import importlib.util

        spec = importlib.util.spec_from_file_location("manual_from_site", REPO / "scripts" / "manual_from_site.py")
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        problem = script.differs_from(site, bundled)
        check("the shipped manual is the website's", problem is None, problem or f"site {site}")

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
        return 1
    print("every check passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT,
                        help="lite (the default: no torch, the classical segmenter only) or normal (Lite plus CUDA "
                             "torch and Cellpose 4; no model ships, Cellpose-SAM downloads at first use). Omnipose "
                             "is never bundled")
    parser.add_argument("--dist", type=Path, default=None,
                        help="where the bundle folder (smtracker/) and its zip are written "
                             "(default packaging/dist); a Normal build is several GB")
    parser.add_argument("--work", type=Path, default=None,
                        help="PyInstaller's work folder (default packaging/build)")
    parser.add_argument("--segment-image", type=Path, default=None,
                        help="a phase .tif for the Normal build's verify to segment with Cellpose-SAM in the frozen app")
    parser.add_argument("--onefile", action="store_true")
    parser.add_argument("--no-clean", dest="clean", action="store_false")
    parser.add_argument("--verify-only", action="store_true",
                        help="skip the build and check the bundle that is already there")
    parser.add_argument("--no-verify", action="store_true",
                        help="build without checking. A build that COMPLETES is not a "
                             "build that RUNS, so this is for iterating, not for "
                             "shipping")
    parser.add_argument("--zip", action="store_true",
                        help="after every check passes, zip the bundle as "
                             "smtracker-<commit>-<date>.zip beside it, for copying to "
                             "another machine")
    parser.add_argument("--installer", action="store_true",
                        help="after every check passes, make the installer beside the bundle: on Windows "
                             "SMTracker-<version>-<commit>-setup.exe with Inno Setup 6 (packaging/smtracker.iss), "
                             "on a Mac SMTracker-<version>-<commit>-macos-<arch>.pkg (pkgbuild and productbuild)")
    parser.add_argument("--macos-target", metavar="VERSION", default=None,
                        help="a Mac: fail the verify unless every binary in the app loads on this macOS (e.g. 13.0). "
                             "The build environment must hold wheels for it: this checks, it cannot fix")
    parser.add_argument("--iscc", type=Path, default=None,
                        help=f"Inno Setup's compiler ISCC.exe, where it is not found by itself (also ${ISCC_ENV})")
    parser.add_argument("--project", help="a real project root, to enable --dry-run")
    parser.add_argument("--condition", help="a condition inside --project")
    parser.add_argument("--prep-folder",
                        help="conditions holding raw movies (ideally one .nd2 and one "
                             ".stk) and no saved curves, to enable --selftest-prep")
    args = parser.parse_args()

    mac_app = sys.platform == "darwin" and not args.onefile
    dist_dir = default_bundle(args.dist) if (args.dist or mac_app) else None
    if not args.verify_only:
        code = build(variant=args.variant, dist=args.dist, work=args.work, onefile=args.onefile,
                     clean=args.clean)
        if code != 0:
            return code
        if os.name == "nt" and not args.onefile:
            print(f"writing the console program: {console_twin(dist_dir)}")
        if mac_app:
            print(f"naming and stamping the app: {finish_app(args.dist)}")
        print("adding the licences, the notices and the source")
        add_legal(dist_dir)
        if mac_app:
            print("signing the app again (ad-hoc), its Resources having changed")
            sign_app(dist_dir)
        if args.no_verify:
            return 0
    code = verify(dist_dir, project=args.project, condition=args.condition,
                  prep_folder=args.prep_folder, variant=args.variant, segment_image=args.segment_image,
                  macos_target=args.macos_target)
    artifacts = []
    if code == 0 and args.zip:
        artifacts.append(zip_bundle(dist_dir))
        print(f"zipped: {artifacts[-1]}")
    if code == 0 and args.installer:
        artifacts.append(make_pkg(dist_dir) if sys.platform == "darwin" else compile_installer(dist_dir, iscc=args.iscc))
        print(f"installer: {artifacts[-1]}")
    if artifacts:
        print(f"checksums: {write_sums(artifacts)}")
    return code


# ---------------------------------------------------------------------------------
# THE INSTALLER (27 Nov 2026, asked: "do we provide zip or isntaller that guides through installation with asking
# what else to download?", then "prepare the installer as discussed").
# ---------------------------------------------------------------------------------
#
# Windows: Inno Setup 6 compiles `packaging/smtracker.iss`, every value passed with /D (`installer_command`): one
# user, no administrator, Start menu and uninstaller, the GPL page, and what else to get (the segmentation pack through
# the program's own command, Fiji's download page), every choice off. A Mac: a .pkg made with the Mac's own pkgbuild
# and productbuild (`make_pkg`, below), offering the same in the Mac's way.

#: Where ISCC.exe is named when it is not on PATH or in Inno Setup's own folders.
ISCC_ENV = "SMTRACKER_ISCC"
ISS_FILE = HERE / "smtracker.iss"


def find_iscc(given=None) -> Path:
    """Inno Setup 6's compiler: ``given``, else ``$SMTRACKER_ISCC``, else PATH, else its usual install folders."""
    candidates = [given, os.environ.get(ISCC_ENV), shutil.which("ISCC")]
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
        if base:
            candidates.append(os.path.join(base, "Inno Setup 6", "ISCC.exe"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise SystemExit(f"Inno Setup 6 was not found: install it (winget install JRSoftware.InnoSetup) and name its "
                     f"ISCC.exe with --iscc or ${ISCC_ENV}")


def _version_and_commit(dist_dir: Path) -> tuple:
    """The bundle's own version and commit (``-edits`` when built from uncommitted changes), from build_info.json."""
    info = json.loads(Layout(dist_dir).info_path.read_text("utf-8"))
    return (info.get("version") or "0"), (info.get("commit") or "unknown") + ("-edits" if info.get("dirty") else "")


def installer_name(dist_dir) -> str:
    """`SMTracker-<version>-<commit>-setup.exe`, the name smtracker.iss gives it (OutputBaseFilename)."""
    version, commit = _version_and_commit(dist_dir)
    return f"SMTracker-{version}-{commit}-setup.exe"


def installer_command(dist_dir, iscc) -> list:
    """The ISCC command for the bundle in ``dist_dir``, writing beside it: the version and commit from its own
    build_info.json, the website and Fiji's page from the package (`manual.SITE_URL`, `fiji.FIJI_WEBSITE`), never
    typed twice."""
    from smtracker.appicon import icon_path
    from smtracker.fiji import FIJI_WEBSITE
    from smtracker.manual import SITE_URL

    dist_dir = Path(dist_dir)
    version, commit = _version_and_commit(dist_dir)
    return [str(iscc), "/Qp",
            f"/DSourceDir={dist_dir}", f"/DOutputDir={dist_dir.parent}",
            f"/DAppVersion={version}", f"/DCommit={commit}",
            f"/DIconFile={icon_path('.ico')}", f"/DSiteUrl={SITE_URL}", f"/DFijiUrl={FIJI_WEBSITE}",
            str(ISS_FILE)]


def compile_installer(dist_dir=None, iscc=None) -> Path:
    """Compile the Windows installer beside the bundle and return its path."""
    dist_dir = Path(dist_dir or (HERE / "dist" / "smtracker"))
    code = subprocess.call(installer_command(dist_dir, find_iscc(iscc)))
    made = dist_dir.parent / installer_name(dist_dir)
    if code != 0 or not made.is_file():
        raise SystemExit(f"Inno Setup stopped with exit {code}; {made.name} {'is' if made.is_file() else 'is not'} there")
    print(f"  {made.stat().st_size / 1e6:.0f} MB")
    return made


# THE MAC INSTALLER (27 Nov 2026): what the Windows installer offers, in the Mac's way. One .pkg (productbuild) holds
# four packages, each a CHOICE on the Installation Type page (`customize="always"` opens it), every extra OFF:
#
#   app      SMTracker.app into /Applications, or ~/Applications for "me only" (no password; the Windows installer's
#            one user, no administrator). Required. Not relocatable: an SMTracker.app elsewhere on the disk (a build
#            folder) is never "upgraded" in its place.
#   segpack  the segmentation pack: opens a Terminal window running the app's OWN `--segmentation-pack install`, so the
#            progress is seen, ending with "press Return to close" (Windows: a console that pauses). Where no pack
#            exists for this Mac (an Intel processor, or a macOS older than the pack's wheels need) it says so there.
#   fiji     Fiji with TrackMate: opens `fiji.FIJI_WEBSITE`.
#   start    open SMTracker when the installer finishes (Windows: "Start SMTracker" on the last page).
#
# The extras are payload-free packages whose postinstall scripts act as the LOGGED-IN USER, never as root (the
# installer runs them as root for an all-users install): the user at the screen (`stat -f%Su /dev/console`) through
# `launchctl asuser` and `sudo -u`, so the pack lands in that user's folder and is remembered in that user's settings.
# A Terminal window is opened by handing it a .command file (`open -a Terminal`), which needs no permission to
# control another app. A script never fails the install: what it could not do it says, and it exits 0.
#
# No desktop shortcut: a Mac app is started from Applications, Launchpad or the Dock. Every value (version, commit,
# processor, oldest macOS, Fiji's page, the pack's oldest macOS) is PASSED IN from the bundle and the package.
# Uninstalling is dragging SMTracker.app to the Trash; settings, caches and a pack stay (README.txt).

MAC_CHOICES = ("app", "segpack", "fiji", "start")


def pkg_name(app, arch: str | None = None) -> str:
    """``SMTracker-<version>-<commit>-macos-<arch>.pkg``, the processor read from the app's program unless given."""
    version, commit = _version_and_commit(app)
    return f"SMTracker-{version}-{commit}-macos-{arch or bundle_arch(app)}.pkg"


def pack_min_macos(spec=None) -> str:
    """The oldest macOS the Apple-silicon segmentation pack's wheels load on: the newest ``macosx_<major>_<minor>`` tag
    among them (`segpack_spec.SPEC`), or "" where no Mac pack is pinned."""
    import re

    if spec is None:
        from smtracker.segpack_spec import SPEC as spec
    newest = None
    for wheels in (spec.get("mac-arm64") or {}).values():
        for wheel in wheels:
            found = re.search(r"macosx_(\d+)_(\d+)_", wheel["filename"])
            if found:
                version = (int(found.group(1)), int(found.group(2)))
                newest = version if newest is None or version > newest else newest
    return f"{newest[0]}.{newest[1]}" if newest else ""


def mac_values(app) -> dict:
    """Every value the installer's texts and scripts use, read from the app and the package, never typed there."""
    from smtracker.fiji import FIJI_WEBSITE
    from smtracker.manual import SITE_URL

    version, commit = _version_and_commit(app)
    arch = bundle_arch(app)
    return {"version": version, "commit": commit, "arch": arch,
            "min_macos": bundle_min_macos(app) or "13.0", "bundle_id": MAC_BUNDLE_ID, "app_name": MAC_APP,
            "fiji_url": FIJI_WEBSITE, "site_url": SITE_URL, "pack_min_macos": pack_min_macos()}


def _xml(text) -> str:
    from xml.sax.saxutils import escape

    return escape(str(text), {'"': "&quot;"})


def distribution_xml(values: dict) -> str:
    """productbuild's distribution file: the welcome, README and licence pages, who may install where, the processor
    and oldest macOS, and the four choices (every extra off)."""
    v = {k: _xml(x) for k, x in values.items()}
    host = ",".join(a for a in values["arch"].split("+"))
    pack = (f"Needs an Apple-silicon Mac with macOS {values['pack_min_macos']} or later; elsewhere the window says so."
            if values.get("pack_min_macos") else "No pack is made for a Mac; the window says so.")
    choices = {
        "app": ("SMTracker", f"SMTracker {values['version']} (build {values['commit']}), the application.", True, False),
        "segpack": ("Segmentation pack (about 2.5 GB)",
                    "PyTorch and Cellpose for Cellpose-SAM segmentation and training, downloaded after the "
                    "installation in a Terminal window that shows the progress. Also later, in the outlines "
                    f"window. {pack}", False, True),
        "fiji": ("Fiji with TrackMate", "For tracking: opens Fiji's download page in your browser.", False, True),
        "start": ("Open SMTracker at the end", "Starts SMTracker when the installer finishes.", False, True),
    }
    lines = ['<?xml version="1.0" encoding="utf-8"?>',
             '<installer-gui-script minSpecVersion="2">',
             f'    <title>SMTracker {v["version"]}</title>',
             '    <welcome file="welcome.txt" mime-type="text/plain"/>',
             '    <readme file="README.txt" mime-type="text/plain"/>',
             '    <license file="LICENSE.txt" mime-type="text/plain"/>',
             f'    <options customize="always" require-scripts="false" hostArchitectures="{host}"/>',
             '    <domains enable_anywhere="false" enable_currentUserHome="true" enable_localSystem="true"/>',
             '    <volume-check>',
             '        <allowed-os-versions>',
             f'            <os-version min="{v["min_macos"]}"/>',
             '        </allowed-os-versions>',
             '    </volume-check>',
             '    <choices-outline>']
    lines += [f'        <line choice="{name}"/>' for name in MAC_CHOICES]
    lines += ['    </choices-outline>']
    for name in MAC_CHOICES:
        title, description, selected, enabled = choices[name]
        lines += [f'    <choice id="{name}" title="{_xml(title)}" description="{_xml(description)}" '
                  f'start_selected="{str(selected).lower()}" enabled="{str(enabled).lower()}">',
                  f'        <pkg-ref id="{_pkg_id(values, name)}"/>',
                  '    </choice>']
    for name in MAC_CHOICES:
        lines.append(f'    <pkg-ref id="{_pkg_id(values, name)}" version="{v["version"]}" onConclusion="none">'
                     f'{name}.pkg</pkg-ref>')
    lines += ['</installer-gui-script>', '']
    return "\n".join(lines)


def _pkg_id(values, name) -> str:
    return values["bundle_id"] if name == "app" else f"{values['bundle_id']}.{name}"


def welcome_text(values: dict) -> str:
    return "\n".join([
        f"SMTracker {values['version']} (build {values['commit']})",
        "",
        "Single-molecule tracking analysis: from raw movies to diffusive-state fractions, condition by condition.",
        "",
        "This installs SMTracker.app: in Applications for all users (asks for your password) or in your own "
        "Applications folder (for you only, no password).",
        "",
        "Installation Type shows what else to get, every choice off: the segmentation pack (PyTorch and Cellpose, "
        "about 2.5 GB), Fiji with TrackMate (its download page), and opening SMTracker at the end. MATLAB with "
        "u-track is licensed and installed separately.",
        "",
        "Removing SMTracker later is dragging it from Applications to the Trash: your settings, caches and a "
        "downloaded pack stay.",
        "",
        f"Licence: GNU General Public License v3 or later. Website: {values['site_url']}",
        ""])


#: The common head of every postinstall script: who is at the screen, and how to act as that user.
_SCRIPT_HEAD = r"""#!/bin/bash
# SMTracker {version} (build {commit}): the installer's "{choice}" choice. Generated by packaging/build.py, every value
# passed in. Acts as the user logged in at the screen, never as root, and never fails the installation.
CONSOLE_USER=$(stat -f%Su /dev/console)
if [ -z "$CONSOLE_USER" ] || [ "$CONSOLE_USER" = "root" ] || [ "$CONSOLE_USER" = "loginwindow" ]; then
    echo "SMTracker {choice}: nobody is logged in at the screen; skipped"
    exit 0
fi
CONSOLE_UID=$(id -u "$CONSOLE_USER")
CONSOLE_HOME=$(dscl . -read "/Users/$CONSOLE_USER" NFSHomeDirectory | awk '{{print $2}}')
as_user() {{
    if [ "$(id -u)" = "0" ]; then
        launchctl asuser "$CONSOLE_UID" sudo -u "$CONSOLE_USER" "$@"
    else
        "$@"
    fi
}}
# where the app was put: the install's own folder ($2: / for all users, the home folder for "me only"), else either
APP=""
for candidate in "${{2%/}}/Applications/{app_name}" "$CONSOLE_HOME/Applications/{app_name}" "/Applications/{app_name}"; do
    if [ -z "$APP" ] && [ -x "$candidate/Contents/MacOS/smtracker" ]; then
        APP="$candidate"
    fi
done
echo "SMTracker {choice}: user $CONSOLE_USER, app ${{APP:-not found}}"
"""

_SCRIPT_BODIES = {
    "fiji": r"""as_user open {fiji_url_q} || echo "SMTracker fiji: the page could not be opened: {fiji_url}"
exit 0
""",
    "start": r"""if [ -n "$APP" ]; then
    as_user open "$APP" || echo "SMTracker start: the app could not be opened"
fi
exit 0
""",
    "segpack": r"""# a .command file opens in Terminal by itself, so this needs no permission to control another app
WINDOW=$(mktemp "/tmp/smtracker-segmentation-pack.XXXXXX") || exit 0
mv "$WINDOW" "$WINDOW.command" && WINDOW="$WINDOW.command"
{{
    echo '#!/bin/bash'
    printf 'APP=%q\n' "$APP"
    cat <<'WINDOW_BODY'
clear
echo "SMTracker {version}: the segmentation pack (PyTorch and Cellpose, about 2.5 GB)"
echo
PACK_MIN={pack_min_macos_q}
HERE_OS=$(sw_vers -productVersion)
at_least() {{ awk -v a="$1" -v b="$2" 'BEGIN {{ split(a, x, "."); split(b, y, "."); exit !((x[1] > y[1]) || (x[1] == y[1] && x[2] + 0 >= y[2] + 0)) }}'; }}
if [ "$(uname -m)" != "arm64" ]; then
    echo "No segmentation pack is made for this Mac: it has an Intel processor, and the pack"
    echo "exists for Apple silicon only. SMTracker works without it (the classical segmenter)."
elif [ -z "$PACK_MIN" ]; then
    echo "No segmentation pack is pinned for a Mac in this version of SMTracker."
    echo "SMTracker works without it (the classical segmenter)."
elif ! at_least "$HERE_OS" "$PACK_MIN"; then
    echo "The segmentation pack needs macOS $PACK_MIN or later (PyTorch and OpenCV are built for it);"
    echo "this Mac runs macOS $HERE_OS. SMTracker works without it (the classical segmenter)."
elif [ -z "$APP" ]; then
    echo "SMTracker.app was not found in Applications, so the pack could not be fetched."
    echo "Open SMTracker and use Segmentation pack in the outlines window."
else
    "$APP/Contents/MacOS/smtracker" --segmentation-pack install
    status=$?
    echo
    if [ $status -eq 0 ]; then echo "The segmentation pack is installed."; else echo "The pack stopped (exit $status): what it said is above. Run it again from the outlines window; a download continues where it stopped."; fi
fi
echo
read -r -p "Press Return to close this window. " _
rm -f "$0"
WINDOW_BODY
}} > "$WINDOW"
chmod 755 "$WINDOW"
chown "$CONSOLE_USER" "$WINDOW" 2>/dev/null
as_user open -a Terminal "$WINDOW" || echo "SMTracker segpack: Terminal could not be opened"
exit 0
""",
}


def postinstall_script(choice: str, values: dict) -> str:
    """The postinstall script of one extra choice (fiji, start, segpack), with every value filled in."""
    import shlex

    if choice not in _SCRIPT_BODIES:
        raise ValueError(f"no script for the choice {choice!r}")
    filled = dict(values, choice=choice, fiji_url_q=shlex.quote(values["fiji_url"]),
                  pack_min_macos_q=shlex.quote(values.get("pack_min_macos") or ""))
    return (_SCRIPT_HEAD + _SCRIPT_BODIES[choice]).format(**filled)


def _component_plist() -> bytes:
    import plistlib

    return plistlib.dumps([{"RootRelativeBundlePath": MAC_APP, "BundleIsRelocatable": False,
                            "BundleIsVersionChecked": False, "BundleHasStrictIdentifier": True,
                            "BundleOverwriteAction": "upgrade"}])


def make_pkg(app=None, out_dir=None) -> Path:
    """The Mac installer, ``SMTracker-<version>-<commit>-macos-<arch>.pkg``, beside the app (or in ``out_dir``)."""
    app = Path(app or default_bundle())
    lay = Layout(app)
    if not lay.mac:
        raise RuntimeError(f"{app} is not a Mac app")
    values = mac_values(app)
    out_dir = Path(out_dir) if out_dir else app.parent
    work = out_dir / "pkg-work"
    if work.exists():
        shutil.rmtree(work)
    root, packages, resources = work / "root", work / "packages", work / "resources"
    for folder in (root, packages, resources):
        folder.mkdir(parents=True)

    def run(command):
        done = subprocess.run([str(c) for c in command], capture_output=True, text=True)
        if done.returncode != 0:
            raise SystemExit(f"{command[0]} stopped with exit {done.returncode}: {done.stderr.strip()}")

    run(["ditto", app, root / MAC_APP])                # ditto keeps the signatures, links and modes
    (work / "component.plist").write_bytes(_component_plist())
    run(["pkgbuild", "--root", root, "--component-plist", work / "component.plist", "--identifier",
         values["bundle_id"], "--version", values["version"], "--install-location", "/Applications",
         packages / "app.pkg"])
    for choice in MAC_CHOICES[1:]:
        scripts = work / f"scripts-{choice}"
        scripts.mkdir()
        script = scripts / "postinstall"
        script.write_text(postinstall_script(choice, values), encoding="utf-8", newline="\n")
        script.chmod(0o755)
        run(["pkgbuild", "--nopayload", "--scripts", scripts, "--identifier", _pkg_id(values, choice),
             "--version", values["version"], "--install-location", "/", packages / f"{choice}.pkg"])
    shutil.copyfile(lay.legal / LICENSE_NAME, resources / LICENSE_NAME)
    shutil.copyfile(lay.legal / README_NAME, resources / README_NAME)
    (resources / "welcome.txt").write_text(welcome_text(values), encoding="utf-8")
    (work / "distribution.xml").write_text(distribution_xml(values), encoding="utf-8")
    out = out_dir / pkg_name(app, values["arch"])
    if out.exists():
        out.unlink()
    run(["productbuild", "--distribution", work / "distribution.xml", "--resources", resources,
         "--package-path", packages, out])
    shutil.rmtree(work)
    print(f"  {out.stat().st_size / 1e6:.0f} MB")
    return out


def write_sums(artifacts, folder=None) -> Path:
    """``SHA256SUMS.txt`` beside the artifacts made (sha256sum's format: ``<64 hex>  <file name>``), for a release to
    carry: the updater (`smtracker.updates`) installs nothing it cannot check against it. Lines of earlier artifacts
    still in the folder are kept, a name made again is replaced."""
    from smtracker import updates
    from smtracker.segpack import sha256_file

    artifacts = [Path(a) for a in artifacts]
    folder = Path(folder) if folder is not None else artifacts[0].parent
    target = folder / updates.SUMS_NAME
    kept = updates.parse_sums(target.read_text("utf-8")) if target.is_file() else {}
    kept = {name: digest for name, digest in kept.items() if (folder / name).is_file()}
    kept.update({a.name: sha256_file(a) for a in artifacts})
    target.write_text(updates.format_sums(sorted(kept.items())), encoding="utf-8", newline="\n")
    return target


def zip_bundle(dist_dir=None) -> Path:
    """The verified bundle as ONE zip named after the commit and the day, so the copy on
    another machine still says which code it is before anyone opens it."""
    lay = layout(dist_dir)
    dist_dir = lay.root
    info = lay.info()
    commit = (info.get("commit") or "unknown") + ("-edits" if info.get("dirty") else "")
    day = str(info.get("built", ""))[:10].replace("-", "") or "undated"
    if lay.mac:
        return _zip_mac_app(lay, f"smtracker-{commit}-{day}-macos-{bundle_arch(lay.root)}")
    # the shipped build is THE build (one build, ruled 26 Nov 2026 evening): smtracker-<commit>-<date>.zip; only a
    # developer's Normal build ends its name with the variant, ...-normal.zip
    variant = info.get("variant", DEFAULT_VARIANT)
    base = dist_dir.parent / f"smtracker-{commit}-{day}{'' if variant == 'lite' else '-' + variant}"
    archive = Path(shutil.make_archive(str(base), "zip", root_dir=dist_dir.parent,
                                       base_dir=dist_dir.name))
    print(f"  {archive.stat().st_size / 1e6:.0f} MB")
    return archive


def _zip_mac_app(lay: Layout, stem: str) -> Path:
    """The Mac zip: a folder "SMTracker" holding SMTracker.app and, visible beside it, the licence, the notices, the
    README and the source zip (the copies inside Contents/Resources stay). Made with ditto, which keeps the app's
    links, modes and signatures (a plain zip breaks the frameworks' links)."""
    staging = lay.root.parent / "zip-staging"
    if staging.exists():
        shutil.rmtree(staging)
    folder = staging / "SMTracker"
    folder.mkdir(parents=True)
    subprocess.run(["ditto", str(lay.root), str(folder / MAC_APP)], check=True)
    for name in (LICENSE_NAME, NOTICES_NAME, README_NAME):
        shutil.copyfile(lay.legal / name, folder / name)
    shutil.copytree(lay.legal / SOURCE_DIR, folder / SOURCE_DIR)
    archive = lay.root.parent / f"{stem}.zip"
    if archive.exists():
        archive.unlink()
    subprocess.run(["ditto", "-c", "-k", "--keepParent", str(folder), str(archive)], check=True)
    shutil.rmtree(staging)
    print(f"  {archive.stat().st_size / 1e6:.0f} MB")
    return archive


if __name__ == "__main__":
    sys.exit(main())
