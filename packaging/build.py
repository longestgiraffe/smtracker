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
file asking for a console, for the commands (`console_twin`).

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

    executable_icon = icon_path(".ico")
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
        # program beside it). Only on Windows: on a Mac --windowed makes an .app bundle, another layout.
        command.append("--windowed")
    if onefile:
        # Slower to start (it unpacks to a temp dir every launch) and harder to
        # debug; the installer will ship the folder build.
        command.append("--onefile")
    for package in COLLECT_ALL:
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

    dist_dir = Path(dist_dir)
    names = set()
    archive = CArchiveReader(str(dist_dir / "smtracker.exe"))
    for entry in archive.toc:
        if entry.upper().startswith("PYZ"):
            names |= {m.split(".")[0] for m in archive.open_embedded_archive(entry).toc}
    internal = dist_dir / "_internal"
    if internal.is_dir():
        for child in internal.iterdir():
            if child.is_dir() and not child.name.endswith((".dist-info", ".egg-info")):
                names.add(child.name)
            elif child.suffix == ".pyd":
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


def notices_text(dists, info: dict) -> str:
    """THIRD-PARTY-NOTICES.txt: a list first (name, version, licence), then every text."""
    rule = "=" * 79
    python = sys.version.split()[0]
    lines = [f"SMTracker {info.get('version', '?')} (commit {info.get('commit') or '?'}): "
             "third-party software in this folder",
             "",
             "SMTracker itself is licensed under the GNU General Public License, version 3 or",
             "later (LICENSE.txt); its source is in the folder 'source'. It contains the",
             "software below, each under its own licence. Qt is used under the GNU Lesser",
             "General Public License v3 through PySide6; its libraries are the separate DLL",
             "files in _internal/PySide6 and may be replaced. Qt's own third-party components",
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
                  "extension modules (.pyd files) in _internal, not linked into SMTracker's own files, so they can be "
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


def readme_text(info: dict) -> str:
    """README.txt: how to start it, and where the licence, notices and source are."""
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


def add_legal(dist_dir=None) -> list:
    """Write LICENSE.txt, THIRD-PARTY-NOTICES.txt, README.txt and the source zip beside the
    program; returns the paths written. Raises where the commit's source cannot be had: a
    copy without it must not go out."""
    dist_dir = Path(dist_dir or (HERE / "dist" / "smtracker"))
    info_path = dist_dir / "_internal" / "smtracker" / "build_info.json"
    info = json.loads(info_path.read_text("utf-8")) if info_path.is_file() else build_info()
    commit = info.get("commit") or _git("rev-parse", "--short", "HEAD")
    if not commit:
        raise RuntimeError("no commit to take the source from: the copy would ship without its source")
    written = [dist_dir / LICENSE_NAME, dist_dir / NOTICES_NAME, dist_dir / README_NAME]
    shutil.copyfile(REPO / "GPL-License.txt", written[0])
    written[1].write_text(notices_text(bundled_distributions(dist_dir), info), encoding="utf-8")
    written[2].write_text(readme_text(info), encoding="utf-8")
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


def verify(dist_dir=None, project=None, condition=None, prep_folder=None, variant=None,
           segment_image=None) -> int:
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
    dist_dir = Path(dist_dir or (HERE / "dist" / "smtracker"))
    binary = dist_dir / ("smtracker.exe" if os.name == "nt" else "smtracker")
    # THE COMMANDS run through the console program on Windows (`console_twin`), as a lab member types them; the
    # Windows app itself is checked by the bare start, a command with nowhere to print, and its own worker pool.
    cli = dist_dir / CONSOLE_NAME if os.name == "nt" else binary
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
    stamp = dist_dir / "_internal" / "smtracker" / "build_info.json"
    stamped = json.loads(stamp.read_text("utf-8")).get("variant") if stamp.is_file() else None
    variant = variant or stamped or DEFAULT_VARIANT
    check("the bundle's own build_info names the variant verified", stamped == variant,
          f"bundle says {stamped!r}, verifying as {variant!r}")
    print(f"  variant: {variant}")

    size = _folder_size_mb(dist_dir)
    print(f"  bundle is {size:.0f} MB")

    internal = dist_dir / "_internal"
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
    licence = dist_dir / LICENSE_NAME
    check("the GPL ships beside the program (LICENSE.txt)", licence.is_file()
          and "GNU GENERAL PUBLIC LICENSE" in licence.read_text("utf-8", errors="replace")[:200])
    notices = dist_dir / NOTICES_NAME
    notice_text = notices.read_text("utf-8", errors="replace") if notices.is_file() else ""
    check("the third-party notices name Qt's licence and carry the LGPL text",
          "PySide6" in notice_text and "GNU LESSER GENERAL PUBLIC LICENSE" in notice_text,
          "" if notice_text else f"no {NOTICES_NAME}")
    sources = sorted((dist_dir / SOURCE_DIR).glob("smtracker-*-source.zip"))
    names = []
    if len(sources) == 1:
        import zipfile

        with zipfile.ZipFile(sources[0]) as archive:
            names = archive.namelist()
    check("the source of this build ships (source/smtracker-<commit>-source.zip)",
          any(n.endswith("src/smtracker/__init__.py") for n in names)
          and any(n.endswith("packaging/build.py") for n in names),
          f"found {[s.name for s in sources] or 'nothing'}")
    check("README.txt says how to start it", (dist_dir / README_NAME).is_file())

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
                             "on a Mac a .dmg")
    parser.add_argument("--iscc", type=Path, default=None,
                        help=f"Inno Setup's compiler ISCC.exe, where it is not found by itself (also ${ISCC_ENV})")
    parser.add_argument("--project", help="a real project root, to enable --dry-run")
    parser.add_argument("--condition", help="a condition inside --project")
    parser.add_argument("--prep-folder",
                        help="conditions holding raw movies (ideally one .nd2 and one "
                             ".stk) and no saved curves, to enable --selftest-prep")
    args = parser.parse_args()

    dist_dir = (args.dist / "smtracker") if args.dist else None
    if not args.verify_only:
        code = build(variant=args.variant, dist=args.dist, work=args.work, onefile=args.onefile,
                     clean=args.clean)
        if code != 0:
            return code
        if os.name == "nt" and not args.onefile:
            print(f"writing the console program: {console_twin(dist_dir)}")
        print("adding the licences, the notices and the source")
        add_legal(dist_dir)
        if args.no_verify:
            return 0
    code = verify(dist_dir, project=args.project, condition=args.condition,
                  prep_folder=args.prep_folder, variant=args.variant, segment_image=args.segment_image)
    artifacts = []
    if code == 0 and args.zip:
        artifacts.append(zip_bundle(dist_dir))
        print(f"zipped: {artifacts[-1]}")
    if code == 0 and args.installer:
        artifacts.append(make_dmg(dist_dir) if sys.platform == "darwin" else compile_installer(dist_dir, iscc=args.iscc))
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
# the program's own command, Fiji's download page), every choice off. A Mac: a .dmg holding the bundle folder and a
# link to Applications (`make_dmg`), made with the Mac's own hdiutil; NOT yet run on a Mac.

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
    info = json.loads((Path(dist_dir) / "_internal" / "smtracker" / "build_info.json").read_text("utf-8"))
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


def make_dmg(dist_dir=None) -> Path:
    """A Mac's disk image: the bundle folder as "SMTracker" beside a link to /Applications, compressed (UDZO)."""
    dist_dir = Path(dist_dir or (HERE / "dist" / "smtracker"))
    version, commit = _version_and_commit(dist_dir)
    staging = dist_dir.parent / "dmg-staging"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(dist_dir, staging / "SMTracker", symlinks=True)
    os.symlink("/Applications", staging / "Applications")
    out = dist_dir.parent / f"SMTracker-{version}-{commit}.dmg"
    code = subprocess.call(["hdiutil", "create", "-volname", f"SMTracker {version}",
                            "-srcfolder", str(staging), "-ov", "-format", "UDZO", str(out)])
    shutil.rmtree(staging)
    if code != 0:
        raise SystemExit(f"hdiutil stopped with exit {code}")
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
    dist_dir = Path(dist_dir or (HERE / "dist" / "smtracker"))
    info_path = dist_dir / "_internal" / "smtracker" / "build_info.json"
    info = json.loads(info_path.read_text("utf-8")) if info_path.is_file() else {}
    commit = (info.get("commit") or "unknown") + ("-edits" if info.get("dirty") else "")
    day = str(info.get("built", ""))[:10].replace("-", "") or "undated"
    # the shipped build is THE build (one build, ruled 26 Nov 2026 evening): smtracker-<commit>-<date>.zip; only a
    # developer's Normal build ends its name with the variant, ...-normal.zip
    variant = info.get("variant", DEFAULT_VARIANT)
    base = dist_dir.parent / f"smtracker-{commit}-{day}{'' if variant == 'lite' else '-' + variant}"
    archive = Path(shutil.make_archive(str(base), "zip", root_dir=dist_dir.parent,
                                       base_dir=dist_dir.name))
    print(f"  {archive.stat().st_size / 1e6:.0f} MB")
    return archive


if __name__ == "__main__":
    sys.exit(main())
