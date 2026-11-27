"""The desktop entry point — and the packaging smoke test that de-risks it.

``smtracker --selftest`` constructs a napari viewer with synthetic layers and
exits 0. That single flag is the whole point of this module existing before the
UI does: napari + Qt + vispy through PyInstaller is the most failure-prone step
in the standalone-installer plan, and it must be proven against ten lines of
code rather than discovered against a finished application.

``smtracker --root <project> --movie series001`` opens the real viewer.
"""

from __future__ import annotations

import os

#: SINGLE-THREADED BLAS, set BEFORE numpy is ever imported (these are read
#: once, at the native library's own init -- setting them later is a no-op).
#: This machine's numpy links Apple's Accelerate (`np.show_config()`), which
#: spawns its OWN native GCD threads for even small linear-algebra steps
#: inside `scipy.optimize.least_squares` -- invisible to Python's GIL and
#: to `parallel._SERIAL_ONLY` alike (that guard stops a SECOND process pool
#: from nesting inside a GUI worker thread; it says nothing about a single
#: fit's OWN native threads). Those threads compete with the GUI thread for
#: actual CPU CORES while a real multistart `fit_sqd` runs on a background
#: QThread -- reported as the SectionPlaceholder's own spinner running
#: smoothly, stalling for a while, then resuming, matching bursts of heavy
#: Accelerate activity during a fit's Jacobian/SVD steps. Undetectable by a
#: headless, offscreen test (three tried, all clean): the stutter is real
#: SCREEN COMPOSITING contention (WindowServer/Metal), which an
#: `QT_QPA_PLATFORM=offscreen` run never touches at all.
#:
#: Global, not scoped to just the GUI-worker path (unlike `_SERIAL_ONLY`,
#: which specifically must NOT apply to `parallel`'s own process-pool
#: workers): `os.environ` mutations here are inherited by every child
#: process too (`multiprocessing`'s "spawn" carries the parent's
#: environment), so this also makes each bootstrap worker single-threaded
#: internally -- the RIGHT default, since `ProcessPoolExecutor` already
#: gives each worker its own core; stacking Accelerate's own threading on
#: top of that would oversubscribe the same cores twice over. `fit_sqd`'s
#: own matrices are tiny (n_pop <= 4 populations), so there is no real
#: multi-core linear algebra to lose here in the first place.
for _var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
from pathlib import Path


def _use_utf8_console() -> None:
    """Windows consoles default to cp1252, which cannot encode 'σ' or 'µ'.

    Every summary in this project reports units and σ, so without this a lab
    user on Windows gets a UnicodeEncodeError instead of a result.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):      # already wrapped or redirected
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smtracker",
        description="SMTracker: single-molecule tracking analysis. With no arguments it opens the stage menu.")
    parser.add_argument(
        "--selftest", action="store_true",
        help="check this installation and exit. A simulated data set goes through the SQD fit and the HMM and "
             "must reproduce the stored numbers, then a napari viewer opens and closes. Exits 0 only if both pass",
    )
    parser.add_argument(
        "--selftest-prep", metavar="FOLDER", type=Path,
        help="open the preparation window hidden on FOLDER, read every movie through its "
             "worker pool, print what was read and exit. Checks that a packaged app can start worker "
             "processes; use a folder without saved bleaching curves",
    )
    parser.add_argument(
        "--simulate", nargs=2, metavar=("RECIPE", "OUT"), type=Path,
        help="build a simulated data set from a recipe file (JSON, `smtracker.generator`) into the new folder OUT, "
             "with its truth, and exit. RECIPE 'example' writes an example recipe to OUT instead",
    )
    parser.add_argument("--root", type=Path, help="project folder")
    parser.add_argument(
        "--deposit", metavar="OUTDIR", type=Path,
        help="run the full pipeline for --condition and write a deposited table to OUTDIR",
    )
    parser.add_argument("--enzyme", help="protein label for --deposit (required; it has no default)")
    parser.add_argument("--sigma-nm", type=float, help="localisation sigma for --deposit")
    parser.add_argument("--erosion-px", type=float, help="per-side erosion for --deposit")
    parser.add_argument("--d-threshold", type=float,
                        help="low-D class boundary on apparent D, for --deposit")
    parser.add_argument("--movie", default="series001", help="movie folder name")
    parser.add_argument("--condition", default=None, help="condition label")
    parser.add_argument(
        "--image-kinds", default="phase,sum",
        help="backgrounds to load into the layer list (comma-separated)",
    )
    parser.add_argument(
        "--layer-controls", action="store_true",
        help="show napari's per-layer editing controls (hidden by default)",
    )
    parser.add_argument(
        "--color-by", default=None, help="track property to colour by",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="dock the single-track inspector; click a track to inspect it",
    )
    parser.add_argument(
        "--hmm-states", type=int, default=2,
        help="fit an HMM with this many states to colour and inspect by",
    )
    parser.add_argument(
        "--browse", action="store_true",
        help="open the classic browser: select conditions and every method view follows",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="start the browser in fast mode (reduced rigor). It is marked on screen and in the "
             "provenance; do not use it for analysis",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="recompute the cache instead of reading it",
    )
    parser.add_argument(
        "--analyses", action="store_true",
        help="dock the analysis panel: every SMTracker 2 tool plus the newer methods, "
             "each naming the .m file it reproduces",
    )
    parser.add_argument(
        "--series-per-day", type=int, default=10,
        help="how many series numbers make one replicate (default 10; some labs "
             "use 8). The blocks present are numbered in the order they occur",
    )
    parser.add_argument(
        "--views", action="store_true",
        help="dock the track views: all tracks on one cell (allTracksonCell.m), one cell "
             "and one track (SMTracker_explorer.m), and playback",
    )
    parser.add_argument(
        "--qc", action="store_true",
        help="dock the QC dashboard: what the length filter removed, density over "
             "time, N per group, and per-unit outliers (flagged, not removed)",
    )
    parser.add_argument(
        "--qc-unit", default="cell", choices=["cell", "day", "movie"],
        help="unit for the outlier panel",
    )
    parser.add_argument(
        "--exclude", action="append", default=[], metavar="KIND:KEY",
        help="exclude a movie, cell or uid, for example 'cell:series004:1'. Can be repeated. "
             "Requires --exclude-reason.",
    )
    parser.add_argument(
        "--exclude-reason", default=None,
        help="why the --exclude units are excluded (required with --exclude; recorded with the result)",
    )
    parser.add_argument(
        "--exclusions", type=Path, default=None,
        help="JSON file of exclusions",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="build the viewer bundle and print it, but open no window. Checks that a packaged app "
             "can read .mat files, decode MCOS, load meshes and read TIFF, which --selftest does not.",
    )
    parser.add_argument(
        "--localization-nm", type=float, default=None,
        help="localisation precision σ in nm (u-track records none). Default: derived "
             "from the pixel size (pixel/√12, the pixelation lower bound). Give a "
             "value only for a measured or estimated σ.",
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="run legacy mode: reproduce the SMTracker 2 output from the same data. "
             "Writes to legacy_out/ only. Every file name and figure title carries the word 'legacy'. "
             "It reproduces the SMTracker 2 method and is not a second analysis; do not use its numbers in a manuscript.",
    )
    parser.add_argument(
        "--prepare", action="store_true",
        help="open the preparation stage: raw .nd2/.stk to cropped single-molecule "
             "stacks, projections and bleaching curves. The crop is set per movie",
    )
    parser.add_argument(
        "--editor-only", action="store_true", dest="editor_only",
        help="skip the segmentation window and open the mesh editor directly, for a "
             "project that already has cell_meshes/",
    )
    parser.add_argument(
        "--segment-classic", action="store_true", dest="segment_classic",
        help="open the classic segmentation window instead of the outlines window",
    )
    parser.add_argument(
        "--editor-classic", action="store_true", dest="editor_classic",
        help="open the classic mesh editor window instead of the outlines window's editor",
    )
    parser.add_argument(
        "--browser-editor", action="store_true", dest="browser_editor",
        help="open the browser-based mesh editor instead of the Qt window",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="print which build this is and exit",
    )
    parser.add_argument(
        "--report", nargs="?", const=Path("."), type=Path, metavar="FOLDER", default=None,
        help="write a report to send when something went wrong, into FOLDER (the current folder if not given), "
             "and exit. It is one zip of the error log, the build, the package versions and the settings file, "
             "with no tracks, images or results",
    )
    parser.add_argument(
        "--check-updates", action="store_true",
        help="ask GitHub at once for the newest release, print the answer (Help's Check for updates) and exit; "
             "installs nothing. Exit 1 when the check did not work",
    )
    parser.add_argument(
        "--segment-check", nargs="?", const="", default=None, metavar="IMAGE",
        help="print which segmentation engines this build carries (classical, Cellpose, Omnipose), torch and "
             "CUDA; with IMAGE, also segment that one image through the windows' own code path (see --model, "
             "--cpu) and exit",
    )
    parser.add_argument(
        "--segmentation-pack", nargs="+", metavar=("ACTION", "DIR"), default=None, dest="segmentation_pack",
        help="the downloadable PyTorch and Cellpose pack: status | list | install [--cuda|--cpu] [--dir DIR] | "
             "install-from WHEELS_DIR [--cuda|--cpu] [--dir DIR] | remove [--dir DIR]; then exit",
    )
    parser.add_argument("--cuda", action="store_true", help="--segmentation-pack: the CUDA (GPU) build")
    parser.add_argument("--dir", type=Path, default=None, metavar="DIR",
                        help="--segmentation-pack: the pack's own folder (default: under the user's local app data)")
    parser.add_argument(
        "--model", default="cpsam", help="the model --segment-check uses: cpsam, classical_sum, "
                                         "bact_phase_omni, or a model file",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="--segment-check on the CPU even where a GPU is present; "
                                         "with --segmentation-pack, the CPU build",
    )
    parser.add_argument(
        "--train-bundle", type=Path, metavar="DIR",
        help="fine-tune from a training bundle the trainer window built, in this process, and exit "
             "(a frozen build has no separate python to run the bundle's train.py)",
    )
    parser.add_argument(
        "--formats", action="store_true",
        help="print which raw acquisition formats this install can read (.nd2 from "
             "Nikon NIS, .stk from MetaMorph, .tif/.tiff) and exit",
    )
    parser.add_argument(
        "--track", action="store_true",
        help="open the preparation and tracking stage: crop and calibrate the movies, "
             "then track with TrackMate (Fiji) or u-track (MATLAB), one condition or a "
             "batch.",
    )
    parser.add_argument(
        "--track-classic", action="store_true",
        help="open the classic preparation and tracking window instead of the standard one",
    )
    return parser


def _headless_reason() -> str | None:
    """Why a GUI folder-picker cannot be shown here, or ``None`` if it can.

    A modal file dialog with no one to answer it hangs forever — this was a real
    full-suite hang, and it would equally hang CI, a batch job, an ssh session, or
    a scheduled run. So before EVER opening one, we ask this: return a short,
    human-readable reason string in any non-interactive context, and ``None`` only
    in a genuine interactive desktop session (including the frozen double-click
    app, which has no TTY but a real display).

    Detected non-interactive: the Qt offscreen/minimal platforms (tests), common
    CI markers, a headless POSIX session (no ``DISPLAY``/``WAYLAND_DISPLAY`` —
    covers ssh-without-X and cron), and an explicit ``SMTRACKER_NONINTERACTIVE``
    override for anything the heuristics miss (e.g. a Windows scheduled task in an
    isolated session). Note: raw ``stdin.isatty()`` is deliberately NOT used — the
    shipped Windows GUI app has no console yet must still show the picker.
    """
    import os
    import sys

    if os.environ.get("SMTRACKER_NONINTERACTIVE"):
        return "SMTRACKER_NONINTERACTIVE is set"
    platform = os.environ.get("QT_QPA_PLATFORM", "").lower()
    if platform in ("offscreen", "minimal"):
        return f"QT_QPA_PLATFORM={platform}"
    for var in ("CI", "GITLAB_CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS"):
        if os.environ.get(var):
            return f"{var} is set (automation)"
    if sys.platform.startswith("linux"):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return "no DISPLAY/WAYLAND_DISPLAY (headless session)"
    return None


#: THERE IS NO FOLDER PICKER HERE ANY MORE, and that is the guarantee rather than an
#: omission. `_pick_project_folder` opened a native dialog before any window existed; it
#: was the last caller of `QFileDialog` in this module, and removing it makes "the window
#: opens first" a property of the SOURCE — `tests/test_launcher.py` asserts this file
#: names no folder dialog at all — instead of something each route has to remember.
#:
#: Every stage window carries its own open control: `segment_gui` and `editor_gui`
#: ("Open folder…"), `prep_gui` ("Choose raw/ folder…"), `tracking_gui` ("Choose
#: folder…") and `open_gui` ("Choose project folder…"), the last of which exists
#: precisely because the comparison browser cannot open without a loaded project.

def _resolve_day_maps_interactive(root, condition_filter=None, remembered=None):
    """Resolve + CONFIRM the replicate-day mapping before the browser loads.

    For each condition with >= 2 proposed days, opens the ``DayMapDialog`` (the
    timestamp proposal, editable, with the confirm gates) and collects the
    confirmed :class:`daymap.DayMapping`. Single-session conditions need no
    mapping and are skipped. Returns ``{condition: DayMapping}`` or None.

    ``remembered`` is what `day_map.json` already holds. A condition that HAS one is not
    re-proposed: it is offered as "keep this, or edit it", which is the shape asked for
    -- "naming convention found, continue or edit the replicate?" -- and keeping is the
    default button, because the remembered mapping is the one previous results were
    reported under and changing it silently would re-number days already published.

    Never runs in a non-interactive context (tests, CI, headless): it returns None
    there so the load falls back to the series default without a modal nobody can
    answer — same rule as the folder picker.
    """
    if _headless_reason() is not None:
        return None
    try:
        from qtpy.QtWidgets import QApplication

        from . import daymap as dm
        from .daymap_widget import DayMapDialog
        from .project import discover_conditions
    except Exception:                                  # pragma: no cover - no Qt
        return None

    try:
        conditions = discover_conditions(root)
    except Exception:
        return None
    if condition_filter:
        conditions = {c: p for c, p in conditions.items() if c == condition_filter}

    QApplication.instance() or QApplication([])
    from .appicon import apply as _apply_icon
    _apply_icon()
    remembered = dict(remembered or {})
    maps = {}
    for condition, path in conditions.items():
        known = remembered.get(condition)
        if known is not None:
            if not _ask_keep_day_map(condition, known):
                # EDIT starts from what is remembered, not from a fresh proposal: the
                # user asked to change a mapping, not to be shown a different one.
                edited = _edit_day_map(condition, path, known)
                if edited is not None and edited.ok:
                    maps[condition] = edited
            continue

        # NOTHING REMEMBERED: ASK, rather than apply the naming rule in silence.
        # Reported: "now it just opened without asking me first and i never set the
        # replicate days. maybe my convention would be different than 0-10 etc." The
        # rule assumes TEN series to a day, which is this lab's convention and not a
        # fact about the data -- at eight, series001-010 is two days rather than one.
        # So the derived answer is shown as a PROPOSAL and confirmed once, then
        # remembered.
        naming = _proposal_from_series_naming(path)
        if naming is not None:
            if _ask_keep_day_map(condition, naming, proposed=True):
                maps[condition] = naming
            else:
                edited = _edit_day_map(condition, path, naming)
                if edited is not None and edited.ok:
                    maps[condition] = edited
            continue
        try:
            proposal = dm.resolve_for_condition(path)
        except Exception:                              # pragma: no cover - IO/mcos
            continue
        if proposal.n_days < 2:
            continue                                   # single session — no mapping
        movie_paths = dm.discover_movie_paths(path)
        # each movie's own frame interval, so the picker can show it per row and
        # REFUSE a day that mixes rates — the day is the analysis unit, and every D
        # scales with 1/tau. Best-effort: an unreadable movieData just means no tau
        # column, never a guessed one.
        tau_by_movie = {}
        try:
            from .utrack_meta import read_project_metadata

            tau_by_movie = {m.movie: m.frame_interval_ms
                            for m in read_project_metadata(path)
                            if m.frame_interval_ms is not None}
        except Exception:                              # pragma: no cover - IO/mcos
            tau_by_movie = {}
        dialog = DayMapDialog(proposal, stamps=dm.read_movie_stamps(movie_paths),
                              movies=list(movie_paths), tau_by_movie=tau_by_movie)
        dialog.setWindowTitle(f"Replicate mapping: {condition}")
        dialog.exec()
        confirmed = dialog.result_mapping()
        if confirmed is not None and confirmed.ok:
            maps[condition] = confirmed
    return maps or None


def _proposal_from_series_naming(condition_root):
    """What the series NAMING implies for one condition, or ``None``.

    The lab's authority -- one block of series numbers is one replicate day, blocks
    numbered in the order they occur. Returned as a mapping to be CONFIRMED, never
    applied on its own: the block size is a convention (ten here, eight elsewhere) and
    nothing in the data says which.
    """
    from . import daymap as dm
    from .grouping import day_map_from_series_blocks

    try:
        movies = list(dm.discover_movie_paths(condition_root))
    except Exception:                                  # pragma: no cover - IO
        return None
    if not movies:
        return None
    try:
        movie_to_day, _spans = day_map_from_series_blocks(movies)
    except Exception:                                  # noqa: BLE001 - advisory
        return None
    mapping = dm.mapping_from_assignments(movie_to_day)
    mapping.source = "manual"
    return mapping if mapping.ok else None


def _ask_keep_day_map(condition: str, mapping, *, proposed: bool = False) -> bool:
    """True to accept the mapping shown, False to edit it.

    KEEP IS THE DEFAULT BUTTON in both cases, for different reasons. A REMEMBERED
    mapping is the one earlier results were reported under, so re-numbering it is the
    change that needs the deliberate press. A PROPOSED one is the lab's own naming
    authority, which is right far more often than not -- but it is shown rather than
    applied, because the block size is a convention and nothing in the data says which.
    """
    from qtpy.QtWidgets import QApplication, QMessageBox

    QApplication.instance() or QApplication([])
    # EVERY DOOR GETS THE ICON. `test_appicon` rediscovers the construction sites from
    # the tree rather than trusting its own table, and it caught this one the moment it
    # was added -- a dialog that opens with the default Qt icon is a different
    # application as far as the taskbar is concerned.
    from .appicon import apply as _apply_icon
    _apply_icon()
    spans = ", ".join(
        f"replicate {d}: {n} movie(s)"
        for d, n in sorted(_movies_per_day(mapping).items()))
    box = QMessageBox()
    box.setWindowTitle(f"Replicates: {condition}")
    if proposed:
        box.setText(f"{condition} has no replicate mapping recorded yet.")
        box.setInformativeText(
            f"From the series numbering, ten to a replicate: {mapping.n_days} replicate(s), "
            f"{spans}.\n\nUse this, or set the replicates yourself? Whichever you choose is "
            f"remembered, so this is asked once.")
        keep = box.addButton("Use this", QMessageBox.AcceptRole)
        box.addButton("Set the replicates…", QMessageBox.RejectRole)
    else:
        box.setText(f"{condition} already has a replicate mapping.")
        box.setInformativeText(f"{mapping.n_days} replicate(s): {spans}\n\n"
                               f"Continue with it, or edit it?")
        keep = box.addButton("Continue", QMessageBox.AcceptRole)
        box.addButton("Edit…", QMessageBox.RejectRole)
    box.setDefaultButton(keep)
    box.exec()
    return box.clickedButton() is keep


def _movies_per_day(mapping) -> dict:
    """``{day: n movies}``, for a one-line summary of a mapping."""
    counts: dict = {}
    for day in (mapping.movie_to_day or {}).values():
        counts[int(day)] = counts.get(int(day), 0) + 1
    return counts


def _edit_day_map(condition: str, path, mapping):
    """Open the day dialog seeded with an existing mapping. None if it was cancelled."""
    from . import daymap as dm
    from .daymap_widget import DayMapDialog

    movie_paths = dm.discover_movie_paths(path)
    try:
        stamps = dm.read_movie_stamps(movie_paths)
    except Exception:                                  # pragma: no cover - IO/mcos
        stamps = []
    dialog = DayMapDialog(mapping, stamps=stamps, movies=list(movie_paths))
    dialog.setWindowTitle(f"Replicate mapping: {condition}")
    dialog.exec()
    return dialog.result_mapping()


def _warn_dialog(message: str) -> None:
    """Best-effort GUI warning; silent if Qt is unavailable (console already told)."""
    try:
        from qtpy.QtWidgets import QApplication, QMessageBox
    except Exception:                                  # pragma: no cover - no Qt
        return
    QApplication.instance() or QApplication([])
    from .appicon import apply as _apply_icon
    _apply_icon()
    QMessageBox.warning(None, "SMTracker: could not load project", message)



def _open_generator():
    """The simulated-data window (`ui.generator.window`). Returns None (closed), "menu" (Back to menu) or
    ``(stage, folder)`` (a built set to open in "track" or "analyse")."""
    from .ui.generator.window import open_generator_window

    application, window = open_generator_window()
    if window is None:
        return None
    application.exec_() if hasattr(application, "exec_") else application.exec()
    if getattr(window, "return_to_menu", False):
        return "menu"
    return getattr(window, "next_stage", None)


def _open_segmentation(args):
    """Open the segmentation stage — the window, or the editor under `--editor-only`.

    ONE ROUTE, called from both places that reach this stage. It used to live inside the
    no-root branch alone, so `--root X --editor-only` fell through to the ANALYSIS route
    and ran a different stage without a word: a flag declared, documented and silently
    ignored on one path, which is this project's most-repeated defect shape
    (`require_all_inside`, `readers.read_tracks`, `fit_by_unit`).
    """
    # THE EDITOR ASKS FOR ITS OWN FOLDER, so it does not go through the project
    # picker's later branches. It scans for `phase/` beside `cell_meshes/`,
    # which is a different question from "which project do I analyse" — a
    # segmentation session often covers a folder with no tracks in it yet.
    #
    # THE Qt EDITOR, once it reached PARITY and not before. It was made the
    # default too early once — with no channel switch, no brightness, contrast
    # or alpha, no draw preview, no merge button, dead keyboard shortcuts and an
    # outline drawn with hooks past both poles — because "the tests pass" was
    # mistaken for "it works". It was reverted the same day.
    #
    # What changed is not confidence, it is a CHECKLIST: `editor_gui.PARITY`
    # lists every control in `editor.html`, and `test_segmentation_editor_gui`
    # reads that page and fails if one has no counterpart here. A missing
    # feature is now a failing test rather than something noticed in use.
    #
    # `--browser-editor` still opens the old one. It works, it is not going
    # away this week, and being one flag from the tool that has been used for
    # real is what makes switching the default a safe thing to do rather than a
    # bet.
    # THE WINDOW OPENS FIRST, AND ITS OWN BUTTON PICKS THE FOLDER.
    #
    # Asked for, and it is the right way round: a folder dialog in front of an
    # application the user has not seen yet gives them nothing to orient by, and if
    # anything then goes slowly there is no window to say so — which is exactly how
    # "i tried to open the segmentation/editor gui -> open folder. now its loading for
    # 10 years and nothing is opening!" happened. Both windows already accept
    # `root=None` and carry an "Open folder…" button, so this costs nothing.
    #
    # `--root` still opens straight onto that folder, so a scripted run is unchanged.
    picked = args.root or None
    if getattr(args, "browser_editor", False):
        from .segmentation.editor import serve

        try:
            serve(root=str(picked) if picked else None)
        except ImportError:
            print("error: the browser editor needs Flask, which is not "
                  "installed.\n       pip install flask   (the Qt editor, "
                  "which is the default, needs nothing extra)", file=sys.stderr)
            return 2
        return 0

    # THE SEGMENTATION WINDOW, which then opens the editor. The door used to
    # go straight to the editor, so the stage offered correction and no way to
    # produce anything to correct. `--editor-only` keeps the direct route for
    # a project whose outlines already exist.
    from qtpy.QtWidgets import QApplication

    if getattr(args, "segment_classic", False) or getattr(args, "editor_classic", False):
        # THE CLASSIC WINDOWS, one flag away while the outlines window is new -- the same
        # way back `--track-classic` keeps for tracking.
        if getattr(args, "editor_classic", False):
            from .segmentation.editor_gui import launch as launch_editor

            launch_editor(root=str(picked) if picked else None)
        else:
            from .segmentation.segment_gui import launch as launch_segment

            launch_segment(root=str(picked) if picked else None)
        application = QApplication.instance()
        if application is not None:
            application.exec_() if hasattr(application, "exec_") \
                else application.exec()
        return 0

    # THE OUTLINES WINDOW, in the workspace's style -- asked for: "finish the
    # segmentation part and mesh editor with guys gui". Segment, Edit outlines and Train
    # in one window; `--editor-only` opens it on the editor. The editor page inherits
    # every handler of the classic editor, so its shortcuts are the same code.
    from .ui.outlines.window import open_outlines_window

    application, window = open_outlines_window(
        str(picked) if picked else None,
        section="edit" if getattr(args, "editor_only", False) else "segment")
    if window is None:
        return 0
    application.exec_() if hasattr(application, "exec_") else application.exec()
    # "BACK TO MENU" IS NOT AN EXIT: the stage chooser again, as the other stages do.
    if getattr(window, "return_to_menu", False):
        return main([])
    return 0


def _remembered_day_maps(root, condition=None, say=print):
    """``({condition: DayMapping}, [conditions with none])`` read from disk.

    A file that cannot be READ is reported and skipped rather than raised: one unreadable
    mapping must not stop a project opening, and the alternative -- silently re-deriving
    over it -- is how two day numberings come to describe one folder.
    """
    from . import daymap as dm
    from .project import discover_conditions

    try:
        conditions = discover_conditions(root)
    except Exception:                                     # noqa: BLE001 - advisory
        return {}, []
    if condition:
        conditions = {c: p for c, p in conditions.items() if c == condition}
    found, missing = {}, []
    for name, path in conditions.items():
        try:
            mapping = dm.load_day_map(path)
        except dm.DayMapError as exc:
            say(f"note: {name}'s remembered replicate mapping is unreadable ({exc}); "
                f"it will be derived instead")
            missing.append(name)
            continue
        if mapping is None:
            missing.append(name)
            continue
        found[name] = mapping
        out = len(set(mapping.held_out) | set(mapping.no_group))
        say(f"replicates for {name}: remembered, "
            f"{mapping.n_days} replicate(s) over {len(mapping.movie_to_day)} movie(s)"
            + (f", {out} left out on the group board" if out else ""))
    return found, missing


def _save_day_maps(root, day_maps, say=print) -> None:
    """Remember every confirmed mapping beside its own condition."""
    from . import daymap as dm
    from .project import discover_conditions

    try:
        conditions = discover_conditions(root)
    except Exception:                                     # noqa: BLE001 - advisory
        return
    for name, mapping in (day_maps or {}).items():
        path = conditions.get(name)
        if path is None or mapping is None:
            continue
        try:
            dm.save_day_map(path, mapping)
        except Exception as exc:                          # noqa: BLE001 - named
            say(f"note: could not remember {name}'s replicate mapping ({exc})")
        else:
            say(f"remembered {name}'s replicate mapping in {dm.DAY_MAP_FILE}")


def _session_project(path):
    """The project a saved session belongs to, loaded as the start screen loads one (the same day maps, the
    session's own trackers and outline tolerance), or None after saying why on the console."""
    from . import options as opts
    from .project import APP_FRAME_INTERVAL, APP_MIXED_TAU, load_project
    from .session_file import SessionError, read_manifest, session_root

    try:
        manifest = read_manifest(path)
        root, conditions = session_root(manifest)
        identity = manifest.get("project") or {}
        sources = {c: t for c, t in (identity.get("trackers") or {}).items() if c in conditions} or None
        tolerance = identity.get("outline_tolerance_nm")
        if tolerance is None:
            from .ui import theme

            tolerance = opts.outline_tolerance_nm(theme.remembered(opts.OPTIONS_KEY))
        print(f"opening the session's project: {root}", flush=True)
        day_maps = resolve_day_maps(root, report=lambda text: print(text, flush=True), interactive=False,
                                    sources=sources)
        from .project import with_replicate_pairs, without_held_out
        from .ui import theme as _theme

        loaded = load_project(root, conditions=tuple(conditions), mixed_tau=APP_MIXED_TAU,
                              frame_interval_from=APP_FRAME_INTERVAL,
                              max_tau_spread=opts.max_tau_spread(_theme.remembered(opts.OPTIONS_KEY)),
                              skip_unsegmented=True,
                              outline_tolerance_nm=tolerance, series_per_day=10, day_maps=day_maps,
                              source=sources or "auto")
        # THE GROUP BOARD'S HOLD-OUTS, remembered in day_map.json (tester D11): what Open workspace left out stays
        # out, so the session's tracks are the ones it was saved with; and the replicates it paired, paired
        return with_replicate_pairs(without_held_out(loaded, day_maps), day_maps)
    except (SessionError, Exception) as exc:                 # noqa: BLE001 - said, and the start screen opens
        print(f"the session could not be opened: {type(exc).__name__}: {exc}", flush=True)
        return None


def resolve_day_maps(root, condition=None, series_per_day=None, report=None,
                     interactive=True, sources=None):
    """The replicate-day mapping: confirmed in a dialog, else derived by naming.

    EXTRACTED so the analysis OPEN WINDOW and the `--root` command line resolve it
    the same way. The dialog half must run on the GUI thread; the load that follows
    it must not, which is why the two are separable at all.

    ``interactive=False`` skips the dialog outright rather than relying on
    `_headless_reason` to suppress it -- for the onboarding window, which has its
    OWN screen for reviewing and editing this same mapping (the Confirm page's
    group board). Popping `DayMapDialog` there as well asked the same question
    twice, once as a modal before the group board even existed and once again on
    it. Remembered mappings still apply, and a first-time project still gets the
    naming authority's derived answer -- only the confirm-now step is skipped.

    ``sources`` is the tracker chosen per condition (`load_project(source=...)`); the days
    are numbered from THAT tracker's movies (`day_maps_from_series_naming`), since a
    condition tracked twice may hold different movies per tracker.
    """
    # Confirm the replicate-day mapping first (interactive only); a single
    # session or a headless run simply gets no day_maps and the series default.
    # WHERE THE REASON GOES. Every line here used to be a `print`, which in the
    # analysis window goes to a stream nobody is looking at -- so when the derivation
    # failed, the window showed only the day guard that fired afterwards and nothing
    # about why the mapping it needed was missing. `report` puts it on screen.
    def say(line):
        print(line, file=sys.stderr if line.startswith("note:") else sys.stdout)
        if report is not None:
            report(line)

    # THREE SOURCES, MERGED PER CONDITION -- never one branch winning the whole project.
    # The first version chose: any remembered mapping at all skipped the derive step, so
    # a project where ONE condition had been confirmed and the others had not came back
    # holding only that one, and the rest reached the loader with no mapping and no word
    # said. Precedence is derived < remembered < confirmed-now, which is least to most
    # deliberate.
    remembered, gone = _remembered_day_maps(root, condition, say)
    confirmed = {}
    if interactive:
        confirmed = _resolve_day_maps_interactive(root, condition,
                                                  remembered=remembered) or {}
        if confirmed:
            # WRITTEN BACK, so the next session does not ask again.
            _save_day_maps(root, confirmed, say)

    # A SAVED MAPPING IS EXTENDED TO MOVIES ADDED SINCE IT WAS SAVED (a new replicate day,
    # or a movie added to a day): it named only the movies present then, and the loader
    # refuses a mapping that leaves one out, so a folder that gained a day could not be
    # opened until its day_map.json was deleted. `daymap.extend_mapping` places each new
    # movie by the series naming and SAYS where.
    stale = [name for name in remembered if name not in confirmed]
    if stale:
        from . import daymap as dm
        from .grouping import day_maps_from_series_naming

        try:
            present = day_maps_from_series_naming(
                root, sources=sources, **({} if series_per_day is None
                                          else {"series_per_day": series_per_day}))
        except Exception as exc:                  # noqa: BLE001 - named on screen
            present = {}
            say(f"note: could not check the saved replicate mappings against the movies "
                f"present ({type(exc).__name__}: {exc})")
        for name in stale:
            if name not in present:
                continue
            try:
                extended, note = dm.extend_mapping(remembered[name], present[name])
            except dm.DayMapError as exc:
                say(f"note: {name}: {exc}")
                continue
            if note:
                remembered[name] = extended
                say(f"note: {name}: {note}")

    settled = {**remembered, **confirmed}
    day_maps = dict(settled)

    # DERIVED ONLY FOR WHAT IS STILL MISSING. Re-deriving a condition somebody confirmed
    # is how two day numberings come to describe one folder, which is the defect the day
    # guard exists for.
    outstanding = [name for name in gone if name not in settled]
    if outstanding or not settled:
        from .grouping import day_maps_from_series_naming

        try:
            # ONLY WHEN ASKED. `series_per_day` is None unless the command line set it,
            # and passing that through OVERRODE the function's own default of 10 --
            # `day_map_from_series_blocks` then compared a series number against None
            # and raised. The caller swallowed it to stderr, so the window showed only
            # the day guard that fired afterwards and none of the reason it had to.
            derived = day_maps_from_series_naming(
                root, sources=sources, **({} if series_per_day is None
                                          else {"series_per_day": series_per_day}))
            spans = derived.pop("__spans__", {})
            for name, mapping in derived.items():
                if name in settled:
                    continue
                if condition and name != condition:
                    continue
                day_maps[name] = mapping
                say(f"replicates for {name}: {spans.get(name, {})}")
        except Exception as exc:                  # noqa: BLE001 - named on screen
            say(f"note: could not derive replicates ({type(exc).__name__}: {exc}). "
                f"The default series rule applies, and it refuses a project whose "
                f"series blocks do not start at 001; pass an explicit replicate mapping.")
    return day_maps or None

def _run_analysis(args) -> int:
    """Load the project, build the viewer bundle and open whatever was asked for.

    Split out of `main` so the entry point reads as what it is, a dispatcher. Only
    this tail moved: the stage routing above it is pinned by four AST tests in
    `test_launcher.py` that walk `main`'s own body, and they exist because two
    whole UIs once became unreachable. Moving `choose_stage` into a helper would
    put the `args.root is None` condition at the call site instead of around it,
    which is a weaker guarantee than the one they make today.

    Takes `args` and nothing else, and returns the process exit code.
    """
    # Imported here, not at module scope: --selftest must not need a project,
    # and the analysis stack must not be paid for by someone who only wants help.
    from .canonical import to_canonical
    from .cellmesh import assign_cells, read_project_cell_meshes
    from .grouping import (UTRACK_GLOB, discover_utrack_project,
                           load_utrack_project)
    from .utrack_meta import acquisition_params_from_project
    from .viewer import build_bundle, launch

    from .grouping import GroupingError

    # `--condition` SCOPES THE READ, it is not only a label. Every call below scanned
    # the WHOLE root and used `--condition` as a name to stamp on whatever it found, so
    # the documented invocation — `--root <project> --condition <label>`, which is the
    # packaging check — refused any project with two conditions: `discover_utrack_project`
    # sees `glucose/tif/series001` and `malate/tif/series001` and rightly says movie
    # names must be unique. It only ever worked on a single-condition root, and the
    # reference project has two.
    #
    # Scoped by NAME MATCH ONLY. If `<root>/<condition>` is not a folder the root is left
    # alone, because a condition label that is not a folder name is a legitimate thing to
    # want (`--root <one condition> --condition Malate` stamps a label on a folder called
    # something else), and silently redirecting that would be worse than not scoping.
    read_root = Path(args.root)
    if args.condition and (read_root / args.condition).is_dir():
        read_root = read_root / args.condition
        print(f"reading {read_root} (scoped by --condition)")

    params = acquisition_params_from_project(
        read_root, localization_precision_nm=args.localization_nm
    )
    # THE DAY MAP MUST REACH THIS PATH TOO. `--browse` resolves it through the picker;
    # this one called `load_utrack_project` with no day map at all, so it fell to the
    # DEFAULT rule -- which numbers a day by its ABSOLUTE block of series numbers and
    # therefore refuses any project whose blocks are not contiguous from the first
    # (series001-010 then 031-050 gives days [1, 4, 5]). The guard was right and the
    # instruction it printed named `day_map_from_series_blocks`, which is IN THE PACKAGE
    # and which this path never called. A guard that names a fix the tool cannot perform
    # is a dead end for the person reading it.
    from .grouping import day_map_from_series_blocks

    day_map = None
    try:
        movies = [str(m) for m, _ in discover_utrack_project(
            str(read_root), pattern=UTRACK_GLOB)]
        if movies:
            day_map, spans = day_map_from_series_blocks(
                movies, series_per_day=args.series_per_day)
            print(f"replicates from the series naming "
                  f"({args.series_per_day} series per replicate): "
                  f"{len(set(day_map.values()))} replicates over {len(movies)} movies")
    except Exception as exc:                          # noqa: BLE001
        print(f"note: could not derive replicates from the series naming ({exc}); "
              f"falling back to the default rule", file=sys.stderr)

    try:
        raw = load_utrack_project(read_root, condition=args.condition,
                                  day=day_map, series_per_day=args.series_per_day)
    except GroupingError as exc:
        # The loader is right to refuse to invent a condition label, but a lab
        # user should meet that as an instruction, not as a stack trace.
        print(f"error: {exc}", file=sys.stderr)
        if args.condition is None:
            print(
                "\nThis project does not label its own conditions, so one must be "
                "given:\n"
                f"    py -m smtracker.app --root {args.root} --condition <label> "
                "--inspect\n"
                "Any label works; it names the group in the output.",
                file=sys.stderr,
            )
        return 2
    # THE OUTLINES AFTER THE TRACKS: the condition check above is what tells a user of a two-condition root to
    # name one. Read first, the outlines of glucose/ and malate/ (both with a series001) stopped the run with a
    # traceback instead (since `read_project_cell_meshes` reads every outline folder under a root and refuses a
    # movie found twice, 20 Oct 2026); before that, the first folder's outlines were used for every movie, silently.
    from .cellmesh import CellMeshError

    try:
        meshes = read_project_cell_meshes(read_root, allow_unsegmented=True)
    except CellMeshError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    canonical = to_canonical(assign_cells(raw, meshes), params)

    if args.exclude or args.exclusions:
        from .exclusions import Exclusion, ExclusionError, ExclusionSet, apply_exclusions

        try:
            excluded = (
                ExclusionSet.from_json(args.exclusions) if args.exclusions
                else ExclusionSet()
            )
            for text in args.exclude:
                if not args.exclude_reason:
                    raise ExclusionError(
                        "--exclude needs --exclude-reason: an unexplained "
                        "exclusion is unreproducible later"
                    )
                excluded.items.append(
                    Exclusion.parse(text, reason=args.exclude_reason)
                )
            canonical = apply_exclusions(canonical, excluded)
        except ExclusionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"exclusions: {canonical.attrs['exclusion_summary']}")

    bundle = build_bundle(
        canonical, movie=args.movie, meshes=meshes, root=read_root,
        image_kinds=tuple(k.strip() for k in args.image_kinds.split(",") if k.strip()),
    )
    print(bundle.summary())
    if args.dry_run:
        return 0

    import napari

    viewer = launch(bundle, color_by=args.color_by,
                    show_layer_controls=args.layer_controls)

    fit = None
    if args.inspect or args.qc:
        # Fitted here, once, and handed to the widgets: a Qt class must not
        # compute anything (see smtracker.widgets).
        from .hmm import HMMError, fit_hmm, step_sequences

        try:
            fit = fit_hmm(
                step_sequences(canonical), n_states=args.hmm_states,
                tau_s=params.tau_s, n_starts=6,
            )
            print(f"HMM D = {fit.D.round(4)}  occupancy = {fit.occupancy.round(3)}")
        except HMMError as exc:
            # A degenerate chain is informative, not fatal: the views still work
            # without per-step states.
            print(f"note: no usable HMM fit ({exc}); continuing without states")
            fit = None

    if args.inspect:
        from .qc import track_summary
        from .widgets import dock_inspector, dock_track_list

        outlines_um = {
            f"{m.movie}:{m.cell_id}": m.polygon * params.um_per_pixel
            for values in meshes.values() for m in values
        }
        inspector = dock_inspector(
            viewer, bundle, canonical, hmm_fit=fit, outlines_um=outlines_um,
            confinement_radius_um=0.1,
        )
        # The track list is the second selection mechanism (BUGS.md B2): it
        # drives the same inspector a canvas click does.
        movie_tracks = canonical[canonical["movie"] == args.movie]
        dock_track_list(
            viewer, track_summary(movie_tracks, hmm_fit=fit),
            on_select=inspector.show_uid,
        )

    if args.analyses:
        from .widgets import dock_analyses

        # The panel reads smtracker.analyses; nothing about which tools exist lives in
        # the Qt layer, so a missing tool is a catalogue entry rather than a UI edit.
        dock_analyses(viewer, canonical)

    if args.views:
        from .cellcoords import project_tracks
        from .widgets import dock_views

        # The cell-frame projection is computed HERE, once, and handed over: a Qt class
        # must not compute anything. Without ribs there is no centreline, so a cell that
        # cannot be projected is dropped by project_tracks rather than guessed at.
        try:
            projected = project_tracks(canonical, meshes, require_centerline=False)
        except Exception as exc:                    # noqa: BLE001
            print(f"note: no cell-frame projection ({exc}); "
                  f"the two static views will say so")
            projected = None
        dock_views(
            viewer, canonical, projected=projected, movie=args.movie,
            background=bundle.image, pixel_size_nm=params.pixel_size_nm,
            frame_interval_ms=params.frame_interval_ms,
        )

    if args.qc:
        from .widgets import dock_qc

        dock_qc(
            viewer, canonical, hmm_fit=fit, unit=args.qc_unit,
            field_shape=bundle.image.shape if bundle.image is not None else None,
        )

    napari.run()
    return 0


def _simulate_command(recipe_path: Path, out: Path) -> int:
    """``--simulate RECIPE OUT``: the generator from the command line (`smtracker.generator`)."""
    from . import generator

    if str(recipe_path) == "example":
        print(f"example recipe written: {generator.save_recipe(generator.example_recipe(), out)}")
        return 0
    try:
        recipe = generator.load_recipe(recipe_path)
        print(generator.describe(recipe))
        generator.build(recipe, out, on_line=print)
    except generator.GeneratorError as exc:
        print(f"not built: {exc}")
        return 2
    print(f"built: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    # HTTPS ON A MAC (27 Nov 2026): the packaged app's OpenSSL has no certificate list of its own there, so every
    # download failed (the segmentation pack, Cellpose-SAM's model, the updater); `netssl` points it at one first.
    from .netssl import use_bundled_certificates

    use_bundled_certificates()
    args = _build_parser().parse_args(argv)

    if getattr(args, "simulate", None) is not None:
        return _simulate_command(*args.simulate)

    if getattr(args, "report", None) is not None:
        from . import crashlog

        print(crashlog.about_text())
        print(f"report written: {crashlog.report_bundle(args.report)}")
        return 0

    if getattr(args, "check_updates", False):
        # one HTTPS request to GitHub, the words Help shows (27 Nov 2026; also the quickest proof that a packaged
        # app's downloads work on a machine, `netssl`)
        from .updates import check_now

        outcome = check_now()
        print(outcome.words)
        return 1 if outcome.kind == "failed" else 0

    if not (getattr(args, "version", False) or getattr(args, "formats", False)):
        # WHAT WENT WRONG IS WRITTEN DOWN (`smtracker.crashlog`): every uncaught error, Qt's own warnings and the
        # stack at a hard crash go to a log beside the settings, which `--report` and each window's "Report a
        # problem" pack to send. It never raises and never stops a start; the two flags that only print skip it.
        from . import crashlog

        crashlog.install(argv)
        if not getattr(args, "legacy", False):
            crashlog.install_qt()

    if getattr(args, "legacy", False):
        # ONE SWITCH (legacy rule 1). Everything downstream of here writes through
        # smtracker.legacy.guard.legacy_path, which refuses any path outside
        # legacy_out/, so the two analyses cannot mix on disk.
        from .legacy.run import run_legacy

        return run_legacy(args)

    if getattr(args, "version", False):
        from .buildstamp import version_text

        print(version_text())
        return 0

    if getattr(args, "segmentation_pack", None) is not None:
        from . import segpack_view

        action, *rest = args.segmentation_pack
        flavour = "cuda" if args.cuda else ("cpu" if args.cpu else None)
        return segpack_view.cli(action, rest, flavour=flavour, folder=args.dir,
                                say=lambda line: print(line, flush=True))

    # A pack the user downloaded earlier is put on sys.path before anything imports torch or cellpose
    # (smtracker.segpack: appended, so the bundle's own modules win).
    from . import segpack

    segpack.activate_remembered()

    if getattr(args, "segment_check", None) is not None:
        from .segmentation import check

        if not args.segment_check:
            print("\n".join(check.engines_report()))
            return 0
        return check.segment_check(args.segment_check, args.model, use_gpu=not args.cpu)

    if getattr(args, "train_bundle", None) is not None:
        from .segmentation import check

        return check.train_check(args.train_bundle)

    if getattr(args, "formats", False):
        # Thin: rawmovie owns the fact, this only prints it.
        from .rawmovie import describe_formats, readable_formats

        print(describe_formats())
        return 0 if all(i["available"] for i in readable_formats().values()) else 1

    if args.selftest:
        # THE ANALYSIS FIRST (`smtracker.selfcheck`, 10 Nov 2026): it needs no display and never raises, so its
        # lines are printed even where the viewer check then fails. Both must hold.
        from . import selfcheck
        from .viewer import selftest

        analysis = selfcheck.main()
        return 0 if (selftest() and analysis) else 1

    if args.selftest_prep is not None:
        from .ui.prep.window import selftest_prep

        return selftest_prep(args.selftest_prep)

    if args.dry_run and args.root is None:
        # --dry-run IS THE PACKAGING CHECK, so it must never reach a dialog. Without a
        # root it would fall through to the launcher and wait on a window: in CI that
        # blocks, and in a desktop session it verifies NOTHING while exiting 0, which is
        # the shape of a check that passes vacuously. It needs a project, so say so.
        print(
            "error: --dry-run needs a project to read, and none was given.\n"
            "       It checks that the analysis stack (.mat, MCOS, meshes, TIFF) works,\n"
            "       so give it a real project:\n"
            "         smtracker --root <project folder> --condition <label> --dry-run",
            file=sys.stderr,
        )
        return 2

    if (getattr(args, "editor_only", False) or getattr(args, "browser_editor", False)
            or getattr(args, "segment_classic", False)
            or getattr(args, "editor_classic", False)):
        # THE SEGMENTATION FLAGS WORK WITH OR WITHOUT `--root`, and they did not before:
        # `--root X --editor-only` ran the analysis and said nothing.
        return _open_segmentation(args)

    if args.root is None:
        # No folder given. In an interactive desktop session, open a picker and go
        # comparison-first on the choice (the "just run smtracker" flow). In ANY
        # non-interactive context, fail loud with an actionable message instead of
        # opening a modal dialog nobody can answer — CI, batch, ssh and scheduled
        # runs must exit nonzero, never block.
        headless = _headless_reason()
        if headless is not None:
            print(
                f"error: no project folder given, and cannot open a folder picker "
                f"({headless}).\n"
                "       pass --root <project folder>, the folder containing your "
                "condition subfolders, e.g.\n"
                "         smtracker --root path/to/project",
                file=sys.stderr,
            )
            return 2
        # THE LAUNCHER: which STAGE, not which analysis. Only the no-argument
        # flow sees it — every existing flag still goes straight where it always
        # did, because a menu in front of a path is how the --analyses catalogue
        # became unreachable once already.
        from .launcher import (STAGE_ANALYSE, STAGE_PREPARE, STAGE_SEGMENT,
                               STAGE_SIMULATE, STAGE_TRACK, choose_stage)

        stage = choose_stage()
        if stage is None:
            print('cancelled.')
            return 0            # closing a chooser is not an error

        # AN UNBUILT STAGE MUST NOT FALL THROUGH TO A BUILT ONE. The fallthrough here
        # was once `else: browse`, so a stage enabled ahead of its window existing
        # would have silently opened the ANALYSIS browser — the user looking at the
        # wrong stage with no error anywhere. The refusal below stays even though the
        # editor now exists, because it is what catches the NEXT stage added.
        if stage == STAGE_SEGMENT:
            return _open_segmentation(args)
        # SIMULATED DATA (25 Nov 2026): the generator's window; a built set opens in the stage it is for, tracks in
        # the analysis (its folder first among the start screen's recent ones) and movies in image preparation.
        if stage == STAGE_SIMULATE:
            after = _open_generator()
            if after is None:
                return 0
            if after == "menu":
                return main([])
            kind, folder = after
            if kind == "track":
                args.root = Path(folder)
                stage = STAGE_TRACK
            else:
                stage = STAGE_ANALYSE

        # THE WINDOW OPENS FIRST, AND ITS OWN BUTTON PICKS THE FOLDER — asked for, for
        # every stage whose window carries one. Preparation has "Choose raw/ folder…"
        # and tracking has "Choose folder…", so a picker in front of either put a modal
        # dialog ahead of an application the user had not seen yet.
        #
        # ANALYSIS IS THE EXCEPTION, and it is not an oversight: `browser` opens onto a
        # project that has ALREADY been read, so there is nothing to put on screen until
        # a folder has been chosen and loaded. Named here rather than left as an
        # asymmetry to be rediscovered.
        if stage == STAGE_TRACK:
            args.track = True
        elif stage == STAGE_PREPARE:
            args.prepare = True
        elif stage == STAGE_ANALYSE:
            # ANALYSIS OPENS A WINDOW FIRST TOO, and it needed one built for it: the
            # browser takes a LOADED project, so there is nothing to show until a
            # folder has been read. The onboarding flow is that front door — choose
            # or reopen a folder, see what is in it IMMEDIATELY from a glob, and
            # watch the numbers fill in as the load runs on a thread, which here is
            # genuinely minutes of work rather than a stray import.
            from .ui.onboarding.window import open_project
            from .ui.workspace.window import open_workspace, take_after_close, take_session_path

            # THE WORKSPACE'S HEADER MENU leads back here: "Open another folder" runs the
            # start screen again, "Back to menu" the stage chooser (asked 15 Oct 2026: "we
            # need a go back button or something to open a new folder").
            session = None
            while True:
                # A SESSION OF ANOTHER PROJECT (the workspace's Open session..., 17 Oct 2026): its folders are
                # loaded the way the start screen loads them, then the session drawn; a load that fails says why
                # and the start screen opens instead.
                project = colours = None
                if session is not None:
                    project = _session_project(session)
                    if project is None:
                        session = None
                if project is None:
                    project, colours = open_project()
                if project is None:
                    from .ui.onboarding.window import take_after_close as start_after_close

                    # THE START SCREEN'S BACK TO MENU (tester B22, 3 Nov 2026) is the stage chooser again
                    if start_after_close() == "menu":
                        return main([])
                    print('cancelled.')
                    return 0        # closing the flow is not an error, as above
                # HANDED OVER LOADED, which is the whole point of the flow: the new
                # workspace shell (sidebar + toolbar; see `ui.workspace`) takes a
                # project rather than a path, so nothing is read twice. It replaces
                # `browser.launch_browser` on this path while the new UI is being
                # ported view by view -- `launch_browser` still exists and is what
                # `--browse` opens directly, below.
                #
                # `open_workspace` SHOWS and returns; the event loop is the caller's,
                # exactly as it was when `open_gui` held this branch.
                application, _window = open_workspace(project, colours=colours)
                if session is not None:
                    _window.open_session(session)
                    session = None
                if application is not None:
                    (application.exec_() if hasattr(application, "exec_")
                     else application.exec())
                after = take_after_close()
                if after == "session":
                    session = take_session_path()
                    continue
                if after == "another":
                    continue
                if after == "menu":
                    return main([])
                return 0
        else:
            # A stage was added to STAGES and not to this routing. Better to say so than
            # to send someone to whichever branch happened to be last.
            print(f"error: no route for stage {stage!r}", file=sys.stderr)
            return 2

    if args.deposit is not None:
        # AT1: the CLI and the GUI call the SAME function with the SAME arguments.
        # Nothing is computed here; this block only parses and prints.
        from .run import config_from_inputs, run_to_deposit

        missing = [n for n in ("root", "condition", "enzyme", "sigma_nm",
                               "erosion_px", "d_threshold")
                   if getattr(args, n, None) is None]
        if missing:
            print(f"--deposit needs {', '.join('--' + m.replace('_', '-') for m in missing)}"
                  "; none of these has a default", file=sys.stderr)
            return 2
        # AU3: the SAME constructor the GUI uses, so the two parameter sets are equal
        # by construction rather than by coincidence.
        cfg = config_from_inputs(
            enzyme=args.enzyme, condition=args.condition, sigma_nm=args.sigma_nm,
            erosion_px=args.erosion_px, d_threshold=args.d_threshold)
        result = run_to_deposit(args.root, cfg, args.deposit)
        print(result.counts_line())
        for K in sorted(result.fits):
            f = result.fits[K]
            occ = ", ".join(f"{100 * o:.1f} %" for o in f.occupancy)
            print(f"  {f.label()}  occupancy {occ}   BIC {f.bic:.1f}")
        print(result.gag_reason if result.gagged
              else "  not gagged - enrichment number emitted")
        prov = result.provenance or {}
        print(f"  commit {prov.get('commit') or 'unknown'}"
              + ("  [uncommitted changes]" if prov.get("dirty") else ""))
        print(f"  wrote {', '.join(sorted(result.written))}")
        return 0

    if getattr(args, 'prepare', False):
        # PREPARATION produces the stacks every later stage reads, so like tracking it
        # is a pipeline stage rather than an analysis and gets its own window.
        from .prep_gui import launch_prep

        return launch_prep(args.root)

    if getattr(args, 'track', False) and not getattr(args, 'track_classic', False):
        # THE NEW PREPARATION WINDOW, in the workspace's style -- asked for: "the gui for
        # image prep is clunky, long load times ... can we use the qt analysis gui from
        # hui for this as well?". It covers movies, crop, calibration and save; its
        # Detect & link and Track pages open the classic window on the same folder until
        # those are ported. `--track-classic` opens the classic window directly.
        from .ui.prep.window import open_prep

        app, window = open_prep(args.root)
        # THE LOOP RUNS ONLY FOR A WINDOW THAT EXISTS, so a stand-in that opens nothing
        # (a test's) returns an exit code instead of blocking on an empty loop.
        if window is None:
            return 0
        code = app.exec()
        # "BACK TO MENU" IS NOT AN EXIT, the classic window's rule: its button asks for the
        # stage chooser again; closing the window any other way leaves the tool.
        if getattr(window, "return_to_menu", False):
            return main([])
        return code

    # `--track-classic` ALONE opens the classic window too, as its help says (tester B23, 3 Nov 2026: with `--root`
    # and without `--track` it fell through to the analysis and opened the napari viewer on series001)
    if getattr(args, 'track', False) or getattr(args, 'track_classic', False):
        # TRACKING is a pipeline stage, not an analysis: it PRODUCES the
        # canonical table that every catalogue entry consumes, and one batch is
        # a quarter of an hour. Its window is its own.
        from .tracking_gui import RETURN_TO_MENU, launch_tracking

        # BACK TO THE MENU, NOT OUT OF THE TOOL. Tracking is a MIDDLE step — what
        # follows is segmentation or analysis — so its Done button asks for the stage
        # chooser again rather than ending the process. Closing the window any other way
        # still exits, which is the distinction: the X means "done with the tool", the
        # button means "done with this stage".
        #
        got = launch_tracking(args.root)
        if got != RETURN_TO_MENU:
            return got
        # RE-ENTER THE NO-ARGUMENT FLOW, which is what "the main menu" IS: the stage
        # chooser, and then whichever window that stage opens. The folder is picked
        # INSIDE that window now rather than ahead of it, which serves the same
        # point — the next stage is often a different condition — without a dialog
        # in front of a window nobody has seen yet.
        #
        # `main([])` and not a loop over the dispatch below, because that dispatch is
        # driven by the `args` flags this branch has already set: looping it would need
        # them cleared, and a flag cleared in one place and read in five is how a stage
        # ends up running twice.
        return main([])

    if args.browse:
        # Self-contained: the browser loads its own multi-condition project and
        # needs no single-movie bundle.
        from .browser import launch_browser
        from .fastmode import BANNER
        from . import options as opts
        from .project import APP_FRAME_INTERVAL, APP_MIXED_TAU, ProjectError, load_project
        from .ui import theme

        day_maps = resolve_day_maps(args.root, args.condition,
                                    args.series_per_day)
        try:
            project = load_project(
                args.root,
                conditions=[args.condition] if args.condition else None,
                localization_precision_nm=args.localization_nm,
                day_maps=day_maps,
                mixed_tau=APP_MIXED_TAU, frame_interval_from=APP_FRAME_INTERVAL,
                max_tau_spread=opts.max_tau_spread(theme.remembered(opts.OPTIONS_KEY)),
                outline_tolerance_nm=opts.outline_tolerance_nm(
                    theme.remembered(opts.OPTIONS_KEY)),
                skip_unsegmented=True,
            )
            # THE GROUP BOARD'S HOLD-OUTS, remembered in day_map.json: left out here too, one rule at every door
            # (ruled 4 Nov 2026)
            from .project import with_replicate_pairs, without_held_out

            project = with_replicate_pairs(without_held_out(project, day_maps), day_maps)
            from .onboarding_view import unsegmented_notice

            notice = unsegmented_notice(project.canonical.attrs.get("dropped_unsegmented"),
                                        list(project.conditions))
            if notice is not None:
                print(f"{notice['title']}: {notice['body']}", flush=True)
        except ProjectError as exc:
            print(f"error: {exc}", file=sys.stderr)
            _warn_dialog(f"Could not load a project from:\n{args.root}\n\n{exc}")
            return 2
        print(project.summary())
        if args.fast:
            print("*** " + BANNER.replace("—", "-") + " ***")
        app, _browser = launch_browser(project, root=args.root, fast=args.fast,
                                       refresh=args.refresh)
        app.exec()
        return 0

    return _run_analysis(args)


if __name__ == "__main__":
    sys.exit(main())
