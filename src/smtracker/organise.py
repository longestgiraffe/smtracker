"""A flat folder of acquisitions becomes a project folder, and says so first.

What a microscope leaves behind is a single folder holding everything::

    conditionA/
        series001.nd2     the acquisition, 4500 frames
        series001.tif     the phase snapshot, ONE page
        series002.nd2
        series002.tif
        ...

What every stage here expects is the layout the lab already uses on its own drives,
confirmed against `D:/data/project/condition`::

    <condition>/
        raw/series001.nd2         the acquisitions
        phase/series001.tif       the single-page companions
        tif/<movie>/...           written by preparation, not here
        cell_meshes/*.mat         written by Oufti, not here

So the gap between the two is a rename, and asking the experimenter to do it by hand
before every project is asking them to do the one step where a mistake is silent: a
`.tif` filed as an acquisition becomes a twentieth movie, and this project has already
measured that exact defect (three callers listing ten `.nd2` beside ten `.tif` and
reporting twenty movies).

Five rules, each because the alternative reads as something it is not.

1. Plan and apply are separate. `plan_layout` reads and touches nothing;
   `apply_layout` performs the plan it is handed. A window shows the plan and the person
   presses. This is the same firewall shape the deposit and the spatial maps use, and it
   is what makes "what is about to happen to my data" answerable BEFORE it happens.
2. Nothing is ever overwritten, and a collision fails the whole plan. Not the one
   file: the whole plan. A half-applied reorganisation leaves a folder in a state
   neither stage understands, with some movies in `raw/` and some beside it, and the
   scan then reports a movie count that is real and wrong.
3. `mode` has no default. Moving is not reversible by pressing anything, and copying
   doubles a folder that is routinely gigabytes. Neither is safe to assume on someone
   else's data, so the caller states it.
4. A file that is not recognised is named and left alone. Silently leaving something
   behind reads as "there was nothing else in the folder".
5. An ALREADY-ORGANISED folder is a NO-OP, not an error. Pointing at the same folder
   twice must be safe, because that is what a person does when they are not sure whether
   the first press worked.

What this does not create, deliberately: `tif/` and `cell_meshes/`. Preparation writes
the first and Oufti the second, and an EMPTY `cell_meshes/` is worse than an absent one,
`cellmesh.read_project_cell_meshes` raises a clear "no 'cell_meshes' folder" today, and
an empty folder would turn that into a subtler failure further down. The plan NAMES both
as still to come rather than making them.

`folder_layout` SAYS WHICH OF FOUR LAYOUTS a condition folder is in (asked 13 Oct 2026: "a
better detection if files are in the correct folder structure or not"): organised, loose
files, half-organised, nothing found. It is `plan_layout` plus two listings, never a second
discovery, and it searches no deeper than the folder's own `raw/` and `phase/` -- the
folders ABOVE a movie carry no meaning, so nothing here walks to an arbitrary depth.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .rawmovie import ACQUISITION_SUFFIXES, discover_raw_movies

__all__ = [
    "OrganiseError",
    "RAW_SUBFOLDER",
    "PHASE_SUBFOLDER",
    "MODES",
    "Move",
    "LayoutPlan",
    "plan_layout",
    "apply_layout",
    "acquisition_folder",
    "FolderState",
    "inspect_folder",
    "KIND_EMPTY",
    "KIND_FLAT_RAW",
    "KIND_ORGANISED_RAW",
    "KIND_PREPARED",
    "KIND_TRACKED",
    "LAYOUT_ORGANISED",
    "LAYOUT_LOOSE",
    "LAYOUT_HALF",
    "LAYOUT_NOTHING",
    "FolderLayout",
    "folder_layout",
]


class OrganiseError(RuntimeError):
    """Raised when a folder cannot be reorganised without guessing or losing data."""


#: The two folders this step fills, named as the lab's own projects name them.
RAW_SUBFOLDER = "raw"
PHASE_SUBFOLDER = "phase"

#: Written by other stages; named in the plan so their absence is not read as an error.
PRODUCED_LATER = (
    ("tif", "the cropped single-molecule stacks, written by preparation"),
    ("cell_meshes", "the Oufti outlines, written by the segmentation step"),
)

MODES = ("move", "copy")


@dataclass(frozen=True)
class Move:
    """One file and where it is going. `kind` is what the file was recognised AS."""

    source: Path
    destination: Path
    kind: str                                     # "acquisition" | "companion"

    @property
    def collides(self) -> bool:
        return self.destination.exists()


@dataclass(frozen=True)
class LayoutPlan:
    """What `apply_layout` would do, in full, before anything is touched."""

    folder: Path
    moves: tuple[Move, ...] = ()
    #: Files in the folder that no rule here recognises. Left alone, never guessed at.
    untouched: tuple[Path, ...] = ()
    #: True when the folder already has `raw/` holding acquisitions and there is nothing
    #: loose to file. Applying such a plan is a no-op rather than an error.
    already_organised: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def collisions(self) -> tuple[Move, ...]:
        return tuple(move for move in self.moves if move.collides)

    @property
    def acquisitions(self) -> tuple[Move, ...]:
        return tuple(m for m in self.moves if m.kind == "acquisition")

    @property
    def companions(self) -> tuple[Move, ...]:
        return tuple(m for m in self.moves if m.kind == "companion")

    def caption(self) -> str:
        """One paragraph a person can check before pressing anything."""
        # THE NOTES ARE THE REASON, so every branch has to carry them. The first draft
        # returned early here and dropped them, which meant the one case that most needs
        # explaining (you are pointing at `raw/` itself) said only "nothing to do".
        if self.already_organised and not self.moves:
            return "\n".join(
                (f"{self.folder.name} is already a project folder: "
                 f"{RAW_SUBFOLDER}/ holds the acquisitions. Nothing to do.",)
                + self.notes)
        if not self.moves:
            return "\n".join(
                (f"No acquisitions found loose in {self.folder.name}. "
                 f"Nothing to organise.",) + self.notes)
        lines = [
            f"{len(self.acquisitions)} acquisition(s) -> {RAW_SUBFOLDER}/",
        ]
        if self.companions:
            lines.append(
                f"{len(self.companions)} single-page companion(s) -> "
                f"{PHASE_SUBFOLDER}/")
        if self.untouched:
            shown = ", ".join(p.name for p in self.untouched[:5])
            more = " ..." if len(self.untouched) > 5 else ""
            lines.append(f"LEFT ALONE ({len(self.untouched)}): {shown}{more}")
        if self.collisions:
            shown = ", ".join(m.destination.name for m in self.collisions[:5])
            more = " ..." if len(self.collisions) > 5 else ""
            lines.append(
                f"REFUSED: {len(self.collisions)} destination(s) already exist: "
                f"{shown}{more}")
        lines.extend(self.notes)
        return "\n".join(lines)


def plan_layout(folder) -> LayoutPlan:
    """What would be filed where. Reads the folder; changes nothing.

    The acquisition/companion split is `rawmovie.discover_raw_movies`' rule and not a
    second copy of it: a `.tif` is a companion only when an acquisition of the same stem
    sits beside it. A folder of `.tif` stacks alone is therefore left entirely alone,
    which is right: those are already-extracted movies, not phase snapshots.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise OrganiseError(f"not a folder: {folder}")

    # POINTING AT THE `raw/` FOLDER ITSELF IS A LIKELY MISTAKE AND A DESTRUCTIVE ONE.
    # It is exactly where `acquisition_folder` sends a caller once a project is
    # organised, so it is the folder a person is most likely to be looking at when they
    # press Organise a second time - and filing it would make `raw/raw/` and leave every
    # scan pointing at an empty level. Treated as already organised, which it is.
    if folder.name.lower() == RAW_SUBFOLDER:
        return LayoutPlan(
            folder=folder, already_organised=True,
            notes=(f"this IS the {RAW_SUBFOLDER}/ folder: its acquisitions are "
                   f"already filed. Organising it would make "
                   f"{RAW_SUBFOLDER}/{RAW_SUBFOLDER}/.",))

    raw_dir = folder / RAW_SUBFOLDER
    organised = raw_dir.is_dir() and any(
        p.suffix.lower() in ACQUISITION_SUFFIXES for p in raw_dir.iterdir()
        if p.is_file())

    try:
        kept, companions = discover_raw_movies(folder)
    except Exception as exc:                           # noqa: BLE001 - re-raised named
        raise OrganiseError(f"cannot list {folder}: {type(exc).__name__}: {exc}")

    # Only ACQUISITIONS are filed. `kept` also contains bare `.tif` stacks when no
    # acquisition shares their stem, and those are movies in their own right - filing
    # them under `raw/` would rename a prepared stack into an acquisition.
    acquisitions = [p for p in kept if p.suffix.lower() in ACQUISITION_SUFFIXES]
    moves = [Move(p, raw_dir / p.name, "acquisition") for p in acquisitions]
    moves += [Move(p, folder / PHASE_SUBFOLDER / p.name, "companion")
              for p in companions]

    filed = {m.source for m in moves}
    untouched = tuple(p for p in sorted(folder.iterdir())
                      if p.is_file() and p not in filed)

    notes = []
    if moves:
        notes.append("still to come: "
                     + "; ".join(f"{name}/ ({why})" for name, why in PRODUCED_LATER))
    if organised and moves:
        notes.append(
            f"NOTE: {RAW_SUBFOLDER}/ already holds acquisitions AND there are loose "
            f"ones beside it. Both sets are real; check which is which before filing.")
    return LayoutPlan(folder=folder, moves=tuple(moves), untouched=untouched,
                      already_organised=organised, notes=tuple(notes))


def apply_layout(plan: LayoutPlan, *, mode: str) -> dict:
    """Carry out `plan`. ``mode`` is "move" or "copy" and has no default.

    Refuses BEFORE touching anything if any destination exists, so a folder is never
    left half-filed. Returns what was done, per file, so the caller can print it rather
    than assert it.
    """
    if mode not in MODES:
        raise OrganiseError(
            f"mode must be one of {MODES}, got {mode!r}. There is no default: moving cannot be undone "
            f"and copying doubles a folder that is often gigabytes, so choose one.")
    if plan.already_organised and not plan.moves:
        return {"mode": mode, "moved": (), "created": (), "folder": str(plan.folder),
                "note": "already a project folder; nothing was touched"}
    if not plan.moves:
        raise OrganiseError(
            f"nothing to organise in {plan.folder}: no acquisition sits loose in it. "
            f"If the movies are already under {RAW_SUBFOLDER}/ this folder is ready.")
    if plan.collisions:
        names = ", ".join(str(m.destination) for m in plan.collisions[:6])
        raise OrganiseError(
            f"{len(plan.collisions)} destination(s) already exist and nothing was "
            f"touched: {names}. Filing over them would replace data, and filing around "
            f"them would leave this folder half-organised, with some movies under "
            f"{RAW_SUBFOLDER}/ and some beside it.")

    created = []
    for directory in sorted({m.destination.parent for m in plan.moves}):
        if not directory.exists():
            directory.mkdir(parents=True)
            created.append(str(directory))

    # ALL OR NOTHING APPLIES TO FAILURES TOO, not only to collisions. Files are filed
    # one at a time, so anything that stops the loop half way leaves some movies under
    # `raw/` and some beside it - the exact state the collision check above exists to
    # prevent, arriving by a different route.
    #
    # AND IT IS NOT HYPOTHETICAL ON WINDOWS: a file another process has open cannot be
    # moved (WinError 32), and this window's own crop panel reads every acquisition in a
    # background thread. So the partial failure is reachable by pressing the button at
    # the wrong moment.
    action = shutil.move if mode == "move" else shutil.copy2
    done = []
    try:
        for item in plan.moves:
            action(str(item.source), str(item.destination))
            done.append((str(item.source), str(item.destination), item.kind))
    except Exception as exc:                           # noqa: BLE001 - re-raised named
        undone, failed_undo = _roll_back(done, mode=mode)
        detail = (f"{len(undone)} already-filed file(s) put back"
                  if not failed_undo else
                  f"The roll-back also failed for {len(failed_undo)} file(s): "
                  + "; ".join(failed_undo[:4])
                  + ". This folder is now half organised and must be checked by hand.")
        raise OrganiseError(
            f"filing stopped at {len(done) + 1} of {len(plan.moves)} "
            f"({type(exc).__name__}: {exc}). {detail}")
    return {
        "mode": mode,
        "moved": tuple(done),
        "created": tuple(created),
        "folder": str(plan.folder),
        "note": (f"{len(plan.acquisitions)} acquisition(s) and "
                 f"{len(plan.companions)} companion(s) {mode}d"),
    }


def acquisition_folder(folder) -> tuple[Path, str]:
    """``(folder holding the acquisitions, a sentence saying which and why)``.

    Needed because organising moves the goalposts. Every caller here (`prepare_batch`,
    the crop panel, the tracking scan) is pointed straight at the folder holding the
    acquisitions, and none of them appends `raw/`. That is fine until the moment someone
    presses Organise: the files leave the folder they are looking at and the next scan
    reports zero movies, which reads as "my data disappeared" rather than "they are one
    level down".

    So this resolves it ONCE, and returns the reason rather than silently descending. A
    folder that quietly redirects itself is indistinguishable from one that found
    nothing, and the difference matters: one means "ready", the other means "wrong
    folder".

    Preference is for the folder AS GIVEN. If it holds acquisitions they are the answer,
    even when a `raw/` also exists: the loose ones are the newer arrivals and choosing
    the subfolder behind the user's back would track the wrong set.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise OrganiseError(f"not a folder: {folder}")

    def acquisitions_in(where):
        if not where.is_dir():
            return []
        return [p for p in sorted(where.iterdir())
                if p.is_file() and p.suffix.lower() in ACQUISITION_SUFFIXES]

    here = acquisitions_in(folder)
    if here:
        return folder, f"{len(here)} acquisition(s) in {folder.name} itself"
    nested = folder / RAW_SUBFOLDER
    found = acquisitions_in(nested)
    if found:
        return nested, (f"{len(found)} acquisition(s) in {folder.name}/"
                        f"{RAW_SUBFOLDER}/: using that, since {folder.name} itself "
                        f"holds none")
    return folder, f"no acquisition in {folder.name} or {folder.name}/{RAW_SUBFOLDER}/"


def _roll_back(done, *, mode: str):
    """Undo what a failed `apply_layout` had already filed.

    ``(undone, failures)``. A MOVE is put back where it came from; a COPY is deleted,
    since the original never left. Rolling back is itself allowed to fail (a locked
    file is exactly what caused the original failure) so the failures are RETURNED and
    named rather than swallowed. A roll-back that silently half-worked would be worse
    than no roll-back at all, because the report would then be wrong as well as the
    folder.
    """
    undone, failures = [], []
    for source, destination, _kind in reversed(done):
        try:
            if mode == "move":
                shutil.move(destination, source)
            else:
                Path(destination).unlink()
        except Exception as exc:                       # noqa: BLE001 - returned, named
            failures.append(f"{destination}: {type(exc).__name__}: {exc}")
        else:
            undone.append(destination)
    return undone, failures


#: What a folder IS, in the order the work happens. `inspect_folder` returns one.
KIND_EMPTY = "empty"
KIND_FLAT_RAW = "flat_raw"
KIND_ORGANISED_RAW = "organised_raw"
KIND_PREPARED = "prepared"
KIND_TRACKED = "tracked"


@dataclass(frozen=True)
class FolderState:
    """What is in this folder, and therefore where the window should start.

    The window must look rather than assume. Pointed at a folder of raw acquisitions it
    used to open on "Prepared stacks", find nothing, and report zero movies, which
    reads as "your data is unreadable" when the truth is "you are on the wrong tab".
    Pointed at an already-prepared project it should not force the crop question again,
    because the crop was already answered and re-answering it silently produces a second
    dataset with the same name.

    So this reports what is present and RECOMMENDS a starting point; it never chooses on
    the user's behalf. Both routes stay available in the window: someone re-cropping a
    prepared project is a real case, which is exactly why the recommendation is a
    default and not a lock.
    """

    folder: Path
    kind: str
    #: Acquisitions loose in the folder, i.e. not yet filed into `raw/`.
    n_loose: int = 0
    #: Acquisitions under `raw/`.
    n_raw: int = 0
    #: Movie folders under `tif/` that hold frames.
    n_prepared: int = 0
    #: Movies carrying tracker output, u-track's `TrackingPackage` or a `trackmate/`.
    n_tracked: int = 0
    has_crops: bool = False
    #: Images under `phase/`, what the segmentation window and the mesh editor draw on.
    n_phase: int = 0
    #: Outline files under `cell_meshes/`, excluding u-track's own `movieList.mat` and
    #: `movieData.mat`, which live there too and are not outlines.
    n_meshes: int = 0

    @property
    def n_acquisitions(self) -> int:
        return self.n_loose + self.n_raw

    @property
    def recommends_raw_source(self) -> bool:
        """Whether the window should start on the RAW source rather than PREPARED.

        The acquisition wins wherever there is one. This used to prefer PREPARED
        whenever prepared stacks existed, so a folder holding both opened on the cropped
        copy: ruled against by the experimenter, and the reasoning holds: `tif/*.tif` is
        a stack written to feed u-track, which cannot read a `.nd2`. TrackMate reads the
        acquisition directly, and the two are VALIDATED element-wise identical over the
        same crop range (one lab movie: 1312 spots, 253 tracks, 922
        localisations each). Given that, tracking a derived copy in preference to the
        thing it was derived from is backwards.

        PREPARED remains one click away and remains the only route for a movie whose
        acquisition is not beside it: an archived project, or a set prepared on another
        machine.

        NO EXCEPTION FOR A MISSING CROP RANGE, ruled by the experimenter on a folder of
        MetaMorph `.stk` files with prepared tifs beside them: "tifs already produced ->
        should ignore the tifs and read the raw stk still". The exception this replaces
        opened such a folder on the tifs whenever no `crops.json` existed, so the window
        read no acquisition, drew no bleaching curve, and the folder looked unreadable.
        A missing range is what the preparation page is FOR: it reads the acquisitions
        and the ranges are set there. Tracking the raw path still refuses a movie with
        no range, by name, and `caption()` says to set them or switch to the stacks.
        """
        return bool(self.n_acquisitions)

    @property
    def needs_organising(self) -> bool:
        return self.kind == KIND_FLAT_RAW

    def caption(self) -> str:
        """One line saying what was found and what happens next."""
        if self.kind == KIND_EMPTY:
            return (f"Nothing trackable in {self.folder.name}: no raw acquisition, no "
                    f"prepared stack. Choose the folder holding one condition's movies.")
        if self.kind == KIND_FLAT_RAW:
            return (f"{self.n_loose} raw acquisition(s), not yet filed. "
                    f"Organise into raw/ + phase/, then set the crop range.")
        if self.kind == KIND_ORGANISED_RAW:
            # A RECORDED CROP SET MEANS THERE IS NOTHING TO DO HERE, and saying so is
            # the point: `crops.json` is loaded either way, so without this line a
            # returning user re-answers a question they already answered. It is the
            # single record of the range precisely so it is not asked twice.
            if self.has_crops:
                return (f"{self.n_raw} raw acquisition(s) in raw/, crop ranges already "
                        f"recorded: nothing to set, go straight to tracking.")
            return (f"{self.n_raw} raw acquisition(s) in raw/; no crop range yet: set "
                    f"one per movie.")
        if self.kind == KIND_PREPARED:
            return self._both_present_caption(
                f"{self.n_prepared} prepared stack(s)")
        return self._both_present_caption(
            f"{self.n_prepared} prepared stack(s), {self.n_tracked} of them already "
            f"carrying tracker output (tracking again overwrites nothing unless you "
            f"tick overwrite)")

    def _both_present_caption(self, prepared_phrase: str) -> str:
        """Say which source will be used, and (when it is not raw) why not.

        A fallback that does not explain itself reads as a preference. Landing on the
        prepared stacks looks like the tool preferring u-track's copy; it is the tool
        having no crop range for the acquisitions, which is a thing the user can fix.
        """
        if not self.n_acquisitions:
            # THE OTHER ROUTE IS STILL NAMED. Re-cropping a prepared project is a real
            # thing to want, and a line that only says what is happening reads as a lock.
            return (f"{prepared_phrase}, no acquisition beside them. Tracking those: "
                    f"switch the source to Raw to change the crop.")
        if self.has_crops:
            # BOTH ARE HERE AND THE ACQUISITION IS THE ONE TO TRACK. The prepared stack
            # is a copy made to feed u-track; TrackMate reads the acquisition itself,
            # and the two are validated element-wise identical over the same range.
            return (f"{self.n_acquisitions} raw acquisition(s) with recorded crop "
                    f"ranges, and {prepared_phrase} made from them. "
                    f"Tracking uses the acquisitions.")
        return (f"{self.n_acquisitions} raw acquisition(s), no crop range recorded "
                f"yet, and {prepared_phrase} beside them. Reading the acquisitions: set "
                f"one crop range per movie before tracking, to drop the dense early frames, "
                f"or switch the source to Prepared.")


def inspect_folder(folder) -> FolderState:
    """Classify a folder so the window can start in the right place.

    Reads directory listings only: no movie is opened, so this is cheap enough to run
    every time the folder changes.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise OrganiseError(f"not a folder: {folder}")

    loose = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in ACQUISITION_SUFFIXES]
    raw_dir = folder / RAW_SUBFOLDER
    in_raw = ([p for p in raw_dir.iterdir()
               if p.is_file() and p.suffix.lower() in ACQUISITION_SUFFIXES]
              if raw_dir.is_dir() else [])

    tif_dir = folder / "tif"
    prepared, tracked = [], []
    if tif_dir.is_dir():
        # ONE MULTI-PAGE TIFF PER MOVIE is what preparation writes, and what the macro
        # wrote before it. The nested folder below is u-track's own working copy of the
        # same movie, so a folder holding both must not count it twice.
        prepared += [p for p in sorted(tif_dir.glob("*.tif"))
                     if not (tif_dir / p.stem).is_dir()]
        for movie in sorted(p for p in tif_dir.iterdir() if p.is_dir()):
            # `tif/<movie>/<movie>/*.tif` is this lab's layout; `tif/<movie>/*.tif`
            # happens too, so both count as prepared.
            frames = list((movie / movie.name).glob("*.tif")) or list(movie.glob("*.tif"))
            if frames:
                prepared.append(movie)
            if (list(movie.glob("TrackingPackage/tracks/*.mat"))
                    or list(movie.glob("trackmate/*.xml"))):
                tracked.append(movie)
    tracked += [p for p in [folder / "trackmate"] if p.is_dir() and list(p.glob("*.xml"))]

    from .movieprep import CROPS_FILENAME

    # ALL THREE HOMES. The set moved from beside the movies, to the condition root, to
    # `bleaching/` where the curves that justify it live, and this check knew only the
    # first two, so a folder whose crop was recorded TODAY reported having none. A
    # location list that is not shared with the thing that writes it drifts the moment
    # the writer moves.
    has_crops = any((folder / where / CROPS_FILENAME).is_file()
                    for where in ("bleaching", ".", RAW_SUBFOLDER))

    if tracked:
        kind = KIND_TRACKED
    elif prepared:
        kind = KIND_PREPARED
    elif loose:
        # LOOSE WINS OVER `raw/`, the same rule `acquisition_folder` uses: the loose
        # files are the newer arrivals and are what the person is looking at.
        kind = KIND_FLAT_RAW
    elif in_raw:
        kind = KIND_ORGANISED_RAW
    else:
        kind = KIND_EMPTY

    phase_dir = folder / PHASE_SUBFOLDER
    n_phase = (sum(1 for q in phase_dir.iterdir()
                   if q.is_file() and q.suffix.lower() in (".tif", ".tiff")
                   and not q.name.startswith("._"))
               if phase_dir.is_dir() else 0)

    from .cellmesh import NOT_OUTLINE_STEMS, mesh_folder

    mesh_dir = mesh_folder(folder)
    n_meshes = (sum(1 for q in mesh_dir.iterdir()
                    if q.is_file() and q.suffix.lower() == ".mat"
                    and q.stem.lower() not in NOT_OUTLINE_STEMS
                    and not q.name.startswith("._"))
                if mesh_dir.is_dir() else 0)

    return FolderState(folder=folder, kind=kind, n_loose=len(loose), n_raw=len(in_raw),
                       n_prepared=len(prepared), n_tracked=len(tracked),
                       has_crops=has_crops, n_phase=n_phase, n_meshes=n_meshes)


def discover_project_folders(parent, *, include_self: bool = True) -> list[FolderState]:
    """Every project folder under ``parent``, so three windows can open a batch.

    Asked for: "for the image prep and cell meshes editor and segmentation step it would
    be cool if multiple folders would be detected and analysed at the same time. a parent
    folder name could differentiate between them". The FOLDER NAME is the label, which is
    already how this project distinguishes conditions everywhere else.

    ONE LEVEL DOWN, NOT A RECURSIVE WALK. A condition folder holds `tif/<movie>/` and
    other nested folders that are not projects, so recursing would offer a user
    `Malate/tif/series001` beside `Malate`. `project.discover_conditions` draws the line
    in the same place and for the same reason.

    A PARENT THAT IS ITSELF A PROJECT RETURNS ONLY ITSELF, so pointing at one folder
    behaves exactly as it did before this existed and no caller has to ask which case it
    is in. That mirrors `discover_conditions`, where a folder holding movies is one
    condition and a folder holding condition folders is a comparison.

    Reads directory listings only. Nothing is opened, so a parent of thirty conditions
    costs the same as one.
    """
    parent = Path(parent)
    if not parent.is_dir():
        raise OrganiseError(f"not a folder: {parent}")

    # ONE PREDICATE FOR BOTH. The first version tested the parent with
    # `kind != KIND_EMPTY` and the children with that OR phase OR meshes, so a condition
    # holding only `phase/` and `cell_meshes/` -- segmented but never tracked, which is
    # exactly the mesh editor's input -- failed the parent test, fell through to the
    # child scan, and refused because its own subfolders are not projects.
    here = inspect_folder(parent)
    if include_self and _is_project(here):
        return [here]

    from .cellmesh import is_mesh_folder_name

    found = []
    for child in sorted(p for p in parent.iterdir() if p.is_dir()):
        if (child.name.startswith(".") or child.name.lower() in _NOT_A_PROJECT
                or is_mesh_folder_name(child.name)):
            continue
        state = inspect_folder(child)
        if _is_project(state):
            found.append(state)
    if not found:
        raise OrganiseError(
            f"no project folders under {parent}: nothing here holds acquisitions, "
            f"prepared stacks, phase images or cell outlines. Choose the folder "
            f"holding your conditions, or one condition directly")
    return found


def _is_project(state: FolderState) -> bool:
    """Does this folder hold anything one of the three stages could open?

    Acquisitions or prepared stacks for preparation, `phase/` for segmentation, and
    `phase/` plus `cell_meshes/` for the editor. A folder failing all of those is not
    a condition, it is a folder.
    """
    return bool(state.kind != KIND_EMPTY or state.n_phase or state.n_meshes)


#: Folder names that are part of a project rather than a project of their own. Without
#: these a parent gets offered its own working folders as if they were conditions.
_NOT_A_PROJECT = frozenset({
    "raw", "phase", "tif", "sum", "cell_meshes", "cell_masks", "cell_meshes_preview",
    "trackmate", "bleaching", "backups", "__pycache__",
})


def plan_layouts(parent) -> dict:
    """``{condition name: LayoutPlan}`` for every condition under ``parent``.

    A microscope leaves a whole session as several sibling folders of loose files, so
    organising them one at a time is four dialogs for one decision. The plan is still
    per folder, because `apply_layout` rolls back per folder and that guarantee is what
    makes a mid-run failure survivable.

    READS AND TOUCHES NOTHING, exactly as `plan_layout` does. A folder already organised
    yields a plan with no moves rather than an error, so a mixed parent, some conditions
    filed and some not, is an ordinary case and not a refusal.
    """
    plans = {}
    for state in discover_project_folders(parent):
        try:
            plans[state.folder.name] = plan_layout(state.folder)
        except OrganiseError:
            continue
    if not plans:
        raise OrganiseError(
            f"nothing under {parent} can be organised: no folder holds loose "
            f"acquisitions or images")
    return plans


def needs_organising(parent) -> list[str]:
    """Conditions under ``parent`` whose files are still loose, by name.

    The question segmentation has to answer: it needs a `phase/` folder per condition,
    and an unorganised condition has its images sitting beside the acquisitions instead.
    Saying which ones, and that organising is the step, beats a channel error naming the
    condition folders as if they were candidate channels.
    """
    out = []
    for name, plan in plan_layouts(parent).items():
        if plan.moves:
            out.append(name)
    return out


def apply_layouts(plans, *, mode: str) -> dict:
    """Apply several `LayoutPlan`s, one condition at a time.

    ``plans`` is what `plan_layouts` returns. ``mode`` has NO DEFAULT for the same reason
    it has none on `apply_layout`: moving is not undone by pressing anything, and copying
    doubles a gigabyte folder.

    PER FOLDER, DELIBERATELY. `apply_layout` files one at a time and rolls that folder
    back if any single file fails, and running them in one flat pass would trade that
    guarantee for nothing. So a failure leaves the folder it happened in untouched, the
    folders already done stay done, and both are named. A half-organised SESSION is
    recoverable because each condition is all-or-nothing; a half-organised FOLDER would
    not be.

    Returns ``{name: {"ok": bool, "moved": int, "error": str | None}}``.
    """
    if mode not in ("move", "copy"):
        raise OrganiseError(
            f"mode must be 'move' or 'copy', got {mode!r}. There is no default: moving "
            f"cannot be undone, and copying doubles the folder")
    out = {}
    for name, plan in plans.items():
        if not plan.moves:
            out[name] = {"ok": True, "moved": 0, "error": None}
            continue
        try:
            apply_layout(plan, mode=mode)
        except Exception as exc:                          # noqa: BLE001 - named, not eaten
            out[name] = {"ok": False, "moved": 0, "error": f"{type(exc).__name__}: {exc}"}
        else:
            out[name] = {"ok": True, "moved": len(plan.moves), "error": None}
    return out


#: The four layouts `folder_layout` tells apart, worded as the rows show them.
LAYOUT_ORGANISED = "organised"
LAYOUT_LOOSE = "loose files"
LAYOUT_HALF = "half-organised"
LAYOUT_NOTHING = "nothing found"

#: What a phase snapshot is saved as; only a listing compares suffixes.
_IMAGE_SUFFIXES = (".tif", ".tiff")


@dataclass(frozen=True)
class FolderLayout:
    """Which layout one condition folder is in, and what filing would move where.

    ``kind`` is one of the four ``LAYOUT_*`` words and ``words`` the line a row shows.
    ``moves`` is ``(file name, destination folder)`` for every file `plan_layout` would
    file; ``stranded`` names loose images whose ACQUISITION is already in ``raw/`` --
    pairing is by stem BESIDE the acquisition, so filing leaves them where they are."""

    folder: Path
    kind: str
    words: str
    n_raw: int = 0
    n_phase: int = 0
    moves: tuple = ()
    stranded: tuple = ()


def _names(items, n: int = 3) -> str:
    items = list(items)
    return ", ".join(items[:n]) + (" ..." if len(items) > n else "")


def _image_stem(path: Path) -> str:
    """``series001`` for ``series001.tif`` and for ``series001.nd2.tif``."""
    inner = Path(path.stem)
    return inner.stem if inner.suffix.lower() in ACQUISITION_SUFFIXES else path.stem


def folder_layout(folder) -> FolderLayout:
    """Which of the four layouts ``folder`` (ONE condition) is in. Listings only.

    - **organised**: acquisitions in ``raw/`` and/or images in ``phase/``, nothing loose;
    - **loose files**: acquisitions (and their snapshots) directly in the folder, no
      ``raw/`` or ``phase/`` -- filing (`apply_layout`) would put them in both;
    - **half-organised**: ``raw/`` or ``phase/`` present AND loose acquisitions or snapshots
      beside them; the words name what filing would move where, and which loose images it
      would NOT move (their acquisition is already filed, so nothing pairs them);
    - **nothing found**: no acquisition here or in ``raw/``, no image in ``phase/``.

    `plan_layout` decides what is loose (so the acquisition/companion rule is
    `rawmovie.discover_raw_movies`', not a second copy); the counts are one listing each of
    ``raw/`` and ``phase/``. Nothing is written, and nothing deeper is searched. A folder
    that cannot be listed is NAMED as nothing found, with the error, never skipped.
    """
    folder = Path(folder)
    try:
        plan = plan_layout(folder)
        raw_dir, phase_dir = folder / RAW_SUBFOLDER, folder / PHASE_SUBFOLDER
        raw = sorted(p.stem for p in (raw_dir.iterdir() if raw_dir.is_dir() else ())
                     if p.is_file() and p.suffix.lower() in ACQUISITION_SUFFIXES)
        n_phase = sum(1 for p in (phase_dir.iterdir() if phase_dir.is_dir() else ())
                      if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
                      and not p.name.startswith("._"))
        filed, filed_stems = {m.source for m in plan.moves}, set(raw)
        stranded = tuple(sorted(
            p.name for p in folder.iterdir()
            if p.is_file() and p not in filed and p.suffix.lower() in _IMAGE_SUFFIXES
            and not p.name.startswith("._") and _image_stem(p) in filed_stems))
    except (OrganiseError, OSError) as exc:
        reason = str(exc) if isinstance(exc, OrganiseError) else f"{type(exc).__name__}: {exc}"
        return FolderLayout(folder=folder, kind=LAYOUT_NOTHING,
                            words=f"{LAYOUT_NOTHING}: {reason}")

    acquisitions = [m.source.name for m in plan.acquisitions]
    snapshots = [m.source.name for m in plan.companions]
    moves = tuple((m.source.name, m.destination.parent.name) for m in plan.moves)
    structured = bool(raw) or n_phase > 0
    have = []
    if raw:
        have.append(f"{len(raw)} acquisition(s) in {RAW_SUBFOLDER}/")
    if n_phase:
        have.append(f"{n_phase} image(s) in {PHASE_SUBFOLDER}/")
    loose = []
    if acquisitions:
        loose.append(f"{len(acquisitions)} acquisition(s) ({_names(acquisitions)}) -> "
                     f"{RAW_SUBFOLDER}/")
    if snapshots:
        loose.append(f"{len(snapshots)} snapshot(s) ({_names(snapshots)}) -> "
                     f"{PHASE_SUBFOLDER}/")
    if (plan.moves or stranded) and structured:
        kind = LAYOUT_HALF
        words = f"{LAYOUT_HALF}: " + " and ".join(have)
        if loose:
            words += "; loose beside them, filing would move " + " and ".join(loose)
        if stranded:
            words += (f"; {len(stranded)} image(s) whose acquisition is already in "
                      f"{RAW_SUBFOLDER}/ ({_names(stranded)}) stay where they are when "
                      f"filed -- move them into {PHASE_SUBFOLDER}/ by hand")
        if acquisitions and raw:
            # `acquisition_folder` PREFERS THE LOOSE ONES: a reader opening this folder
            # sees only them, and raw/'s set stays unread until they are filed.
            words += (f". Opened as it is, only the {len(acquisitions)} loose "
                      f"acquisition(s) are read, not the {len(raw)} in {RAW_SUBFOLDER}/")
    elif plan.moves:
        kind = LAYOUT_LOOSE
        what = f"{len(acquisitions)} acquisition(s)"
        if snapshots:
            what += f" and {len(snapshots)} snapshot(s)"
        words = (f"{LAYOUT_LOOSE}: {what} directly in the folder; filing would put them in "
                 + (f"{RAW_SUBFOLDER}/ and {PHASE_SUBFOLDER}/" if snapshots
                    else f"{RAW_SUBFOLDER}/"))
    elif structured:
        kind = LAYOUT_ORGANISED
        words = f"{LAYOUT_ORGANISED}: " + ", ".join(have)
        if not raw:
            words += f", no acquisition (none here or in {RAW_SUBFOLDER}/)"
        elif not n_phase:
            words += f", no {PHASE_SUBFOLDER}/ images"
    else:
        kind = LAYOUT_NOTHING
        words = (f"{LAYOUT_NOTHING}: no acquisition here or in {RAW_SUBFOLDER}/, no image "
                 f"in {PHASE_SUBFOLDER}/")
    return FolderLayout(folder=folder, kind=kind, words=words, n_raw=len(raw),
                        n_phase=n_phase, moves=moves, stranded=stranded)


def describe_layouts(plans) -> str:
    """What `apply_layouts` would do, in words, for a confirmation the user reads.

    Counted per KIND rather than as one total: "20 files" says nothing about whether the
    acquisitions or the phase images are the ones about to move.
    """
    lines = []
    total = 0
    for name, plan in plans.items():
        if not plan.moves:
            lines.append(f"  {name}: already organised")
            continue
        kinds: dict = {}
        for move in plan.moves:
            kinds[move.kind] = kinds.get(move.kind, 0) + 1
        detail = ", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items()))
        lines.append(f"  {name}: {detail}")
        total += len(plan.moves)
    lines.append(f"{total} file(s) in {len(plans)} folder(s); "
                 f"acquisitions go to raw/, images to phase/")
    return "\n".join(lines)
