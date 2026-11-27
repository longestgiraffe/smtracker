"""Which folder holds which CHANNEL: resolved by name, corrected by pointing.

The model declares its roles; the project supplies folders; this maps one to THE
OTHER. A model trained on `[phase, sum]` is not interchangeable with one trained on
`sum` alone, and handing either the wrong input produces MASKS rather than an error,
which is why the model carries what it needs rather than the window guessing.

The rule, in the experimenter's words: a model trained with `phase` + `sum` opening a
project that has `phase/` and `sum/` knows what to do. Where the naming differs (the
fluorophore channel called something other than `sum`) `phase` still resolves itself and
the user points at the folder for the role that did not.

Automatic where it is certain, manual where it is not, and never a guess between. A
role with exactly one candidate resolves. A role with several is left UNRESOLVED and
named, because picking the first alphabetically would be a guess wearing the appearance
of a default, and this project already records what a silently-wrong channel costs: a
DNA-bound protein segmented on its own fluorescence returns outlines at 48 % of the true
cell area, which looks plausible and is wrong.

`sum` and `phase` are this lab's convention and not a law, so the aliases below are a
starting point rather than a closed set, and any role can be overridden with a path.

PHASE SNAPSHOTS BESIDE THE MOVIES ARE READ WHERE THEY LIE (13 Oct 2026). The new
preparation window's Save files nothing (ruled: it stays that way), so a folder straight off
the microscope keeps ``series001.nd2`` and its snapshot ``series001.tif`` side by side, and a
rule that knew only a ``phase/`` FOLDER could never segment it. Where no folder is named like
phase, the role resolves to the condition folder ITSELF, limited to the snapshots filing
would have put in ``phase/`` (`organise.plan_layout`'s companions: one rule, not a second
copy), so a ``.tif`` movie or a stray file is never taken for a field. A ``phase/`` folder,
the old layout, still wins wherever it exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ChannelRole",
    "ROLE_ALIASES",
    "ChannelResolution",
    "folders_in",
    "resolve_channels",
    "loose_snapshots",
    "movie_names_beside",
    "BESIDE_THE_MOVIES",
]

#: What a role reading the snapshots beside the movies shows where a folder name would.
BESIDE_THE_MOVIES = "snapshots beside the movies"

#: The two roles this project's models are trained on. A role is what the MODEL asks
#: for; a folder name is what a MICROSCOPIST happened to type.
ChannelRole = str

#: Folder names that are taken to mean a role WITHOUT asking. Lower-cased comparison.
#: Deliberately short: a name not on this list is not a failure, it is a question.
ROLE_ALIASES: dict[ChannelRole, tuple[str, ...]] = {
    "phase": ("phase", "brightfield", "bf", "trans", "dic"),
    "sum": ("sum", "fluorescence", "fluo", "epi", "signal", "proj"),
}

#: Folders that are never a channel, however they are spelled. These sit beside the
#: image folders in every project this reads.
NOT_CHANNELS = frozenset({
    "cell_meshes", "cell_meshes_preview", "tif", "raw", "movies", "movielist",
    "movielist", "bleaching", "results", "training_data", "trackmate",
    ".smtracker_warm", "__pycache__",
})


@dataclass
class ChannelResolution:
    """Which folder each role resolved to, and what could not be decided.

    `unresolved` is a mapping from role to the CANDIDATES that were seen, so the window
    can offer them rather than making the user browse from scratch. An empty candidate
    list means nothing plausible was found at all, which is a different message.
    """

    root: Path
    resolved: dict[ChannelRole, Path] = field(default_factory=dict)
    unresolved: dict[ChannelRole, list[Path]] = field(default_factory=dict)
    ignored: list[Path] = field(default_factory=list)
    #: Roles whose candidates MATCHED an alias, so more than one was plausible. A role
    #: absent from this set was not recognised at all, which is a different message:
    #: "ambiguous" said of a single candidate reads as though the tool had two answers.
    ambiguous: set = field(default_factory=set)
    #: ``{role: stems}`` for a role reading the condition folder ITSELF (its loose phase
    #: snapshots): only these are its fields. Absent for a role reading a channel folder.
    stems_of: dict = field(default_factory=dict)
    #: ``{role: {field: file}}`` for those loose snapshots: a field's name is its MOVIE's
    #: (`snapshot_field`), which is not always the file's stem.
    files_of: dict = field(default_factory=dict)
    #: ``{role: [folders]}`` named like the role but holding none of the movies' images, passed over for the one
    #: that does (`resolve_channels`); named by `describe` so the choice can be checked.
    passed_over: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.unresolved

    def label(self, role) -> str | None:
        """What a role reads, as a window names it: the folder, or the loose snapshots."""
        if role in self.stems_of:
            return BESIDE_THE_MOVIES
        path = self.resolved.get(role)
        return path.name if path is not None else None

    def image_path(self, role, name: str) -> Path | None:
        """The image file of field ``name`` in ``role``, or None: a loose snapshot by its
        record (``series001.nd2.tif`` is field ``series001``), else ``<folder>/<name>.tif``,
        else, in a folder too, the name an export kept its acquisition suffix in
        (``<folder>/series001.nd2.tif``; 19 Oct 2026: a filed snapshot keeps its name)."""
        from ..rawmovie import ACQUISITION_SUFFIXES, same_movie

        if role in self.files_of:
            files = self.files_of[role]
            hit = same_movie(name, files)              # by series number when the name differs
            return files.get(hit) if hit is not None else None
        folder = self.resolved.get(role)
        if folder is None:
            return None
        for infix in ("", *sorted(ACQUISITION_SUFFIXES)):
            for suffix in (".tif", ".tiff"):
                candidate = Path(folder) / f"{name}{infix}{suffix}"
                if candidate.exists():
                    return candidate
        # THE SAME MOVIE UNDER ANOTHER NAME (23 Oct 2026): ``phase/001.tif`` for field ``series001_YFPLaser``, or
        # ``sum/series001_YFPLaser.tif`` for field ``001`` -- one image with that series number, never a guess
        hit = same_movie(name, image_stems(folder))
        if hit is not None and hit != name:
            return self.image_path(role, hit)
        return None

    def stems(self, role) -> list[str]:
        """The fields a resolved role holds."""
        if role in self.stems_of:
            return sorted(self.stems_of[role])
        return image_stems(self.resolved[role])

    def describe(self) -> str:
        """One line per role, for the console. Says WHY, not just what."""
        lines = []
        for role in sorted(self.resolved):
            others = ", ".join(p.name for p in self.passed_over.get(role, ()))
            lines.append(f"  {role:6s} -> {self.label(role)}"
                         + (f" (the one holding the movies' images; {others} holds none)" if others else ""))
        for role, candidates in sorted(self.unresolved.items()):
            names = ", ".join(p.name for p in candidates)
            # WHERE TO ANSWER IT, not just that it is open. The window used to carry a
            # "Point at a folder…" button; the folder is a column in the condition table
            # now, so the guidance names that instead of a control that is gone.
            if not candidates:
                lines.append(f"  {role:6s} -> not found; set the folder in the table")
            elif role in self.ambiguous:
                lines.append(f"  {role:6s} -> ambiguous ({names}); "
                             f"set the folder in the table")
            else:
                lines.append(f"  {role:6s} -> no folder is named like {role!r}; "
                             f"set the folder in the table, one of: {names}")
        return "\n".join(lines) if lines else "  no channels found"


def folders_in(root) -> list[Path]:
    """Sub-folders that could plausibly hold images, in name order."""
    from ..cellmesh import is_mesh_folder_name

    root = Path(root)
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir()
                   if p.is_dir() and p.name.lower() not in NOT_CHANNELS
                   and p.name.lower().replace(" ", "") not in NOT_CHANNELS     # `movie list`
                   and not is_mesh_folder_name(p.name) and not p.name.startswith(".")),
                  key=lambda p: p.name.lower())


def resolve_channels(root, roles, *, overrides=None,
                     aliases=None) -> ChannelResolution:
    """Map each required role to a folder under `root`.

    `roles` is what the model declares it needs, e.g. `("phase", "sum")`.
    `overrides` is `{role: path}` from the user pointing at a folder; an override always
    wins and is never second-guessed, including when it names a folder whose name says
    something else: the person looking at the images knows what is in them.

    A role is resolved automatically ONLY when exactly one folder matches its aliases, or
    exactly one of several holds the movies' images. Two matches is a question, not a coin
    toss.
    """
    from ..rawmovie import same_movie

    root = Path(root)
    aliases = aliases or ROLE_ALIASES
    overrides = {k: Path(v) for k, v in (overrides or {}).items()}
    present = folders_in(root)

    out = ChannelResolution(root=root)
    taken: set[Path] = set()
    movies = None

    for role in roles:
        if role in overrides:
            out.resolved[role] = overrides[role]
            taken.add(overrides[role])
            continue
        wanted = tuple(a.lower() for a in aliases.get(role, (role,)))
        matches = [p for p in present
                   if p.name.lower() in wanted and p not in taken]
        if len(matches) > 1:
            # THE DATA DECIDES, NOT THE NAME (23 Oct 2026): one user's conditions hold ``sum/`` (our projections,
            # ``series001_YFPLaser.tif``) beside ``epi/`` (his epifluorescence snapshots, ``YFP-1.tif``, or only dated
            # sub-folders), both named like "sum". Of several, the ONE whose images are the movies' (by name or series
            # number, `rawmovie.same_movie`) is the channel; the others are named, never silently dropped. Where
            # none or several hold them, it stays the question it was.
            if movies is None:
                tracked, _, acquired = movie_names_beside(root)
                movies = tracked + acquired
            holding = [p for p in matches
                       if movies and any(same_movie(s, movies) is not None for s in image_stems(p))]
            if len(holding) == 1:
                out.passed_over[role] = [p for p in matches if p != holding[0]]
                matches = holding
        if len(matches) == 1:
            out.resolved[role] = matches[0]
            taken.add(matches[0])
        elif not matches and role == "phase" and (snaps := loose_snapshots(root)):
            out.resolved[role] = root
            out.stems_of[role] = {snapshot_field(p) for p in snaps}
            out.files_of[role] = {snapshot_field(p): p for p in snaps}
        else:
            # Ambiguous OR unrecognised. Offer every folder not already claimed, because
            # where nothing matched, the one the user wants is by definition not named
            # like the role.
            if len(matches) > 1:
                out.ambiguous.add(role)
            out.unresolved[role] = matches or [p for p in present if p not in taken]

    out.ignored = [p for p in present if p not in taken]
    return out


def movie_names_beside(folder) -> list:
    """The names a condition's movies go by beyond its snapshots, most authoritative first: as TRACKED (``tif/``'s
    stacks and u-track folders, TrackMate's ``trackmate/<movie>.xml``), as ``sum/`` images, as acquisitions
    (``raw/``, else loose beside). Three lists; empty where a folder is absent or cannot be read."""
    from ..organise import ACQUISITION_SUFFIXES

    folder = Path(folder)
    suffixes = tuple(s.lower() for s in ACQUISITION_SUFFIXES)
    images = (".tif", ".tiff")

    def listing(path, keep):
        try:
            return sorted({name for p in Path(path).iterdir() if (name := keep(p))})
        except OSError:
            return []
    tracked = (listing(folder / "tif", lambda p: p.name if p.is_dir()
                       else (p.stem if p.suffix.lower() in images else None))
               + listing(folder / "trackmate", lambda p: p.stem if p.suffix.lower() == ".xml" else None))
    summed = listing(folder / "sum", lambda p: p.stem if p.suffix.lower() in images else None)
    acquired = (listing(folder / "raw", lambda p: p.stem if p.suffix.lower() in suffixes else None)
                or listing(folder, lambda p: p.stem if p.is_file() and p.suffix.lower() in suffixes else None))
    return [tracked, summed, acquired]


def snapshot_field(path) -> str:
    """The FIELD a snapshot belongs to: its stem without an acquisition suffix an export kept.

    ``series001.nd2.tif`` is ``series001``'s snapshot (`rawmovie.discover_raw_movies` already
    reads it so), and its outlines must be ``cell_meshes/series001.mat``: the analysis finds a
    movie's outlines by the MOVIE's name. Found 14 Oct 2026 on one user's folders, where three
    outline files were written as ``series001.nd2.mat`` and would have been missed."""
    from ..rawmovie import ACQUISITION_SUFFIXES

    stem = Path(path).stem
    for suffix in ACQUISITION_SUFFIXES:
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def loose_snapshots(folder) -> list[Path]:
    """The phase snapshots lying directly in ``folder`` beside its acquisitions: what filing
    would move to ``phase/`` (`organise.plan_layout`), TIFFs only. Empty where there are none
    or the folder cannot be read (the role then stays unresolved and is named)."""
    from ..organise import OrganiseError, plan_layout

    try:
        plan = plan_layout(folder)
    except OrganiseError:
        return []
    return sorted(m.source for m in plan.companions
                  if m.source.suffix.lower() in (".tif", ".tiff"))


def image_stems(folder, suffixes=(".tif", ".tiff")) -> list[str]:
    """The field names in a channel folder, without extension, in name order."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted({p.stem for p in folder.iterdir()
                   if p.suffix.lower() in suffixes and not p.name.startswith(".")})


def common_fields(resolution: ChannelResolution) -> list[str]:
    """Field names present in EVERY resolved channel.

    The intersection, not the union: a model wanting two channels cannot run on a field
    that has one. The fields that are missing somewhere are the caller's to report, a
    channel folder short of a few images is exactly the kind of thing that otherwise
    shows up as an unexplained drop in cell count.
    """
    from ..rawmovie import same_movie

    if not resolution.resolved:
        return []
    roles = list(resolution.resolved)
    stems = {role: set(resolution.stems(role)) for role in roles}
    # ONE MOVIE UNDER EACH FOLDER'S NAME FOR IT (23 Oct 2026): one user's ``phase/001.tif`` and
    # ``sum/series001_YFPLaser.tif`` intersected to nothing, so a phase + sum model found "no sum" for every field.
    # Matched by series number, and the field takes the name the movie's tracks go by (not a bare snapshot number),
    # because that is the name its outlines are filed under and the analysis reads.
    # A phase-only model meets the bare number alone, so the movie's name is looked up beside (`movie_names_beside`).
    out = set()
    beside = None
    for name in stems[roles[0]]:
        matched = [same_movie(name, stems[role]) for role in roles]
        if all(m is not None for m in matched):
            named = next((m for m in matched if not m.isdigit()), name)
            if named.isdigit():
                beside = movie_names_beside(resolution.root) if beside is None else beside
                named = next((hit for names in beside if (hit := same_movie(named, names))), named)
            out.add(named)
    return sorted(out)


def missing_fields(resolution: ChannelResolution) -> dict[str, list[str]]:
    """Per channel, the fields the OTHER channels have and it does not."""
    from ..rawmovie import same_movie

    if len(resolution.resolved) < 2:
        return {}
    stems = {role: set(resolution.stems(role)) for role in resolution.resolved}
    out = {}
    for role, found in stems.items():
        # a field the other channels hold is missing here only if no image here is that movie (by series number)
        missing = sorted({n for other, names in stems.items() if other != role for n in names
                          if same_movie(n, found) is None})
        if missing:
            out[role] = missing
    return out
