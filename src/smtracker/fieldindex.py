"""ONE field index per condition: which files belong to the same field, decided once.

WHY THIS EXISTS. A field's files are spread across folders and written by different
programs -- the microscope's acquisition, a phase snapshot, our projection, Oufti's
outlines, u-track's folder, TrackMate's XML -- and NINE functions each derived "which
field is this file" their own way (`rawmovie.movie_name`, `rawmovie.movie_names`,
`rawmovie.discover_raw_movies`, `channels.snapshot_field`, `channels.resolve_channels`,
`editor_gui._load_channels`, `editor_project.scan_fields`, `cellmesh.outline_coverage`,
`dayfolders.movie_name`, `trackmate_run.discover_movies`). Every naming defect of 17-18
Oct 2026 was that shape, three times in two days:

    the classical segmenter did not know snapshots can lie loose beside the movies;
    the editor's overlay did not know the channel folders can sit beside the image;
    the overlay still did not know `series001.nd2.tif` is field `series001` --
        and `channels.snapshot_field` had handled exactly that since 14 Oct.

The knowledge existed and was not shared. Teaching a new convention (`phase001`,
`series1`) to nine private rules is nine chances to miss one, so the rules live here and
the consumers ask.

ASKED FOR (18 Oct 2026): "what if other people use different naming conventions ... the
numbering should be the important one?" Yes -- but as the LAST rule, never the first, and
never silently. The ordered rules are:

    1. the exact stem                       series001.tif        -> series001
    2. minus an acquisition suffix          series001.nd2.tif    -> series001
    3. the channel token swapped            phase001.tif         -> sum001 in sum/
    4. the NUMERIC TAIL, if unambiguous     phase001.tif         -> 001

**RULE 4 IS GUARDED THREE WAYS, because a wrong numeric match is the only failure here
that CORRUPTS rather than hides** -- it would draw one cell's outline on another cell's
image with nothing wrong on screen:

    the LAST run of digits, never "the number": `20260904_series001` is field 001, not
        field 20260904, and a date-prefixed name is the common case;
    UNIQUENESS: if two candidates reduce to one key (`series1`, `series01`, `series001`)
        the pairing is REFUSED and both are named -- two matches is a question, not a
        coin toss, which is this project's rule for channel folders already;
    LAST RESORT: reachable only when 1-3 found nothing, so a folder that is named
        consistently never reaches it.

AND THE PAIRING IS REPORTED, not assumed. `FieldIndex.report()` says what paired, what
did not, and what was refused, so a window can show it. A pairing nobody can see is a
pairing nobody can check, and the cell filter already discards whole movies in silence.

WHAT THIS DOES NOT DO. It does not rename anything, and it does not change what we
WRITE: our own outputs are already named by the field rather than by a source file's
quirk (`series001.nd2.tif` produced `sum/series001.tif`, `cell_meshes/series001.mat`,
`tif/series001/`), which is why the meshes and the analysis were unaffected by that
stray suffix. The work is entirely on the READING side.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field as _field
from pathlib import Path

__all__ = [
    "FieldIndex",
    "Pairing",
    "IMAGE_SUFFIXES",
    "field_key",
    "numeric_tail",
    "candidate_keys",
    "match_in",
    "build_index",
]

#: What an image of a field may be called. Our own writers use `.tif`.
IMAGE_SUFFIXES = (".tif", ".tiff")

#: The last run of digits in a stem, which is the field's number where there is one.
#: ANCHORED AT THE END so a date or a position prefix cannot win: `20260904_series001`
#: gives 001, and `pos2_series010` gives 010.
_TAIL = re.compile(r"(\d+)(?!.*\d)")


def numeric_tail(stem: str) -> str | None:
    """The LAST run of digits in ``stem``, or None. Never "the number" in it.

    `20260904_series001` -> `001`; `pos2_series010` -> `010`; `phase1` -> `1`;
    `overview` -> None. A stem with no digits has no number and must fall through to a
    refusal rather than to an arbitrary pairing.
    """
    found = _TAIL.search(str(stem))
    return found.group(1) if found else None


def field_key(path) -> str:
    """The field a file belongs to, by rules 1-2: the stem without an acquisition suffix.

    This is `segmentation.channels.snapshot_field`'s rule, lifted here so every consumer
    shares it instead of keeping a copy. Rules 3 and 4 are about matching ACROSS folders
    and need the candidates, so they live in `match_in`.
    """
    from .rawmovie import ACQUISITION_SUFFIXES

    stem = Path(path).stem
    for suffix in ACQUISITION_SUFFIXES:
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def candidate_keys(key: str, *, want: str | None = None, have: str | None = None) -> list:
    """The names ``key`` might wear in another role's folder, best first.

    ``have``/``want`` are the role TOKENS (`phase`, `sum`): a microscopist who writes
    `phase001.tif` writes `sum001.tif` beside it, and one who writes `series001_phase`
    writes `series001_sum`. Both forms are tried, and the bare key first, since a folder
    named after the channel usually repeats the field name unchanged.
    """
    out = [key]
    if want and have:
        for form in (f"_{have}", have):
            if form and form in key:
                swapped = key.replace(form, f"_{want}" if form.startswith("_") else want)
                if swapped not in out:
                    out.append(swapped)
    return out


@dataclass(frozen=True)
class Pairing:
    """How one file was paired to a field, and by which rule -- so it can be shown.

    ``rule`` is one of ``exact``, ``suffix``, ``token``, ``number``. A window prints it
    beside an odd pairing, because "field series001 reads series001.nd2.tif" is checkable
    and a silent normalisation is not.
    """

    field: str
    path: Path
    rule: str


def match_in(key: str, folder, *, want: str | None = None, have: str | None = None,
             suffixes=IMAGE_SUFFIXES) -> tuple:
    """Find ``key``'s file in ``folder``. Returns ``(path, rule)``, or ``(None, why)``.

    The ordered rules of the module docstring. ``why`` is ``"missing"`` where nothing
    plausible was there, or ``"ambiguous: a, b"`` where the numeric rule found more than
    one -- which is REFUSED rather than resolved.
    """
    folder = Path(folder)
    try:
        present = [p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in suffixes]
    except OSError:
        return None, "missing"
    if not present:
        return None, "missing"

    by_key: dict = {}
    for p in present:
        by_key.setdefault(field_key(p), []).append(p)

    # 1-2: the exact stem, then the stem minus an acquisition suffix (both via field_key).
    #  3 : the channel token swapped.
    for candidate in candidate_keys(key, want=want, have=have):
        hits = by_key.get(candidate)
        if hits and len(hits) == 1:
            rule = "exact" if candidate == key else "token"
            if rule == "exact" and hits[0].stem != key:
                rule = "suffix"          # series001.nd2.tif reached key series001
            return hits[0], rule
        if hits:
            names = ", ".join(sorted(h.name for h in hits))
            return None, f"ambiguous: {names}"

    # 4: the numeric tail, LAST and only when exactly one candidate carries it.
    number = numeric_tail(key)
    if number is None:
        return None, "missing"
    wanted = int(number)
    numbered: dict = {}
    for candidate, hits in by_key.items():
        tail = numeric_tail(candidate)
        if tail is not None and int(tail) == wanted:
            numbered.setdefault(candidate, []).extend(hits)
    flat = [p for hits in numbered.values() for p in hits]
    if len(flat) == 1:
        return flat[0], "number"
    if len(flat) > 1:
        # `series1`, `series01` and `series001` all reduce to 1. Naming both is the whole
        # point: a coin toss here pairs the wrong images and nothing on screen is wrong.
        names = ", ".join(sorted(p.name for p in flat))
        return None, f"ambiguous: {names}"
    return None, "missing"


@dataclass
class FieldIndex:
    """``{field: {role: path}}`` for one condition, with what did not pair.

    ``rules`` records how each file was reached, so an odd pairing is shown rather than
    normalised in silence. ``refused`` holds the ambiguities, by field and role.
    """

    root: Path
    fields: dict = _field(default_factory=dict)
    rules: dict = _field(default_factory=dict)
    refused: dict = _field(default_factory=dict)
    #: Where each role's files were found: a folder, or the condition itself (loose).
    sources: dict = _field(default_factory=dict)

    def path(self, name: str, role: str):
        """One field's file for one role, or None."""
        return self.fields.get(name, {}).get(role)

    def names(self) -> list:
        return sorted(self.fields)

    def rule_for(self, name: str, role: str) -> str | None:
        return self.rules.get((name, role))

    def odd(self) -> list:
        """Pairings a window should SHOW: anything not reached by the exact stem."""
        return sorted((n, r, self.rules[(n, r)]) for (n, r) in self.rules
                      if self.rules[(n, r)] != "exact")

    def report(self) -> str:
        """What paired, what did not, what was refused -- the line a window prints."""
        roles = sorted({r for by_role in self.fields.values() for r in by_role})
        counts = {r: sum(1 for by_role in self.fields.values() if by_role.get(r))
                  for r in roles}
        parts = [f"{len(self.fields)} field(s)"]
        parts += [f"{counts[r]} {r}" for r in roles]
        line = ", ".join(parts)
        if self.refused:
            line += f"; {len(self.refused)} refused as ambiguous"
        odd = self.odd()
        if odd:
            line += f"; {len(odd)} paired by name rather than exactly"
        return line


def build_index(root, *, roles=("phase", "sum"), bright: str = "phase") -> FieldIndex:
    """Index one CONDITION folder: its fields and each role's file.

    The BRIGHT role decides the fields, because that is the image a field is edited and
    segmented on; the others are matched to it. Its images come from a channel folder or
    from the condition itself (`channels.resolve_channels` already knows both layouts and
    is asked rather than reimplemented here).
    """
    from .segmentation.channels import resolve_channels

    root = Path(root)
    index = FieldIndex(root=root)
    resolution = resolve_channels(root, (bright,))
    bright_folder = resolution.resolved.get(bright)
    if bright_folder is None:
        # NO BRIGHT IMAGES IS AN ORDINARY STATE, not an error: a condition can be opened
        # before its phase snapshots are filed. `stems()` raises for an unresolved role,
        # so it is asked only once the role is there, and the empty index reports itself.
        index.sources[bright] = None
        index.refused[(None, bright)] = "unresolved: " + (
            ", ".join(p.name for p in resolution.unresolved.get(bright, ())) or "nothing found")
        return index
    names = resolution.stems(bright)
    index.sources[bright] = bright_folder
    for name in sorted(names):
        image = resolution.image_path(bright, name)
        if image is None:
            continue
        index.fields[name] = {bright: Path(image)}
        index.rules[(name, bright)] = ("exact" if Path(image).stem == name else "suffix")

    for role in roles:
        if role == bright:
            continue
        folder = _role_folder(root, role, bright_folder)
        index.sources[role] = folder
        if folder is None:
            continue
        for name in index.names():
            found, rule = match_in(name, folder, want=role, have=bright)
            if found is not None:
                index.fields[name][role] = found
                index.rules[(name, role)] = rule
            elif rule.startswith("ambiguous"):
                index.refused[(name, role)] = rule
    return index


def _role_folder(root: Path, role: str, bright_folder):
    """The folder holding ``role``'s images: beside the bright images, or in the condition.

    BOTH LAYOUTS, nearest first -- the same rule the editor's partner lookup needed. A
    folder beside the image belongs to this condition; one a level up might belong to a
    sibling, which is how a condition called `sum` could otherwise be read as another
    condition's channel.
    """
    from .segmentation.channels import ROLE_ALIASES

    wanted = {a.lower() for a in ROLE_ALIASES.get(role, (role,))}
    bases = []
    if bright_folder is not None:
        bases.append(Path(bright_folder))
        bases.append(Path(bright_folder).parent)
    bases.append(root)
    seen = set()
    for base in bases:
        if base in seen:
            continue
        seen.add(base)
        try:
            for child in base.iterdir():
                if child.is_dir() and child.name.lower() in wanted:
                    return child
        except OSError:
            continue
    return None
