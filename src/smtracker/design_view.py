"""The analysis door's "What do the folders mean?" page, its rules and words, with no Qt (phase 2 of the folder
designs, 24 Oct 2026; `folderdesign` is phase 1).

WHEN IT ASKS (`design_question`): never for a folder whose movies sit one level down or in the folder itself (the
conditions are its folders: nothing to name), never for one whose saved design fits, never for one the user said to
load without a design (`decline`, remembered in the personal library's file). It asks when the movies sit two or
more levels down, and when a saved design no longer fits (with its problems, so it can be fixed here rather than by
deleting a file). The rows start from the first library design that fits, else from `folderdesign.propose`.

WHAT IT SHOWS: one row per level -- a few of its folder names, how many there are, the role (Factor / Replicate /
Skip) and a factor's name -- and under them what the design makes (`preview`): the conditions and their factors, or
why it cannot work. One walk of the folders (`folderdesign.movie_holders`) serves every preview.

WHAT "USE" DOES (`accept`): writes `folderdesign.DESIGN_FILE` into the project folder and, when a name is given, adds
the design to the personal library. The door then looks at the folder again, now through the design.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import folderdesign as fd

__all__ = ["ROLE_WORDS", "ROLE_TIPS", "SET_ASIDE_FILE", "design_question", "preview", "design_from", "accept",
           "decline", "declined", "library_levels", "set_aside"]

#: Where a saved design that no longer fits goes when the folder is loaded without it: renamed, never deleted.
SET_ASIDE_FILE = "smtracker_design (not used).json"

ROLE_WORDS = {"factor": "Factor", "replicate": "Replicate", "skip": "Skip"}
ROLE_TIPS = {
    "factor": "These folders name something you compare, such as a protein, a medium, a strain or a treatment. "
              "Give it a name. A condition is one combination of factor values.",
    "replicate": "Each folder is one replicate (an independent culture) of the condition above it, whatever "
                 "its name (a date, a number). Use it only directly below the last factor.",
    "skip": "A folder that only wraps the next level, such as 'results'. Its name is not used.",
}
#: How many folder names a row shows before "...".
EXAMPLES = 4


def _levels_of(design) -> list:
    return [{"role": level.role, "name": level.name} for level in design.levels]


def library_levels(name: str) -> list:
    """A library design's levels as the page's rows hold them."""
    return _levels_of(fd.library()[name])


def _rows(holders, depth: int) -> list:
    rows = []
    for i in range(depth):
        names = sorted({p[i] for p in holders if len(p) > i})
        shown = ", ".join(names[:EXAMPLES]) + (", ..." if len(names) > EXAMPLES else "")
        rows.append({"level": i + 1, "examples": shown, "count": len(names)})
    return rows


def design_question(root) -> dict | None:
    """What the page asks about ``root`` (module docstring), or None when it need not ask. Plain values only:
    ``{"root", "holders", "rows": [{"level", "examples", "count"}], "levels": [{"role", "name"}], "library": [names],
    "from": "library name" | "proposed" | "saved", "why", "problems", "saved_file"}``. Walks the folders once."""
    root = Path(root)
    holders = fd.movie_holders(root)
    if not holders:
        return None
    depth = max(len(p) for p in holders)
    try:
        saved = fd.read(root)
    except fd.DesignError as exc:
        saved, broken = None, str(exc)
    else:
        broken = ""
    names = list(fd.library())
    base = {"root": str(root), "holders": [list(p) for p in holders], "rows": _rows(holders, depth),
            "library": names, "problems": [],
            # a design file is there (fitting or not, readable or not): "load without" sets it aside (`set_aside`)
            "saved_file": (root / fd.DESIGN_FILE).is_file()}
    if saved is not None:
        result = fd.apply(saved, root, holders=holders)
        if result.ok:
            return None
        return {**base, "levels": _pad(_levels_of(saved), depth), "from": "saved",
                "why": "The folder design saved with this project does not fit its folders. Fix it here, or load "
                       "the folder without a design.", "problems": list(result.problems)}
    if declined(root) and not broken:
        return None
    if depth < 2 and not broken:
        return None
    matches = fd.matching(root, holders=holders)
    if matches:
        name, design = matches[0]
        levels, source = _levels_of(design), name
    else:
        proposed = fd.propose(root, holders=holders)
        levels = _levels_of(proposed) if proposed is not None else [{"role": "factor", "name": f"factor {i + 1}"}
                                                                     for i in range(depth)]
        source = "proposed"
    depths = sorted({len(p) for p in holders})
    why = (f"The movies sit {depth} folders below {root.name or root}. Say what each level means; the conditions "
           f"and the factors you can group charts by follow from it.")
    if len(depths) > 1:
        why = (f"The movies sit at different depths below {root.name or root} ({', '.join(map(str, depths))} "
               f"folders). A design needs them all at one depth. Open a folder further down, or load without one.")
    if broken:
        why = broken + ". Make another design here, or load the folder without one."
    return {**base, "levels": _pad(levels, depth), "from": source, "why": why}


def _pad(levels: list, depth: int) -> list:
    levels = list(levels)[:depth]
    while len(levels) < depth:
        levels.append({"role": "factor", "name": f"factor {len(levels) + 1}"})
    return levels


def design_from(levels) -> fd.FolderDesign:
    """The page's rows as a design."""
    return fd.FolderDesign(tuple(fd.Level(str(row["role"]), str(row.get("name") or "").strip()
                                          if row["role"] == "factor" else "") for row in levels))


def preview(question: dict, levels) -> dict:
    """What ``levels`` make of the folders: ``{"ok", "text", "conditions": [{"key", "factors", "replicates"}],
    "problems"}``, the conditions in load order."""
    design = design_from(levels)
    result = fd.apply(design, question["root"], holders=[tuple(p) for p in question["holders"]])
    if not result.ok:
        problems = result.problems or ["the design finds no condition"]
        return {"ok": False, "text": "This design cannot load the folders: " + "; ".join(problems[:3])
                + (" ..." if len(problems) > 3 else ""), "conditions": [], "problems": problems}
    conditions = []
    names = design.factor_names
    for c in result.conditions:
        factors = " · ".join(f"{n} {v}" for n, v in zip(names, c.values))
        conditions.append({"key": c.key, "factors": factors, "replicates": len(c.replicates)})
    shown = ", ".join(c["key"] for c in conditions[:6]) + (f" and {len(conditions) - 6} more"
                                                          if len(conditions) > 6 else "")
    reps = (" Replicates: one per folder at the replicate level." if design.replicate_index is not None else
            " Replicates: found inside each condition, as without a design.")
    text = (f"{len(conditions)} condition{'s' if len(conditions) != 1 else ''}: {shown}. Grouped by "
            f"{' and '.join(names)}.{reps}")
    return {"ok": True, "text": text, "conditions": conditions, "problems": []}


def accept(question: dict, levels, *, remember_as: str = "", by: str = "") -> Path:
    """Save the design with the project (and in the library under ``remember_as``); the path written. Raises
    `folderdesign.DesignError` when it does not fit."""
    design = design_from(levels)
    path = fd.write(design, question["root"], by=by)
    if remember_as.strip():
        fd.remember(remember_as.strip(), design)
    _set_declined(question["root"], False)
    return path


# --------------------------------------------------------------------------- "load without a design", remembered
def _declined_file() -> Path:
    return fd.library_path().with_name("designs_declined.json")


def _read_declined() -> list:
    path = _declined_file()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []                       # a lost "do not ask" only asks again: never worth refusing a load over
    return [str(p) for p in data] if isinstance(data, list) else []


def _set_declined(root, on: bool) -> None:
    key = str(Path(root).resolve())
    current = [p for p in _read_declined() if p != key]
    if on:
        current.append(key)
    path = _declined_file()
    if not on and not path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=1), encoding="utf-8")


def set_aside(root) -> Path | None:
    """Rename the project's design file to `SET_ASIDE_FILE` (replacing an older one set aside) so nothing reads it,
    and remember the folder as loading without a design. The path it went to, or None when there was none."""
    source = Path(root) / fd.DESIGN_FILE
    if not source.is_file():
        return None
    target = Path(root) / SET_ASIDE_FILE
    if target.exists():
        target.unlink()
    source.rename(target)
    fd.forget_declared()
    decline(root)
    return target


def decline(root) -> None:
    """Remember that ``root`` loads without a design: the page will not ask again for it."""
    _set_declined(root, True)


def declined(root) -> bool:
    return str(Path(root).resolve()) in _read_declined()
