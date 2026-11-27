"""The outlines window's rules for AUTOMATIC SEGMENTATION, with no Qt in them.

The workspace-style outlines window (`ui.outlines`) shows what this module decides, in
the way `prep_model` serves the preparation window. The rules are the classic
segmentation window's (`segment_gui`), carried over and named:

- THE FOLDER FOLLOWS THE MODEL unless it was chosen by hand: `bact_phase_omni` reads
  phase contrast and `bact_fluor_omni` the summed fluorescence -- different IMAGES, not
  different weights over one -- so a model change re-derives the folder, and a folder
  picked by hand is named as re-derived rather than silently dropped.
- AN UNRESOLVED FOLDER STAYS UNSET. `channels.resolve_channels` refuses to guess between
  two candidates ("two matches is a question, not a coin toss"), and a default here would
  answer it for the user and feed the run an override nothing checks.
- AN OVERRIDE IS NEVER SECOND-GUESSED: the person looking at the images knows what is in
  them.
- WHAT THE GRID SHOWS IS WHAT THE RUN GETS.

AND ONE THAT IS NEW: A RUN DOES NOT OVERWRITE OUTLINES UNASKED. `segment_folder` writes
`cell_meshes/<field>.mat` over whatever is there -- including outlines corrected by hand
in the editor, which carry a `.curated.json` marker. The classic window warned in its
console; here a run without "Replace existing outlines" segments only the fields that have
no outline yet -- CONTINUE, asked for after a half-segmented condition (5 of 10) could only
be finished by ticking Replace -- and keeps every outline already there, hand-corrected ones
included. A condition with every field outlined waits for Replace, and the hand-corrected
ones are counted by name.

Pure where it can be; the run itself (`segment_condition`) is called on a thread.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["OutlineError", "OutlineCondition", "MESHES_FOLDER", "models", "roles_for",
           "default_model", "usable_model", "can_run", "runtime_words", "saved_engine",
           "open_outlines", "resolve", "set_model", "set_folder", "copy_settings",
           "existing_outlines", "missing_fields", "run_fields", "run_action", "readiness",
           "segment_condition", "STOPPED", "progress_words", "field_line",
           "NEXT_STEPS", "next_step"]

MESHES_FOLDER = "cell_meshes"
#: What a field carries when a run was stopped before reaching it.
STOPPED = "not segmented: the run was stopped before this field"


class OutlineError(RuntimeError):
    """A refusal, worded for the person at the window."""


@dataclass
class OutlineCondition:
    """One condition: where it is, the model it is set to, and what that resolves to."""

    name: str
    folder: Path
    model: str = "bact_phase_omni"
    #: ``{role: folder}`` chosen BY HAND; empty when the folder follows the model.
    overrides: dict = field(default_factory=dict)
    resolution: object = None
    fields: list = field(default_factory=list)
    #: ``(n_outlines, n_curated)`` already in ``cell_meshes/``.
    outlines: tuple = (0, 0)
    #: The classical segmenter's settings (`classical_view`), set by the window before a run.
    classical: dict | None = None

    @property
    def roles(self) -> tuple:
        return roles_for(self.model)


def models() -> list:
    """``[(name, roles, label)]``: the models THIS INSTALL CAN RUN, then the saved ones.

    Cellpose-SAM is listed where cellpose 4 is installed, the Omnipose models (and the recipes built
    on them) only where Omnipose is, the classical segmenter always (`backends.available_backends`): a
    list never offers what would refuse at the first field. A saved model READS what it was trained on
    (its record's ``channels``); one with no record of that reads phase, the conservative default the
    classic window used. A saved model is listed where an engine that can run it is installed (its base
    model names the engine; one whose base says nothing is listed when either engine is)."""
    from . import backends
    from .segment import CELLPOSE, CLASSICAL, PRETRAINED, RECIPES

    have = set(backends.available_backends())
    out = []
    if backends.CELLPOSE_BACKEND in have:
        out += [(name, tuple(roles), f"{words} (reads {'+'.join(roles)})")
                for name, (roles, words) in CELLPOSE.items()]
    if backends.OMNIPOSE_BACKEND in have:
        out += [(name, tuple(roles), f"{name} (reads {'+'.join(roles)})")
                for name, roles in PRETRAINED.items()]
        out += [(name, tuple(roles), f"{words} (reads {'+'.join(roles)})")
                for name, (roles, words) in RECIPES.items()]
    out += [(name, tuple(roles), words) for name, (roles, words) in CLASSICAL.items()]
    if have & {backends.CELLPOSE_BACKEND, backends.OMNIPOSE_BACKEND}:
        try:
            from .models import list_models

            for saved in list_models():
                engine = saved_engine(saved)
                if engine is not None and engine not in have:
                    continue
                roles = tuple(saved.channels) or ("phase",)
                out.append((str(saved.weights), roles,
                            f"{saved.name} (trained, reads {'+'.join(roles)})"))
        except Exception:                               # noqa: BLE001 - a store is optional
            pass
    return out


def default_model() -> str:
    """The model a condition starts with where nothing was chosen: the Omnipose phase model where Omnipose is
    installed (as before), else Cellpose-SAM, else the classical segmenter: the first that THIS INSTALL CAN RUN."""
    names = [name for name, _roles, _label in models()]
    for preferred in ("bact_phase_omni", "cpsam", "classical_sum"):
        if preferred in names:
            return preferred
    return names[0]


def usable_model(model) -> str:
    """``model`` where this install lists it, else `default_model` (a standard saved on a machine that had
    Omnipose must not leave a Normal build with a model it cannot run)."""
    names = {name for name, _roles, _label in models()}
    return str(model) if str(model) in names else default_model()


def can_run(model) -> bool:
    """Whether ``model`` can segment in this install: judged by NAME, nothing is read. A file is judged by the
    saved-model record that names it, else by whether any model engine is here (a run refuses by name when the file
    is the other engine's)."""
    from . import backends
    from .segment import CELLPOSE, CLASSICAL, PRETRAINED, RECIPES

    have = set(backends.available_backends())
    model = str(model)
    if model in CLASSICAL:
        return True
    if model in CELLPOSE:
        return backends.CELLPOSE_BACKEND in have
    if model in PRETRAINED or model in RECIPES:
        return backends.OMNIPOSE_BACKEND in have
    try:
        from .models import list_models

        for saved in list_models():
            if str(saved.weights) == model:
                engine = saved_engine(saved)
                return engine in have if engine is not None else len(have) > 1
    except Exception:                                   # noqa: BLE001 - a store is optional
        pass
    return len(have) > 1


def runtime_words() -> str:
    """What the Segment page says when the models cannot run in this install; ``""`` where some can. The packaged
    program says that the segmentation pack (PyTorch and Cellpose) is downloaded on request (`smtracker.segpack`)."""
    from . import backends

    if len(backends.available_backends()) > 1:
        return ""
    return backends.PACK_WORDS if backends._frozen() else (
        "No segmentation model can run in this Python: neither Cellpose 4 nor Omnipose is installed.")


def saved_engine(saved):
    """The engine a saved model runs on, from the base it was fine-tuned from (the name, never the file:
    a listing reads no weights); ``None`` where the record does not say."""
    from . import backends
    from .segment import CELLPOSE, PRETRAINED, RECIPES

    if getattr(saved, "engine", ""):
        return str(saved.engine)
    base = str(saved.base_model or "")
    if base in CELLPOSE:
        return backends.CELLPOSE_BACKEND
    if base in PRETRAINED or base in RECIPES:
        return backends.OMNIPOSE_BACKEND
    return None


def roles_for(model: str) -> tuple:
    for name, roles, _label in models():
        if name == model:
            return roles
    return ("phase",)


def existing_outlines(folder) -> tuple:
    """``(outlines, corrected by hand)`` in ``<folder>/cell_meshes``: a listing only."""
    from .editor_project import CURATED_SUFFIX

    from ..cellmesh import mesh_folder

    meshes = mesh_folder(folder)
    if not meshes.is_dir():
        return (0, 0)
    mats = [p for p in meshes.glob("*.mat") if not p.name.startswith("._")]
    curated = sum(1 for p in mats if p.with_name(p.name + CURATED_SUFFIX).exists())
    return (len(mats), curated)


def missing_fields(cond: OutlineCondition) -> list:
    """The condition's fields with no ``cell_meshes/<field>.mat`` yet (a failed field wrote
    none, so it is retried): a listing only. An outline filed under the movie's number (``001.mat`` for
    ``series001_YFPLaser``, 23 Oct 2026) counts, as the analysis pairs it (`cellmesh.outline_coverage`)."""
    from ..cellmesh import outline_coverage

    have = set(outline_coverage(cond.folder, cond.fields).with_outlines) if cond.fields else set()
    return [f for f in cond.fields if f not in have]


def run_fields(cond: OutlineCondition, *, replace: bool = False) -> list:
    """What a run segments: every field with Replace, otherwise only those not outlined."""
    return list(cond.fields) if replace else missing_fields(cond)


def run_action(cond: OutlineCondition, *, replace: bool = False) -> str:
    """The row button's word: "Continue" where some fields are outlined and a run would
    finish the rest, else "Segment"."""
    if replace or not cond.fields:
        return "Segment"
    todo = len(missing_fields(cond))
    return "Continue" if 0 < todo < len(cond.fields) else "Segment"


def resolve(cond: OutlineCondition) -> OutlineCondition:
    """Re-resolve the condition's images for its model and overrides, and its fields."""
    from .channels import common_fields, resolve_channels

    cond.resolution = resolve_channels(cond.folder, cond.roles, overrides=cond.overrides)
    cond.fields = common_fields(cond.resolution) if cond.resolution.ok else []
    cond.outlines = existing_outlines(cond.folder)
    return cond


def open_outlines(root, *, model: str | None = None) -> dict:
    """``{condition: OutlineCondition}`` for a folder of conditions, or the folder itself.
    Every condition starts with ``model`` (the user's standard, `options`), else the
    pretrained phase model.

    A condition folder is one whose children have channels -- decided by LOOKING, as
    `segment_gui.describe_folder` does, never by depth. Raises `OutlineError` naming the
    folder when nothing in it can be segmented."""
    from .channels import folders_in, resolve_channels

    root = Path(root)
    model = model or default_model()

    def is_condition(folder) -> bool:
        # AN AMBIGUOUS PHASE FOLDER IS STILL A CONDITION: `phase/` beside `brightfield/`
        # is a question for the user to answer in the grid, and a folder that is not
        # listed cannot be asked about (the first draft dropped it, and a test caught it).
        got = resolve_channels(folder, ("phase",))
        return bool(got.resolved) or "phase" in got.ambiguous or (folder / "sum").is_dir()

    if is_condition(root):
        return {root.name: resolve(OutlineCondition(name=root.name, folder=root,
                                                    model=model))}
    out, where = {}, {}
    expanded = {}
    for child in folders_in(root):
        if is_condition(child):
            expanded[child] = [child]
        else:
            # A GROUP OF CONDITIONS (one protein's media, a user's `protein/condition/...`, 15 Oct 2026) opens as
            # its conditions, as the analysis does (`project.discover_conditions`).
            expanded[child] = [g for g in folders_in(child) if is_condition(g)]
    # A CONDITION WHOSE REPLICATE DAYS ARE SUB-FOLDERS (D:/data/project/condition/1, /2, /3) is not a group: its
    # day folders are named alike in every condition (1, 2, 3), which a group's conditions never are. Where the
    # expanded names would collide, the window opens as it did before the groups rule (only folders that are
    # conditions themselves), so nothing that opened before is refused now.
    names = [m.name for members in expanded.values() for m in members]
    if len(names) != len(set(names)):
        expanded = {child: [child] for child in expanded if is_condition(child)}
    for members in expanded.values():
        for member in members:
            out[member.name] = resolve(OutlineCondition(name=member.name, folder=member, model=model))
            where[member.name] = member
    if not out:
        raise OutlineError(f"nothing to segment in {root.name}: no phase/ folder and no phase "
                           f"snapshot beside the movies, in it or in its sub-folders.")
    return out


def set_model(cond: OutlineCondition, model: str) -> str:
    """Switch the model, and RE-DERIVE the folder: the models read different images.
    Returns a note when a folder chosen by hand was re-derived, else ``""``."""
    had = dict(cond.overrides)
    cond.model = model
    cond.overrides = {}
    resolve(cond)
    if had:
        return (f"{cond.name}: {', '.join(Path(p).name for p in had.values())}/ was set by hand "
                f"and has been re-derived for {model}. Set it again if it was right.")
    return ""


def set_folder(cond: OutlineCondition, folder) -> None:
    """The folder the model's (first) role reads, chosen by hand. Never second-guessed."""
    cond.overrides = {cond.roles[0]: Path(folder)}
    resolve(cond)


def copy_settings(conditions: dict, source: str) -> list:
    """The source's model, and its folder BY NAME, onto every other condition. A condition
    with no folder of that name keeps its own. Returns the names that changed."""
    src = conditions[source]
    hand = {role: Path(p).name for role, p in src.overrides.items()}
    changed = []
    for name, cond in conditions.items():
        if name == source:
            continue
        before = (cond.model, dict(cond.overrides))
        cond.model = src.model
        cond.overrides = {role: cond.folder / folder for role, folder in hand.items()
                          if (cond.folder / folder).is_dir()}
        resolve(cond)
        if (cond.model, dict(cond.overrides)) != before:
            changed.append(name)
    return changed


def readiness(cond: OutlineCondition, *, replace: bool = False) -> tuple:
    """``(ready, words)``: whether a run may start, and the row's status either way."""
    res = cond.resolution
    if res is None or not res.ok:
        unresolved = ", ".join(sorted(res.unresolved)) if res is not None else "the images"
        return False, f"set the folder it reads ({unresolved})"
    if not cond.fields:
        return False, "no field is present in every channel it reads"
    n, curated = cond.outlines
    if replace:
        words = f"ready: {len(cond.fields)} field(s)"
        if n:
            words += (f", replacing {n} outline file(s)"
                      + (f" ({curated} corrected by hand)" if curated else ""))
        return True, words
    todo = missing_fields(cond)
    kept = len(cond.fields) - len(todo)
    if not todo:
        return False, (f"all {len(cond.fields)} field(s) outlined"
                       + (f", {curated} corrected by hand" if curated else "")
                       + "; tick Replace to segment again")
    if kept:
        return True, (f"continue: {len(todo)} of {len(cond.fields)} field(s) left; the "
                      f"{kept} outlined are kept"
                      + (f" ({curated} corrected by hand)" if curated else ""))
    return True, f"ready: {len(cond.fields)} field(s)"


class _Stopped(Exception):
    pass


def segment_condition(cond: OutlineCondition, *, use_gpu: bool, replace: bool = False,
                      on_field=None, stop=None, on_status=None) -> dict:
    """Segment ``cond`` into ``cell_meshes/`` (`segment.segment_folder`): every field with
    ``replace``, otherwise only the fields with no outline yet (`run_fields`), keeping the
    rest. Each field is reported as a plain row through ``on_field(done, total, row)``;
    ``stop()`` is asked after each field. Returns ``{condition, rows, cells, failed,
    stopped, kept, summary}``. Runs on a THREAD: the model is loaded once, on the first
    field."""
    from .segment import config_for, prepare_model, segment_folder

    targets = run_fields(cond, replace=replace)
    if not targets:
        raise OutlineError(f"{cond.name}: every field already has an outline. Tick Replace to "
                           f"segment again.")
    kept = len(cond.fields) - len(targets)
    rows = []
    cfg = config_for(cond.model, tuple(cond.roles), use_gpu=bool(use_gpu), classical=cond.classical)
    # THE ENGINE AND THE WEIGHTS FIRST, said as they happen (Cellpose-SAM's 1.2 GB download at first use,
    # then the load), so the first field's time is the field's. A refusal here is named, never a fallback.
    try:
        prepare_model(cfg, on_status)
    except Exception as exc:                            # noqa: BLE001 - one named refusal
        raise OutlineError(f"{cond.name}: {exc}") from exc

    def progress(done, total, result):
        row = {"field": result.field, "cells": int(result.cells),
               "masks": int(result.masks_found), "rejected": dict(result.rejected),
               "seconds": float(result.seconds), "error": result.error,
               "alignment": result.alignment,
               "shift": float(getattr(result, "alignment_shift_px", 0.0) or 0.0),
               "fill": result.fill}
        rows.append(row)
        if on_field is not None:
            on_field(done, total, row)
        if stop is not None and stop() and done < total:
            raise _Stopped()

    stopped = 0
    try:
        from ..cellmesh import mesh_folder

        report = segment_folder(cond.resolution, mesh_folder(cond.folder), cfg,
                                fields=targets, progress=progress)
        summary = report.summary()
    except _Stopped:
        stopped = len(targets) - len(rows)
        summary = (f"Stopped after {len(rows)} of {len(targets)} field(s); "
                   f"{stopped} not segmented")
    if kept:
        summary += f"\n{kept} field(s) that already had outlines were kept"
    cond.outlines = existing_outlines(cond.folder)
    return {"condition": cond.name, "rows": rows,
            "cells": sum(r["cells"] for r in rows),
            "failed": [r["field"] for r in rows if r["error"]],
            "stopped": stopped, "kept": kept, "summary": summary}


def progress_words(name: str, done: int, total: int, rows: list) -> str:
    """The progress line under the bar: where the run is, how fast it goes, what is left.

    THE FIRST FIELD IS NOT A RATE: it also loads the model (seconds to a minute), so the
    estimate uses the fields after it once there are any, and says so until then."""
    if not rows:
        return (f"{name}: loading the model and segmenting the first of {total} field(s). "
                f"The first field is the slow one.")
    last = rows[-1]
    timed = [r["seconds"] for r in rows[1:] if r["seconds"]] or [rows[0]["seconds"]]
    rate = sum(timed) / len(timed) if timed else 0.0
    left = rate * (total - done)
    text = f"{name}: field {done} of {total}"
    if rate:
        text += f", {rate:.1f} s a field"
        if done < total:
            text += f", about {left:.0f} s left" if left < 120 else f", about {left / 60:.0f} min left"
    text += (f"; last: {last['field']}, failed ({last['error']})" if last["error"] else
             f"; last: {last['field']}, {last['cells']} cell(s)")
    return text


def field_line(done: int, total: int, row: dict) -> str:
    """One field's line in the log."""
    head = f"[{done}/{total}] {row['field']}: "
    if row["error"]:
        return head + f"failed: {row['error']}"
    text = head + f"{row['cells']} cells from {row['masks']} masks"
    if row["rejected"]:
        text += "   (rejected: " + ", ".join(f"{n} {k}" for k, n in row["rejected"].items()) + ")"
    text += f"   {row['seconds']:.1f} s"
    # FLAGGED ONLY WHEN A SHIFT WAS MEASURED AND IS NOT "aligned", the classic rule: a
    # "not checked" verdict is information, not a warning.
    if row.get("shift") and not str(row.get("alignment", "")).startswith("aligned"):
        text +=f"\n     ! {row['alignment']}"
    return text


# ---------------------------------------------------------------- the Next button
#: The outlines window's steps and where Next goes: ``(target, title)``; None closes the
#: window back to the menu. Training is optional, so Edit outlines ends the stage.
NEXT_STEPS = {"segment": ("edit", "Edit outlines"),
              "edit": (None, "Done, back to the menu")}


def next_step(section: str, states: dict, *, running: str | None = None,
              unsaved: bool = False) -> dict | None:
    """``{target, label, ready, why}`` for ``section``'s Next button, or None for a page
    without one (Train). Asked for (13 Oct 2026): "a next button in the image prep/tracking
    and segmentation gui when the current screen is ready and done". From the counts the
    window already holds (`OutlineCondition.outlines`), so nothing is read.

    - Segment: nothing running and every field of every condition outlined;
    - Edit outlines: no unsaved edit.
    """
    if section not in NEXT_STEPS:
        return None
    target, title = NEXT_STEPS[section]
    step = {"target": target, "label": f"Next: {title}" if target else title}
    why = ""
    if not states:
        why = "open a folder first"
    elif section == "segment":
        if running:
            why = f"segmenting {running}"
        else:
            missing = {n: len(c.fields) - c.outlines[0] for n, c in states.items()
                       if c.outlines[0] < len(c.fields)}
            if missing:
                why = (f"{sum(missing.values())} field(s) in {len(missing)} condition(s) have "
                       f"no outlines yet: segment them, or draw them in the editor")
    elif section == "edit" and unsaved:
        why = "there are unsaved edits; use Save all first"
    step.update(ready=not why, why=why)
    return step

