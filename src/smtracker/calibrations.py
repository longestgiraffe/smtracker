"""WHICH PER-SPOT CALIBRATION A PROJECT USES, and the library of calibrations made in the app (6 Nov 2026). Headless:
no Qt.

Asked: "people should choose (internal calibration this folder), own folder. measure on current data set can be
tricky? whats the best angle here?", then "yes to all" to: make a calibration only from a separate stationary
measurement, keep every one made in a library, and choose one explicitly per project.

* THE LIBRARY -- calibrations measured in the app (`calibration_view.measure_folder` on a stationary measurement),
  each with a name, the day it was made and its record, kept with the user's settings (ui.json, `LIBRARY_KEY`), never
  beside the data. `library`, `add`, `remove`, `entry`.
* THE CHOICE, per project (`condition_order.project_key`, the workspace's key; `CHOICE_KEY`): ``builtin`` (the
  tracker's own, `precision.PRESETS`: the stationary synthetic set), ``project`` (the project's
  `precision.PROJECT_FILE`) or ``library`` with an ``id``. NEVER CHOSEN, a project reads as before (the file where
  one is found, else built-in): a new setting changes nothing made before. `chosen`, `choose`, `current`.
* `resolve(project, conditions, trackers)` -- the ONE calibration a Per spot Run uses, every section's
  (`workspace_view.run_precision` calls it); a choice that can no longer be honoured (the file gone, the library entry
  removed) is refused by name, never replaced in silence.
* `options(project, trackers)` -- the rows the dialog offers (plain dicts).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

__all__ = ["LIBRARY_KEY", "CHOICE_KEY", "KINDS", "library", "add", "remove", "entry", "chosen", "choose",
           "current", "resolve", "options", "curve_words"]

#: ui.json keys (`ui.theme.remember`), isolated per test by the suite's conftest.
LIBRARY_KEY = "precision_library"
CHOICE_KEY = "precision_choice"
KINDS = ("builtin", "project", "library")


def _remembered(key: str):
    from .ui import theme

    value = theme.remembered(key)
    return value


def _remember(key: str, value) -> None:
    from .ui import theme

    theme.remember(key, value)


# -- the library ---------------------------------------------------------------------------------------------- #
def library() -> list:
    """Every calibration in the library, oldest first: ``{"id", "name", "made", "c_nm", "p", "source", "record"}``."""
    value = _remembered(LIBRARY_KEY)
    return [dict(e) for e in value if isinstance(e, dict) and e.get("id")] if isinstance(value, list) else []


def entry(entry_id: str) -> dict | None:
    return next((e for e in library() if e["id"] == str(entry_id)), None)


def add(payload: dict, name: str) -> dict:
    """Put a measured calibration (`calibration_view.measure`'s payload, ``ok``) in the library under ``name``; its
    source -- what every caption that uses it says -- becomes "<name> (<what it was measured on>)"."""
    import uuid

    if not payload.get("ok"):
        raise ValueError("nothing measured to add")
    name = " ".join(str(name).split()) or str(payload.get("name") or "calibration")
    item = {"id": uuid.uuid4().hex[:12], "name": name, "made": date.today().isoformat(),
            "c_nm": float(payload["c_nm"]), "p": float(payload["p"]),
            "source": f"{name} ({payload['source']})", "record": dict(payload.get("record") or {})}
    _remember(LIBRARY_KEY, library() + [item])
    return item


def remove(entry_id: str) -> None:
    """Take a calibration out of the library. A project that chose it is then refused by name at its next Run."""
    _remember(LIBRARY_KEY, [e for e in library() if e["id"] != str(entry_id)])


def _calibration(item: dict):
    from .precision import Calibration

    return Calibration(c_nm=float(item["c_nm"]), p=float(item["p"]), source=str(item["source"]),
                       record=dict(item.get("record") or {}))


# -- the choice, per project ------------------------------------------------------------------------------------ #
def project_key(project) -> str | None:
    from .condition_order import project_key as key

    return key(list((getattr(project, "roots", None) or {}).values()))


def chosen(project) -> dict | None:
    """The project's choice (``{"kind", "id"?}``), None where none was ever made."""
    key = project_key(project)
    stored = _remembered(CHOICE_KEY)
    choice = (stored if isinstance(stored, dict) else {}).get(str(key)) if key else None
    return dict(choice) if isinstance(choice, dict) and choice.get("kind") in KINDS else None


def choose(project, kind: str, entry_id: str | None = None) -> dict:
    """Make ``kind`` (``library``: with its ``entry_id``) the project's calibration from now on."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if kind == "library" and entry(entry_id or "") is None:
        raise ValueError(f"no calibration {entry_id!r} in the library")
    key = project_key(project)
    if not key:
        raise ValueError("this project has no folders to remember a choice by")
    choice = {"kind": kind, **({"id": str(entry_id)} if kind == "library" else {})}
    stored = _remembered(CHOICE_KEY)
    stored = dict(stored) if isinstance(stored, dict) else {}
    stored[key] = choice
    _remember(CHOICE_KEY, stored)
    return choice


def _project_file(project, conditions=None):
    from .precision import project_file

    roots = getattr(project, "roots", None) or {}
    names = list(roots) if conditions is None else [c for c in conditions if c in roots]
    return project_file([roots[c] for c in names])


def current(project) -> dict:
    """The choice IN FORCE: the one made, else what a project that never chose reads (`resolve`): ``project`` where
    its file is found, else ``builtin``. ``{"kind", "id"?, "made": bool}``."""
    made = chosen(project)
    if made is not None:
        return {**made, "made": True}
    from .precision import PrecisionError

    try:
        found = _project_file(project)
    except PrecisionError:
        found = None
    return {"kind": "project" if found is not None else "builtin", "made": False}


def resolve(project, conditions, trackers):
    """The calibration a Per spot Run of ``conditions`` (tracked by ``trackers``) uses: the project's choice, else as
    before a choice existed (`precision.calibration_for`: its file, else built-in). RAISES `precision.PrecisionError`
    by name where the choice cannot be honoured."""
    from .precision import PrecisionError, builtin_for, calibration_for, load

    choice = chosen(project)
    roots = getattr(project, "roots", None) or {}
    if choice is None:
        return calibration_for([roots[c] for c in conditions if c in roots], trackers)
    if choice["kind"] == "builtin":
        return builtin_for(trackers)
    if choice["kind"] == "project":
        found = _project_file(project, conditions)
        if found is None:
            raise PrecisionError("this project's calibration file (precision_calibration.json) is not in the "
                                 "conditions' folders or the folder above them: put it back, or choose "
                                 "another calibration (the menu under the project's name, Per-spot calibration)")
        return load(found)
    item = entry(choice.get("id", ""))
    if item is None:
        raise PrecisionError("the calibration chosen for this project was removed from the library: choose another "
                             "(the menu under the project's name, Per-spot calibration)")
    return _calibration(item)


# -- what the dialog offers ------------------------------------------------------------------------------------- #
def project_trackers(project, conditions) -> set:
    """The trackers ``conditions`` were tracked by, as `project.load_project` records them
    (``canonical.attrs["trackers"]``); a condition without a record reads ``utrack``."""
    canonical = getattr(project, "canonical", None)
    trackers = (canonical.attrs.get("trackers") if canonical is not None else None) or {}
    return {str(trackers.get(c, "utrack")) for c in conditions}


def curve_words(c_nm: float, p: float) -> str:
    return f"σ true = {float(c_nm):.3g} × reported^{float(p):.3g} nm"


def options(project, trackers) -> list:
    """The rows of the choice, in order: built-in, the project's file where one is found (or its refusal), then the
    library. Each ``{"kind", "id"?, "title", "detail", "available", "selected"}``."""
    from .precision import PRESETS, PrecisionError, load

    now = current(project)
    kinds = sorted({str(t) for t in trackers} & set(PRESETS)) or ["utrack"]
    detail = "; ".join(f"{'u-track' if k == 'utrack' else 'TrackMate'}: {curve_words(PRESETS[k].c_nm, PRESETS[k].p)}"
                       for k in kinds)
    rows = [{"kind": "builtin", "title": "Built-in",
             "detail": f"{detail} · a stationary synthetic set",
             "available": True, "selected": now["kind"] == "builtin"}]
    try:
        found = _project_file(project)
        if found is not None:
            cal = load(found)
            rows.append({"kind": "project", "title": "This project's file",
                         "detail": f"{curve_words(cal.c_nm, cal.p)} · {cal.source} · {found}",
                         "available": True, "selected": now["kind"] == "project"})
        elif now["kind"] == "project":
            rows.append({"kind": "project", "title": "This project's file", "detail": "file not found",
                         "available": False, "selected": True})
    except (PrecisionError, OSError, ValueError, TypeError) as exc:
        rows.append({"kind": "project", "title": "This project's file", "detail": str(exc), "available": False,
                     "selected": now["kind"] == "project"})
    for item in library():
        rows.append({"kind": "library", "id": item["id"], "title": item["name"],
                     "detail": f"{curve_words(item['c_nm'], item['p'])} · made {item['made']}",
                     "available": True, "selected": now["kind"] == "library" and now.get("id") == item["id"]})
    if now["kind"] == "library" and not any(r.get("id") == now.get("id") for r in rows):
        rows.append({"kind": "library", "id": now.get("id"), "title": "The calibration chosen before",
                     "detail": "removed from the library", "available": False, "selected": True})
    return rows
