"""A workspace SESSION on disk (v2's File > Save session / Load session; added 17 Oct 2026): the project it belongs
to, every section's settings and results, written once and read back without recomputing. Headless.

THE FILE, ``<name>.smtsession``, is a zip of two members:

- ``manifest.json`` -- readable by anyone: the format and its version, when and by which build it was written, and the
  PROJECT: its condition folders (``roots``), the tracker read per condition, the track cap, the exclusions in force,
  the condition labels, and per condition the number of tracks and localisations and a CONTENT FINGERPRINT
  (`provenance.hash_frame` of the condition's analysis table: uid, frame, x, y, amp).
- ``state.pkl`` -- the sections' settings and results, as the window holds them (plain dicts and lists; the SQD
  bubbles' dataclasses), pickled.

FORMAT 2 (4 Nov 2026) adds the ADDED PAGES of SQD and the Spatial Map (``"subpages"`` in the state: the page model,
`subpages.SubpageSet.state`, and per page key its own settings and results). A session without the key is a format-1
session and is read exactly as before, with no added pages; one written by this build is refused by an older build as
newer, by the same rule as any newer session.

A SESSION'S RESULTS BELONG TO ITS DATA. Before anything is drawn, `check_project` compares the open project with the
manifest: the same folders, and per condition the same fingerprint. A condition whose tracks changed since (a
re-tracking, other exclusions, another cap) is NAMED and the session is refused for it -- results are never drawn over
data they were not computed from. The window applies the track cap and the exclusions recorded before it checks.

READING IS RESTRICTED (`load_state`): a pickle can run code when it is read, and a session file is a file someone may
mail around. The reader accepts only plain Python types, numpy arrays, pandas' tables (a DataFrame or Series, their
indexes and the column arrays pandas pickles them with: `_PANDAS`, added 30 Oct 2026 -- the Kinetics section's
slow-fraction result holds two, so a session or composition holding it was refused), and SMTracker's own DATACLASSES
and enums; anything else -- a function, a class from another package -- is refused with its name, so a session from a
newer or tampered build fails loud instead of executing.

ONE ROUTE FOR SMTRACKER'S OWN FILES OF RESULTS (30 Oct 2026): the zip of a manifest and a restricted pickle is
`write_bundle` / `read_bundle`, which a COMPOSITION FILE (`smtracker.composer`, ``.composer``: the Composer's panels and
settings) uses too, with its own format name and version; the refusals are the same sentences with the file's name.

Invariants: no Qt; nothing here draws or recomputes; a refusal raises `SessionError` with a sentence a user can act on.
"""
from __future__ import annotations

import dataclasses
import enum
import io
import json
import pickle
import zipfile
from datetime import datetime
from pathlib import Path

__all__ = ["SessionError", "FORMAT", "VERSION", "SUFFIX", "project_identity", "write_session", "read_manifest",
           "load_state", "check_project", "session_root", "write_bundle", "read_bundle"]

FORMAT = "smtracker-session"
VERSION = 2
SUFFIX = ".smtsession"


class SessionError(RuntimeError):
    """Raised when a session cannot be written, read or applied honestly."""


def project_identity(project, conditions) -> dict:
    """What a session's results were computed from: the folders and per condition the counts and the fingerprint."""
    from .canonical import without_attrs
    from .provenance import hash_frame

    roots = {str(c): str(Path(r).resolve()) for c, r in (getattr(project, "roots", None) or {}).items()}
    per = {}
    for c in conditions:
        table = without_attrs(project.condition(c))
        per[str(c)] = {"n_tracks": int(table["uid"].nunique()) if "uid" in table.columns else 0,
                       "n_localisations": int(len(table)),
                       "fingerprint": hash_frame(table) if len(table) else ""}
    trackers = {}
    try:
        trackers = {str(k): str(v) for k, v in (project.canonical.attrs.get("trackers") or {}).items()}
    except AttributeError:
        pass
    tolerance = None
    try:
        tolerance = project.canonical.attrs.get("outline_tolerance_nm")
    except AttributeError:
        pass
    return {"roots": roots, "conditions": per, "trackers": trackers,
            "outline_tolerance_nm": None if tolerance is None else float(tolerance)}


def session_root(manifest: dict) -> tuple:
    """``(project folder, conditions)`` to load a session's project from: the folder holding its condition folders
    (their common parent) and the conditions it holds."""
    import os

    roots = list(((manifest.get("project") or {}).get("roots") or {}).values())
    conditions = list(((manifest.get("project") or {}).get("conditions") or {}).keys())
    if not roots:
        raise SessionError("the session names no folders")
    missing = [r for r in roots if not Path(r).is_dir()]
    if missing:
        raise SessionError("the session's folders are not here: " + ", ".join(missing[:3])
                           + " (moved, renamed, or on a drive that is not connected)")
    root = Path(os.path.commonpath(roots)) if len(roots) > 1 else Path(roots[0]).parent
    if str(root) in roots:
        root = root.parent
    return root, conditions


def write_session(path, *, identity: dict, state: dict, labels: dict, track_cap=None, exclusions=None,
                  view: str = "", build: str = "") -> Path:
    """Write the session to ``path`` (``.smtsession`` added when missing). Returns the path written."""
    path = Path(path)
    if path.suffix.lower() != SUFFIX:
        path = path.with_name(path.name + SUFFIX)
    manifest = {"format": FORMAT, "version": VERSION, "saved_at": datetime.now().isoformat(timespec="seconds"),
                "build": build, "project": identity, "labels": {str(k): str(v) for k, v in labels.items()},
                "track_cap": track_cap, "exclusions": list(exclusions or []), "view": view,
                "sections": sorted(k for k, v in state.get("sections", {}).items() if v)}
    return write_bundle(path, manifest, state, what="session")


def write_bundle(path, manifest: dict, state, *, what: str, check: bool = False) -> Path:
    """``manifest`` (JSON) and ``state`` (pickled) zipped to ``path``, written under a temporary name and moved into
    place, so a file under the real name is always whole. ``what`` names the file in a refusal ("session",
    "composition"). ``check``: read the pickle back with the restricted reader first (`read_bundle`'s), so a file is
    never written that could not be opened again -- refused naming what it holds instead."""
    path = Path(path)
    try:
        blob = pickle.dumps(state, protocol=4)
    except Exception as exc:                                # noqa: BLE001 - named: what could not be kept
        raise SessionError(f"a result could not be written into the {what} ({type(exc).__name__}: {exc})") from exc
    if check:
        _restricted_loads(blob, what, path.name)
    tmp = path.with_name(path.name + ".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2, default=str))
        z.writestr("state.pkl", blob)
    tmp.replace(path)                                       # never a half-written file under the real name
    return path


def _unreadable(exc) -> str:
    """Why a file could not be opened, in words a user reads (3 Nov 2026, reported: the Composer said
    "(BadZipFile: File is not a zip file)")."""
    if isinstance(exc, zipfile.BadZipFile):
        return "it is not a file SMTracker writes, or it is damaged"
    if isinstance(exc, KeyError):
        return "it holds no SMTracker contents list"
    if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
        return "its contents list is damaged"
    if isinstance(exc, OSError):
        return f"it could not be read: {(exc.strerror or str(exc)).rstrip('.').lower()}"
    return str(exc)


def _bundle_manifest(path, *, fmt: str, version: int, what: str, oldest: int = 1) -> dict:
    """The manifest of an SMTracker ``what`` file, checked to be one this build reads: its format, and a version
    from ``oldest`` to ``version`` (a session's manifest without one has always been read: ``oldest`` 0)."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as z:
            manifest = json.loads(z.read("manifest.json").decode("utf-8"))
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SessionError(f"{path.name} is not an SMTracker {what}: {_unreadable(exc)}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != fmt:
        raise SessionError(f"{path.name} is not an SMTracker {what} "
                           f"(format {manifest.get('format') if isinstance(manifest, dict) else None!r})")
    try:
        found = int(manifest.get("version", 0))
    except (TypeError, ValueError):
        found = -1                                          # not a number: no version this build knows
    if found > version:
        raise SessionError(f"{path.name} was written by a newer SMTracker ({what} version "
                           f"{manifest.get('version')}, this build reads {version}): update to open it")
    if found < oldest:
        raise SessionError(f"{path.name} names no {what} version this build knows ({manifest.get('version')!r})")
    return manifest


def read_bundle(path, *, fmt: str, version: int, what: str, oldest: int = 1) -> tuple:
    """``(manifest, state)`` of an SMTracker ``what`` file: the manifest checked (`_bundle_manifest`), the state read
    with the restricted reader (see the module head). Every failure is a `SessionError` naming the file."""
    path = Path(path)
    manifest = _bundle_manifest(path, fmt=fmt, version=version, what=what, oldest=oldest)
    try:
        with zipfile.ZipFile(path) as z:
            blob = z.read("state.pkl")
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise SessionError(f"{path.name}: the {what} could not be read: {_unreadable(exc)}") from exc
    return manifest, _restricted_loads(blob, what, path.name)


def read_manifest(path) -> dict:
    """The session's manifest, checked to be one this build reads."""
    return _bundle_manifest(path, fmt=FORMAT, version=VERSION, what="session", oldest=0)


#: numpy's own reconstruction entry points, by module (the names numpy 1.x and 2.x pickle under).
_NUMPY = {("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
          ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
          ("numpy._core.multiarray", "scalar"), ("numpy.core.numeric", "_frombuffer"),
          ("numpy._core.numeric", "_frombuffer"), ("numpy", "float64"), ("numpy", "int64"), ("numpy", "bool_")}
_BUILTINS = {"set", "frozenset", "tuple", "list", "dict", "complex", "slice", "range", "bytearray", "float", "int",
             "bool", "str", "bytes"}
#: pandas' own tables and their reconstruction entry points (the names pandas 3 pickles under, public and internal),
#: each of which builds only these classes: no callable from elsewhere is reachable through them.
_PANDAS = {("pandas", n) for n in ("DataFrame", "Series", "Index", "RangeIndex", "MultiIndex", "Categorical",
                                   "CategoricalDtype", "StringDtype")} | {
    ("pandas.arrays", "DatetimeArray"), ("pandas.arrays", "StringArray"),
    ("pandas.core.frame", "DataFrame"), ("pandas.core.series", "Series"),
    ("pandas.core.indexes.base", "Index"), ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.range", "RangeIndex"), ("pandas.core.indexes.multi", "MultiIndex"),
    ("pandas.core.arrays.categorical", "Categorical"), ("pandas.core.dtypes.dtypes", "CategoricalDtype"),
    ("pandas.core.arrays.string_", "StringDtype"), ("pandas.core.arrays.string_", "StringArray"),
    ("pandas.core.arrays.datetimes", "DatetimeArray"),
    ("pandas.core.internals.managers", "BlockManager"), ("pandas.core.internals.managers", "SingleBlockManager"),
    ("pandas._libs.internals", "_unpickle_block"), ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked")}


class _Restricted(pickle.Unpickler):
    #: What the file is, for a refusal ("session", "composition").
    what = "session"

    def find_class(self, module, name):
        if module == "builtins" and name in _BUILTINS:
            return super().find_class(module, name)
        if module == "collections" and name == "OrderedDict":
            return super().find_class(module, name)
        if (module, name) in _NUMPY or (module, name) in _PANDAS:
            return super().find_class(module, name)
        if module == "smtracker" or module.startswith("smtracker."):
            obj = super().find_class(module, name)
            if isinstance(obj, type) and (dataclasses.is_dataclass(obj) or issubclass(obj, enum.Enum)):
                return obj
        what = self.what
        raise SessionError(f"the {what} holds {module}.{name}, which a {what} may not carry; it was not read "
                           f"(a {what} from another build, or not one SMTracker wrote)")


def _restricted_loads(blob: bytes, what: str, name: str):
    """``blob`` unpickled by the restricted reader; a failure is a `SessionError` naming ``name``."""
    reader = _Restricted(io.BytesIO(blob))
    reader.what = what
    try:
        return reader.load()
    except SessionError:
        raise
    except Exception as exc:                                # noqa: BLE001 - named
        raise SessionError(f"{name}: the {what} could not be read ({type(exc).__name__}: {exc})") from exc


def load_state(path) -> dict:
    """The sections' settings and results, read with the restricted reader (see the module head)."""
    path = Path(path)
    _manifest, state = read_bundle(path, fmt=FORMAT, version=VERSION, what="session", oldest=0)
    if not isinstance(state, dict):
        raise SessionError(f"{path.name}: not a session's results")
    return state


def check_project(manifest: dict, identity: dict) -> list:
    """Why the open project is not the session's, one sentence each; empty when it is."""
    want = manifest.get("project") or {}
    problems = []
    if sorted((want.get("roots") or {}).values()) != sorted((identity.get("roots") or {}).values()):
        problems.append("the session belongs to other folders: " + ", ".join(sorted((want.get("roots") or {}).values())))
        return problems
    have = identity.get("conditions") or {}
    for c, w in (want.get("conditions") or {}).items():
        h = have.get(c)
        if h is None:
            problems.append(f"{c}: not in the open project")
        elif h.get("fingerprint") != w.get("fingerprint"):
            problems.append(f"{c}: its tracks differ from the saved session ({w.get('n_tracks'):,} tracks "
                            f"saved, {h.get('n_tracks'):,} loaded)")
    return problems
