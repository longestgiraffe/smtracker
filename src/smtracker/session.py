"""Session restore: reopen where you left off.

Remembers the last project, the selected condition/tier, and which panels were
open, so relaunching lands the user back where they were rather than at a blank
window. Small and JSON-backed; useful for demo continuity and daily use.

Deliberately tolerant on load: a corrupt or partial session file yields sensible
defaults rather than an error, because losing your place should never block
opening the app. Saving is best-effort for the same reason.

THE RECENT-PROJECTS LIST HAS TWO WRITERS, `remember_project` (a folder opened) and
`forget_project` (the home screen's x, 14 Oct 2026). Reported: "sometimes i see
last opened, sometimes i dont". The comparison browser builds its state from a blank
`SessionState()` and saved ALL of it, so every time it opened -- including in the test
suite, which wrote to the user's real file -- the list was replaced by an empty one
(found at 15:24 on 10 Oct 2026: `recent_projects: []` beside the browser's own settings,
written by a test run). `save_session` now keeps the list already on disk unless it is
handed one, and `SMTRACKER_SESSION` points the file elsewhere (the suite sets it).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["RECENT_LIMIT", "SESSION_ENV", "SessionState", "default_session_path",
           "forget_project", "load_session", "recent_projects", "remember_project",
           "save_session"]

#: Points the session file elsewhere, so a test never writes the user's own.
SESSION_ENV = "SMTRACKER_SESSION"


@dataclass
class SessionState:
    """What to restore on relaunch."""

    last_project: str | None = None
    last_condition: str | None = None
    last_tier: str = "quick"
    open_panels: list = field(default_factory=list)
    window_geometry: dict | None = None      # {x, y, w, h}, restored if present
    # --- comparison-first state (drives every method view) --------------- #
    selected_conditions: list = field(default_factory=list)
    #: condition -> model choice (see comparison.MODEL_CHOICES); "best" default
    per_condition_model: dict = field(default_factory=dict)
    lock_model: str | None = None            # one model forced on all if set
    mobility_threshold_um2_s: float = 0.15   # the comparable slow-fraction basis
    reference_condition: str | None = None   # Δ / forest reference
    forest_methods: list = field(default_factory=list)  # ticked method rows
    journal_preset: str = "screen"           # figstyle export preset
    journal_width: str = "single"            # single- or double-column
    spatial_source: str = "appd"             # spatial map population source
    spatial_mode: str = "enrichment"         # spatial map: enrichment / density
    fraction_uncertainty: str = "analytic"   # fraction view CI tier
    sqd_cis: bool = False                     # SQD bubble error bars (slow)
    sqd_fraction_weighting: str = "track"     # fraction estimand: track=molecule, step=displacement/time
    ecdf_subsample: bool = False              # ECDF-subsample the SQD point fit
    #: Projects opened before, MOST RECENT FIRST. Each entry is
    #: ``{"path", "opened_at", "conditions", "series"}`` -- the last two cached
    #: from the discovery that ran when it was opened, so the home screen can
    #: describe a folder without globbing four drives to draw a list.
    #:
    #: CACHED FACTS GO STALE, and the screen must treat them that way: they say
    #: what was there LAST time, and the folder may since have moved, grown or
    #: gone. What is never stale is the path, which is why it is the key.
    recent_projects: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def default_session_path() -> Path:
    """Per-user session file (under the platform config dir), or `SMTRACKER_SESSION`."""
    override = os.environ.get(SESSION_ENV)
    if override:
        return Path(override).expanduser()
    try:
        from platformdirs import user_config_dir

        base = Path(user_config_dir("smtracker", "smtracker"))
    except Exception:                                  # pragma: no cover - fallback
        base = Path.home() / ".smtracker"
    base.mkdir(parents=True, exist_ok=True)
    return base / "session.json"


def load_session(path=None) -> SessionState:
    """Load the saved session, or defaults. Never raises on a bad file."""
    path = Path(path) if path is not None else default_session_path()
    if not path.is_file():
        return SessionState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return SessionState()
        return SessionState.from_dict(data)
    except (OSError, ValueError):
        return SessionState()


#: How many projects the home screen remembers. Ten is a list you can read;
#: fifty is a search problem, and the answer to a search problem is the folder
#: picker sitting right beside it.
RECENT_LIMIT = 10


def remember_project(root, *, conditions=0, series=0, path=None) -> list:
    """Record a project as just-opened and return the new list, newest first.

    MOST RECENT FIRST, AND EACH PATH ONCE. Reopening a folder moves it to the
    top rather than adding a second row: a list that can hold the same project
    three times is a list that pushes the other two off the end.

    Paths are compared RESOLVED, so ``D:/data/run`` and ``D:/data/./run`` are
    one entry -- on Windows the same folder has more spellings than anywhere
    else, and two rows for one project is exactly the kind of thing nobody
    reports as a bug and everybody notices.

    Best-effort, like everything else here: a project that cannot be recorded
    still opens.
    """
    from datetime import datetime, timezone

    state = load_session(path)
    try:
        key = str(Path(root).resolve())
    except OSError:                                    # pragma: no cover - unresolvable
        key = str(root)
    kept = [e for e in state.recent_projects
            if isinstance(e, dict) and not _same_path(e.get("path"), key)]
    entry = {
        "path": key,
        "opened_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "conditions": int(conditions),
        "series": int(series),
    }
    state.recent_projects = [entry, *kept][:RECENT_LIMIT]
    save_session(state, path, recents=state.recent_projects)
    return state.recent_projects


def forget_project(root, *, path=None) -> list:
    """Take a project off the list and return the new list, newest first.

    Asked for (14 Oct 2026): "the remove x in the recent when loading analysis tool is
    missing, to remove links to recent data not needed anymore". Only the ENTRY goes: the
    folder, its tracks and its caches are not touched. Matched as `remember_project`
    matches, so any spelling of the folder removes its one row; a folder that is gone (an
    unplugged drive) is removed the same way."""
    state = load_session(path)
    kept = [e for e in state.recent_projects
            if isinstance(e, dict) and not _same_path(e.get("path"), str(root))]
    state.recent_projects = kept
    save_session(state, path, recents=kept)
    return kept


def _same_path(a, b) -> bool:
    """Two spellings of one folder. Never raises: a bad entry is simply not a match."""
    if not a or not b:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except (OSError, ValueError):                      # pragma: no cover - odd entry
        return str(a) == str(b)


def recent_projects(path=None) -> list:
    """The remembered projects, newest first, skipping anything malformed."""
    state = load_session(path)
    return [e for e in state.recent_projects
            if isinstance(e, dict) and e.get("path")]


def save_session(state: SessionState, path=None, *, recents: list | None = None) -> Path | None:
    """Persist the session. Best-effort: a write failure is swallowed.

    The recent-projects list written is ``recents`` when given (only `remember_project`
    and `forget_project` give it) and otherwise the list ALREADY ON DISK -- never ``state``'s own copy, which
    may be a blank default's empty list or one read before another window remembered a
    project. See the module docstring for the defect this closes."""
    path = Path(path) if path is not None else default_session_path()
    payload = state.to_dict()
    payload["recent_projects"] = (list(recents) if recents is not None
                                  else load_session(path).recent_projects)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
    except OSError:                                    # pragma: no cover - best effort
        return None
