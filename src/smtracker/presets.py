"""Saved detection settings, so one session's choices serve the next.

What a preset is not. This module replaces a dropdown that offered "Glucose" and
"Malate": one lab's CONDITIONS, presented as though they were general settings. They
were the two thresholds hand-matched to u-track's yield (60 and 130), and they are real
measurements; but a condition name is not a setting, and anybody else opening the tool
would be asked to pick between two words meaningless to them. `trackmate_run.
UTRACK_COMPARABLE` keeps those numbers as the record they always were.

What replaces it is the actual choice: let the movies decide, or decide yourself.

    ADAPTIVE   threshold = k x sigma_LoG(this movie), measured per batch
    MANUAL     type the numbers

and then, because a session's settings are worth reusing across the days of one
experiment, any number of SAVED presets alongside.

Where they live, and why not the installation folder. The experimenter suggested "the
smtracker installation folder or something", and the "or something" is the right
instinct: a frozen install commonly sits under Program Files, which is READ-ONLY to the
user who runs it, and reinstalling or upgrading replaces that folder wholesale, taking
every saved preset with it. A per-user config directory survives both. It is also the
only choice that behaves the same in the source tree and in the bundle.

Every preset carries a comment and its provenance. A threshold with no note is a number
somebody typed once, and six months later nobody can say whether it was measured, copied
from a paper, or guessed. `provenance` records which ("computed from movies (k=7.2,
sigma_LoG 36.7)" or "typed") so a preset cannot quietly launder a guess into a
convention.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = [
    "PresetError",
    "ADAPTIVE",
    "MANUAL",
    "BUILT_IN",
    "SavedPreset",
    "preset_dir",
    "list_presets",
    "broken_presets",
    "load_preset",
    "save_preset",
    "delete_preset",
]


class PresetError(RuntimeError):
    """Raised when a preset cannot be read, written or named honestly."""


#: The two that are not saved files: they describe HOW the numbers are chosen.
ADAPTIVE = "Adaptive: compute from movies"
MANUAL = "Manual: type the numbers"
BUILT_IN = (ADAPTIVE, MANUAL)

#: Environment override, so a test never writes to the real user directory and a lab
#: can point a shared folder at it if they want presets to travel between machines.
PRESET_ENV = "SMTRACKER_PRESETS"


def preset_dir() -> Path:
    """Where saved presets live. Created on demand, never in the install folder."""
    override = os.environ.get(PRESET_ENV)
    if override:
        return Path(override).expanduser()
    # Per-user, per-platform, and writable by the person running the tool. Windows gives
    # APPDATA; everything else follows the XDG convention with a plain-dotfile fallback.
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "SMTracker" / "presets"
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "smtracker" / "presets"


@dataclass(frozen=True)
class SavedPreset:
    """One saved set of detection and linking settings."""

    name: str
    #: What it is for, in the user's words. Not optional in spirit: the field may be
    #: empty, but the window asks, because a nameless number is unusable later.
    comment: str = ""
    #: HOW the threshold was arrived at: "computed from movies (…)" or "typed". A
    #: preset that does not say cannot be told apart from a guess six months on.
    provenance: str = ""
    settings: dict = field(default_factory=dict)

    def summary(self) -> str:
        bits = [f"{k} {v}" for k, v in sorted(self.settings.items())]
        return f"{self.name}: {', '.join(bits)}" + (
            f": {self.comment}" if self.comment else "")


def _path_for(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in " ._-").strip()
    if not safe:
        raise PresetError(
            f"{name!r} has no usable characters for a filename; a preset needs a name "
            f"somebody can recognise in a list")
    return preset_dir() / f"{safe}.json"


def _read_folder():
    """Every file in the preset folder, split into readable and not."""
    folder = preset_dir()
    if not folder.is_dir():
        return [], []
    good, broken = [], []
    for path in sorted(folder.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            good.append(SavedPreset(
                name=str(payload.get("name") or path.stem),
                comment=str(payload.get("comment", "")),
                provenance=str(payload.get("provenance", "")),
                settings=dict(payload.get("settings", {}))))
        except Exception as exc:                    # noqa: BLE001 - kept and named
            broken.append(f"{path.name} ({type(exc).__name__}: {exc})")
    return good, broken


def list_presets() -> tuple[SavedPreset, ...]:
    """Every readable saved preset. A corrupt file is skipped, never fatal.

    A corrupt preset must not stop the window opening. It must not VANISH either, so
    `broken_presets` names them and the window reports both.
    """
    good, _ = _read_folder()
    return tuple(good)


def broken_presets() -> tuple[str, ...]:
    """Preset files that could not be read, each with the reason.

    SEPARATE from `list_presets` rather than smuggled onto its result: the caller that
    fills a dropdown wants the usable ones, and the caller that tells the user something
    is wrong wants these. Attaching them to the same tuple served neither.
    """
    _, broken = _read_folder()
    return tuple(broken)


def load_preset(name: str) -> SavedPreset:
    path = _path_for(name)
    if not path.is_file():
        have = ", ".join(p.name for p in list_presets()) or "none saved"
        raise PresetError(f"no preset named {name!r}; have {have}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise PresetError(f"{path}: not readable JSON ({exc})") from exc
    return SavedPreset(name=str(payload.get("name") or path.stem),
                       comment=str(payload.get("comment", "")),
                       provenance=str(payload.get("provenance", "")),
                       settings=dict(payload.get("settings", {})))


def save_preset(name: str, settings: dict, *, comment: str = "",
                provenance: str = "", overwrite: bool = False) -> Path:
    """Write a preset. Refuses to replace one silently.

    Overwriting is explicit because two presets under one name is exactly the ambiguity
    this project refuses everywhere else: afterwards nobody can say which run used which
    numbers.
    """
    if name in BUILT_IN:
        raise PresetError(
            f"{name!r} is one of the built-in choices ({', '.join(BUILT_IN)}) and "
            f"describes how the numbers are chosen, not what they are; pick "
            f"another name")
    if not settings:
        raise PresetError("refusing to save a preset with no settings in it")
    path = _path_for(name)
    if path.exists() and not overwrite:
        raise PresetError(
            f"a preset named {name!r} already exists at {path}. Saving over it would "
            f"leave two different settings under one name, which nothing downstream "
            f"could tell apart: pass overwrite=True to replace it deliberately")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(SavedPreset(name=name, comment=comment, provenance=provenance,
                                 settings=dict(settings)))
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def delete_preset(name: str) -> None:
    path = _path_for(name)
    if not path.is_file():
        raise PresetError(f"no preset named {name!r} to delete")
    path.unlink()
