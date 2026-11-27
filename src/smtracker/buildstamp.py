"""Which code is running, short enough for a window title.

WHY THIS EXISTS. This repository can be started three ways that look identical on screen:
the editable install (always current), and two PyInstaller bundles under `packaging/`
that were frozen on their build day and can never contain anything written afterwards.
Nothing in any window said which was open.

That cost the better part of a day. A defect was fixed, verified from source through
every entry point, and reported as still present -- with a screenshot that could not
distinguish "the fix is not in this build" from "the fix does not work". Every check run
in response verified the SOURCE rather than the BINARY, which is the one thing a
screenshot could have settled in a glance.

So the stamp goes in the TITLE BAR, where a screenshot always catches it. It answers one
question -- which code is this -- and it must never be able to raise or to be slow, since
a window that fails to open because it could not name itself is a worse defect than the
one it was added to diagnose.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

from .childproc import NO_WINDOW

__all__ = ["BUILD_INFO_NAME", "build_info", "build_stamp", "code_root", "newest_edit",
           "version_text"]

#: What `packaging/build.py` writes beside this module inside a bundle: the commit it was
#: built from. A bundle has no repository and no git, so without this file a build sent to
#: another machine cannot say which code it is -- asked for directly ("dont forget which
#: version is it").
BUILD_INFO_NAME = "build_info.json"

#: Wall-clock seconds allowed for the `git` call. A repository on a slow or disconnected
#: network drive must not hold a window shut, so this is short and its failure is a
#: missing word in a title rather than an error.
GIT_TIMEOUT_S = 2.0


def code_root() -> Path:
    """The folder the running `smtracker` package was imported from."""
    return Path(__file__).resolve().parent


def newest_edit(root: Path | None = None) -> _dt.datetime | None:
    """When the newest module here was last written, or None if that cannot be read.

    The package's own mtime is not enough: installing or copying rewrites a directory's
    timestamp without changing a line of code. The newest `.py` is what actually moves
    when someone edits, which is the question being asked.
    """
    root = root or code_root()
    try:
        stamps = [q.stat().st_mtime for q in root.glob("*.py")]
    except OSError:
        return None
    if not stamps:
        return None
    return _dt.datetime.fromtimestamp(max(stamps))


def _commit(root: Path) -> str:
    """The short commit of the repository the code sits in, or "" when there is none.

    A frozen bundle has no repository and no git, and that is an ORDINARY state rather
    than a failure: the edit time still identifies the build.
    """
    try:
        got = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_S, check=False, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return ""
    return got.stdout.strip() if got.returncode == 0 else ""


def build_info(root: Path | None = None) -> dict | None:
    """The stamp `packaging/build.py` wrote into a bundle, or None: a source run, or a
    bundle built before the stamp existed. Never raises."""
    try:
        path = (root or code_root()) / BUILD_INFO_NAME
        if not path.is_file():
            return None
        info = json.loads(path.read_text(encoding="utf-8"))
        return info if isinstance(info, dict) else None
    except (OSError, ValueError):
        return None


def _stamped(info: dict) -> str:
    """`frozen build 8621aea, built 08 Oct 2026 15:42` -- `+edits` after the commit when
    the tree it was built from had uncommitted changes, since the commit alone would then
    name code that is not quite what shipped."""
    commit = str(info.get("commit") or "no commit")
    if info.get("dirty"):
        commit += "+edits"
    # THE VARIANT: ONE build ships (ruled 26 Nov 2026 evening, PyTorch and Cellpose downloaded on demand), so the shipped
    # one ("lite" inside build.py) names no variant; only the developer's Normal build says so. A stamp without the key
    # (a build before the variants) names none either.
    if info.get("variant") and info["variant"] != "lite":
        commit += f" ({str(info['variant']).capitalize()})"
    try:
        built = _dt.datetime.fromisoformat(str(info.get("built"))).strftime("%d %b %Y %H:%M")
    except ValueError:
        built = "unknown time"
    return f"frozen build {commit}, built {built}"


def build_stamp(*, frozen: bool | None = None) -> str:
    """One short phrase naming the running code. Never raises.

    ``frozen`` is taken from the interpreter when not given, so a bundle says so in the
    title instead of leaving the reader to infer it from a missing commit.
    """
    try:
        import sys

        if frozen is None:
            frozen = bool(getattr(sys, "frozen", False))
        root = code_root()
        if frozen:
            info = build_info(root)
            if info is not None:
                return _stamped(info)
        edited = newest_edit(root)
        when = edited.strftime("%d %b %H:%M") if edited else "unknown"
        if frozen:
            return f"frozen build, {when}"
        commit = _commit(root)
        return f"{commit} · {when}" if commit else f"source, {when}"
    except Exception:                                     # noqa: BLE001 - never fatal
        # A window must open even when it cannot name itself. This is the one place in
        # this project where swallowing is right, and it is narrow: the cost of failing
        # is a missing suffix in a title.
        return "unknown build"


def version_text() -> str:
    """What `smtracker --version` prints: the stamp, the stamp's details when this is a
    stamped build, and the folder the code was imported from."""
    import sys

    frozen = bool(getattr(sys, "frozen", False))
    lines = [f"SMTracker -- {build_stamp(frozen=frozen)}"]
    info = build_info() if frozen else None
    if info is not None:
        lines.append(f"commit {info.get('commit')} of {info.get('commit_date', '?')}, "
                     f"branch {info.get('branch', '?')}, version {info.get('version', '?')}")
        if info.get("variant"):
            lines.append(f"variant: {info['variant']}"
                         + (" (the classical segmenter only; no torch)" if info["variant"] == "lite"
                            else " (with Cellpose 4 and CUDA torch; Cellpose-SAM downloads at first use)"
                            if info["variant"] == "normal" else ""))
        if info.get("dirty"):
            lines.append("built from a tree with uncommitted edits to src/ or packaging/")
    lines.append(f"code: {code_root()}")
    return "\n".join(lines)
