"""What went wrong, written to a file a lab member can send.

WHY THIS EXISTS (10 Nov 2026, the software review's item 2). A packaged app's console closes with it: an uncaught
error went to a window that is gone, and a Qt abort (a `QThread` destroyed while running, a `QDialog` before a
`QApplication`) ends the process with no traceback at all. The only report a user could make was "it closed".
`buildstamp` answers WHICH code is running; this answers WHAT it said before it stopped, in one file beside the UI's
own settings, and packs that file with the build and the versions into one zip to send.

WHAT IS WRITTEN, to `log_path()` (``%APPDATA%/SMTracker/logs/smtracker.log`` on Windows; `LOG_DIR_ENV` moves it):

  - a SESSION HEADER at every start: the time, `buildstamp.build_stamp()`, Python, the platform, the arguments;
  - every UNCAUGHT EXCEPTION, with its traceback: the main thread's (`sys.excepthook`, which is also where PySide
    sends an error raised inside a slot) and any other thread's (`threading.excepthook`);
  - Qt's own WARNINGS, CRITICALS AND FATALS (`install_qt`): the last line Qt says before it aborts is its reason
    ("QThread: Destroyed while thread is still running"), and with a fatal the Python stack at that moment;
  - the program's own PYTHON WARNINGS (`warnings.showwarning`, 27 Nov 2026), each text at most `QT_REPEATS`
    times: since the packaged smtracker.exe is a Windows app with no console, the warnings this project raises
    instead of swallowing ("said, not swallowed") had nowhere else to go. smtracker-console.exe still shows them;
  - on a hard crash (a segfault, an abort) the PYTHON STACK OF EVERY THREAD (`faulthandler`), which is the only
    trace such an end leaves.

Every hook CHAINS: what was installed before still runs, so the console of a source run reads as it always did.

IT CAN NEVER BE THE FAILURE. A window that does not open because its log could not be written is a worse defect
than the one the log is for (`buildstamp`'s rule): every function here returns quietly on an unwritable folder, and
`install` returns None then. That is the second place in this project where swallowing is right, and it is as
narrow: the cost of failing is a missing log.

NO DATA. The log holds error texts and tracebacks, so FOLDER AND FILE NAMES appear in it; the report adds the build,
the package versions and the UI's settings file (recent folders, options). No tracks, no images, no results. The
window that makes a report says so before it is sent (`REPORT_HOLDS`).

A hook installed AFTER ours by another library belongs to that library: napari's viewer shows an error in its own
notification area. The analysis windows are plain Qt and reach this log.

Imports with Qt blocked: `install_qt` imports Qt itself, and does nothing where there is none.
"""

from __future__ import annotations

import atexit
import datetime as _dt
import faulthandler
import os
import platform
import sys
import threading
import traceback
import zipfile
from pathlib import Path

__all__ = ["LOG_DIR_ENV", "LOG_NAME", "MAX_BYTES", "ISSUE_BUTTON", "PACKAGES", "REPORT_HOLDS", "about_text", "failed_session", "install",
           "install_qt", "issue_url", "log_dir", "log_path", "note", "report_bundle", "report_name", "saved_text", "uninstall"]

#: Environment override for the log folder, so a test never writes the real user's log -- the same pattern as
#: `ui.theme.UI_SETTINGS_ENV` and `presets.PRESET_ENV`.
LOG_DIR_ENV = "SMTRACKER_LOG_DIR"

LOG_NAME = "smtracker.log"
#: The log before the last rotation: one generation is kept, so a report always holds the session that failed.
OLD_NAME = "smtracker.old.log"

#: The log is rotated at a start that finds it larger than this. A session writes a few hundred bytes unless
#: something fails, so this is months of ordinary use.
MAX_BYTES = 1_000_000

#: One Qt message is written this many times a session, then counted only: a warning raised on every repaint
#: would otherwise bury the line that matters and fill the disk.
QT_REPEATS = 3

#: The packages whose versions a report names: the ones a fit, a file reader or a window depends on.
PACKAGES = ("numpy", "scipy", "pandas", "PySide6", "qtpy", "napari", "saspt", "tifffile", "nd2", "mat73",
            "matplotlib", "pyqtgraph", "scikit-image", "torch", "omnipose", "cellpose")

#: What a report holds, said to the person before they send it.
REPORT_HOLDS = ("The report holds the error log, which build this is, the versions of Python and its packages, "
                "and SMTracker's settings file. Folder and file names appear in the log and the settings. "
                "It holds no tracks, no images and no results.")

_state: dict = {}
_lock = threading.Lock()


def log_dir() -> Path:
    """The per-user folder the log is kept in: beside ``ui.json``, never in the install folder."""
    override = os.environ.get(LOG_DIR_ENV)
    if override:
        return Path(override).expanduser()
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "SMTracker" / "logs"
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "smtracker" / "logs"


def log_path() -> Path:
    return log_dir() / LOG_NAME


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _header(argv) -> str:
    try:
        from .buildstamp import build_stamp

        stamp = build_stamp()
    except Exception:                                      # noqa: BLE001 - the log must open without a name
        stamp = "unknown build"
    args = " ".join(str(a) for a in (sys.argv[1:] if argv is None else argv)) or "(no arguments)"
    return (f"\n==== {_now()}  SMTracker {stamp}\n"
            f"     Python {platform.python_version()} on {platform.platform()}"
            f"{', frozen' if getattr(sys, 'frozen', False) else ''}\n"
            f"     arguments: {args}\n")


def note(text: str) -> None:
    """Write one dated entry to the open log; nothing where no log is open. Never raises."""
    handle = _state.get("handle")
    if handle is None:
        return
    try:
        with _lock:
            handle.write(f"[{_now()}] {text.rstrip()}\n")
            handle.flush()
    except (OSError, ValueError):
        pass


def install(argv=None, *, fault: bool = True) -> Path | None:
    """Open the log for this session and send every uncaught error to it. Returns the log's path, or None where it
    could not be opened (the app runs on without one). A second call in one process returns the open log.
    `faulthandler` is pointed at the log only where nobody enabled it before (a test runner, ``-X faulthandler``:
    theirs is left alone), and never with ``fault`` False."""
    if _state.get("handle") is not None:
        return _state["path"]
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size > MAX_BYTES:
            os.replace(path, path.with_name(OLD_NAME))
        handle = open(path, "a", encoding="utf-8", errors="replace")       # noqa: SIM115 - held for the session
        handle.write(_header(argv))
        handle.flush()
    except OSError:
        return None
    _state.update(handle=handle, path=path, excepthook=sys.excepthook, thread_hook=threading.excepthook, qt_seen={})

    def on_error(kind, value, tb):
        note("UNCAUGHT ERROR\n" + "".join(traceback.format_exception(kind, value, tb)))
        previous = _state.get("excepthook") or sys.__excepthook__
        previous(kind, value, tb)

    def on_thread_error(args):
        if args.exc_type is not SystemExit:
            name = getattr(args.thread, "name", "?")
            note(f"UNCAUGHT ERROR in thread {name}\n"
                 + "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))
        previous = _state.get("thread_hook") or threading.__excepthook__
        previous(args)

    sys.excepthook, threading.excepthook = on_error, on_thread_error
    _state.update(on_error=on_error, on_thread_error=on_thread_error)

    import warnings

    _state.update(showwarning=warnings.showwarning, warn_seen={})

    def on_warning(message, category, filename, lineno, file=None, line=None):
        # the program's own warnings ("said, not swallowed"): the Windows app has no console to say them in
        text = f"{getattr(category, '__name__', category)}: {message}"
        seen = _state.setdefault("warn_seen", {})
        seen[text] = seen.get(text, 0) + 1
        if seen[text] <= QT_REPEATS:
            note(f"WARNING {text} ({filename}:{lineno})")
        previous = _state.get("showwarning")
        if previous is not None:
            previous(message, category, filename, lineno, file, line)

    warnings.showwarning = on_warning
    _state["on_warning"] = on_warning
    if fault and not faulthandler.is_enabled():
        try:
            # the Python stack of every thread on a segfault or an abort: the only trace a Qt abort leaves
            faulthandler.enable(file=handle, all_threads=True)
            _state["fault"] = True
        except (OSError, ValueError, RuntimeError):
            pass
    return path


def install_qt() -> bool:
    """Send Qt's warnings, criticals and fatals to the open log; what handled them before still does (Qt's own
    printing to stderr where nothing did). False where no log is open or there is no Qt. Never raises."""
    if _state.get("handle") is None or "qt_handler" in _state:
        return "qt_handler" in _state
    try:
        from qtpy.QtCore import QtMsgType, qInstallMessageHandler
    except Exception:                                      # noqa: BLE001 - no Qt, or a binding that will not load
        return False
    words = {QtMsgType.QtWarningMsg: "warning", QtMsgType.QtCriticalMsg: "critical", QtMsgType.QtFatalMsg: "FATAL"}
    seen = _state["qt_seen"]

    def handler(kind, context, message):
        word = words.get(kind)
        if word is not None:
            count = seen.get(message, 0) + 1
            if len(seen) < 2000 or message in seen:
                seen[message] = count
            if count <= QT_REPEATS or word == "FATAL":
                where = ""
                try:
                    if context is not None and context.file:
                        where = f" ({context.file}:{context.line})"
                except Exception:                          # noqa: BLE001 - a context without a file
                    where = ""
                note(f"Qt {word}: {message}{where}"
                     + (" [said again; not written from here on]" if count == QT_REPEATS and word != "FATAL" else "")
                     # Qt ends the process after a fatal, and on Windows that end raises no signal `faulthandler`
                     # could catch: the Python stack is written HERE, while there still is one
                     + ("\nPython stack when Qt said so:\n" + "".join(traceback.format_stack()[:-1])
                        if word == "FATAL" else ""))
        previous = _state.get("qt_previous")
        if previous is not None:
            previous(kind, context, message)
        elif sys.stderr is not None:
            try:
                print(message, file=sys.stderr)
            except (OSError, ValueError):
                pass

    try:
        _state["qt_previous"] = qInstallMessageHandler(handler)
    except Exception:                                      # noqa: BLE001
        return False
    _state["qt_handler"] = handler
    # Qt may still speak while the interpreter is being taken down, and a Python handler must not be called then
    atexit.register(_restore_qt)
    return True


def _restore_qt() -> None:
    if _state.pop("qt_handler", None) is None:
        return
    try:
        from qtpy.QtCore import qInstallMessageHandler

        qInstallMessageHandler(_state.pop("qt_previous", None))
    except Exception:                                      # noqa: BLE001
        pass


def uninstall() -> None:
    """Close the log and put back what `install` replaced (for tests; a session keeps its log to the end)."""
    handle = _state.get("handle")
    if handle is None:
        return
    _restore_qt()
    if _state.get("fault"):
        try:
            faulthandler.disable()
        except (OSError, ValueError, RuntimeError):
            pass
    if sys.excepthook is _state.get("on_error"):
        sys.excepthook = _state["excepthook"]
    if threading.excepthook is _state.get("on_thread_error"):
        threading.excepthook = _state["thread_hook"]
    import warnings

    if warnings.showwarning is _state.get("on_warning"):
        warnings.showwarning = _state["showwarning"]
    try:
        handle.close()
    except OSError:
        pass
    _state.clear()


# -- the report ------------------------------------------------------------------------------------------------ #
def about_text() -> str:
    """Which build, which Python, which packages: the first file of a report, and what `--report` prints."""
    from importlib import metadata

    try:
        from .buildstamp import version_text

        lines = [version_text()]
    except Exception:                                      # noqa: BLE001
        lines = ["SMTracker, unknown build"]
    lines += [f"written {_now()}",
              f"Python {platform.python_version()} ({platform.architecture()[0]}), {platform.platform()}",
              f"executable: {sys.executable}",
              f"log: {log_path()}", "", "packages:"]
    for name in PACKAGES:
        try:
            lines.append(f"  {name} {metadata.version(name)}")
        except Exception:                                  # noqa: BLE001 - not installed, or no metadata in a bundle
            lines.append(f"  {name} (not found)")
    return "\n".join(lines) + "\n"


def report_name(now: _dt.datetime | None = None) -> str:
    return f"SMTracker report {(now or _dt.datetime.now()).strftime('%Y-%m-%d %H%M')}.zip"


def saved_text(path) -> tuple:
    """The title and the sentences a window shows once a report is written: where it is, what it holds."""
    return ("Report saved",
            f"{path}\n\n{REPORT_HOLDS}\n\n"
            f"{ISSUE_BUTTON} opens a public report on GitHub with the end of the log already in it; drag this file "
            "into it to add the rest. Nothing is sent until you press Submit there.")


# ------------------------------------------------------------------ the report on GitHub (27 Nov 2026)
# Asked: "why not send a log or something to my git?", then "u are right": no key in the program (the code is public,
# and GitHub cancels a key it finds there), so the button OPENS the repository's new-issue page with the version and
# the end of the log filled in; the person reads it, deletes what they would rather not show, and submits it with
# their own GitHub account. Nothing leaves the computer unseen.

#: The button's words, in every window that shows a saved report.
ISSUE_BUTTON = "Report on GitHub"

#: GitHub opens a new-issue address of up to about 8,000 characters, and encoding can make a log's characters up to
#: three times longer: the log's part is cut to this many from its end.
ISSUE_LOG_CHARS = 2000

#: What a log entry worth reporting starts with (`note`'s callers: errors, the program's warnings, Qt's messages).
_ENTRY_MARKS = ("UNCAUGHT ERROR", "] WARNING ", "] Qt ")


def failed_session(text: str, limit: int = ISSUE_LOG_CHARS) -> str:
    """The most recent session of a log that holds an error or a warning (the one that failed, usually not the session
    writing the report), else the last session; at most ``limit`` characters from its end."""
    sessions = [s for s in text.split("\n==== ") if s.strip()]
    if not sessions:
        return ""
    chosen = next((s for s in reversed(sessions) if any(m in s for m in _ENTRY_MARKS)), sessions[-1]).strip()
    return chosen if len(chosen) <= limit else "...\n" + chosen[-limit:]


def issue_url(repository: str | None = None, log_text: str | None = None) -> str:
    """The repository's new-issue address with the title, the version, the system and the end of the log filled in.
    No path of the program's own goes in (`about_text` has them; the zip carries it); the log's own folder names
    are said to be public in the text itself."""
    from urllib.parse import urlencode

    from . import __version__

    if repository is None:
        from .updates import REPOSITORY as repository
    if log_text is None:
        try:
            log_text = log_path().read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_text = ""
    try:
        from .buildstamp import build_stamp

        stamp = build_stamp()
    except Exception:                                      # noqa: BLE001 - a report must open without a name
        stamp = "unknown build"
    body = "\n".join([
        "**What were you doing when it went wrong?**", "", "",
        "**What did you expect, and what happened instead?**", "", "",
        f"**Version:** SMTracker {__version__}, {stamp}",
        f"**System:** {platform.platform()}, Python {platform.python_version()}", "",
        "**The end of the log.** Everything here is public, and a log shows folder and file names: delete any you "
        "would rather not show.",
        "```", failed_session(log_text) or "(the log is empty)", "```", "",
        "The whole report (Report a problem saves it as a zip) can be dragged into this box."])
    return f"https://github.com/{repository}/issues/new?" + urlencode(
        {"title": "Problem: ", "labels": "bug", "body": body})


def _settings_file() -> Path | None:
    try:
        from .ui.theme import settings_path

        path = settings_path()
        return path if path.is_file() else None
    except Exception:                                      # noqa: BLE001
        return None


def report_bundle(dest) -> Path:
    """Write the report: one zip holding ``about.txt`` (`about_text`), the log and the one before it, and the UI's
    settings file. ``dest`` is the zip to write, or a folder to write `report_name()` into. Returns the zip's path.
    Raises OSError where the zip cannot be written: this is something a person asked for, so it says why."""
    dest = Path(dest)
    if dest.is_dir() or dest.suffix.lower() != ".zip":
        dest.mkdir(parents=True, exist_ok=True)
        dest = dest / report_name()
    handle = _state.get("handle")
    if handle is not None:
        try:
            handle.flush()
        except (OSError, ValueError):
            pass
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("about.txt", about_text())
        found = False
        for name in (LOG_NAME, OLD_NAME):
            path = log_dir() / name
            try:
                if path.is_file():
                    bundle.writestr(name, path.read_text(encoding="utf-8", errors="replace"))
                    found = True
            except OSError as exc:
                bundle.writestr(f"{name}.unreadable.txt", str(exc))
        if not found:
            bundle.writestr("no_log.txt", f"No log was found in {log_dir()}.\n")
        settings = _settings_file()
        if settings is not None:
            try:
                bundle.writestr("ui.json", settings.read_text(encoding="utf-8", errors="replace"))
            except OSError as exc:
                bundle.writestr("ui.json.unreadable.txt", str(exc))
    return dest
