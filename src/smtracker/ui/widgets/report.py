"""The "Report a problem" action every window shares: ask where, write the report, say what it holds.

Thin: what a report is and what it holds is `smtracker.crashlog`'s (`report_bundle`, `saved_text`); this asks for
the place with the system's own save dialog and returns what happened, and each window says it in its own way (the
workspace in a notice over its page, the first window in a message box, since it has no page to lay one over).
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QUrl
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import QFileDialog, QMessageBox

from ... import crashlog

__all__ = ["save_report", "report_from_dialog", "open_issue"]


def _start_folder() -> Path:
    desktop = Path.home() / "Desktop"
    return desktop if desktop.is_dir() else Path.home()


def save_report(parent, dest=None) -> tuple:
    """Write the report to ``dest`` (asked for when None). Returns ``(path, title, body)``: the zip written and the
    words for it, or ``(None, title, why)`` where it could not be written, or ``(None, "", "")`` on Cancel."""
    if dest is None:
        dest, _ = QFileDialog.getSaveFileName(parent, "Save the report as",
                                              str(_start_folder() / crashlog.report_name()), "Zip file (*.zip)")
        if not dest:
            return None, "", ""
        if not str(dest).lower().endswith(".zip"):
            dest = f"{dest}.zip"
    try:
        path = crashlog.report_bundle(dest)
    except OSError as exc:
        return None, "The report could not be saved", f"{dest}\n\n{exc}"
    return (path, *crashlog.saved_text(path))


def open_issue(parent=None) -> bool:
    """Open the repository's new-issue page with the version and the end of the log filled in
    (`crashlog.issue_url`); nothing is sent until the person submits it there. Returns whether a browser took it."""
    url = crashlog.issue_url()
    if QDesktopServices.openUrl(QUrl(url)):
        return True
    QMessageBox.information(parent, crashlog.ISSUE_BUTTON, "Open this address in a browser:\n"
                            + url.split("?", 1)[0])
    return False


def report_from_dialog(parent, dest=None):
    """The action for a window without a page to lay a notice over: the save dialog, then a message box with
    "Report on GitHub" beside OK once the report is saved."""
    path, title, body = save_report(parent, dest)
    if title:
        box = QMessageBox(QMessageBox.Information if path else QMessageBox.Warning, title, body, QMessageBox.Ok,
                          parent)
        github = box.addButton(crashlog.ISSUE_BUTTON, QMessageBox.ActionRole) if path else None
        box.exec_() if hasattr(box, "exec_") else box.exec()
        if github is not None and box.clickedButton() is github:
            open_issue(parent)
    return path
