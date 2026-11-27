"""The update notice and the check's wiring (thin Qt): everything it says and decides is `smtracker.updates`'.

Three things live here, and each only paints or starts what the headless module decided:
- `start_check(dialog)`: at the start, in a background thread (the window never waits for it), asks `updates.run_check`
  and, when a newer release is out, has the dialog show an `UpdateNotice`. Silent on every other answer.
- `check_for_updates(parent, t)`: the Help menu's third entry. Checks at once and ALWAYS answers.
- `UpdateNotice` / `UpdateDialog` and the install flow (`install`): "Download and install" downloads and checks the
  installer with progress and Cancel (`updates.download_installer`), says "SMTracker closes and the installer
  starts", starts it and quits the application. Nothing installs without that button; elsewhere than Windows it opens the
  release page.

QThread is not used: a plain daemon thread hands its answer to the window through a signal on a QObject the window owns
(a QThread destroyed while running aborts Qt), and an answer for a window already closed is dropped.
"""
from __future__ import annotations

import threading

from qtpy.QtCore import QObject, Qt, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (QApplication, QDialog, QFrame, QHBoxLayout, QMessageBox, QPlainTextEdit,
                            QVBoxLayout)

from ... import updates
from ..theme import typography as ty
from .controls import Button
from .primitives import Card, Progress, label

__all__ = ["UpdateNotice", "UpdateDialog", "DownloadDialog", "start_check", "check_for_updates", "install"]


#: The notes box's height: about six lines; longer notes scroll.
NOTES_HEIGHT = 96


class _Relay(QObject):
    """Carries one background answer to the window's thread."""

    done = Signal(object)
    progress = Signal(object)


def _background(owner, work, on_done) -> _Relay:
    """Run ``work()`` in a thread and call ``on_done(result)`` in the window's thread. An exception is DELIVERED as
    the result (the receiver shows it), never swallowed. The relay is a child of ``owner`` and returned."""
    relay = _Relay(owner)
    relay.done.connect(on_done)

    def run():
        try:
            result = work(relay)
        except Exception as exc:                                  # handed to the window, which says it
            result = exc
        try:
            relay.done.emit(result)
        except RuntimeError:                                      # the window is gone: nothing to tell
            pass

    threading.Thread(target=run, name="smtracker-updates", daemon=True).start()
    return relay


def _open_page(url: str, parent) -> None:
    if not QDesktopServices.openUrl(QUrl(url)):
        QMessageBox.information(parent, "Update", f"Open this address in a browser:\n{url}")


def _notes_text(release: updates.Release) -> str:
    notes = updates.short_notes(release.notes)
    return notes or "No release notes were published."


class UpdateNotice(Card):
    """A small notice inside a window: the title, the release notes, Download and install, Not now."""

    def __init__(self, t, release: updates.Release, *, on_install=None, on_later=None, parent=None):
        super().__init__(t, shadowed=False, parent=parent)
        self.t = t
        self.release = release
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(8)
        self.title_label = label(updates.words_for("newer", version=release.version), ty.ui("sm", "semibold"),
                                 t.foreground)
        self.title_label.setWordWrap(True)
        lay.addWidget(self.title_label)
        self.have_label = label(f"You have {updates.current_version()}.", ty.ui("xs"), t.muted_foreground)
        lay.addWidget(self.have_label)
        # a read-only text box of a fixed height, not a wrapped label: a wrapped label inside a card is squeezed
        # to a line or two when the window has no height to spare, and the longer notes scroll here
        self.notes_view = QPlainTextEdit(_notes_text(release))
        self.notes_view.setReadOnly(True)
        self.notes_view.setFrameShape(QFrame.NoFrame)
        self.notes_view.setFont(ty.ui("xs"))
        self.notes_view.setStyleSheet(f"QPlainTextEdit {{ color:{t.muted_foreground}; background:transparent; }}")
        self.notes_view.setFixedHeight(NOTES_HEIGHT)
        lay.addWidget(self.notes_view)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addStretch(1)
        self.later_button = Button(t, "Not now", variant="ghost", size="sm")
        self.later_button.clicked.connect(on_later or self.dismiss)
        row.addWidget(self.later_button)
        self.install_button = Button(t, "Download and install" if updates.installs_here() else "Open the release page",
                                     variant="default", size="sm")
        self.install_button.clicked.connect(on_install or (lambda: install(self.window(), t, release)))
        row.addWidget(self.install_button)
        lay.addLayout(row)

    def dismiss(self) -> None:
        """Not now: gone until the next day's check."""
        self.setVisible(False)
        self.deleteLater()


class UpdateDialog(QDialog):
    """The same notice as a dialog, for the Help entry."""

    def __init__(self, t, release: updates.Release, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Check for updates")
        self.setStyleSheet(f"QDialog {{ background:{t.background}; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 16)
        self.notice = UpdateNotice(t, release, on_later=self.reject, parent=self,
                                   on_install=lambda: (self.accept(), install(parent or self, t, release)))
        lay.addWidget(self.notice)
        self.setMinimumWidth(460)


class DownloadDialog(QDialog):
    """The installer arriving: a bar, what is happening, Cancel. Closing it is Cancel."""

    def __init__(self, t, release: updates.Release, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Downloading the update")
        self.setStyleSheet(f"QDialog {{ background:{t.background}; }}")
        self.cancel_event = threading.Event()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 16)
        lay.setSpacing(10)
        lay.addWidget(label(f"Downloading SMTracker {release.version}", ty.ui("sm", "semibold"), t.foreground))
        self.bar = Progress(t, 0.0)
        lay.addWidget(self.bar)
        self.detail = label("Starting…", ty.ui("xs", tabular=True), t.muted_foreground)
        self.detail.setWordWrap(True)
        lay.addWidget(self.detail)
        row = QHBoxLayout()
        row.addStretch(1)
        self.cancel_button = Button(t, "Cancel", variant="outline", size="sm")
        self.cancel_button.clicked.connect(self.reject)
        row.addWidget(self.cancel_button)
        lay.addLayout(row)
        self.setMinimumWidth(420)

    def set_progress(self, done: int, total: int) -> None:
        from ... import segpack

        self.bar.set_value(done / total if total else 0.0)
        self.detail.setText(f"{segpack.sizes_words(done)} of {segpack.sizes_words(total)}")

    def reject(self) -> None:
        self.cancel_event.set()
        super().reject()


# ------------------------------------------------------------------- the flows
def _confirm_restart(parent) -> bool:
    box = QMessageBox(parent)
    box.setWindowTitle("Install the update")
    box.setText("SMTracker closes and the installer starts.")
    box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    return box.exec() == QMessageBox.Ok


def _quit_application() -> None:
    app = QApplication.instance()
    if app is not None:
        app.closeAllWindows()
        app.quit()


def _refused(parent, exc: Exception, release: updates.Release) -> None:
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle("Update")
    box.setText(str(exc))
    page = box.addButton("Open the release page", QMessageBox.AcceptRole)
    box.addButton("Close", QMessageBox.RejectRole)
    box.exec()
    if box.clickedButton() is page:
        _open_page(release.page_url, parent)


def install(parent, t, release: updates.Release) -> None:
    """The button: download and check the installer, say what happens next, start it and quit. Elsewhere than
    Windows, open the release page."""
    if not updates.installs_here():
        _open_page(release.page_url, parent)
        return
    dialog = DownloadDialog(t, release, parent)

    def work(relay):
        return updates.download_installer(release, on_progress=lambda done, total: relay.progress.emit((done, total)),
                                          cancel=dialog.cancel_event)

    def done(result):
        if dialog.cancel_event.is_set():
            return
        dialog.accept()
        if isinstance(result, Exception):
            from ...segpack import PackCancelled

            if not isinstance(result, PackCancelled):
                _refused(parent, result, release)
            return
        if _confirm_restart(parent):
            try:
                updates.start_installer(result)
            except updates.UpdateError as exc:
                _refused(parent, exc, release)
                return
            _quit_application()

    relay = _background(dialog, work, done)
    relay.progress.connect(lambda pair: dialog.set_progress(*pair))
    dialog.exec()


def start_check(dialog) -> None:
    """At the start: ask in the background (once a day, frozen builds, option on) and let ``dialog`` show the notice
    when a newer release is out. Silent otherwise; a failure went to the crash log."""
    from ... import options
    from .. import theme

    enabled = options.updates_at_start(theme.remembered(options.OPTIONS_KEY))
    if not enabled or not updates.automatic_allowed():
        return

    def got(result):
        if isinstance(result, updates.Outcome) and result.kind == "newer":
            dialog.show_update(result.release)

    _background(dialog, lambda relay: updates.run_check(enabled=True), got)


def check_for_updates(parent, t) -> None:
    """The Help menu's entry: checks at once and always answers."""
    if updates.runs_from_source():
        QMessageBox.information(parent, "Check for updates", updates.words_for("source"))
        return
    QApplication.setOverrideCursor(Qt.WaitCursor)

    def got(result):
        QApplication.restoreOverrideCursor()
        if isinstance(result, Exception):
            QMessageBox.warning(parent, "Check for updates", updates.words_for("failed", reason=str(result)))
        elif result.kind == "newer":
            UpdateDialog(t, result.release, parent).exec()
        elif result.kind == "failed":
            QMessageBox.warning(parent, "Check for updates", result.words)
        else:
            QMessageBox.information(parent, "Check for updates", result.words)

    holder = getattr(parent, "_update_relays", None)
    if holder is None:
        holder = parent._update_relays = []
    holder.append(_background(parent, lambda relay: updates.check_now(), got))
