"""The running-job card and the finished pop-up, shared by every window.

Asked for (13 Oct 2026): a more obvious way to see that something is running and how far
it is ("condition 5/10 ... total 5/40 ... a bar filling"), and a pop-up when something
that ran for long has finished.

THIN QT: the words and numbers are `smtracker.run_progress`'s payloads; this paints them.
- `RunProgress` -- a card under a window's header, hidden while nothing runs: a spinner
  and the job's title, a bar filling over the whole job (busy without a fill while the
  size is unknown), the group and unit in hand, and the time so far with an estimate.
- `FinishedDialog` -- an in-window `OverlayDialog` (the UI designer's), never a `QDialog`: a modal
  OS dialog would block a headless test forever. The window also flashes on the taskbar
  (`QApplication.alert`), so a finished job behind another program is still seen.
"""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout

from .. import theme
from ..theme import typography as ty
from ...run_progress import NOTIFY_AFTER_S
from .controls import Button
from .dialogs import CloseButton, OverlayDialog
from .primitives import Card, Progress, Spinner, label
from .settings import CUT_INK

__all__ = ["FinishedDialog", "RunProgress", "notify_after_s", "show_finished"]


class RunProgress(Card):
    """The job in hand: title, bar, detail, timing. ``set_payload(None)`` hides it."""

    def __init__(self, t, parent=None):
        super().__init__(t, shadowed=False, parent=parent)
        self.t = t
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 12)
        lay.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(8)
        self.spinner = Spinner(t.foreground, 16)
        head.addWidget(self.spinner)
        self.title = label("", ty.ui("sm", "semibold"), t.foreground)
        head.addWidget(self.title)
        head.addStretch(1)
        self.timing = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        head.addWidget(self.timing)
        lay.addLayout(head)
        self.bar = Progress(t, 0.0)
        lay.addWidget(self.bar)
        self.detail = label("", ty.ui("xs", tabular=True), t.muted_foreground)
        self.detail.setWordWrap(True)
        lay.addWidget(self.detail)
        self.setVisible(False)

    def set_payload(self, payload: dict | None) -> None:
        if not payload:
            self.setVisible(False)
            return
        self.title.setText(payload.get("title", ""))
        self.timing.setText(payload.get("timing", ""))
        self.detail.setText(payload.get("detail", ""))
        fraction = payload.get("fraction")
        self.bar.setVisible(fraction is not None)
        if fraction is not None:
            self.bar.set_value(float(fraction))
        self.setVisible(True)


class FinishedDialog(OverlayDialog):
    """A job that ran for long has ended: what it did, how long it took, one button."""

    def __init__(self, t, payload: dict, *, content=None, parent=None):
        super().__init__(t, content, parent=parent)
        self._card.setMinimumWidth(480)
        self._card.setMaximumWidth(720)
        head = QHBoxLayout()
        title = label(payload.get("title", "Finished"), ty.ui("lg", "semibold"),
                      CUT_INK if payload.get("failed") else t.foreground)
        head.addWidget(title)
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        lines = list(payload.get("lines") or [])
        if lines:
            body = label("\n".join(lines[:12]) + ("\n…" if len(lines) > 12 else ""),
                         ty.ui("sm"), t.foreground)
            body.setWordWrap(True)
            body.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.card_lay.addWidget(body)
        if payload.get("elapsed"):
            self.card_lay.addWidget(label(payload["elapsed"], ty.ui("xs", tabular=True),
                                          t.muted_foreground))
        foot = QHBoxLayout()
        foot.addStretch(1)
        ok = Button(t, "OK", variant="default")
        ok.clicked.connect(self._dismiss)
        foot.addWidget(ok)
        self.card_lay.addLayout(foot)


def notify_after_s() -> float:
    """How long a job must run before its end gets a pop-up: the value the Options screen
    SAVED (``ui.json["options"]["notify_after_s"]``, `options.notify_after`), else a
    top-level ``notify_after_s`` from before the Options screen, else
    `run_progress.NOTIFY_AFTER_S`. Every window reads this one copy. (Until 3 Nov 2026 it
    read only the top-level key, so a value saved in Options did nothing.)"""
    from ... import options as opts

    saved = opts.notify_after(theme.remembered(opts.OPTIONS_KEY))
    if saved is not None:
        return saved
    value = theme.remembered("notify_after_s")
    try:
        return float(value) if value is not None and float(value) >= 0 else NOTIFY_AFTER_S
    except (TypeError, ValueError):
        return NOTIFY_AFTER_S


def show_finished(window, t, payload: dict | None):
    """Open the pop-up over ``window``'s content (when there is a payload) and flash the
    taskbar. Returns the dialog, or None."""
    if not payload:
        return None
    QApplication.alert(window)
    dialog = FinishedDialog(t, payload, content=window.centralWidget(), parent=window)
    dialog.closed.connect(dialog.deleteLater)
    dialog.open()
    return dialog
