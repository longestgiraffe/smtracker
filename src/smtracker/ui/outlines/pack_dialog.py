"""The segmentation pack's window: the offer, the download, and the place to see it and remove it.

Thin Qt: every rule and every word is in `smtracker.segpack` / `smtracker.segpack_view`. One in-window overlay
(`OverlayDialog`, like the other dialogs) with three faces:

- OFFER, where no pack is installed: what the pack is, the two builds (GPU support / CPU only, the one that fits this
  computer chosen, the reason said), the sizes, the folder and the free space on its drive, and [Download] [No];
- PROGRESS: a bar over the whole pack, the file and bytes, the time left, [Cancel] (what arrived is kept);
- STATUS, where a pack is installed (or damaged, or another version): where it is and what it holds, [Remove], [Close].

The download runs on a plain Python thread and reports through a lock-held snapshot polled by a timer, so no Qt object
is touched from the worker and no ``QThread`` can be destroyed while running.
"""
from __future__ import annotations

import threading
from pathlib import Path

from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import QFileDialog, QHBoxLayout, QVBoxLayout

from ... import segpack, segpack_view
from ..theme import typography as ty
from ..widgets.controls import Button, Segmented
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.primitives import Progress, label

__all__ = ["PackDialog"]


class PackDialog(OverlayDialog):
    """``installed`` is emitted with the pack's folder after a successful install; ``removed`` after a removal."""

    installed = Signal(str)
    removed = Signal()

    def __init__(self, t, *, content=None, parent=None, gpu=None, spec=None):
        super().__init__(t, content, parent=parent)
        self._spec = spec
        self._card.setMinimumWidth(520)
        self._card.setMaximumWidth(640)
        self._worker = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._error = None
        self._result = None
        self._pace = None
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._poll)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        self.title_label = label(segpack_view.OFFER_TITLE, ty.ui("lg", "semibold"), t.foreground)
        self.title_label.setWordWrap(True)
        head.addWidget(self.title_label, 1)
        self._close_btn = CloseButton(t)
        self._close_btn.clicked.connect(self._close_clicked)
        head.addWidget(self._close_btn)
        self.card_lay.addLayout(head)

        self.body_label = label("", ty.ui("sm"), t.muted_foreground)
        self.body_label.setWordWrap(True)
        self.card_lay.addWidget(self.body_label)
        self.choice = None
        self._rows = []
        self.choice_note = label("", ty.ui("xs"), t.muted_foreground)
        self.choice_note.setWordWrap(True)
        self.card_lay.addWidget(self.choice_note)
        self.size_label = label("", ty.ui("sm"), t.foreground)
        self.size_label.setWordWrap(True)
        self.card_lay.addWidget(self.size_label)
        self.folder_label = label("", ty.ui("sm"), t.foreground)
        self.folder_label.setWordWrap(True)
        self.card_lay.addWidget(self.folder_label)
        self.space_label = label("", ty.ui("xs"), t.muted_foreground)
        self.space_label.setWordWrap(True)
        self.card_lay.addWidget(self.space_label)
        self.bar = Progress(t, 0.0)
        self.card_lay.addWidget(self.bar)
        self.progress_label = label("", ty.ui("sm"), t.muted_foreground)
        self.progress_label.setWordWrap(True)
        self.card_lay.addWidget(self.progress_label)

        foot = QHBoxLayout()
        foot.setContentsMargins(0, 0, 0, 0)
        foot.setSpacing(8)
        self.change_button = Button(t, "Change folder…", size="sm")
        self.change_button.clicked.connect(self._pick_folder)
        foot.addWidget(self.change_button)
        foot.addStretch(1)
        self.remove_button = Button(t, "Remove the pack", variant="outline")
        self.remove_button.clicked.connect(self._remove)
        foot.addWidget(self.remove_button)
        self.no_button = Button(t, "No", variant="outline")
        self.no_button.clicked.connect(self._close_clicked)
        foot.addWidget(self.no_button)
        self.go_button = Button(t, "Download", variant="default")
        self.go_button.clicked.connect(self._start)
        foot.addWidget(self.go_button)
        self.card_lay.addLayout(foot)

        self._gpu = gpu
        self._flavour = None
        self._folder = None
        self._proposal_note = ""
        self._show_first()

    # -- which face ---------------------------------------------------------------- #
    def _show_first(self) -> None:
        remembered = segpack.remembered_pack()
        folder = Path(remembered["folder"]) if remembered else None
        st = segpack.status(folder) if folder is not None else None
        if st is not None and st.state != "absent":
            self._show_status(st)
        else:
            self._show_offer()

    def _set_visible(self, **flags) -> None:
        for name, on in flags.items():
            getattr(self, name).setVisible(bool(on))

    def _show_offer(self) -> None:
        self.title_label.setText(segpack_view.OFFER_TITLE)
        self.body_label.setText(segpack_view.OFFER_BODY + " " + segpack_view.WHY_NOT_NOW)
        try:
            rows = segpack_view.flavour_rows(self._gpu, spec=self._spec)
        except segpack.SegPackError as exc:
            self.body_label.setText(str(exc))
            self._set_visible(choice_note=0, size_label=0, folder_label=0, space_label=0, bar=0, progress_label=0,
                              change_button=0, remove_button=0, go_button=0)
            self.no_button.set_text("Close")
            return
        self._rows = rows
        if self.choice is not None:
            self.card_lay.removeWidget(self.choice)
            self.choice.deleteLater()
        self.choice = Segmented(self.t, [r["title"] for r in rows], current=0)
        self.choice.changed.connect(self._flavour_changed)
        self.card_lay.insertWidget(self.card_lay.indexOf(self.choice_note), self.choice)
        self._flavour_changed(0)
        self._set_visible(choice_note=1, size_label=1, folder_label=1, space_label=1, bar=0, progress_label=0,
                          change_button=1, remove_button=0, go_button=1, no_button=1)
        self.no_button.set_text("No")

    def _flavour_changed(self, index: int) -> None:
        row = self._rows[index]
        self._flavour = row["flavour"]
        recommended = next(r for r in self._rows if r["recommended"])
        self.choice_note.setText((recommended["reason"] + " ") if recommended["reason"] else "")
        self.size_label.setText(f"{row['title']}: {row['words']}")
        folder, why = segpack.propose_folder(self._flavour, spec=self._spec)
        if self._folder is None or not getattr(self, "_chosen_by_user", False):
            self._folder = folder
            self._proposal_note = why
        self._paint_folder()

    def _paint_folder(self) -> None:
        self.folder_label.setText(f"Folder: {self._folder}")
        ok, line = segpack_view.folder_check(self._flavour, self._folder, spec=self._spec)
        self.space_label.setText((self._proposal_note + " " if self._proposal_note else "") + line)
        self.go_button.setEnabled(ok)

    def _pick_folder(self) -> None:
        start = str(Path(self._folder).parent) if self._folder else str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Folder for the segmentation pack", start)
        if not chosen:
            return
        self._chosen_by_user = True
        self._folder = Path(chosen) / segpack.pack_folder_name(self._flavour)
        self._proposal_note = ""
        self._paint_folder()

    def _show_status(self, st) -> None:
        self._status = st
        self.title_label.setText("Segmentation pack")
        self.body_label.setText(st.words())
        self.size_label.setText("Packages: " + ", ".join(st.packages) if st.packages else "")
        self._set_visible(choice_note=0, size_label=bool(st.packages), folder_label=0, space_label=0, bar=0,
                          progress_label=0, change_button=0, remove_button=1, go_button=0, no_button=1)
        if self.choice is not None:
            self.choice.setVisible(False)
        self.no_button.set_text("Close")

    # -- the download -------------------------------------------------------------- #
    def _start(self) -> None:
        self._cancel.clear()
        self._error = self._result = self._latest = None
        self._pace = segpack_view.Pace()
        folder, flavour = Path(self._folder), self._flavour
        self.body_label.setText("Downloading. You can cancel; what has arrived is kept and a later download continues it.")
        self._set_visible(choice_note=0, size_label=0, folder_label=0, space_label=0, bar=1, progress_label=1,
                          change_button=0, remove_button=0, go_button=0)
        if self.choice is not None:
            self.choice.setVisible(False)
        self.no_button.set_text("Cancel")
        self.bar.set_value(0.0)
        self.progress_label.setText("Starting.")

        def tick(p):
            with self._lock:
                self._latest = p

        def work():
            try:
                self._result = segpack.install(flavour, folder, on_progress=tick, cancel=self._cancel, spec=self._spec)
            except segpack.SegPackError as exc:
                self._error = exc

        self._worker = threading.Thread(target=work, name="segpack-install", daemon=True)
        self._worker.start()
        self._timer.start()

    def _poll(self) -> None:
        with self._lock:
            p = self._latest
        if p is not None:
            fraction, text = segpack_view.progress_line(p, self._pace)
            self.bar.set_value(fraction)
            self.progress_label.setText(text)
        if self._worker is not None and self._worker.is_alive():
            return
        self._timer.stop()
        self._worker = None
        if self._error is not None:
            cancelled = isinstance(self._error, segpack.PackCancelled)
            self.body_label.setText(("Cancelled. " if cancelled else "The download failed. ") + str(self._error))
            self._set_visible(bar=0, progress_label=0, go_button=1, no_button=1, change_button=0)
            self.go_button.set_text("Continue" if cancelled else "Try again")
            self.no_button.set_text("Close")
            return
        folder = self._result
        segpack.remember_pack(folder, self._flavour)
        segpack.activate(folder)
        st = segpack.status(folder)
        self._set_visible(bar=0, progress_label=0, go_button=0, change_button=0)
        self.title_label.setText("The segmentation pack is installed")
        self.body_label.setText(segpack_view.done_words(st))
        self.no_button.set_text("OK")
        self.installed.emit(str(folder))

    def _remove(self) -> None:
        st = getattr(self, "_status", None)
        if st is None:
            return
        try:
            gone = segpack.remove(st.folder)
        except segpack.SegPackError as exc:
            self.body_label.setText(str(exc))
            return
        self.body_label.setText("Removed " + ", ".join(str(g) for g in gone) + ". Segmentation models stop being "
                                "offered the next time SMTracker starts.")
        self._set_visible(remove_button=0, size_label=0)
        self.no_button.set_text("Close")
        self.removed.emit()

    # -- leaving ------------------------------------------------------------------- #
    def _close_clicked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._cancel.set()                            # the poll shows the cancellation, the dialog stays
            self.progress_label.setText("Cancelling after the block in hand.")
            return
        self._acted = True
        self._dismiss()

    def _on_dismiss_without_action(self) -> None:
        pass
