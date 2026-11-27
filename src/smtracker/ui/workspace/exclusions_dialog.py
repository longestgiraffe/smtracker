"""The Exclusions dialog (asked 16 Oct 2026: "Exclusions in the workspace -- exclude a track, cell or movie with a
reason ... the fractions shown with and without it"), opened from QC's "Contributes nothing" card.

A DRAFT the user edits: QC's flagged movies and cells offered as suggestions (`exclusions_view.suggestions`), any
movie, cell or track by its key, each with a written reason; the draft listed with what each removes
(`exclusions_view.rows`); **Preview** asks the window for the with-and-without fit (`exclusions_view.preview`, a
pool job) and shows it here; **Apply** hands the draft to the window, which rebuilds the workspace around it (as
for the track cap). Nothing changes until Apply; Cancel and the corner X discard the draft. Every rule and number
is `exclusions_view`'s (thin Qt: no numpy or pandas here).
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLineEdit, QScrollArea, QVBoxLayout, QWidget

from ...exclusions_view import KIND_LABELS, KINDS, key_problem, rows, suggestions
from ..theme import typography as ty
from ..widgets.controls import Button, Segmented
from ..widgets.dialogs import CloseButton, OverlayDialog
from ..widgets.primitives import discard, fit_label, label
from .toolbar import rem

__all__ = ["ExclusionsDialog"]


class _Field(QFrame):
    """A one-line text input in the dialogs' Input style."""

    changed = Signal(str)

    def __init__(self, t, placeholder: str, parent=None):
        super().__init__(parent)
        self.setObjectName("xfield")
        self.setStyleSheet(f"QFrame#xfield {{ background:{t.card}; border:1px solid {t.input};"
                           f" border-radius:{t.radius_md:g}px; }}")
        self.setFixedHeight(t.control_height)
        self.edit = QLineEdit("")
        self.edit.setFrame(False)
        self.edit.setFont(ty.ui("sm"))
        self.edit.setPlaceholderText(placeholder)
        self.edit.setStyleSheet(f"QLineEdit {{ background:transparent; color:{t.foreground}; border:none;"
                                f" padding:0px; }}")
        self.edit.textChanged.connect(self.changed.emit)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.addWidget(self.edit)

    def text(self) -> str:
        return self.edit.text()

    def set_text(self, text: str) -> None:
        self.edit.setText(text)


class ExclusionsDialog(OverlayDialog):
    """``project``: the project WITHOUT exclusions (the window's capped one), where every key is checked and every
    count made. ``in_force``: the items applied now. ``remembered``: the last Apply's items, offered when nothing is
    in force. ``qc_payloads``: QC's per-condition results, for the suggestions."""

    #: The draft's items, on Apply.
    applied = Signal(list)
    #: The draft's items, for the window's with-and-without job.
    preview_requested = Signal(list)

    def __init__(self, t, *, project, in_force=(), remembered=(), qc_payloads=None, content=None, parent=None):
        super().__init__(t, content, parent=parent)
        self.t = t
        self._project = project
        self._in_force = [dict(i) for i in in_force]
        self._remembered = [dict(i) for i in remembered]
        self._qc = qc_payloads or {}
        self.draft: list = [dict(i) for i in in_force]
        head = QHBoxLayout()
        head.addWidget(label("Exclusions", ty.ui("base", "semibold"), t.foreground))
        head.addStretch(1)
        close = CloseButton(t)
        close.clicked.connect(self._dismiss)
        head.addWidget(close)
        self.card_lay.addLayout(head)
        self.card_lay.addWidget(fit_label(
            "A movie, cell or track left out of every analysis, each with its reason. Nothing is deleted: remove an "
            "exclusion to put it back. Nothing changes until Apply.", ty.ui("sm"), t.muted_foreground))

        self._restore = QWidget()
        self._restore.setStyleSheet("background:transparent;")
        rl = QHBoxLayout(self._restore)
        rl.setContentsMargins(0, 0, 0, 0)
        self._restore_text = label("", ty.ui("sm"), t.foreground)
        rl.addWidget(self._restore_text, 1)
        restore = Button(t, "Restore", variant="outline", size="small")
        restore.clicked.connect(self._on_restore)
        rl.addWidget(restore)
        self.card_lay.addWidget(self._restore)

        self.card_lay.addWidget(label("Excluded", ty.ui("sm", "medium"), t.muted_foreground))
        self._list_host = QWidget()
        self._list_host.setStyleSheet("background:transparent;")
        self._list = QVBoxLayout(self._list_host)
        self._list.setContentsMargins(0, 0, 0, 0)
        self._list.setSpacing(rem(0.3))
        self._list.setAlignment(Qt.AlignTop)                # rows from the top, however few
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background:transparent; }")
        scroll.setWidget(self._list_host)
        scroll.setFixedHeight(rem(6))
        self.card_lay.addWidget(scroll)

        self.card_lay.addWidget(label("Suggested by QC", ty.ui("sm", "medium"), t.muted_foreground))
        self._suggest_host = QWidget()
        self._suggest_host.setStyleSheet("background:transparent;")
        self._suggest = QVBoxLayout(self._suggest_host)
        self._suggest.setContentsMargins(0, 0, 0, 0)
        self._suggest.setSpacing(rem(0.2))
        self._suggest.setAlignment(Qt.AlignTop)
        # IN A SCROLL AREA, as the list above (found by the overnight tester, 4 Nov 2026: eight suggestions laid in the
        # card itself were squeezed at 1366 x 768, their Use buttons cut at the bottom)
        suggest_scroll = QScrollArea()
        suggest_scroll.setWidgetResizable(True)
        suggest_scroll.setFrameShape(QScrollArea.NoFrame)
        suggest_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        suggest_scroll.setStyleSheet("QScrollArea { background:transparent; }")
        suggest_scroll.setWidget(self._suggest_host)
        suggest_scroll.setFixedHeight(rem(8))
        self._suggest_scroll = suggest_scroll
        self.card_lay.addWidget(suggest_scroll)

        self.card_lay.addWidget(label("Add", ty.ui("sm", "medium"), t.muted_foreground))
        add_row = QHBoxLayout()
        add_row.setSpacing(rem(0.5))
        self._kind = Segmented(t, [KIND_LABELS[k] for k in KINDS], current=1, size="sm")
        add_row.addWidget(self._kind)
        self._key = _Field(t, "its key, e.g. glucose_series004:3")
        add_row.addWidget(self._key, 2)
        self._reason = _Field(t, "why (required)")
        add_row.addWidget(self._reason, 2)
        add = Button(t, "Add", variant="outline", size="small")
        add.clicked.connect(self._on_add)
        add_row.addWidget(add)
        self.card_lay.addLayout(add_row)
        self.hint = label("", ty.ui("xs"), t.muted_foreground)
        self.card_lay.addWidget(self.hint)

        self.preview_text = fit_label("", ty.ui("xs", tabular=True), t.foreground)
        self.card_lay.addWidget(self.preview_text)
        footer = QHBoxLayout()
        self._preview_btn = Button(t, "Preview with and without", variant="outline")
        self._preview_btn.clicked.connect(self._on_preview)
        footer.addWidget(self._preview_btn)
        footer.addStretch(1)
        cancel = Button(t, "Cancel", variant="outline")
        cancel.clicked.connect(self._dismiss)
        self._apply_btn = Button(t, "Apply", variant="default")
        self._apply_btn.clicked.connect(self._on_apply)
        footer.addWidget(cancel)
        footer.addWidget(self._apply_btn)
        self.card_lay.addLayout(footer)
        self._row_widgets: list = []
        self._suggest_widgets: list = []
        self._refresh()

    # -- the draft -------------------------------------------------------------------------------------------- #
    def _refresh(self) -> None:
        for w in self._row_widgets + self._suggest_widgets:
            discard(w)
        self._row_widgets, self._suggest_widgets = [], []
        for i, r in enumerate(rows(self._project, self.draft)):
            row = QWidget()
            row.setStyleSheet("background:transparent;")
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            text = (f"{r['kind_label']} {r['shown']} · {r['condition'] or '?'} · {r['tracks']:,} "
                    f"track{'' if r['tracks'] == 1 else 's'}: {r['reason']}")
            # WRAPPED: the lists scroll only up and down, so a one-line row wider than the dialog was cut with its
            # button (the GUI sweep, 5 Nov 2026: a QC suggestion's range and its Use button off the right edge)
            lay.addWidget(fit_label(text, ty.ui("sm", tabular=True), self.t.foreground), 1)
            remove = Button(self.t, "Remove", variant="ghost", size="small")
            remove.clicked.connect(lambda _c=False, k=i: self._remove(k))
            lay.addWidget(remove)
            self._list.addWidget(row)
            self._row_widgets.append(row)
        if not self.draft:
            empty = label("Nothing excluded.", ty.ui("sm"), self.t.muted_foreground)
            self._list.addWidget(empty)
            self._row_widgets.append(empty)
        found = suggestions(self._qc, self.draft)
        for s in found[:8]:
            row = QWidget()
            row.setStyleSheet("background:transparent;")
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(fit_label(f"{KIND_LABELS[s['kind']]} {s['shown']} · {s['condition']} · {s['why']}",
                                ty.ui("sm", tabular=True),
                                self.t.foreground), 1)
            use = Button(self.t, "Use", variant="ghost", size="small")
            use.clicked.connect(lambda _c=False, s=s: self._use(s))
            lay.addWidget(use)
            self._suggest.addWidget(row)
            self._suggest_widgets.append(row)
        if not found:
            none = label("Nothing flagged" + ("" if self._qc else " (run QC to see what it flags)") + ".",
                         ty.ui("sm"), self.t.muted_foreground)
            self._suggest.addWidget(none)
            self._suggest_widgets.append(none)
        offer = bool(self._remembered) and not self._in_force and self.draft != self._remembered
        self._restore.setVisible(offer)
        if offer:
            n = len(self._remembered)
            self._restore_text.setText(f"{n} exclusion{'' if n == 1 else 's'} from the last time you applied some.")
        self._apply_btn.setEnabled(self.draft != self._in_force)
        self.preview_text.setText("")

    def _remove(self, index: int) -> None:
        del self.draft[index]
        self._refresh()

    def _use(self, s: dict) -> None:
        self._kind.select(KINDS.index(s["kind"]))
        self._key.set_text(s["key"])
        self.hint.setText(f"{s['why']}. Write why it should go, then Add.")

    def _on_restore(self) -> None:
        self.draft = [dict(i) for i in self._remembered]
        self._refresh()

    def _on_add(self) -> None:
        kind, key, reason = KINDS[self._kind.current()], self._key.text().strip(), self._reason.text().strip()
        problem = key_problem(self._project, kind, key)
        if not problem and not reason:
            problem = "write why: an exclusion without a reason cannot be reproduced later"
        if not problem and any(i["kind"] == kind and i["key"] == key for i in self.draft):
            problem = "already excluded"
        if problem:
            self.hint.setText(problem)
            return
        self.draft.append({"kind": kind, "key": key, "reason": reason})
        self._key.set_text("")
        self._reason.set_text("")
        self.hint.setText("")
        self._refresh()

    # -- preview and apply ------------------------------------------------------------------------------------- #
    def _on_preview(self) -> None:
        if not self.draft:
            self.preview_text.setText("Nothing to preview: the draft excludes nothing.")
            return
        self.preview_text.setText("fitting with and without…")
        self._preview_btn.setEnabled(False)
        self.preview_requested.emit([dict(i) for i in self.draft])

    def set_preview(self, result: dict) -> None:
        """The window's with-and-without result (`exclusions_view.preview`)."""
        self._preview_btn.setEnabled(True)
        self.preview_text.setText("\n".join(result["lines"]) + f"\n({result['caption']})")

    def set_preview_error(self, text: str) -> None:
        self._preview_btn.setEnabled(True)
        self.preview_text.setText(f"Preview failed: {text}")

    def _on_apply(self) -> None:
        self._acted = True
        self.applied.emit([dict(i) for i in self.draft])
        self._dismiss()
