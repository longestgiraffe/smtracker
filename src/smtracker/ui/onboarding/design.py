"""What do the folders mean? The analysis door's folder-design page (phase 2 of the folder designs, 24 Oct 2026).

Asked: "a builder for folder structures so other people can keep their experimental folder designs ... their own
design can be saved". One row per folder level -- a few of its names, the role (Factor / Replicate / Skip), a
factor's name -- and under them, live, what the design makes: the conditions with their factors, or why it cannot
load the folders. Saved designs from the personal library are one click away; "Save as" adds this one. "Use this
design" saves it with the project (`smtracker_design.json`) and the door looks at the folder again through it;
"Load without a design" loads it the automatic way and is remembered for this folder.

Every rule and word is `design_view`'s (`design_question`, `preview`, `accept`, `decline`); this only lays them out.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QLineEdit, QSizePolicy, QVBoxLayout, QWidget

from ...chart_text import FLAG_COLOR
from ...design_view import ROLE_TIPS, ROLE_WORDS, accept, decline, library_levels, preview
from ..theme import typography as ty
from ..widgets.controls import Button, Segmented
from ..widgets.primitives import Card, ShadowHost, band, icon, label, row

__all__ = ["DesignPage"]

_ROLES = ("factor", "replicate", "skip")


def _field(t, text: str = "", placeholder: str = "") -> QLineEdit:
    edit = QLineEdit(text)
    edit.setPlaceholderText(placeholder)
    edit.setFont(ty.ui("sm"))
    edit.setFixedHeight(30)
    edit.setStyleSheet(
        f"QLineEdit {{ background:{t.background}; color:{t.foreground}; border:1px solid {t.input};"
        f" border-radius:{t.radius_md:g}px; padding:0px 8px; }}"
        f"QLineEdit:disabled {{ color:{t.muted_foreground}; background:{t.muted}; }}")
    return edit


class DesignPage(ShadowHost):
    """Emits ``designed()`` once the design is saved with the project, ``skipped()`` to load without one, or
    ``cancelled()``."""

    designed = Signal()
    skipped = Signal()
    cancelled = Signal()

    def __init__(self, t, question: dict, parent=None):
        super().__init__(t, parent)
        self.t = t
        self.question = question
        self.levels = [dict(level) for level in question["levels"]]
        self._roles: list = []
        self._names: list = []
        self.setStyleSheet(f"background:{t.background};")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(48, 40, 48, 40)
        lay.setSpacing(0)
        lay.addStretch(1)
        root = Path(question["root"])
        head = label("What do the folders mean?", ty.ui("2xl", "semibold"), t.foreground)
        head.setAlignment(Qt.AlignHCenter)
        sub = label(question["why"], ty.ui("sm"), t.muted_foreground)
        sub.setAlignment(Qt.AlignHCenter)
        sub.setWordWrap(True)
        # A FIXED WIDTH, not a maximum: a wrapped label centred by its layout is given its one-line width and
        # cuts its second line off (seen on the first render)
        sub.setFixedWidth(720)
        sub.setMinimumHeight(sub.heightForWidth(720))     # its wrapped height: the layout gives it one line
        # each straight into the page's layout: a `col` holding a wrapped label kept its one-line height
        lay.addWidget(head, alignment=Qt.AlignHCenter)
        lay.addSpacing(6)
        lay.addWidget(sub, alignment=Qt.AlignHCenter)
        lay.addSpacing(20)

        card = Card(t)
        card.setMaximumWidth(760)
        card.setMinimumWidth(560)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(band(
            row(icon("folder-open", 14, t.muted_foreground),
                label(root.name or str(root), ty.ui("sm", "semibold"), t.foreground), None,
                label("Each level from the top down", ty.ui("xs"), t.muted_foreground),
                spacing=8, margins=(16, 12, 16, 12)),
            "designHead", f"background:transparent; border-bottom:1px solid {t.border};"))
        for i, (shown, level) in enumerate(zip(question["rows"], self.levels)):
            roles = Segmented(t, [ROLE_WORDS[r] for r in _ROLES], current=_ROLES.index(level["role"]), size="sm",
                              tips=[ROLE_TIPS[r] for r in _ROLES])
            roles.changed.connect(lambda k, i=i: self._set(i, role=_ROLES[k]))
            name = _field(t, level.get("name", ""), "name, e.g. protein")
            name.textEdited.connect(lambda text, i=i: self._set(i, name=text))
            name.setFixedWidth(150)
            self._roles.append(roles)
            self._names.append(name)
            examples = label(f"{shown['examples']}  ({shown['count']})", ty.ui("xs"), t.muted_foreground)
            examples.setToolTip(shown["examples"])
            cl.addWidget(band(
                row(label(f"Level {shown['level']}", ty.ui("sm", "medium"), t.foreground), examples, None, roles,
                    name, spacing=12, margins=(16, 8, 16, 8)),
                "designRow", "background:transparent;"))
        if question.get("library"):
            picks = [label("Saved designs:", ty.ui("xs"), t.muted_foreground)]
            for saved in question["library"][:6]:
                button = Button(t, saved, variant="outline", size="sm")
                button.clicked.connect(lambda _c=False, n=saved: self._use_saved(n))
                picks.append(button)
            picks.append(None)
            cl.addWidget(band(row(*picks, spacing=6, margins=(16, 8, 16, 10)), "designSaved",
                              f"background:transparent; border-top:1px solid {t.border};"))
        holder = QWidget()
        holder.setStyleSheet("background:transparent;")
        hl = QVBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(card, alignment=Qt.AlignHCenter)
        lay.addWidget(holder)
        lay.addSpacing(14)

        self.preview = label("", ty.ui("sm"), t.foreground)
        self.preview.setAlignment(Qt.AlignHCenter)
        self.preview.setWordWrap(True)
        self.preview.setFixedWidth(760)
        lay.addWidget(self.preview, alignment=Qt.AlignHCenter)
        lay.addSpacing(12)
        self.save_as = _field(t, "", "Save to my designs as ... (optional)")
        self.save_as.setFixedWidth(320)
        lay.addWidget(self.save_as, alignment=Qt.AlignHCenter)
        lay.addSpacing(16)

        self.back = Button(t, "Back", variant="outline")
        self.back.clicked.connect(self.cancelled.emit)
        self.skip = Button(t, "Load without a design", variant="outline")
        self.skip.clicked.connect(self._skip)
        self.use = Button(t, "Use this design", variant="default", glyph="folder-open")
        self.use.clicked.connect(self._use)
        lay.addWidget(row(None, self.back, self.skip, self.use, None, spacing=8))
        lay.addStretch(1)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._refresh()

    # -- the rows ------------------------------------------------------------- #
    def _set(self, i: int, **changes) -> None:
        self.levels[i].update(changes)
        self._refresh()

    def _use_saved(self, name: str) -> None:
        self.levels = [dict(level) for level in library_levels(name)][:len(self._roles)]
        for roles, edit, level in zip(self._roles, self._names, self.levels):
            roles.blockSignals(True)
            roles.select(_ROLES.index(level["role"]))
            roles.blockSignals(False)
            edit.setText(level.get("name", ""))
        self._refresh()

    def _refresh(self) -> None:
        for edit, level in zip(self._names, self.levels):
            edit.setEnabled(level["role"] == "factor")
        self.result = preview(self.question, self.levels)
        colour = self.t.foreground if self.result["ok"] else FLAG_COLOR
        self.preview.setStyleSheet(f"color:{colour}; background:transparent;")
        self.preview.setText(self.result["text"])
        self.preview.setMinimumHeight(self.preview.heightForWidth(760))
        self.use.setEnabled(self.result["ok"])

    # -- the answers ---------------------------------------------------------- #
    def _use(self) -> None:
        from ...folderdesign import DesignError

        if not self.result["ok"]:
            return
        try:
            accept(self.question, self.levels, remember_as=self.save_as.text())
        except (DesignError, OSError) as exc:
            self.preview.setStyleSheet(f"color:{FLAG_COLOR}; background:transparent;")
            self.preview.setText(f"Not saved: {exc}")
            return
        self.designed.emit()

    def _skip(self) -> None:
        try:
            decline(self.question["root"])
        except OSError:
            pass                    # "do not ask again" lost: the page only asks again next time
        self.skipped.emit()
