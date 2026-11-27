"""The SQD section's Signed jumps page (agreed 17 Oct 2026: the along / across signed jump-distance fit as a
cross-check). THIN QT: every number, word and bar comes from `jd_view` (`compute_jd_view` per condition,
`card_payload`, `summary_table`); the histogram is the Jump distances page's own `JumpHistogram`, its x axis signed."""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QScrollArea, QStackedWidget, QVBoxLayout, QWidget

from ...condition_order import apply as order_apply
from ...jd_view import JD_AXES, JD_AXIS_LABELS, MAX_POP, READING_NOTES, card_payload, summary_table
from ..theme import typography as ty
from ..widgets.cardgrid import CardGrid
from ..widgets.controls import Segmented
from ..widgets.primitives import discard, label
from ..widgets.settings import InfoLink
from .jumps_chart import JumpHistogram, _Table
from .qc_chart import _DATA, _Caption, _PanelCard
from .section_placeholder import SectionPlaceholder
from .toolbar import rem

__all__ = ["JDSection"]


class _JDBody(QWidget):
    """One condition: a message, or the signed histogram, the table and its line."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(rem(1.25), 0, rem(1.25), rem(1.0))
        lay.setSpacing(rem(0.5))
        self.message = _Caption(t, size="sm")
        self.message.setText("computing…")
        lay.addWidget(self.message)
        self.histogram = JumpHistogram(t)
        self.histogram.setFixedHeight(280)
        lay.addWidget(self.histogram)
        self.table = _Table(t)
        lay.addWidget(self.table)
        self.caption = _Caption(t, size="xs")
        lay.addWidget(self.caption)
        self._show(False)

    def _show(self, on: bool) -> None:
        for w in (self.histogram, self.table, self.caption):
            w.setVisible(on)
        self.message.setVisible(not on)

    def set_message(self, text: str) -> None:
        self.message.setText(text)
        self._show(False)

    def set_card(self, payload: dict) -> None:
        if "error" in payload:
            self.set_message(payload["error"])
            return
        self.histogram.set_payload(payload["histogram"])
        self.table.set_table(payload["columns"], payload["rows"])
        self.caption.setText(payload["caption"])
        self._show(True)


class JDSection(QWidget):
    """The Signed jumps page: a placeholder until the first result, then the switches (axis, populations), the
    table across conditions and one card per condition, filling in as each lands."""

    run_requested = Signal()
    cancel_requested = Signal()
    #: A card's Export, with the condition's key; the table's Export.
    export_requested = Signal(str)
    table_export_requested = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.placeholder = SectionPlaceholder(
            t, title="Signed jumps",
            description="Each condition's jumps along the cell, across it and on the camera's axes against 1-, 2- "
                        "and 3-population normal mixtures. It cross-checks the SQD fractions on another "
                        "observable and shows how much the cell's width confines fast molecules.",
            icon_name="chart")
        self.placeholder.run_requested.connect(self.run_requested.emit)
        self.placeholder.cancel_requested.connect(self.cancel_requested.emit)
        page_widget = QWidget()
        page_widget.setStyleSheet("background:transparent;")
        page = QVBoxLayout(page_widget)
        page.setContentsMargins(0, 0, 0, rem(0.5))
        page.setSpacing(rem(1.0))
        controls = QHBoxLayout()
        controls.setSpacing(rem(0.5))
        controls.addWidget(label("Jumps", ty.ui("sm"), t.muted_foreground))
        self.axis_choice = Segmented(t, [JD_AXIS_LABELS[a] for a in JD_AXES], current=0, size="sm")
        self.axis_choice.changed.connect(lambda _i: self._redraw())
        controls.addWidget(self.axis_choice)
        controls.addSpacing(rem(1.0))
        controls.addWidget(label("Populations", ty.ui("sm"), t.muted_foreground))
        self.pop_choice = Segmented(t, [str(k) for k in range(1, MAX_POP + 1)], current=1, size="sm")
        self.pop_choice.changed.connect(lambda _i: self._redraw())
        controls.addWidget(self.pop_choice)
        controls.addStretch(1)
        page.addLayout(controls)
        self.reading = InfoLink(t, "How to read this page", READING_NOTES)
        page.addWidget(self.reading)
        summary_host = QWidget()
        summary_host.setStyleSheet("background:transparent;")
        summary_lay = QVBoxLayout(summary_host)
        summary_lay.setContentsMargins(rem(1.25), 0, rem(1.25), rem(1.0))
        self.summary = _Table(t)
        summary_lay.addWidget(self.summary)
        self.summary_card = _PanelCard(t, "Across conditions", summary_host, flush=False)
        self.summary_card.enable_export("Export table (PDF, SVG, PNG, TIFF)")
        self.summary_card.export_requested.connect(self.table_export_requested.emit)
        page.addWidget(self.summary_card)
        holder = QWidget()
        holder.setStyleSheet("background:transparent;")
        self._grid = CardGrid(holder, min_width=rem(30), max_columns=2, spacing=rem(1.0))
        page.addWidget(holder)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background:transparent; }")
        host = QWidget()
        host.setStyleSheet("background:transparent;")
        host_lay = QVBoxLayout(host)
        host_lay.setContentsMargins(0, 0, 0, 0)
        host_lay.addWidget(page_widget)
        host_lay.addStretch(1)
        scroll.setWidget(host)
        self._pages = QStackedWidget()
        self._pages.addWidget(self.placeholder)
        self._pages.addWidget(scroll)
        outer.addWidget(self._pages)
        self._cards: dict[str, tuple] = {}
        self._results: dict[str, dict] = {}
        self._conditions: list[str] = []
        self._labels: dict[str, str] = {}
        self._colours: dict[str, str] = {}
        self._ever_shown = False

    # -- the run ---------------------------------------------------------------------------------------------- #
    def begin_run(self, conditions: list, labels: dict | None = None, colours: dict | None = None) -> None:
        for card, _body in self._cards.values():
            discard(card)
        self._cards, self._results = {}, {}
        self._conditions = list(conditions)
        self._labels = dict(labels or {})
        self._colours = dict(colours or {})
        for key in conditions:
            body = _JDBody(self.t)
            card = _PanelCard(self.t, f"Signed jumps · {self._labels.get(key, key)}", body, flush=False)
            card.enable_export()
            card.export_requested.connect(lambda k=key: self.export_requested.emit(k))
            self._cards[key] = (card, body)
            self._grid.addWidget(card)
        self.summary.set_table([], [])
        self.placeholder.set_running(True)
        self._pages.setCurrentIndex(1 if self._ever_shown else 0)

    def set_order(self, order) -> None:
        """The conditions in ``order`` (`condition_order`, the Conditions tool's arrows, 3 Nov 2026): the cards and the
        table across conditions move; nothing is refitted or recomputed."""
        self._conditions = order_apply(self._conditions, order)
        self._cards = {k: self._cards[k] for k in order_apply(list(self._cards), order)}
        self._grid.reorder([card for card, _body in self._cards.values()])
        if self._cards:
            self._redraw()

    def set_running(self, on: bool) -> None:
        self.placeholder.set_running(on)

    def set_colours(self, colours: dict) -> None:
        self._colours.update(colours)
        self._redraw()

    def set_payload(self, condition: str, result: dict) -> None:
        if condition not in self._cards:
            return
        self._results[condition] = result
        self._ever_shown = True
        self._pages.setCurrentIndex(1)
        self._redraw()

    def set_error(self, condition: str, message: str) -> None:
        if condition not in self._cards:
            return
        self._results[condition] = {"error": message}
        self._ever_shown = True
        self._pages.setCurrentIndex(1)
        self._redraw()

    def axis(self) -> str:
        return JD_AXES[self.axis_choice.current()]

    def populations(self) -> int:
        return self.pop_choice.current() + 1

    def _redraw(self) -> None:
        axis, k = self.axis(), self.populations()
        for key, (_card, body) in self._cards.items():
            result = self._results.get(key)
            if result is None:
                body.set_message("computing…")
            elif "error" in result and not result.get("axes"):
                body.set_message(result["error"])
            else:
                body.set_card(card_payload(result, axis, k, self._colours.get(key, _DATA)))
        done = [c for c in self._conditions if c in self._results]
        table = summary_table(self._results, done, self._labels, axis, k)
        self.summary.set_table(table["columns"], table["rows"])
        self.summary_card.set_caption(table["caption"])

    def has_result(self) -> bool:
        return self._ever_shown

    def export_entries(self, key: str) -> list:
        """What the card ``key``'s Export draws: ``[{"key", "name", "colour", "result", "axis", "selected"}]`` -- the
        axis and the count the page shows (the Export dialog may pick another count) -- or ``[]`` without a fit."""
        result = self._results.get(key)
        if not result or (result.get("error") and not result.get("axes")):
            return []
        axis = (result.get("axes") or {}).get(self.axis()) or {}
        if "error" in axis or not axis.get("fits"):
            return []
        return [{"key": key, "name": self._labels.get(key, key), "colour": self._colours.get(key, _DATA),
                 "result": result, "axis": self.axis(), "selected": self.populations()}]

    def table_export(self) -> dict | None:
        """The table across conditions, as the "table" figure's payload, or None before a result."""
        done = [c for c in self._conditions if c in self._results]
        if not done:
            return None
        table = summary_table(self._results, done, self._labels, self.axis(), self.populations())
        return {"title": "Signed jumps", "columns": list(table["columns"]), "rows": [list(r) for r in table["rows"]],
                "caption": table["caption"]}

    def cards(self) -> dict:
        return {k: card for k, (card, _b) in self._cards.items()}
