"""The Clustering section (v2's Clustering tab, 17 Oct 2026): k-means on the tracks' MSD curves scored against chance.
Two modes: each condition alone (a card per condition), or SHARED CLUSTERS (7 Nov 2026): the centroids fitted once on
the pooled tracks of every condition, ONE card holding the shares per condition and replicate, the figure of them and,
on request, Welch on the replicates. The Run's settings are asked in a dialog (`clustering_dialog`); the page shows
them in one line. A card whose clusters do not beat chance says so first and keeps the clusters behind "Show the
clusters anyway", greyed. THIN QT: every number, row and word comes from `clustering_view` (`compute_clustering_view`,
`card_payload`, `describe_config`) and `clustering_compare` (`compute_shared_clustering`, `shared_card_payload`,
`shares_payload`, `all_pairs_table`)."""
from __future__ import annotations

from qtpy.QtCore import QPointF, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QPen
from qtpy.QtWidgets import (QGraphicsOpacityEffect, QHBoxLayout, QScrollArea, QStackedWidget, QVBoxLayout, QWidget)

from ...axes_view import apply_range, axis_rows, check
from ...axes_view import empty as axes_empty
from ...clustering_compare import all_pairs_table, shared_card_payload, shares_payload
from ...clustering_view import (CLUSTER_INKS, READING_NOTES, STANDARD_CONFIG, card_payload, cluster_inks,
                                config_args, describe_config, section_config)
from ..theme import typography as ty
from ..widgets.axes_popover import AxesPopover
from ..widgets.cardgrid import CardGrid
from ..widgets.controls import Button, Segmented
from ..widgets.primitives import discard, label
from ..widgets.settings import InfoLink
from .jumps_chart import _JumpAxes, _Table
from .qc_chart import _Caption, _PanelCard
from .section_placeholder import SectionPlaceholder
from .toolbar import rem

__all__ = ["ClusteringSection", "ClusterCurves", "ClusterShares", "CLUSTER_INKS", "SHARED_KEY"]

#: The key of the one card the shared mode draws (it is not a condition).
SHARED_KEY = "*shared*"
#: How faint a card's clusters are drawn when they do not beat chance.
GREYED = 0.45


def _ink(pl, j: int) -> str:
    """Cluster ``j``'s ink: the payload's (the palette, fitted to the page: `clustering_view.cluster_inks`)."""
    inks = pl.get("inks") or CLUSTER_INKS
    return inks[j % len(inks)]


class ClusterCurves(_JumpAxes):
    """Each cluster's ensemble MSD against lag time (µm² over seconds), a line and its points."""

    _ROOMY_H = 140
    #: Typed ranges (the card's Axes, `axes_view`): ``{"x": (lo, hi), "y": (lo, hi)}``, an end None = automatic.
    axes: dict = {}

    def _paint(self, p, box: QRectF, pl) -> None:
        curves = pl["curves"]
        xs = [t for c in curves for t, _m in c]
        ys = [m for c in curves for _t, m in c]
        if not xs:
            return
        typed = self.axes or {}
        x0, x1 = apply_range(0.0, max(xs) * 1.05, typed.get("x"))
        if typed.get("y") is None:
            lo, hi = self._grid(p, box, 0.0, max(ys) * 1.08, "MSD (µm²)")
        else:
            lo, hi = self._grid_exact(p, box, *apply_range(0.0, max(ys) * 1.08, typed["y"]), "MSD (µm²)")

        def px(x):
            return box.left() + (x - x0) / (x1 - x0) * box.width()

        def py(v):
            return self._map_y(box, v, lo, hi)

        for i, curve in enumerate(curves):
            ink = QColor(_ink(pl, i))
            p.setPen(QPen(ink, 1.8))
            for (ta, ma), (tb, mb) in zip(curve, curve[1:]):
                p.drawLine(QPointF(px(ta), py(ma)), QPointF(px(tb), py(mb)))
            p.setPen(Qt.NoPen)
            p.setBrush(ink)
            for t, m in curve:
                p.drawEllipse(QPointF(px(t), py(m)), 3.0, 3.0)
        entries = [(_ink(pl, i), f"cluster {i + 1}", "line") for i in range(len(curves))]
        self._legend(p, entries, QPointF(box.left() + 6, box.top() + 4), horizontal=False)
        self._x_numeric(p, box, x0, x1, "Lag time (s)")


class ClusterShares(_JumpAxes):
    """Per condition, each shared cluster's share of tracks: one dot per replicate, the mean as a bar and the SD as
    whiskers. ``payload``: `clustering_compare.shares_payload`. A cluster keeps its ink in every condition; a dot's
    outline is its condition's colour."""

    _ROOMY_H = 170

    def _paint(self, p, box: QRectF, pl) -> None:
        columns, k = pl["columns"], pl["k"]
        if not columns:
            return
        top = 100.0 * max(max(max(c["mean"]) + (max(c["sd"]) if c["sd"] else 0.0),
                              max(max(r["shares"]) for r in c["replicates"])) for c in columns)
        lo, hi = self._grid(p, box, 0.0, max(10.0, top * 1.08), "Share of tracks (%)")
        n = len(columns)
        group = box.width() / n
        slot = group * 0.8 / k

        def py(v):
            return self._map_y(box, 100.0 * v, lo, hi)

        for ci, col in enumerate(columns):
            left = box.left() + ci * group + group * 0.1
            edge = QColor(col["colour"]) if col.get("colour") else None
            for j in range(k):
                ink = QColor(_ink(pl, j))
                x = left + (j + 0.5) * slot
                bar = slot * 0.7
                p.setPen(QPen(ink, 2.2))
                p.drawLine(QPointF(x - bar / 2, py(col["mean"][j])), QPointF(x + bar / 2, py(col["mean"][j])))
                if col["sd"] is not None:
                    p.setPen(QPen(ink, 1.2))
                    a, b = py(max(0.0, col["mean"][j] - col["sd"][j])), py(col["mean"][j] + col["sd"][j])
                    p.drawLine(QPointF(x, a), QPointF(x, b))
                    for y in (a, b):
                        p.drawLine(QPointF(x - bar / 4, y), QPointF(x + bar / 4, y))
                m = len(col["replicates"])
                for r, rep in enumerate(col["replicates"]):
                    dx = (r - (m - 1) / 2) * min(6.0, bar / max(1, m))
                    p.setPen(QPen(edge, 1.2) if edge is not None else Qt.NoPen)
                    fill = QColor(ink)
                    fill.setAlpha(190)
                    p.setBrush(fill)
                    p.drawEllipse(QPointF(x + dx, py(rep["shares"][j])), 3.0, 3.0)
        self._legend(p, [(_ink(pl, j), f"cluster {j + 1}", "dot") for j in range(k)],
                     QPointF(box.right() - 6, box.top() + 4), horizontal=False, right_aligned=True)
        self._x_categories(p, box, [c["name"] for c in columns])


class _ClusterBody(QWidget):
    """One condition: a message, or the headline, the verdict and the k table, then (behind "Show the clusters anyway"
    where they do not beat chance, greyed) the clusters' curves and table."""

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        self._dark = getattr(t, "color_scheme", "light") == "dark"
        lay = QVBoxLayout(self)
        lay.setContentsMargins(rem(1.25), 0, rem(1.25), rem(1.0))
        lay.setSpacing(rem(0.5))
        self.message = _Caption(t, size="sm")
        self.message.setText("computing…")
        lay.addWidget(self.message)
        self.headline = label("", ty.ui("base", "semibold"), t.foreground)
        self.headline.setWordWrap(True)
        lay.addWidget(self.headline)
        self.verdict = _Caption(t, size="sm")
        lay.addWidget(self.verdict)
        self.selection = _Table(t)
        lay.addWidget(self.selection)
        row = QHBoxLayout()
        self.toggle = Button(t, "Show the clusters anyway", variant="outline", size="sm")
        self.toggle.clicked.connect(self._toggle)
        row.addWidget(self.toggle)
        row.addStretch(1)
        self._toggle_row = QWidget()
        self._toggle_row.setLayout(row)
        lay.addWidget(self._toggle_row)
        #: Everything about the clusters themselves: collapsed and greyed when they do not beat chance.
        self.detail = QWidget()
        self.detail.setStyleSheet("background:transparent;")
        self.detail_lay = QVBoxLayout(self.detail)
        self.detail_lay.setContentsMargins(0, 0, 0, 0)
        self.detail_lay.setSpacing(rem(0.5))
        self.curves = ClusterCurves(t)
        self.curves.setFixedHeight(220)
        self.detail_lay.addWidget(self.curves)
        self.table = _Table(t)
        self.detail_lay.addWidget(self.table)
        lay.addWidget(self.detail)
        self.caption = _Caption(t, size="xs")
        lay.addWidget(self.caption)
        self._lay = lay
        self._negative = False
        self._expanded = False
        self._ready = False
        self._effect = QGraphicsOpacityEffect(self.detail)
        self._effect.setOpacity(1.0)
        self.detail.setGraphicsEffect(self._effect)
        self._apply()

    # -- what is shown --------------------------------------------------------------------------------------------- #
    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._apply()

    def _apply(self) -> None:
        """The message alone before a result; after it the headline, verdict, k table and caption, and the clusters
        (shown, greyed when they do not beat chance, behind the toggle)."""
        ready = self._ready
        self.message.setVisible(not ready)
        for w in (self.headline, self.verdict, self.selection, self.caption):
            w.setVisible(ready)
        collapsed = ready and self._negative and not self._expanded
        self._toggle_row.setVisible(ready and self._negative)
        self.toggle.set_text("Hide the clusters" if self._expanded else "Show the clusters anyway")
        self.detail.setVisible(ready and not collapsed)
        self._effect.setOpacity(GREYED if self._negative else 1.0)
        self._effect.setEnabled(self._negative)

    def set_message(self, text: str) -> None:
        self.message.setText(text)
        self._ready = False
        self._apply()

    def set_card(self, card: dict) -> None:
        if "error" in card:
            self.set_message(card["error"])
            return
        self._negative = card.get("exceeds_null") is False
        self._expanded = False
        self.headline.setText(card["headline"])
        self.verdict.setText(card["lines"][0])
        self.selection.set_table(card["selection_columns"], card["selection_rows"])
        self.curves.set_payload({"curves": card["curves"], "inks": cluster_inks(len(card["curves"]), self._dark)})
        self.table.set_table(card["columns"], card["rows"])
        self.caption.setText("\n".join(card["lines"][1:]))
        self._ready = True
        self._apply()


class _SharedBody(_ClusterBody):
    """The shared mode's card: the headline, verdict and selection (the pool), then the shares per condition (the
    figure, the table and, opt-in, Welch on the replicates, every pair) and the pooled clusters' curves and table."""

    def __init__(self, t, parent=None):
        super().__init__(t, parent)
        self.shares_chart = ClusterShares(t)
        self.shares_chart.setFixedHeight(260)
        self.share_table = _Table(t)
        self.compare_row = QWidget()
        row = QHBoxLayout(self.compare_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(rem(0.5))
        row.addWidget(label("Test conditions", ty.ui("sm"), t.muted_foreground))
        self.compare_choice = Segmented(t, ["Off", "Welch on replicates"], current=0, size="sm")
        row.addWidget(self.compare_choice)
        row.addStretch(1)
        self.compare_table = _Table(t)
        self.compare_caption = _Caption(t, size="xs")
        # the shares first, the pool's curves after: the comparison is the reason for this mode
        for i, w in enumerate((self.shares_chart, self.share_table, self.compare_row, self.compare_table,
                               self.compare_caption)):
            self.detail_lay.insertWidget(i, w)
        self.compare_choice.changed.connect(lambda _i: self._refresh_compare())
        self._result: dict | None = None
        self._names: dict = {}
        self._apply()

    def set_shared(self, result: dict, labels: dict, colours: dict | None = None) -> None:
        card = shared_card_payload(result)
        if "error" in card:
            self.set_message(card["error"])
            return
        self._result, self._names = result, dict(labels)
        self.set_card(card)
        self.share_table.set_table(card["share_columns"], card["share_rows"])
        self.shares_chart.set_payload(shares_payload(result, labels, colours, dark=self._dark))
        self._refresh_compare()

    def _refresh_compare(self) -> None:
        on = self.compare_choice.current() == 1 and self._result is not None
        self.compare_table.setVisible(on)
        self.compare_caption.setVisible(on)
        if not on:
            return
        table = all_pairs_table(self._result, self._names)
        self.compare_table.set_table(table["columns"], table["rows"])
        self.compare_caption.setText("\n".join(table["lines"]))

    def set_colours(self, labels: dict, colours: dict) -> None:
        if self._result is not None:
            self.shares_chart.set_payload(shares_payload(self._result, labels, colours, dark=self._dark))

    def _apply(self) -> None:
        super()._apply()
        if hasattr(self, "compare_table"):
            self._refresh_compare()


class ClusteringSection(QWidget):
    """The Clustering section: a placeholder until the first result, then the Run's settings in one line, how to read
    it, and one card per condition (or one shared card), filling in as each lands."""

    run_requested = Signal()
    cancel_requested = Signal()
    #: A card's Export (the condition's key, or `SHARED_KEY`): the window opens the Export dialog on it.
    export_requested = Signal(str)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.setStyleSheet("background:transparent;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._typed: dict = {}
        self._axes_popover = None
        self._axes_key = ""
        self.placeholder = SectionPlaceholder(
            t, title="Clustering",
            description="k-means on each track's MSD curve, as SMTracker 2's Clustering tab, scored against chance: "
                        "k-means always finds clusters, so the question is whether it finds more than chance. Shared "
                        "clusters put every condition on the same clusters so their shares can be compared.",
            icon_name="chart")
        self.placeholder.run_requested.connect(self.run_requested.emit)
        self.placeholder.cancel_requested.connect(self.cancel_requested.emit)
        page_widget = QWidget()
        page_widget.setStyleSheet("background:transparent;")
        page = QVBoxLayout(page_widget)
        page.setContentsMargins(0, 0, 0, rem(0.5))
        page.setSpacing(rem(1.0))
        self.summary = label("", ty.ui("sm"), t.muted_foreground)
        self.summary.setWordWrap(True)
        page.addWidget(self.summary)
        self.reading = InfoLink(t, "How to read this section", READING_NOTES)
        page.addWidget(self.reading)
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
        self._order: list = []
        self.last_config: dict | None = None
        self._ever_shown = False
        #: The Run's settings as the dialog last committed them (`clustering_dialog`): the standard before any.
        self._config = dict(STANDARD_CONFIG)
        self._show_summary()

    # -- the settings ----------------------------------------------------------------------------------------- #
    def config(self) -> dict:
        """What a Run uses (`clustering_view.section_config`): the dialog's last committed settings."""
        return dict(self._config)

    def set_config(self, config: dict) -> None:
        """Put a saved Run's settings back (a session, a new page). A setting the section cannot offer is refused by
        name, never half-applied."""
        self._config = section_config(**config_args(config))
        self._show_summary()

    def _show_summary(self) -> None:
        self.summary.setText(describe_config(self._config))

    # -- the run ---------------------------------------------------------------------------------------------- #
    def begin_run(self, conditions: list, labels: dict | None = None, config: dict | None = None) -> None:
        """A Run begins: one card per condition, or (shared clusters) ONE card, keyed `SHARED_KEY`, holding all of
        them. ``conditions`` are the conditions run either way."""
        for card, _body in self._cards.values():
            discard(card)
        self._cards, self._results = {}, {}
        self._labels = dict(labels or {})
        if config is not None:
            self.set_config(config)
        self.last_config = dict(self._config)
        shared = self.last_config.get("mode") == "shared"
        self._order = list(conditions)
        self._conditions = [SHARED_KEY] if shared else list(conditions)
        for key in self._conditions:
            if shared:
                body = _SharedBody(self.t)
                title = "Clustering · shared clusters"
            else:
                body = _ClusterBody(self.t)
                title = f"Clustering · {self._labels.get(key, key)}"
            card = _PanelCard(self.t, title, body, flush=False)
            card.enable_export()
            card.export_requested.connect(lambda k=key: self.export_requested.emit(k))
            card.enable_inspect("Axes", glyph="ruler")
            card.inspect_requested.connect(lambda k=key: self.open_axes(k))
            body.curves.axes = dict(self._typed.get(key) or {})
            self._cards[key] = (card, body)
            self._grid.addWidget(card)
        self.placeholder.set_running(True)
        self._pages.setCurrentIndex(1 if self._ever_shown else 0)

    def set_order(self, order) -> None:
        """The conditions in ``order`` (`condition_order`, the Conditions tool's arrows, 3 Nov 2026): the cards move;
        nothing is refitted or recomputed. The shared card redraws its columns in that order."""
        from ...condition_order import apply

        if SHARED_KEY in self._cards:
            self._order = apply(self._order, order)
            result = self._results.get(SHARED_KEY)
            if result is not None and "error" not in result:
                self.set_payload(SHARED_KEY, dict(result, conditions=apply(list(result["conditions"]), order)))
            return
        self._conditions = apply(self._conditions, order)
        self._cards = {k: self._cards[k] for k in apply(list(self._cards), order)}
        self._results = {k: self._results[k] for k in apply(list(self._results), order)}
        self._grid.reorder([card for card, _body in self._cards.values()])

    def set_colours(self, colours: dict) -> None:
        """The conditions' colours (the Conditions tool's): the shared figure's dots wear them as outlines."""
        self._colours = dict(colours or {})
        card = self._cards.get(SHARED_KEY)
        if card is not None:
            card[1].set_colours(self._labels, self._colours)

    def set_running(self, on: bool) -> None:
        self.placeholder.set_running(on)

    def set_payload(self, condition: str, result: dict) -> None:
        if condition not in self._cards:
            return
        self._results[condition] = result
        self._ever_shown = True
        self._pages.setCurrentIndex(1)
        body = self._cards[condition][1]
        if condition == SHARED_KEY:
            body.set_shared(result, self._labels, self._colours)
        else:
            body.set_card(card_payload(result))

    # -- typed ranges and the export (4 Nov 2026, rail review B3) ---------------------------------------------- #
    def typed_axes(self, key: str) -> dict:
        """The ranges typed for condition ``key``'s curves (empty: automatic)."""
        return dict(self._typed.get(key) or {})

    def set_typed_axes(self, key: str, axes) -> None:
        """Ranges for ``key``'s chart (`axes_view.check`, refused by name), drawn now; all None: automatic."""
        axes = check(axis_rows("clusters"), axes or {})
        if axes_empty(axes):
            self._typed.pop(key, None)
        else:
            self._typed[key] = axes
        if key in self._cards:
            chart = self._cards[key][1].curves
            chart.axes = dict(self._typed.get(key) or {})
            chart.update()

    def open_axes(self, key: str) -> None:
        """The Axes popover for condition ``key``'s curves, under its card's button."""
        if self._axes_popover is None:
            self._axes_popover = AxesPopover(self.t, self)
            self._axes_popover.applied.connect(lambda axes: self.set_typed_axes(self._axes_key, axes))
        self._axes_key = key
        card = self._cards.get(key)
        self._axes_popover.show_for(axis_rows("clusters"), self.typed_axes(key),
                                    None if card is None else card[0]._inspect)

    def axes_state(self) -> dict:
        return {k: dict(v) for k, v in self._typed.items()}

    def set_axes_state(self, state: dict) -> None:
        for key, axes in (state or {}).items():
            self.set_typed_axes(key, axes)

    def export_entries(self, key: str) -> list:
        """What condition ``key``'s Export draws: ``[{"key", "name", "result"}]`` (the card as drawn,
        `clustering_view.card_payload`), or empty before its result. The shared card's entries are its curves
        (`shared_card_payload`); `shares_export` has its figure."""
        result = self._results.get(key)
        if result is None or "error" in result:
            return []
        if key == SHARED_KEY:
            return [{"key": key, "name": "Shared clusters", "result": shared_card_payload(result)}]
        return [{"key": key, "name": self._labels.get(key, key), "result": card_payload(result)}]

    def shares_export(self) -> dict | None:
        """The shared figure's export payload: ``{"entries", "shares", "card"}``, or None without a shared result."""
        result = self._results.get(SHARED_KEY)
        if result is None or "error" in result:
            return None
        shares = shares_payload(result, self._labels, self._colours)
        entries = [{"key": c["key"], "name": c["name"], "colour": c["colour"], "result": c} for c in shares["columns"]]
        return {"entries": entries, "shares": shares, "card": shared_card_payload(result)}

    def set_error(self, condition: str, message: str) -> None:
        self.set_payload(condition, {"error": message})

    def has_result(self) -> bool:
        return self._ever_shown

    def results(self) -> dict:
        return dict(self._results)

    def conditions(self) -> list:
        return list(self._conditions)

    def cards(self) -> dict:
        return {k: card for k, (card, _b) in self._cards.items()}
