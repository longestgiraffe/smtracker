"""napari dock widgets: a THIN Qt shell over the pure inspector.

The rule this module exists to keep
------------------------------------
Nothing is computed inside a Qt class. The widget's whole job is:

    selection event  ->  viewer.nearest_track  ->  inspector.inspect_track
                     ->  plots.draw_inspection ->  canvas.draw()

Every one of those four is pure numpy/pandas/matplotlib and is tested headless.
So the interactive layer adds no untested logic: if the inspector is right, the
widget is right, and a click that selects the wrong track fails in
``test_viewer.TestSelection`` rather than by looking subtly wrong on screen.

This is the only module in the package that imports Qt at module scope, and
nothing in the analysis spine imports it.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .analyses import TABS, AnalysisError, build_input, by_tab, result_rows
from .inspector import inspect_track
from .plots import (draw_all_tracks_on_cell, draw_cell_explorer,
                    draw_inspection, draw_qc_dashboard)
from .qc import (
    group_counts,
    length_distribution,
    QCError,
    in_cell_share,
    localizations_per_frame,
    track_summary,
    unit_summary,
)
from .viewer import nearest_track

__all__ = [
    "TrackInspectorWidget",
    "TrackListWidget",
    "HIGHLIGHT_LAYER",
    "HIGHLIGHT_CELL_LAYER",
    "QCDashboardWidget",
    "dock_inspector",
    "dock_track_list",
    "dock_qc",
    "AnalysisPanelWidget",
    "dock_analyses",
    "TrackViewsWidget",
    "dock_views",
]

#: The selected track and its cell, drawn on top in colours nothing else uses.
HIGHLIGHT_LAYER = "selected track"
HIGHLIGHT_CELL_LAYER = "selected cell"
HIGHLIGHT_TRACK_COLOR = (1.0, 0.0, 1.0, 1.0)      # magenta, RGBA
HIGHLIGHT_CELL_COLOR = "#ffff00"                  # yellow


HIGHLIGHT_COLORMAP = "smtracker_highlight"


def _flat_colormap() -> str:
    """Register (once) a colormap that is magenta at *every* position, by name.

    Two reasons it has to be flat and registered rather than, say, napari's
    built-in ``magenta``:

    - a Tracks layer colours by a property through a colormap, and with a
      **single** selected track that property has one unique value, so the
      normalisation is degenerate and a ramp can land anywhere on it, including
      black, i.e. an invisible highlight;
    - napari's Tracks layer looks its colormap up **by name** in
      ``AVAILABLE_COLORMAPS`` and rejects a ``Colormap`` object outright
      (``TypeError: unhashable type``), so passing one is not an option.
    """
    from napari.utils import Colormap
    from napari.utils.colormaps import AVAILABLE_COLORMAPS

    if HIGHLIGHT_COLORMAP not in AVAILABLE_COLORMAPS:
        AVAILABLE_COLORMAPS[HIGHLIGHT_COLORMAP] = Colormap(
            colors=[HIGHLIGHT_TRACK_COLOR, HIGHLIGHT_TRACK_COLOR],
            name=HIGHLIGHT_COLORMAP,
        )
    return HIGHLIGHT_COLORMAP


_PLACEHOLDER = (
    "Click a track in the viewer to inspect it.\n\n"
    "The step trace is drawn against the localization-floor band, so a "
    "'slow' call can be judged against what the microscope can actually resolve."
)


class TrackInspectorWidget(QWidget):
    """Shows the single-track inspector for whatever the user clicked."""

    def __init__(self, viewer, bundle, canonical, *, hmm_fit=None,
                 confinement_radius_um=None, outlines_um=None,
                 max_distance_px=3.0, tail_length=10, parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self._bundle = bundle
        self._canonical = canonical
        self._hmm_fit = hmm_fit
        self._confinement_radius_um = confinement_radius_um
        self._outlines_um = outlines_um or {}
        self._max_distance_px = max_distance_px
        self._tail_length = tail_length

        self._status = QLabel(_PLACEHOLDER)
        self._status.setWordWrap(True)
        self._figure = Figure(figsize=(9.0, 5.4), layout=None)
        self._canvas = FigureCanvasQTAgg(self._figure)

        layout = QVBoxLayout()
        layout.addWidget(self._status)
        layout.addWidget(self._canvas)
        self.setLayout(layout)

    # -- selection ---------------------------------------------------------- #

    def on_click(self, layer, event):
        """napari mouse-drag callback: resolve the cursor to a track and show it.

        The cursor position is ``(t, y, x)`` in data coordinates, which is
        already the bundle's coordinate system: the bundle put tracks in pixels
        with the same origin offset, so no conversion happens here. Conversions
        belong in ``viewer``, not in a callback.
        """
        position = self._viewer.cursor.position
        frame, y, x = (position[0], position[-2], position[-1])
        self.show_at(y, x, frame=frame)
        yield                      # napari mouse callbacks are generators

    def show_at(self, y_px, x_px, *, frame=None) -> bool:
        """Inspect whatever is at ``(y_px, x_px)``. Returns whether it found one."""
        hit = nearest_track(
            self._bundle, y_px, x_px, frame=frame,
            max_distance_px=self._max_distance_px,
        )
        if hit is None:
            self._status.setText(
                f"No track within {self._max_distance_px:g} px of "
                f"(y={y_px:.1f}, x={x_px:.1f}) at frame {frame}."
            )
            return False
        self.show_uid(hit.uid)
        return True

    def show_uid(self, uid: str) -> None:
        """Inspect one track by uid. All computation happens in ``inspector``."""
        inspection = inspect_track(
            self._canonical, uid, hmm_fit=self._hmm_fit,
            confinement_radius_um=self._confinement_radius_um,
        )
        self._goto_frame(int(inspection.frames[0]))
        self._highlight(uid)
        self._status.setText(
            f"{inspection.uid}   ·   {inspection.disagreement()}\n"
            f"floor: {inspection.floor_verdict()}"
        )
        draw_inspection(
            inspection, self._figure,
            outline_um=self._outlines_um.get(inspection.cell),
        )
        self._canvas.draw_idle()


    def _highlight(self, uid: str) -> None:
        """Draw the selected track, and its cell, in prominent colours.

        Selection is otherwise invisible: with hundreds of overlapping tracks in
        a field, "which one did I just click?" and "which cell is it in?" cannot
        be answered from the inspector panel alone.

        The track is drawn as a **Tracks** layer, not a static path, so it
        appears, moves and fades with the time slider exactly like every other
        track: only magenta. A path would sit on screen for the whole movie,
        which both misreads as "this molecule is everywhere at once" and hides
        *when* the track exists. The cell outline stays a Shapes polygon,
        because a cell genuinely does not move.

        Geometry comes from ``ViewerBundle``; nothing is computed here.
        """
        try:
            rows = self._bundle.track_rows(uid)
            outline = self._bundle.cell_outline(uid)
        except Exception:                     # pragma: no cover - defensive
            return

        self._highlight_track(rows)
        self._highlight_cell(outline)
        # Adding a layer makes it active, and an active Shapes layer puts an
        # editing toolbar back on screen. Hand focus back to the tracks layer.
        layers = getattr(self._viewer, "layers", {})
        if "tracks" in layers:
            layers.selection.active = layers["tracks"]

    def _highlight_track(self, rows) -> None:
        """The selected track as its own Tracks layer, in a flat magenta."""
        if len(rows) < 2:
            return
        layers = getattr(self._viewer, "layers", {})
        if HIGHLIGHT_LAYER in layers:
            layers[HIGHLIGHT_LAYER].data = rows
            return
        layer = self._viewer.add_tracks(
            rows, name=HIGHLIGHT_LAYER, colormap=_flat_colormap(),
            tail_length=self._tail_length, tail_width=4, blending="translucent",
        )
        layer.interactive = False

    def _highlight_cell(self, outline) -> None:
        """The selected track's cell outline: static, because a cell is."""
        if outline is None or len(outline) < 3:
            return
        layers = getattr(self._viewer, "layers", {})
        if HIGHLIGHT_CELL_LAYER in layers:
            layers[HIGHLIGHT_CELL_LAYER].data = [outline]
            return
        layer = self._viewer.add_shapes(
            [outline], shape_type="polygon", name=HIGHLIGHT_CELL_LAYER,
            edge_color=HIGHLIGHT_CELL_COLOR, face_color="transparent",
            edge_width=0.6,
        )
        layer.editable = False
        layer.interactive = False

    def _goto_frame(self, frame: int) -> None:
        """Move the viewer's time slider to ``frame``.

        Picking a track from the *list* is otherwise blind: the track may start
        at frame 800 while the viewer sits at frame 0, so nothing appears and
        the selection looks broken. The lookup itself is `ViewerBundle`'s job,
        no arithmetic happens here.
        """
        dims = getattr(self._viewer, "dims", None)
        if dims is None or getattr(dims, "ndim", 0) < 3:
            return                      # no time axis to move
        try:
            dims.set_current_step(0, frame)
        except Exception:               # pragma: no cover - cosmetic navigation
            pass


def dock_inspector(viewer, bundle, canonical, *, layer_name="tracks", **kwargs):
    """Add the inspector to ``viewer`` and wire it to clicks on the tracks layer."""
    widget = TrackInspectorWidget(viewer, bundle, canonical, **kwargs)
    viewer.window.add_dock_widget(widget, name="Track inspector", area="right")
    if layer_name in viewer.layers:
        viewer.layers[layer_name].mouse_drag_callbacks.append(widget.on_click)
    return widget


class TrackListWidget(QWidget):
    """A sortable list of tracks: the second selection mechanism (BUGS.md B2).

    Clicking a track in the canvas only reaches what you can hit; with a median
    of 6 localizations per track, the informative long tracks are the hard ones
    to hit. This lists them with their facts so they can be found by *property*
    (longest, most switches, slowest) and sorted by any column.

    Computes nothing: the table comes from ``qc.track_summary``.
    """

    #: Columns shown, in order. Kept short on purpose: the list is for finding
    #: a track, not for reading its whole story.
    COLUMNS = [
        ("uid", "track"),
        ("cell", "cell"),
        ("n_localizations", "N"),
        ("n_steps", "steps"),
        ("n_gaps", "gaps"),
        ("median_step_um", "median step [µm]"),
        ("appd_um2_per_s", "appD [µm²/s]"),
        ("slow_fraction", "slow"),
        ("n_switches", "switches"),
    ]

    def __init__(self, summary, *, on_select=None, parent=None):
        super().__init__(parent)
        self._summary = summary
        self._on_select = on_select

        columns = [c for c in self.COLUMNS if c[0] in summary.columns]
        self._keys = [c[0] for c in columns]

        self._label = QLabel(
            f"{len(summary)} tracks: click a row to inspect, click a header to sort"
        )
        self._table = QTableWidget(len(summary), len(columns))
        self._table.setHorizontalHeaderLabels([c[1] for c in columns])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._fill(summary)
        self._table.setSortingEnabled(True)
        self._table.itemSelectionChanged.connect(self._selection_changed)

        layout = QVBoxLayout()
        layout.addWidget(self._label)
        layout.addWidget(self._table)
        self.setLayout(layout)

    def _fill(self, summary) -> None:
        for row, record in enumerate(summary.to_dict("records")):
            for column, key in enumerate(self._keys):
                value = record[key]
                item = QTableWidgetItem()
                # setData with the native value keeps numeric columns sorting
                # numerically; a plain string sorts 10 before 9.
                if isinstance(value, (int, float)) and value == value:
                    item.setData(Qt.DisplayRole, round(float(value), 4)
                                 if isinstance(value, float) else int(value))
                else:
                    item.setData(Qt.DisplayRole, str(value))
                item.setData(Qt.UserRole, str(record["uid"]))
                self._table.setItem(row, column, item)

    def selected_uid(self) -> str | None:
        items = self._table.selectedItems()
        return items[0].data(Qt.UserRole) if items else None

    def _selection_changed(self) -> None:
        uid = self.selected_uid()
        if uid is not None and self._on_select is not None:
            self._on_select(uid)


class QCDashboardWidget(QWidget):
    """The "should I trust this dataset?" view. Computes nothing itself.

    Every panel is produced by ``smtracker.qc`` and drawn by
    ``smtracker.plots.draw_qc_dashboard``; this only owns the canvas.
    """

    def __init__(self, canonical, *, hmm_fit=None, unit="cell", min_steps=None,
                 field_shape=None, max_per_frame=None, parent=None):
        super().__init__(parent)
        self._figure = Figure(figsize=(11.0, 7.0))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._status = QLabel("")

        layout = QVBoxLayout()
        layout.addWidget(self._status)
        layout.addWidget(self._canvas)
        self.setLayout(layout)
        self.refresh(canonical, hmm_fit=hmm_fit, unit=unit, min_steps=min_steps,
                     field_shape=field_shape, max_per_frame=max_per_frame)

    def refresh(self, canonical, *, hmm_fit=None, unit="cell", min_steps=None,
                field_shape=None, max_per_frame=None) -> None:
        summary = track_summary(canonical, hmm_fit=hmm_fit)
        lengths = length_distribution(canonical, min_steps=min_steps, summary=summary)
        density = localizations_per_frame(
            canonical, field_shape=field_shape, max_per_frame=max_per_frame
        )
        counts = group_counts(canonical, summary=summary)
        units = unit_summary(canonical, unit=unit, summary=summary)

        flagged = units.loc[units["flagged"], unit].astype(str).tolist()
        self._status.setText(
            f"{summary.attrs['n_tracks']} tracks / "
            f"{summary.attrs['n_localizations']} localizations | "
            + (f"flagged {unit}s: {', '.join(flagged)}" if flagged
               else f"no {unit} flagged")
            + ": flags are for inspection; nothing is excluded"
        )
        # UNAVAILABLE IS A STATE, NOT AN ERROR. A canonical table loaded through a path
        # that does not carry the cell-filter counts is UNMEASURED; the panel says so
        # rather than being dropped, because a missing panel reads as "nothing to see".
        try:
            in_cell = in_cell_share(canonical)
        except QCError:
            in_cell = None
        draw_qc_dashboard(
            self._figure, lengths=lengths, density=density, counts=counts, units=units,
            in_cell=in_cell,
        )
        self._canvas.draw_idle()


def dock_track_list(viewer, summary, *, on_select=None, **kwargs):
    """Add the track list to ``viewer``."""
    widget = TrackListWidget(summary, on_select=on_select, **kwargs)
    viewer.window.add_dock_widget(widget, name="Tracks", area="left")
    return widget


def dock_qc(viewer, canonical, **kwargs):
    """Add the QC dashboard to ``viewer``."""
    widget = QCDashboardWidget(canonical, **kwargs)
    viewer.window.add_dock_widget(widget, name="QC", area="bottom")
    return widget

class RunToDepositWidget(QWidget):
    """AT1: the acquisition-folder → deposited-table path, from the GUI.

    THIN-QT, and deliberately so: this class collects arguments and renders a
    :class:`smtracker.run.RunResult`. It computes NOTHING. Every number it shows comes
    from ``run.run_to_deposit``, which is the same function ``smtracker --deposit``
    calls with the same arguments: a GUI that computed a different number from the CLI
    would be worse than no GUI at all, so there is one implementation and two callers.

    Four things it must make visible, each a state that has bitten this project
    silently:

    * **gagging is never silent**, a gagged condition names the condition, its visit
      count and the floor. "Insufficient data" never looks like "not yet run";
    * **K is always displayed**: both fits are shown, the widget never selects one, and
      no fraction appears without its K beside it;
    * **counts are on screen**, cells, replicate days, tracks and shell visits, at the
      point the number is shown;
    * **provenance**, the commit and dirty flag that produced the deposit.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        from qtpy.QtWidgets import QLabel, QPlainTextEdit, QVBoxLayout

        layout = QVBoxLayout(self)
        self._status = QLabel("not yet run")
        self._counts = QLabel("")
        self._gag = QLabel("")
        self._fits = QPlainTextEdit()
        self._fits.setReadOnly(True)
        self._prov = QLabel("")
        for w in (self._status, self._counts, self._gag, self._fits, self._prov):
            layout.addWidget(w)

    @staticmethod
    def build_config(**inputs):
        """Assemble a RunConfig from user-visible inputs, by DELEGATION (AU3).

        The assembly lives in the package (``run.config_from_inputs``) and the CLI
        calls the same one, so the two parameter sets are equal by construction.
        Nothing is parsed or computed in this class.
        """
        from .run import config_from_inputs
        return config_from_inputs(**inputs)

    def run(self, condition_root, config, outdir, *, day_map=None):
        """Call the packaged pipeline and render its result. No computation here."""
        from .run import run_to_deposit

        self._status.setText("running...")
        result = run_to_deposit(condition_root, config, outdir, day_map=day_map)
        self.show_result(result)
        return result

    def show_result(self, result) -> None:
        """Render a RunResult. Pure display: every value is read, none is derived."""
        self._status.setText(
            f"deposited: {', '.join(sorted(result.written))}"
            if result.written else "run complete (nothing written)")
        self._counts.setText(result.counts_line())
        # "insufficient" and "not yet run" must not look alike
        self._gag.setText(result.gag_reason if result.gagged
                          else "not gagged - enrichment number emitted")
        lines = []
        for K in sorted(result.fits):
            f = result.fits[K]
            occ = ", ".join(f"{100 * o:.1f} %" for o in f.occupancy)
            D = ", ".join(f"{d:.4g}" for d in f.D)
            lines.append(f"{f.label()}  occupancy {occ}   D {D} um2/s   "
                         f"BIC {f.bic:.1f}   n = {f.n_tracks} tracks, "
                         f"{f.n_steps} steps")
        if not lines:
            lines = ["no fit converged"]
        self._fits.setPlainText(chr(10).join(lines))
        prov = result.provenance or {}
        dirty = prov.get("dirty")
        self._prov.setText(
            f"commit {prov.get('commit') or 'unknown'}"
            + ("  [WORKING TREE DIRTY]" if dirty else ""))


# =====================================================================================
# THE ANALYSIS PANEL - every v2 tool, findable
# =====================================================================================
#
# This widget OWNS NO KNOWLEDGE. Which analyses exist, what they consume, what their
# parameters cost and which ones refuse a default all live in ``smtracker.analyses``,
# which is pure and tested headless. The panel reads that catalogue and builds controls
# from it, so adding an analysis is a catalogue entry and never a Qt edit -- and the claim
# "someone who used v2 finds their tool" is checked against ``docs/v2_to_v3_map.md`` by
# the suite rather than by looking at a tab bar.
#
# THE REQUIRED FIELDS ARE THE POINT. Thirteen parameters have no default, and Run stays
# disabled until every one is filled. A pre-filled dZ box would invent a measurement: the
# corrected slow fraction swings by tens of points
# from one number the data cannot supply. The same is true of the
# confinement radius, the null D and the gamma shape. Each field shows its own note.


class _MappingProject:
    """The two things `catalogue_view` asks a project for, over a plain mapping.

    Lets the panel reuse the tested pure builder instead of looping over conditions in a
    Qt class, which is the rule this module exists to keep.
    """

    def __init__(self, tables, projected=None):
        self._tables = dict(tables)
        self._projected = projected
        self.conditions = list(self._tables)

    def condition(self, name):
        return self._tables[name]

    def projected(self, name):
        """The table WITH cell coordinates, from whoever can supply it.

        The panel can be constructed from bare tables, which cannot be projected -
        projecting needs the outlines and a table does not carry them. So this raises
        with the reason rather than letting `project_tracks` fail three frames down on a
        missing argument.
        """
        if self._projected is None:
            raise AnalysisError(
                "this analysis needs cell-frame coordinates, which are added from the cell "
                "outlines. The panel was built from tables alone, so it has no outlines "
                "to project with; open the project through the browser, which supplies "
                "them")
        return self._projected(name)


class AnalysisPanelWidget(QWidget):
    """Browse and run the analyses. A thin shell over ``smtracker.analyses``."""

    def __init__(self, canonical, *, parent=None, on_result=None, only=None,
                 projected=None):
        """``canonical`` is a table OR ``{condition: table}``; ``only`` restricts the
        list to those analysis keys.

        Side by side is the default. Every analysis runs on every loaded condition, and a
        single table is the case where a single condition was loaded -- which the folder
        layout already distinguishes. Computing one of four answers a question nobody
        asked.

        ``only`` exists so the browser can put one v2 GROUP in each tab: the group is the
        tab, so the panel does not also need a group selector.
        """
        super().__init__(parent)
        if isinstance(canonical, dict):
            self._canonicals = dict(canonical)
        elif canonical is None:
            self._canonicals = {}
        else:
            label = "all"
            try:
                names = list(dict.fromkeys(canonical["condition"]))
                if len(names) == 1:
                    label = str(names[0])
            except Exception:                          # noqa: BLE001
                pass
            self._canonicals = {label: canonical}
        #: ``name -> projected table``, supplied by whoever owns the cell outlines. None
        #: when the panel was built from bare tables; the analyses that need it then say
        #: so instead of raising something obscure.
        self._projected = projected
        self._only = list(only) if only else None
        self._canonical = (next(iter(self._canonicals.values()))
                           if self._canonicals else None)
        self._on_result = on_result
        self._analysis = None
        self._entries = ()
        self._fields = {}
        self._result = None
        self._last_input = {}
        self._last_errors = {}

        layout = QVBoxLayout(self)

        self._tab = QComboBox()
        self._tab.addItems(list(TABS))
        self._tab.currentTextChanged.connect(self._tab_changed)
        self._tab_label = QLabel("v2 tab")
        layout.addWidget(self._tab_label)
        layout.addWidget(self._tab)
        if self._only is not None:
            # The GROUP is the tab, so a group selector inside it would be a second way
            # to answer a question already answered.
            self._tab_label.setVisible(False)
            self._tab.setVisible(False)

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.currentRowChanged.connect(self._selected)
        layout.addWidget(self._list)

        self._provenance = QLabel("")
        self._provenance.setWordWrap(True)
        self._provenance.setTextFormat(Qt.RichText)
        layout.addWidget(self._provenance)

        self._form_host = QWidget()
        self._form = QFormLayout(self._form_host)
        layout.addWidget(self._form_host)

        buttons = QHBoxLayout()
        self._run = QPushButton("Run")
        self._run.setEnabled(False)
        self._run.clicked.connect(self._run_clicked)
        buttons.addWidget(self._run)
        layout.addLayout(buttons)

        # A RESULT IS SHOWN AS WHAT IT IS. One analysis has a drawing that takes its
        # result; the other twenty-one return a table, and a table shown as a table is
        # honest where an invented chart reads as a finding. The panel says which of the
        # two it is doing rather than leaving the reader to guess.
        self._figure = Figure(figsize=(5, 3), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setVisible(False)
        layout.addWidget(self._canvas)

        self._table = QTableWidget()
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setVisible(False)
        layout.addWidget(self._table)

        self._output = QTextEdit()
        self._output.setReadOnly(True)
        layout.addWidget(self._output)

        self._tab_changed(self._tab.currentText())

    # ------------------------------------------------------------------ browsing
    def _tab_changed(self, tab):
        self._list.clear()
        entries = by_tab(tab)
        if self._only is not None:
            from .analyses import ANALYSES

            keep = set(self._only)
            entries = tuple(a for a in ANALYSES if a.key in keep)
        self._entries = entries
        for a in self._entries:
            self._list.addItem(a.label + "   [" + a.status + "]")
        self._analysis = None
        self._clear_form()
        self._provenance.setText("")
        self._run.setEnabled(False)

    def _selected(self, row):
        if row < 0 or row >= len(self._entries):
            return
        self._analysis = self._entries[row]
        self._provenance.setText(self._provenance_text(self._analysis))
        self._build_form(self._analysis)
        self._check_ready()

    @staticmethod
    def _provenance_text(a):
        """What the panel must say about an entry before it is run.

        The legacy source and the route back to the v2 behaviour are not decoration: a UI
        that shows only a label cannot distinguish the legacy tool from an improved method
        wearing its name, which is the confusion legacy mode exists to prevent.
        """
        parts = ["<b>" + a.label + "</b> &mdash; " + a.status]
        if a.legacy_source:
            parts.append("reproduces <tt>" + a.legacy_source + "</tt>")
        else:
            parts.append("<i>not in v2</i>")
        parts.append(a.summary)
        if a.legacy_route:
            parts.append("<b>v2 behaviour:</b> " + a.legacy_route)
        if a.caveat:
            parts.append("<b>Caveat:</b> " + a.caveat)
        return "<br><br>".join(parts)

    # ------------------------------------------------------------------- the form
    def _clear_form(self):
        self._fields = {}
        while self._form.count():
            item = self._form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _build_form(self, a):
        self._clear_form()
        for p in a.parameters:
            if p.kind == "choice":
                field = QComboBox()
                field.addItems([str(c) for c in p.choices])
                if p.default is not None:
                    field.setCurrentText(str(p.default))
                field.currentTextChanged.connect(self._check_ready)
            elif p.kind == "bool":
                field = QCheckBox()
                field.setChecked(bool(p.default))
                field.stateChanged.connect(self._check_ready)
            else:
                field = QLineEdit()
                # A required parameter renders EMPTY. Pre-filling it would turn a choice
                # the data cannot make into a number the user never saw.
                field.setText("" if p.default is None else str(p.default))
                if p.required:
                    field.setPlaceholderText("required - no default")
                field.textChanged.connect(self._check_ready)
            field.setToolTip(p.note)
            label = QLabel(p.label + (" *" if p.required else ""))
            label.setToolTip(p.note)
            label.setWordWrap(True)
            self._form.addRow(label, field)
            self._fields[p.name] = (p, field)

    def _value(self, p, field):
        # The widget's value goes through Parameter.coerce, so a pick-list of
        # comma-separated counts reaches the function as a tuple of ints rather than as
        # the string that broke the clustering fit.
        if p.kind in ("choice", "int_tuple", "str_tuple"):
            return p.coerce(field.currentText())
        if p.kind == "bool":
            return p.coerce(field.isChecked())
        text = field.text().strip()
        if not text:
            return None
        return p.coerce(text)

    def parameter_values(self):
        """Current values, ``None`` where a required field is still empty."""
        out = {}
        for name, (p, field) in self._fields.items():
            try:
                out[name] = self._value(p, field)
            except ValueError:
                out[name] = None
        return out

    def missing_required(self):
        """Required parameters not yet filled. Run stays disabled while any remain."""
        values = self.parameter_values()
        return tuple(name for name, (p, _) in self._fields.items()
                     if p.required and values.get(name) is None)

    def _check_ready(self, *_):
        self._run.setEnabled(self._analysis is not None and not self.missing_required())

    # ---------------------------------------------------------------------- running
    def _run_clicked(self):
        self._output.setPlainText("running...")
        try:
            self._result = self.run()
        except Exception as exc:                    # noqa: BLE001 - shown, never hidden
            # FAIL LOUD, in the panel. A GUI that swallows an analysis error leaves a
            # stale number on screen looking like a fresh one.
            self._output.setPlainText(type(exc).__name__ + ": " + str(exc))
            return
        self._output.setPlainText(self.describe(self._result))
        self.render(self._result)
        if self._on_result is not None:
            self._on_result(self._analysis, self._result)

    def run(self):
        """Build the input, call the target. No computation happens in this class."""
        if self._analysis is None:
            raise AnalysisError("no analysis selected")
        missing = self.missing_required()
        if missing:
            raise AnalysisError(
                "these have no default and must be chosen: " + ", ".join(missing))
        # EVERY CONDITION, through the pure builder. The panel does not loop over
        # conditions itself: `catalogue_view` keeps a failing condition NAMED beside the
        # ones that worked, refuses missing required parameters before anything runs, and
        # is tested without Qt.
        from .catalogue_view import catalogue_view

        view = catalogue_view(_MappingProject(self._canonicals, self._projected),
                              list(self._canonicals),
                              analysis_key=self._analysis.key,
                              params=self.parameter_values())
        self._last_input = {k: v["input"] for k, v in view["per_condition"].items()}
        self._last_errors = view["failed"]
        return {k: v["result"] for k, v in view["per_condition"].items()}

    def render(self, results):
        """One panel per condition, side by side, in the order they were loaded."""
        if not isinstance(results, dict):
            return self._render_one(results)
        drawing = self._analysis.resolve_draw()
        names = [n for n in self._canonicals if n in results]
        if drawing is not None and names:
            self._figure.clear()
            axes = self._figure.subplots(1, len(names), squeeze=False)[0]
            for ax, name in zip(axes, names):
                payload = self._analysis.draw_payload(results[name])
                if self._analysis.draw_wants_data():
                    drawing(ax, payload, data=self._last_input.get(name))
                else:
                    drawing(ax, payload)
                # The condition goes on the CENTRED title; every caption these drawings
                # write is a LEFT title, so the two cannot collide.
                ax.set_title(name, fontsize=7)
            self._canvas.draw()
            self._canvas.setVisible(True)
            self._fill_table(results)
            self._table.setVisible(True)
            return "figure"
        self._canvas.setVisible(False)
        self._fill_table(results)
        self._table.setVisible(True)
        return "table"

    def _render_one(self, result):
        """The single-result path, for a caller holding a bare result."""
        drawing = self._analysis.resolve_draw()
        if drawing is not None:
            # BOTH, not one or the other. The figure carries the fractions and the D's in
            # its legend, but bic, rss and n_converged_starts live only in the fields, and
            # hiding them to make room would lose the numbers that say whether the fit
            # converged. The table stays underneath.
            self._figure.clear()
            ax = self._figure.add_subplot(111)
            # A drawing that asks for `data` gets the analysis input too: a fitted curve
            # with no measured points beside it cannot be judged.
            if self._analysis.draw_wants_data():
                drawing(ax, self._analysis.draw_payload(result), data=self._last_input)
            else:
                drawing(ax, self._analysis.draw_payload(result))
            self._canvas.draw()
            self._canvas.setVisible(True)
            self._fill_table(result)
            self._table.setVisible(True)
            return "figure"
        self._canvas.setVisible(False)
        self._fill_table(result)
        self._table.setVisible(True)
        return "table"

    def _fill_table(self, results, max_rows=200):
        """One CONDITION column per result, fields down the rows."""
        if not isinstance(results, dict):
            return self._fill_table_one(results, max_rows=max_rows)

        frames = {k: v for k, v in results.items() if hasattr(v, "columns")}
        if frames:
            first = next(iter(frames))
            self._fill_table_one(frames[first], max_rows=max_rows)
            if len(frames) > 1:
                self._table.setHorizontalHeaderLabels(
                    [f"{first}: {c}" for c in list(frames[first].columns)])
            return

        from .analyses import result_rows

        per = {name: dict(result_rows(value)) for name, value in results.items()}
        fields = list(dict.fromkeys(k for d in per.values() for k in d))
        names = [n for n in self._canonicals if n in per]
        self._table.clear()
        self._table.setColumnCount(1 + len(names))
        self._table.setHorizontalHeaderLabels(["field"] + names)
        self._table.setRowCount(len(fields))
        for r, field in enumerate(fields):
            self._table.setItem(r, 0, QTableWidgetItem(field))
            for c, name in enumerate(names, start=1):
                self._table.setItem(r, c, QTableWidgetItem(per[name].get(field, "")))

    def _fill_table_one(self, result, max_rows=200):
        # `or []` on a pandas Index raises: its truth value is ambiguous. Ask for the
        # attribute, then convert, then test the list.
        raw = getattr(result, "columns", None)
        columns = [] if raw is None else list(raw)
        if not columns:
            # NOT A TABLE, SO SHOW WHAT IT IS. Fifteen analyses return a fit object or a
            # dict; looking for .columns and giving up rendered an empty box for the SQD
            # fit, which is the fraction readout.
            pairs = result_rows(result)
            self._table.clear()
            self._table.setColumnCount(2)
            self._table.setHorizontalHeaderLabels(["field", "value"])
            self._table.setRowCount(len(pairs))
            for r, (name, value) in enumerate(pairs):
                self._table.setItem(r, 0, QTableWidgetItem(name))
                self._table.setItem(r, 1, QTableWidgetItem(value))
            return
        rows = result.head(max_rows)
        self._table.clear()
        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels([str(c) for c in columns])
        self._table.setRowCount(len(rows))
        for r, (_, row) in enumerate(rows.iterrows()):
            for c, column in enumerate(columns):
                self._table.setItem(r, c, QTableWidgetItem(str(row[column])))

    def describe(self, results):
        """A short summary naming every condition and every failure."""
        if isinstance(results, dict):
            lines = [self._analysis.label, "",
                     f"conditions computed: {', '.join(results)}"]
            for name, message in (self._last_errors or {}).items():
                lines.append(f"  {name}: FAILED - {message}")
            if self._analysis.draw:
                lines.append("shown as: a figure, one panel per condition ("
                             + self._analysis.draw + ")")
            else:
                lines.append("shown as: the result TABLE, one column per condition -- "
                             "this analysis has no figure yet")
            for name, value in results.items():
                n = len(value) if hasattr(value, "__len__") else 0
                if n > 200:
                    lines.append(f"{name}: table shows the first 200 of {n} rows")
            if self._analysis.caveat:
                lines += ["", "CAVEAT: " + self._analysis.caveat]
            return "\n".join(lines)
        return self._describe_one(results)

    def _describe_one(self, result):
        """A short text summary. Drawing is the caller's job, via ``on_result``."""
        lines = [self._analysis.label, ""]
        attrs = getattr(result, "attrs", None)
        if attrs:
            for key, value in attrs.items():
                if isinstance(value, (int, float, str, bool)):
                    lines.append(str(key) + ": " + str(value))
        shape = getattr(result, "shape", None)
        if shape is not None:
            lines.append("shape: " + str(shape))
        if self._analysis.draw:
            lines.append("shown as: a figure (" + self._analysis.draw + ")")
        else:
            n = len(result) if hasattr(result, "__len__") else 0
            lines.append("shown as: the result TABLE -- this analysis has no figure yet")
            if n > 200:
                # Silent truncation reads as "that is all there is".
                lines.append("table shows the first 200 of " + str(n) + " rows")
        if self._analysis.caveat:
            lines += ["", "CAVEAT: " + self._analysis.caveat]
        return "\n".join(lines)


def dock_analyses(viewer, canonical, *, on_result=None, area="right",
                  name="analyses", **kwargs):
    """Dock the analysis panel."""
    widget = AnalysisPanelWidget(canonical, on_result=on_result, **kwargs)
    viewer.window.add_dock_widget(widget, area=area, name=name)
    return widget


# =====================================================================================
# THE TRACK VIEWS - the two static ones and the player, without a second window
# =====================================================================================
#
# These were built, tested and fast, and there was no way to reach any of them from the
# command line. That is its own kind of missing feature.
#
# ALL THREE ARE MATPLOTLIB, so this dock costs no Qt window and no OpenGL context; the
# napari viewer it sits inside has already paid for those. Measured: the pooled view draws
# a lab project's tracks in 0.045 s, the cell explorer in 0.026 s, and the player runs 1530 to 2884
# frames per second across 28 movies, 31 to 58 times the 50 fps acquisition.


class TrackViewsWidget(QWidget):
    """Pooled tracks, one cell, or playback. Drawing lives in ``plots``/``playback``."""

    VIEWS = ("all tracks on one cell", "single cell and track", "playback")

    def __init__(self, canonical, *, parent=None, projected=None, movie=None,
                 background=None, pixel_size_nm=None, frame_interval_ms=None):
        super().__init__(parent)
        self._canonical = canonical
        self._projected = projected          # cell-frame table, for the two static views
        self._movie = movie
        self._background = background
        self._pixel_size_nm = pixel_size_nm
        self._frame_interval_ms = frame_interval_ms
        self._player = None

        layout = QVBoxLayout(self)
        self._view = QComboBox()
        self._view.addItems(list(self.VIEWS))
        self._view.currentTextChanged.connect(self._view_changed)
        layout.addWidget(self._view)

        self._cell = QComboBox()
        # Refill the track list FIRST: changing the cell leaves the previous cell's uid
        # selected, and draw_cell_explorer rightly refuses a track that is not in the
        # cell. Redrawing without refilling turns a stale selection into an error message.
        self._cell.currentTextChanged.connect(self._cell_changed)
        self._cell_label = QLabel("cell")
        layout.addWidget(self._cell_label)
        layout.addWidget(self._cell)

        self._uid = QComboBox()
        self._uid.currentTextChanged.connect(self._redraw)
        self._uid_label = QLabel("track")
        layout.addWidget(self._uid_label)
        layout.addWidget(self._uid)

        # A MOVIE PICKER, because playback draws ONE movie's tracks over ONE movie's
        # image. Handed a whole condition the player rightly refuses ("30 movies in one
        # table"), and a refusal with no way to satisfy it is a dead end.
        self._movie = QComboBox()
        self._movie.currentTextChanged.connect(self._movie_changed)
        self._movie_label = QLabel("movie")
        layout.addWidget(self._movie_label)
        layout.addWidget(self._movie)

        self._frame = QLineEdit()
        self._frame.setPlaceholderText("frame")
        self._frame.textChanged.connect(self._redraw)
        self._frame_label = QLabel("frame")
        layout.addWidget(self._frame_label)
        layout.addWidget(self._frame)

        if "movie" in canonical.columns:
            self._movie.blockSignals(True)
            self._movie.addItems(sorted(str(m) for m in canonical["movie"].unique()))
            if movie is not None and str(movie) in [
                    self._movie.itemText(i) for i in range(self._movie.count())]:
                self._movie.setCurrentText(str(movie))
            self._movie.blockSignals(False)

        self._figure = Figure(figsize=(5, 3), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        layout.addWidget(self._canvas)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._view_changed(self._view.currentText())

    # --------------------------------------------------------------- which controls
    def _view_changed(self, view):
        static_one = view == "single cell and track"
        play = view == "playback"
        for w in (self._cell, self._cell_label, self._uid, self._uid_label):
            w.setVisible(static_one)
        for w in (self._frame, self._frame_label, self._movie, self._movie_label):
            w.setVisible(play)
        if static_one:
            self._fill_cells()
        self._redraw()

    def select(self, *, view=None, cell=None, uid=None):
        """Open on one view, cell and track -- what the Spatial Map's spot card does
        when it hands a track over. Each step goes through the combo boxes, so the
        cell list is refilled before the track is chosen, as a click would."""
        if view is not None:
            self._view.setCurrentText(view)
        if cell is not None:
            self._cell.setCurrentText(str(cell))
        if uid is not None:
            self._uid.setCurrentText(str(uid))

    def _movie_changed(self, *_):
        # A new movie needs a new player: the old one holds the previous movie's arrays.
        self._player = None
        self._redraw()

    def _fill_cells(self):
        if self._projected is None:
            return
        self._cell.blockSignals(True)
        self._cell.clear()
        self._cell.addItems(sorted(str(c) for c in self._projected["cell"].dropna()
                                   .unique()))
        self._cell.blockSignals(False)
        self._fill_uids()

    def _cell_changed(self, *_):
        self._fill_uids()
        self._redraw()

    def _fill_uids(self):
        if self._projected is None or not self._cell.currentText():
            return
        rows = self._projected[self._projected["cell"] == self._cell.currentText()]
        self._uid.blockSignals(True)
        self._uid.clear()
        self._uid.addItems(sorted(str(u) for u in rows["uid"].unique()))
        self._uid.blockSignals(False)

    # ----------------------------------------------------------------------- drawing
    def _redraw(self, *_):
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        try:
            self._draw(ax)
        except Exception as exc:                    # noqa: BLE001 - shown, never hidden
            # A view that silently draws nothing looks like an empty dataset.
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", wrap=True,
                    fontsize=6, transform=ax.transAxes)
            ax.set_axis_off()
            self._status.setText(type(exc).__name__ + ": " + str(exc))
        self._canvas.draw()

    def _draw(self, ax):
        view = self._view.currentText()
        if view == "playback":
            return self._draw_playback(ax)
        if self._projected is None:
            raise ValueError(
                "these views need cell-frame coordinates, which cellcoords."
                "project_tracks supplies. Drawing raw positions on a model cell would "
                "put every track in the camera's frame and still look plausible")
        if view == "all tracks on one cell":
            draw_all_tracks_on_cell(ax, self._projected)
            density = getattr(ax, "saturation_localisations_per_pixel", None)
            self._status.setText(
                "" if density is None else
                ("%.1f localisations per camera pixel of cell area. Above 1.0 the ink "
                 "cannot separate tracks, and neither can the detector, so this shows "
                 "the envelope rather than individual paths." % density))
            return
        cell = self._cell.currentText()
        uid = self._uid.currentText() or None
        if not cell:
            raise ValueError("no cell selected")
        draw_cell_explorer(ax, self._projected, cell=cell, uid=uid)
        self._status.setText(ax.get_title(loc="left"))

    def _draw_playback(self, ax):
        from .playback import TrackPlayer

        if self._pixel_size_nm is None:
            raise ValueError(
                "no pixel size, and playback will not guess one: the conversion is the "
                "one place a viewer can be wrong by a whole cell")
        if self._player is None:
            rows = self._canonical
            chosen = self._movie.currentText() if self._movie.count() else None
            if chosen and "movie" in rows.columns:
                rows = rows[rows["movie"].astype(str) == chosen]
            self._player = TrackPlayer(
                rows, pixel_size_nm=self._pixel_size_nm, background=self._background,
                frame_interval_ms=self._frame_interval_ms)
        self._player.invalidate()
        self._player.attach(ax)
        text = self._frame.text().strip()
        frame = self._player.first_frame if not text else int(text)
        self._player.draw_frame(frame, blit=False)
        self._status.setText(self._player.caption())


def dock_views(viewer, canonical, *, area="right", name="track views", **kwargs):
    """Dock the track views."""
    widget = TrackViewsWidget(canonical, **kwargs)
    viewer.window.add_dock_widget(widget, area=area, name=name)
    return widget
