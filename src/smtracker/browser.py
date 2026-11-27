"""Browser windows: comparison-first by default, per-condition on drill-down.

The tool measures the SHIFT between conditions, so the DEFAULT window is
``ComparisonBrowser``: a condition multi-select pinned at the top drives every
method tab at once, and one condition is just the N = 1 case. ``MethodBrowser``
(the earlier per-condition scaffold) remains as the DRILL-DOWN detail, clicking
a bubble / violin opens that one condition's per-cell views.

Both obey the thin-shell rule of ``smtracker.widgets``: **nothing is computed
inside a Qt class.** A tab holds a matplotlib canvas and a Compute button;
pressing it calls a pure controller (``comparison_controller`` /
``browser_controller``, numpy/pandas only, headless-tested) that returns a draw
function + a status string. A tab that resolves to the wrong number fails a
controller test, not by looking subtly wrong on screen, and an AST test keeps
numpy/pandas/scipy out of this file.

Compute is **on demand**, per tab: the expensive methods idle until their tab's
button is pressed, so opening the window is instant. A ``ResultCache`` memoizes by
provenance, so a condition already computed is served instantly on the next press.

FEEL still deferred: live drag, animated transitions and publication styling are
left for the experimenter. The drill-down, the pickers, the model dropdowns and the forest
toggles ARE wired functionally here: rough, not polished.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class _ComputeSignals(QObject):
    """Signals from a compute worker back to the GUI thread (Item 7a)."""

    # (generation token, result_or_None, error_or_None)
    done = Signal(int, object, object)


class _ComputeRunnable(QRunnable):
    """Runs one view's compute OFF the GUI thread, then signals the result.

    No nested pools (Item 7e): pools are disabled globally for the duration
    (`parallel.set_serial_only`), so a bootstrap inside a per-condition fit cannot
    spawn a process pool from this worker thread. Cancellation is cooperative,
    the compute checks its cancel hook between conditions and raises, which is
    caught here and reported as a cancel (nothing is painted or cached).
    """

    def __init__(self, compute, gen, signals):
        super().__init__()
        self._compute = compute
        self._gen = gen
        self._signals = signals

    def run(self):                                     # pragma: no cover - thread
        from . import parallel
        from .comparison import CancelledError

        parallel.set_serial_only(True)
        try:
            result = self._compute()
            self._signals.done.emit(self._gen, result, None)
        except CancelledError:
            self._signals.done.emit(self._gen, None, "cancelled")
        except Exception as exc:                       # noqa: BLE001 - surfaced
            self._signals.done.emit(self._gen, None, exc)
        finally:
            parallel.set_serial_only(False)

from . import browser_controller as ctrl
from . import comparison_controller as cctrl
from .comparison import (DEFAULT_D_SHARING_SCOPE, MODEL_CHOICES,
                         ComparisonSpec)
from .fastmode import BANNER, FastProfile
from .figstyle import JOURNAL_PRESETS

__all__ = ["MethodBrowser", "ComparisonBrowser", "launch_browser"]

#: Mobility-threshold slider range (µm²/s) and the integer-slider mapping. The
#: slider carries thousandths of a µm²/s, so 0.15 µm²/s <-> slider value 150.
_THRESH_MIN = 0.01
_THRESH_MAX = 1.0


def _thresh_to_slider(value: float) -> int:
    return int(round(1000 * value))


def _slider_to_thresh(value: int) -> float:
    return value / 1000.0


def _drill_detail(selection) -> str:
    """Short suffix naming the finer identity a click carried (method / vs-ref /
    population), for the drill window title. Empty for a plain condition click."""
    bits = []
    method = getattr(selection, "method", None)
    reference = getattr(selection, "reference", None)
    pop = getattr(selection, "pop", None)
    if method:
        bits.append(method)
    if reference:
        bits.append(f"vs {reference}")
    if pop:
        bits.append(f"pop {pop}")
    return f": {' · '.join(bits)}" if bits else ""


#: Views whose numbers depend on the mobility threshold (so a threshold drag
#: redraws them). Apparent-D and kinetics do not depend on it and are skipped.
_THRESHOLD_DEPENDENT = set()   # threshold control removed (Amendment 4)


def _glossary_tooltip(keys) -> str:
    """Hover text for a tab from the glossary (compact blocks). Empty if none."""
    if not keys:
        return ""
    from .glossary import tooltip

    parts = []
    for key in keys:
        try:
            parts.append(tooltip(key))
        except Exception:                              # pragma: no cover - missing key
            continue
    return "\n\n".join(parts)


class _MethodTab(QWidget):
    """A canvas + a compute button + a status line. Computes nothing itself."""

    def __init__(self, title, compute, *, note="", parent=None):
        super().__init__(parent)
        self._compute = compute            # pure callable -> (draw_fn, status)
        self._figure = Figure(figsize=(8.0, 5.0))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._status = QLabel(note or "press Compute")
        self._status.setWordWrap(True)

        button = QPushButton(f"Compute: {title}")
        button.clicked.connect(self.run)
        export = QPushButton("Export figure")
        export.clicked.connect(self.export)
        self._title = title

        layout = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(button)
        top.addWidget(export)
        top.addWidget(self._status, stretch=1)
        layout.addLayout(top)
        layout.addWidget(self._canvas, stretch=1)
        self.setLayout(layout)

    def export(self, *, path=None) -> list:
        """Export the current canvas to vector + high-DPI raster. Rough wiring,
        the final look tuning is left for later; the PIPELINE is what matters."""
        from .figstyle import export_figure

        if not self._figure.axes:
            self._status.setText("nothing to export: compute first")
            return []
        stem = path or (self._title.split(":")[0].strip().replace(" ", "_") or "figure")
        written = export_figure(self._figure, stem, formats=("png", "svg", "pdf"))
        self._status.setText("exported: " + ", ".join(p.name for p in written))
        return written

    def run(self) -> bool:
        """Compute this tab's analysis and paint it. Returns success."""
        try:
            draw, status = self._compute()
        except Exception as exc:                       # noqa: BLE001 - surfaced
            self._status.setText(f"failed: {exc}")
            return False
        self._figure.clear()
        draw(self._figure)
        self._canvas.draw_idle()
        self._status.setText(status)
        return True


class MethodBrowser(QWidget):
    """One window, every existing analysis reachable in its own tab."""

    #: (label, controller attribute). Each controller fn returns (draw, status).
    TABS = [
        ("Fractions (SQD + cross-method)", "fractions"),
        ("SuperPlot: distribution", "superplot_distribution"),
        ("SuperPlot: fraction", "superplot_fraction"),
        ("Kinetics (HMM)", "kinetics"),
        ("State count (saSPT)", "state_count"),
        ("Spatial maps", "spatial"),
        ("Confinement", "confinement"),
        ("Clustering", "clustering"),
        ("Spot-On", "spoton"),
        ("QC dashboard", "qc"),
    ]

    #: A clicked method opens ON its own tab, the identity survives the click
    #: instead of landing on the generic first tab. A clicked population (spatial
    #: panel, no method) opens the spatial map.
    METHOD_TO_TAB = {"SQD": "fractions", "HMM": "kinetics",
                     "appD": "superplot_distribution", "saSPT": "state_count"}

    def __init__(self, project, *, condition=None, n_boot=50, selection=None,
                 parent=None):
        super().__init__(parent)
        self._project = project
        self._selection = selection            # the drill identity, or None
        self._condition = (getattr(selection, "condition", None)
                           or condition or project.conditions[0])
        self._n_boot = n_boot

        self._picker = QComboBox()
        self._picker.addItems(list(project.conditions))
        self._picker.setCurrentText(self._condition)
        self._picker.currentTextChanged.connect(self._on_condition)

        self._tabs = QTabWidget()
        self._method_tabs = {}
        for label, attr in self.TABS:
            fn = getattr(ctrl, attr)
            tab = _MethodTab(label, self._bind(fn), note=self._note(attr))
            self._method_tabs[attr] = tab
            self._tabs.addTab(tab, label)

        # Inert placeholder for the feel/aesthetic layer (left for the experimenter).
        placeholder = QLabel(
            "Placeholder: drill-down, live sensitivity sliders, publication "
            "styling and guided navigation go here (not built in the scaffold)."
        )
        placeholder.setWordWrap(True)
        self._tabs.addTab(placeholder, "· polish (later) ·")

        header = QHBoxLayout()
        header.addWidget(QLabel("condition:"))
        header.addWidget(self._picker)
        # The drill filter chip: names what was clicked (method / population) and
        # widens back to the whole condition in one click. Only shown when drilled.
        self._filter_chip = QLabel()
        self._filter_chip.setStyleSheet(
            "QLabel { background:#2d3748; color:white; padding:2px 8px;"
            " border-radius:8px; }")
        self._filter_clear = QPushButton("✕ widen")
        self._filter_clear.setToolTip("Clear the drill filter: back to the whole condition")
        self._filter_clear.clicked.connect(self._clear_filter)
        header.addSpacing(12)
        header.addWidget(self._filter_chip)
        header.addWidget(self._filter_clear)
        header.addStretch(1)
        header.addWidget(QLabel(f"n_boot={n_boot} (dev)"))

        layout = QVBoxLayout()
        layout.addLayout(header)
        layout.addWidget(self._tabs, stretch=1)
        self.setLayout(layout)

        self._apply_selection()                # chip + focus the clicked tab

    def _apply_selection(self):
        """Reflect the drill selection: show the chip and open the tab the clicked
        method (or population) implies, so the click lands you on what you clicked."""
        sel = self._selection
        if sel is None:
            self._filter_chip.setVisible(False)
            self._filter_clear.setVisible(False)
            return
        bits = [self._condition]
        method = getattr(sel, "method", None)
        pop = getattr(sel, "pop", None)
        kind = getattr(sel, "kind", None)
        # A population index restricts the view to ONE population ONLY for a genuine
        # spatial-panel click (each panel IS a population). A bubble/table click
        # drilling into the CONDITION view must show ALL of that condition's
        # populations, so its pop is dropped here: it does not narrow the chip and
        # does not redirect to the per-population spatial tab (Item 5). The clicked
        # population still rides the window title, informationally.
        pop_restricts = pop if kind == "spatial_panel" else None
        if method:
            bits.append(method)
        if pop_restricts:
            bits.append(f"pop {pop_restricts}")
        self._filter_chip.setText("drilled: " + " · ".join(bits))
        self._filter_chip.setVisible(True)
        self._filter_clear.setVisible(True)
        # focus the tab the identity points at
        target = self.METHOD_TO_TAB.get(method) if method else None
        if target is None and pop_restricts:
            target = "spatial"
        if target and target in self._method_tabs:
            self._tabs.setCurrentWidget(self._method_tabs[target])

    def _clear_filter(self):
        """Widen back to the whole condition: the chip's one-click escape."""
        self._selection = None
        self._apply_selection()

    # -- wiring ----------------------------------------------------------- #

    def _bind(self, fn):
        """Freeze the current project/condition/n_boot into a zero-arg compute."""
        return lambda: fn(self._project, self._condition, n_boot=self._n_boot)

    def _note(self, attr):
        expensive = {"kinetics", "state_count", "spatial", "confinement",
                     "clustering", "spoton"}
        return ("expensive: computes on demand; press Compute"
                if attr in expensive else "press Compute")

    def _on_condition(self, condition):
        self._condition = condition
        # Changing the condition by hand invalidates a mark-level drill filter.
        if self._selection is not None:
            self._selection = None
            self._apply_selection()
        # Re-bind every tab to the new condition; they recompute on next press.
        for attr, tab in self._method_tabs.items():
            tab._compute = self._bind(getattr(ctrl, attr))

    def compute_all(self) -> dict:
        """Run every tab once: used by the launch smoke test."""
        return {attr: tab.run() for attr, tab in self._method_tabs.items()}


def launch_browser(project, *, root=None, condition=None, n_boot=50, show=True,
                   session=None, fast=False, refresh=False):
    """Open the COMPARISON-first browser (the default entry window).

    ``condition`` pre-ticks a single condition when given; otherwise the session
    (or all conditions) drives the initial selection. ``fast`` starts in the
    honesty-guarded dev fast mode. ``refresh`` rebuilds the warm cache instead of
    serving it. The warm cache lives BESIDE the data
    (``<project>/.smtracker_warm``) so it travels with the SSD; ``root`` names the
    project folder, else it is derived from the loaded conditions.
    """
    from qtpy.QtWidgets import QApplication

    from .warmcache import WarmCache, warm_root_for

    if root is None:
        # The project root is the common parent of the condition folders.
        root = next(iter(project.roots.values())).parent
    app = QApplication.instance() or QApplication([])
    from .appicon import apply as _apply_icon
    _apply_icon()
    # Per-condition results cache inside each condition's own folder (so they
    # travel with it); the cross-condition views cache at the project root.
    cache = WarmCache(warm_root_for(root), refresh=refresh,
                      condition_roots=dict(project.roots))
    browser = ComparisonBrowser(project, n_boot=n_boot, session=session,
                                initial=[condition] if condition else None,
                                fast=fast, cache=cache)
    browser.setWindowTitle("SMTracker · comparison")
    if show:
        browser.show()
    return app, browser


def launch_method_browser(project, *, condition=None, n_boot=50, show=True):
    """Open the per-condition MethodBrowser directly (the drill-down detail)."""
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from .appicon import apply as _apply_icon
    _apply_icon()
    browser = MethodBrowser(project, condition=condition, n_boot=n_boot)
    browser.setWindowTitle("SMTracker · method browser (first draft)")
    if show:
        browser.show()
    return app, browser


# --------------------------------------------------------------------------- #
# Comparison-first browser
# --------------------------------------------------------------------------- #


#: Fit All order, cheapest first, so SQD/appD populate while saSPT/HMM/spatial
#: are still going. PER-CONDITION only: the cross-condition views (fraction, heatmap,
#: forest) were removed (Amendment 4), so they are not fit here.
FIT_ALL_ORDER = ["sqd_view", "appd_view", "saspt_view",
                 "kinetics_view", "spatial_view"]


class _FitAllSignals(QObject):
    """Signals from the Fit All batch worker (Item 8)."""

    # (index, total, attr, status): an item is about to compute
    started = Signal(int, int, str)
    # (attr, result_or_None, error_or_None, skipped)
    item_done = Signal(str, object, object, bool)
    # (summary dict)
    finished = Signal(object)


class _FitAllRunnable(QRunnable):
    """Compute EVERY view in the background, cheapest first (Item 8).

    Owns pool safety by running serial (``set_serial_only``) so it never nests a
    process pool from this worker thread. Each unit: skipped if already cached and
    valid; otherwise computed and cached AS IT FINISHES (so a cancel keeps the
    completed units). Failures do not abort: a failed unit is recorded with its
    reason and the batch continues. Cancellation is cooperative (checked between
    units and, inside a unit, between conditions).
    """

    def __init__(self, jobs, signals, is_cancelled):
        super().__init__()
        self._jobs = jobs                  # list of (attr, compute, is_cached)
        self._signals = signals
        self._is_cancelled = is_cancelled

    def run(self):                                     # pragma: no cover - thread
        from . import parallel

        parallel.set_serial_only(True)
        summary = {"done": [], "skipped": [], "failed": [], "cancelled": False}
        total = len(self._jobs)
        try:
            for i, (attr, compute, is_cached) in enumerate(self._jobs):
                if self._is_cancelled():
                    summary["cancelled"] = True
                    break
                self._signals.started.emit(i, total, attr)
                if is_cached():
                    summary["skipped"].append(attr)
                    self._signals.item_done.emit(attr, None, None, True)
                    continue
                try:
                    result = compute()
                    summary["done"].append(attr)
                    self._signals.item_done.emit(attr, result, None, False)
                except Exception as exc:               # noqa: BLE001 - continue
                    summary["failed"].append((attr, str(exc)))
                    self._signals.item_done.emit(attr, None, exc, False)
        finally:
            parallel.set_serial_only(False)
            self._signals.finished.emit(summary)


class _ComparisonTab(QWidget):
    """A canvas + Compute + Export for ONE comparison view. Computes nothing.

    ``compute`` is a zero-arg callable → ``(draw, status)`` that reads the live
    spec each time it runs, so a control change needs no rebinding, the next
    Compute press just picks up the current selection. ``on_drill`` is called with
    the condition a canvas click lands on (see ``condition_at_x``).
    """

    def __init__(self, title, compute, *, provenance=None, on_drill=None,
                 conditions=None, preset=None, note="", local=None, parent=None):
        super().__init__(parent)
        self._compute = compute
        self._provenance = provenance          # zero-arg -> Provenance, for export
        self._on_drill = on_drill
        self._conditions = conditions          # zero-arg -> list, for drill hit-test
        self._preset = preset                  # zero-arg -> (preset_name, width)
        self._title = title
        self._figure = Figure(figsize=(8.0, 5.0))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.mpl_connect("button_press_event", self._clicked)
        self._canvas.mpl_connect("motion_notify_event", self._hover)
        self._status = QLabel(note or "press Compute")
        self._status.setWordWrap(True)

        button = QPushButton(f"Compute: {title}")
        button.clicked.connect(self.run_async)
        export = QPushButton("Export figure")
        export.clicked.connect(self.export)

        # Responsiveness (Item 7): compute off the GUI thread, per-tab progress +
        # cancel, stale-result guard by generation token.
        self._gen = 0                          # generation counter (stale guard)
        self._cancelled = False
        self._pool = QThreadPool.globalInstance()
        self._signals = _ComputeSignals()
        self._signals.done.connect(self._on_done)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)          # indeterminate (honest: no % faked)
        self._progress.setVisible(False)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.cancel)
        self._cancel_btn.setVisible(False)

        # Layout: canvas gets the room; this view's OWN controls sit in a strip
        # directly beneath it (same place every tab); the action row is last. The
        # global drivers live in the window's top strip, never duplicated here.
        layout = QVBoxLayout()
        layout.addWidget(self._canvas, stretch=1)
        if local is not None:
            layout.addWidget(local)
        actions = QHBoxLayout()
        actions.addWidget(button)
        actions.addWidget(export)
        actions.addWidget(self._progress, stretch=1)
        actions.addWidget(self._cancel_btn)
        actions.addWidget(self._status, stretch=1)
        layout.addLayout(actions)
        self.setLayout(layout)

    def has_drawn(self) -> bool:
        """Whether this tab has been computed at least once (has a live figure).
        Used to gate live-redraw: don't fit a never-opened tab on a slider drag."""
        return bool(self._figure.axes)

    def _message(self, text, *, status=None):
        """Render a clear message ON the canvas (never a stale/blank plot or a
        traceback). Used for empty states, progress, and honest failures."""
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=11,
                color="#4a5568", transform=ax.transAxes)
        self._canvas.draw_idle()
        self._status.setText(status if status is not None else text.replace("\n", " "))

    def run(self) -> bool:
        # Honest empty state: nothing to compute without a condition.
        conditions = self._conditions() if self._conditions else None
        if conditions is not None and len(conditions) == 0:
            self._message("Select at least one condition to compare.")
            return False
        # Visible progress: show it and let the event loop paint before blocking.
        # Only under a REAL display, a headless (offscreen/minimal) platform paints
        # nothing, and pumping the event loop there can hit a native access
        # violation once napari/GL state has accumulated across a full test run
        # (a flaky segfault, seen in CI). Skipping it headless removes that vector
        # and costs the real GUI nothing.
        self._message("computing…", status="computing…")
        import os

        if os.environ.get("QT_QPA_PLATFORM", "").lower() not in ("offscreen", "minimal"):
            try:
                from qtpy.QtWidgets import QApplication

                QApplication.processEvents()
            except Exception:                      # pragma: no cover - no Qt loop
                pass
        try:
            draw, status = self._compute()
            self._figure.clear()
            draw(self._figure)
            self._canvas.draw_idle()
            self._status.setText(status)
            # Capture EXACTLY what was plotted, so a later export re-uses this
            # draw + provenance rather than re-deriving from a picker the user may
            # have changed in between (data integrity, see export()).
            self._last_draw = draw
            self._last_provenance = getattr(self._figure,
                                            "_smtracker_provenance", None)
            return True
        except Exception as exc:               # noqa: BLE001 - surfaced, not swallowed
            # A clear message on the canvas, not a stack trace or a misleading plot.
            self._message(f"Could not compute this view:\n{exc}",
                          status=f"failed: {exc}")
            self._last_draw = None
            self._last_provenance = None
            return False

    # -- async path (Item 7): compute off the GUI thread --------------------- #

    def run_async(self) -> bool:
        """Dispatch the compute to a worker thread so the GUI stays interactive.

        Bumps the generation token (invalidating any in-flight run), shows the
        per-tab progress + cancel, and injects a cancel hook the compute checks
        between conditions. The result is delivered by signal to ``_on_done``,
        which DROPS it if the token no longer matches (stale-result guard).
        """
        conditions = self._conditions() if self._conditions else None
        if conditions is not None and len(conditions) == 0:
            self._message("Select at least one condition to compare.")
            return False
        self._gen += 1
        gen = self._gen
        self._cancelled = False

        def should_cancel(gen=gen):
            return self._cancelled or self._gen != gen

        def compute():
            return self._compute(should_cancel)

        self._progress.setVisible(True)
        self._cancel_btn.setVisible(True)
        self._status.setText("computing…")
        self._pool.start(_ComputeRunnable(compute, gen, self._signals))
        return True

    def _on_done(self, gen, result, error):
        # STALE-RESULT GUARD: a result from a superseded run never reaches the
        # canvas. The user changed something (or cancelled) → token advanced.
        if gen != self._gen:
            return
        self._progress.setVisible(False)
        self._cancel_btn.setVisible(False)
        if error == "cancelled":
            self._status.setText("cancelled: previous figure kept")
            return
        if error is not None:
            self._message(f"Could not compute this view:\n{error}",
                          status=f"failed: {error}")
            self._last_draw = None
            self._last_provenance = None
            return
        draw, status = result
        self.apply_result(draw, status)

    def apply_result(self, draw, status):
        """Paint a computed (draw, status) and capture it for export. Shared by the
        async worker and Fit All."""
        self._figure.clear()
        draw(self._figure)
        self._canvas.draw_idle()
        self._status.setText(status)
        self._last_draw = draw
        self._last_provenance = getattr(self._figure, "_smtracker_provenance", None)

    def cancel(self):
        """Cooperative cancel: stop before the next condition, drop the in-flight
        result (token bumped), keep the previous figure. Writes no whole-view
        result."""
        self._cancelled = True
        self._gen += 1                         # any arriving result is now stale
        self._progress.setVisible(False)
        self._cancel_btn.setVisible(False)
        self._status.setText("cancelling…")

    def export(self) -> list:
        """Export the view. Data integrity: the export re-uses the draw + the
        provenance captured when the figure was plotted, never a fresh compute from
        the live picker, so the exported figure and its provenance companion can
        never describe a different condition than what is on screen.

        For a journal preset the SAME captured draw is re-rendered under that
        preset's rcParams at the journal column size (rcParams only affect newly
        drawn artists, so styling the existing figure would do nothing). 'screen'
        exports the current canvas as-is."""
        from .figstyle import export_figure, figure_size_inches, style

        if not self._figure.axes or getattr(self, "_last_draw", None) is None:
            self._status.setText("nothing to export: compute first")
            return []
        stem = self._title.split(":")[0].strip().replace(" ", "_") or "figure"
        # the provenance of the DATA ACTUALLY PLOTTED, not a live re-derivation
        prov = getattr(self, "_last_provenance", None)
        if prov is None and self._provenance:
            prov = self._provenance()
        preset, width = self._preset() if self._preset else ("screen", "single")

        if preset == "screen":
            figure = self._figure
            written = export_figure(figure, stem, formats=("png", "svg", "pdf"),
                                    provenance=prov)
        else:
            # re-render the SAME captured draw at journal size (NOT a fresh compute)
            with style(preset):
                figure = Figure(figsize=figure_size_inches(preset, width=width))
                self._last_draw(figure)
                written = export_figure(figure, f"{stem}_{preset}_{width}",
                                        formats=("png", "svg", "pdf"), provenance=prov)
        self._status.setText(
            f"exported ({preset}): " + ", ".join(p.name for p in written))
        return written

    def _resolve(self, event):
        """The Selection under an event, via the view's OWN hit-test (attached
        from its plotted geometry), or the legacy column fallback if a view
        attached none. None off-axes or on no mark. Shared by click and hover so
        the two can never disagree about what is under the cursor."""
        if event.inaxes is None:
            return None
        from . import hittest as _hittest

        picker = _hittest.hit_test_of(self._figure)
        if picker is not None:
            return picker.pick(event.xdata, event.ydata, event.inaxes)
        if event.xdata is not None:
            conditions = self._conditions() if self._conditions else []
            cond = cctrl.condition_at_x(conditions, event.xdata)
            return _hittest.Selection(condition=cond) if cond else None
        return None

    def _clicked(self, event) -> None:
        if self._on_drill is None:
            return
        selection = self._resolve(event)
        if selection is None:
            return
        # In-place row↔bubble highlight for views that support it (the bubble +
        # its side table), reusing the SAME resolved selection: no second mapping,
        # no recompute. A no-op (returns False) on every other view.
        from . import compare_plots as _cp

        if _cp.rehighlight(self._figure, selection):
            self._canvas.draw_idle()
        self._on_drill(selection)

    def _hover(self, event) -> None:
        """Affordance rule: a mark that encodes a data identity shows a hand
        cursor + a tooltip naming what a click would drill into; everything else
        stays a plain arrow with no tooltip. ONLY real hit-testable marks get the
        hand: inert-but-inviting teaches distrust of clicks, so it is avoided by
        driving the affordance off the SAME resolve the click uses."""
        selection = self._resolve(event) if self._on_drill is not None else None
        if selection is not None:
            self._canvas.setCursor(Qt.PointingHandCursor)
            self._canvas.setToolTip(
                f"Drill into {selection.condition}{_drill_detail(selection)}")
        else:
            self._canvas.setCursor(Qt.ArrowCursor)
            self._canvas.setToolTip("")


class BubbleInspector(QWidget):
    """A lightweight NON-MODAL panel: click a bubble → its per-replicate fit.

    A single reused instance (never a second app window). It floats above the main
    window (Qt.Tool) and is shown WITHOUT stealing focus (WA_ShowWithoutActivating)
    so rapid clicking between bubbles keeps working. ESC or the close box dismisses
    it. It computes NOTHING: the payload is a cache read, drawn via the pure
    ``plots.draw_replicate_fit``; a missing payload shows a message, never a fit.
    """

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("Replicate fit")
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.resize(560, 460)
        self._figure = Figure(figsize=(5.6, 4.6))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout = QVBoxLayout()
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._status)
        self.setLayout(layout)

    def show_payload(self, payload) -> None:
        from .plots import draw_replicate_fit

        self._figure.clear()
        draw_replicate_fit(self._figure, payload)
        self._canvas.draw_idle()
        self._status.setText("")
        self._show_without_stealing_focus()

    def show_message(self, text) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=10,
                color="#4a5568", transform=ax.transAxes)
        ax.set_axis_off()
        self._canvas.draw_idle()
        self._show_without_stealing_focus()

    def _show_without_stealing_focus(self):
        self.show()
        self.raise_()                      # float above, but do NOT activateWindow

    def keyPressEvent(self, event):        # pragma: no cover - Qt event
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


class _ToolsTab(QWidget):
    """Every v2 tool, grouped as v2 grouped them, over EVERY condition at once.

    The groups are the outer tabs -- Import / SLA / MSD / Clustering / GMM / SQD / APPD /
    SDA / Beyond v2 -- because that is where a v2 user already looks for a tool. Sixteen
    tools across one tab bar is a wall; two clicks through a familiar grouping is not.

    Side by side is the default. Every analysis runs on every loaded condition and the
    results are drawn beside each other. Computing one condition when four were loaded
    answers a question nobody asked. The exception is a project that HAS one condition,
    which the folder layout already distinguishes.

    The track views are the one PER-CONDITION thing and keep their own picker: they show
    one cell in one movie, which has no side-by-side form.
    """

    def __init__(self, project, *, parent=None, show_legacy=False):
        super().__init__(parent)
        self._project = project
        self._show_legacy = bool(show_legacy)
        self._panels = {}

        from .analyses import TABS

        layout = QVBoxLayout(self)

        self._banner = QLabel("")
        self._banner.setWordWrap(True)
        layout.addWidget(self._banner)

        # THE LAYER LINE, on every tab. "n = 3 days" reads as a weakness until the three
        # days are visibly in agreement, and the between-day share is what makes the
        # choice of unit a decision rather than a preference.
        self._layers = QLabel("")
        self._layers.setWordWrap(True)
        self._layers.setTextFormat(Qt.RichText)
        layout.addWidget(self._layers)

        row = QHBoxLayout()
        self._legacy_box = QCheckBox("show legacy versions")
        self._legacy_box.setChecked(self._show_legacy)
        self._legacy_box.setToolTip(
            "Older v2 behaviours that a current tool replaces. Off by default so the "
            "recommended version is what a user meets; every one names the .m file it "
            "reproduces and how it differs.")
        self._legacy_box.stateChanged.connect(self._rebuild)
        row.addWidget(self._legacy_box)
        row.addStretch()
        layout.addLayout(row)

        self._groups = QTabWidget()
        layout.addWidget(self._groups, stretch=1)

        self._tables = {}
        for name in self._project.conditions:
            try:
                self._tables[name] = self._project.condition(name)
            except Exception:                          # noqa: BLE001 - named in the banner
                continue
        self._banner.setText(
            "Analyses run on ALL " + str(len(self._tables)) + " loaded conditions and are "
            "drawn side by side: " + ", ".join(self._tables) + ".")
        self._rebuild()
        self._fill_layer_line()

    # ------------------------------------------------------------------ the groups
    def _rebuild(self, *_):
        from .analyses import TABS, by_tab
        from .widgets import AnalysisPanelWidget, TrackViewsWidget

        self._show_legacy = self._legacy_box.isChecked()
        current = self._groups.tabText(self._groups.currentIndex())
        self._groups.clear()
        self._panels = {}

        for group in TABS:
            entries = [a for a in by_tab(group)
                       if self._show_legacy or a.status != "legacy only"]
            if not entries:
                continue
            # The PROJECT supplies the projected tables, because it owns the cell
            # outlines and the panel does not. Without this the jump-density map is a
            # button that raises for every condition.
            panel = AnalysisPanelWidget(self._tables or None,
                                        only=[a.key for a in entries],
                                        projected=self._project.projected)
            self._panels[group] = panel
            self._groups.addTab(panel, group)

        views = self._build_views()
        if views is not None:
            self._groups.addTab(views, "Track views")

        for i in range(self._groups.count()):
            if self._groups.tabText(i) == current:
                self._groups.setCurrentIndex(i)
                break

    def _build_views(self):
        """One condition at a time, with its own picker: a track view shows one cell in
        one movie, which has no side-by-side form."""
        from .widgets import TrackViewsWidget

        if not self._tables:
            return None
        holder = QWidget()
        box = QVBoxLayout(holder)
        row = QHBoxLayout()
        row.addWidget(QLabel("condition"))
        self._view_condition = QComboBox()
        self._view_condition.addItems(list(self._tables))
        self._view_condition.currentTextChanged.connect(self._views_condition_changed)
        row.addWidget(self._view_condition, stretch=1)
        row.addStretch()
        box.addLayout(row)
        self._views_host = QVBoxLayout()
        box.addLayout(self._views_host, stretch=1)
        self._views_widget = None
        self._views_condition_changed(self._view_condition.currentText())
        return holder

    def _views_condition_changed(self, condition):
        from .cellcoords import project_tracks
        from .widgets import TrackViewsWidget

        if self._views_widget is not None:
            self._views_widget.setParent(None)
            self._views_widget = None
        table = self._tables.get(condition)
        if table is None:
            return
        projected = None
        try:
            meshes = getattr(self._project, "meshes", None)
            if meshes is not None:
                projected = project_tracks(table, meshes, require_centerline=False)
        except Exception as exc:                       # noqa: BLE001
            print(f"note: no cell-frame projection for {condition} ({exc})")
        params = getattr(self._project, "params", None)
        self._views_widget = TrackViewsWidget(
            table, projected=projected,
            pixel_size_nm=getattr(params, "pixel_size_nm", None),
            frame_interval_ms=getattr(params, "frame_interval_ms", None))
        self._views_host.addWidget(self._views_widget)

    # -------------------------------------------------------------- the layer line
    def _fill_layer_line(self):
        """Day-to-day against cell-to-cell spread, and the yield per cell per minute.

        Computed ONCE and shown above every group, because both are properties of the
        EXPERIMENT rather than of whichever analysis is open.

        The density is here rather than in a QC panel. The QC tab is four panels by
        ruling ("keep the QC slim for now"), and this number is not a pass/fail check,
        it is the sample size on a common footing, which a reader needs beside every
        side-by-side comparison rather than behind a separate tab. It is also the one
        line that shows two conditions are not comparable in yield, which no fraction
        can.
        """
        from .catalogue_view import density_summary, layer_summary

        try:
            summary = layer_summary(self._project, list(self._tables))
        except Exception as exc:                       # noqa: BLE001
            self._layers.setText(f"spread not measurable ({type(exc).__name__})")
            return
        # A FAILING DENSITY MUST NOT COST THE SPREAD LINE. They are independent
        # measurements and one is not a precondition for the other, so a condition whose
        # density cannot be formed still shows its spread.
        try:
            density = density_summary(self._project, list(self._tables))
        except Exception as exc:                       # noqa: BLE001
            density = {name: {"caption": f"density not measurable "
                                         f"({type(exc).__name__})"}
                       for name in summary}
        lines = []
        for name, report in summary.items():
            line = f"<b>{name}</b>: {report['caption']}"
            yielded = density.get(name, {}).get("caption")
            if yielded:
                line += f"<br>&nbsp;&nbsp;&nbsp;&nbsp;yield: {yielded}"
            lines.append(line)
        self._layers.setText("<br>".join(lines))


class _QCTab(QWidget):
    """"Should I trust this dataset?" -- in the window people actually open.

    The defect this closes. `smtracker.qc`, `plots.draw_qc_dashboard` and
    `widgets.QCDashboardWidget` were built, measured and tested, and reachable from
    exactly two places: `MethodBrowser` (the DRILL-DOWN window, which opens only after a
    click on a result) and the `--qc` flag, which the no-argument flow never sets. So the
    check documented as "run BEFORE any condition comparison" sat behind the comparison.
    The experimenter found it, as with the tools tab and the track viewer.

    One condition at a time, deliberately, and this is the exception to side-by-side
    rather than a lapse from it: the dashboard is four panels, so three conditions is
    twelve, and the question it answers ("do I trust THIS dataset") is asked of one
    dataset. The N-per-group panel already shows every group's counts inside one
    condition, which is where a movie contributing nothing becomes visible.

    Four panels, on purpose -- ruled by the experimenter, "keep the QC slim for now, we
    can extend it later on the go". Three built modules were offered and deliberately not
    wired, recorded here rather than left to be rediscovered:

      `cellmesh.assignment_funnel`  the cell filter's loss, which its own docstring calls
                                    the largest silent data loss in the pipeline: 5-10 %
                                    of tracks normally, 59 % once, with no message. The
                                    only candidate whose failure mode is "the numbers are
                                    wrong" rather than "the error bars are wrong".
      `aggregate.poolability`       do days differ as much as conditions -- the question
                                    behind every pooled analysis.
      `qc.proposed_from_flags`      closes the loop from a flag to an exclusion carrying
                                    a written reason. Today the tab flags and stops.

    Nothing BETWEEN-CONDITION goes in here. QC is where a difference test is most
    tempting to add, and which statistic answers "are these two different" is still the
    experimenter's open decision (Amendment 4).
    """

    def __init__(self, project, *, parent=None):
        super().__init__(parent)
        self._project = project
        self._widget = None

        layout = QVBoxLayout(self)
        note = QLabel(
            "Run this BEFORE comparing conditions. Every panel prints its own N and its "
            "own threshold. Outliers are FLAGGED, never dropped: a flag means “look at "
            "this”, and whether it is an artefact is not a question a threshold can "
            "answer. To act on one, exclude it explicitly with a written reason.")
        note.setWordWrap(True)
        layout.addWidget(note)

        row = QHBoxLayout()
        row.addWidget(QLabel("condition"))
        self._condition = QComboBox()
        self._condition.addItems(list(project.conditions))
        self._condition.currentTextChanged.connect(self._changed)
        row.addWidget(self._condition, stretch=1)
        row.addWidget(QLabel("outlier unit"))
        self._unit = QComboBox()
        # The unit the flag is computed over. Cell is the default because it is the
        # bootstrap unit and the level the documented flags were measured at; movie is
        # here because a whole bad field of view is the other thing this panel catches.
        self._unit.addItems(["cell", "movie", "day"])
        self._unit.currentTextChanged.connect(self._changed)
        row.addWidget(self._unit)
        row.addStretch()
        layout.addLayout(row)

        # THE CONTROL LINE. Asked for: "we should include an option for qc with no cell
        # control and WT control." A control is a folder like any other condition, so it
        # is picked from what is LOADED rather than hunted for, which also makes the
        # refusal honest when it is absent ("load it as a condition") instead of a guess
        # at where it might live.
        #
        # NONE IS THE DEFAULT, and it is not laziness: a project's own WT can be in LB
        # while the samples are in minimal, so an unasked-for comparison would put a
        # number on screen that OVERSTATES contamination with nothing to say why.
        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("control"))
        self._control = QComboBox()
        self._control.addItem("none", None)
        for name in project.conditions:
            self._control.addItem(name, name)
        self._control.currentIndexChanged.connect(self._changed)
        control_row.addWidget(self._control, stretch=1)
        control_row.addWidget(QLabel("medium: sample"))
        self._sample_medium = QComboBox()
        self._sample_medium.addItems(["(not set)", "minimal", "LB", "none"])
        self._sample_medium.currentTextChanged.connect(self._changed)
        control_row.addWidget(self._sample_medium)
        control_row.addWidget(QLabel("control"))
        self._control_medium = QComboBox()
        self._control_medium.addItems(["(not set)", "minimal", "LB", "none"])
        self._control_medium.currentTextChanged.connect(self._changed)
        control_row.addWidget(self._control_medium)
        control_row.addStretch()
        layout.addLayout(control_row)

        # intensity is a button, not a line. It reads frames: 11 000 spots over four
        # movies is tens of seconds, and this project has just paid for running that
        # kind of work on the GUI thread. The rate line above is instant and stays
        # automatic; this one is asked for.
        intensity_row = QHBoxLayout()
        self._intensity_button = QPushButton("Compare spot intensity")
        self._intensity_button.setToolTip(
            "Reads the raw frames for both groups and integrates each spot. Slow, and "
            "it needs the laser power: intensity scales with illumination, so without "
            "it this compares the lasers rather than the samples.")
        self._intensity_button.clicked.connect(self._run_intensity)
        intensity_row.addWidget(self._intensity_button)
        intensity_row.addWidget(QLabel("laser"))
        self._laser_mw = QDoubleSpinBox()
        self._laser_mw.setRange(0.0, 10000.0)
        self._laser_mw.setDecimals(1)
        self._laser_mw.setSuffix(" mW")
        self._laser_mw.setSpecialValueText("not set")
        intensity_row.addWidget(self._laser_mw)
        intensity_row.addStretch()
        layout.addLayout(intensity_row)
        self._intensity_line = QLabel("")
        self._intensity_line.setWordWrap(True)
        self._intensity_line.setStyleSheet("color: #444;")
        layout.addWidget(self._intensity_line)

        self._control_line = QLabel("")
        self._control_line.setWordWrap(True)
        self._control_line.setStyleSheet("color: #444;")
        layout.addWidget(self._control_line)

        self._host = QVBoxLayout()
        layout.addLayout(self._host, stretch=1)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._changed(self._condition.currentText())

    def _run_intensity(self):
        """Read frames for the sample and the control, off the GUI thread."""
        control = self._control.currentData()
        if control is None:
            self._intensity_line.setText("pick a control first")
            return
        condition = self._condition.currentText()
        laser = float(self._laser_mw.value())
        self._intensity_button.setEnabled(False)
        self._intensity_line.setText("reading frames…")

        class _Signals(QObject):
            done = Signal(str)

        class _Job(QRunnable):
            def __init__(self, project):
                super().__init__()
                self._project = project
                self.signals = _Signals()

            def run(self):                             # noqa: D102 - QRunnable protocol
                try:
                    text = ctrl.control_intensity(
                        self._project, condition, control, laser_power_mw=laser)
                except Exception as exc:               # noqa: BLE001 - named on screen
                    text = f"{type(exc).__name__}: {exc}"
                self.signals.done.emit(text)

        job = _Job(self._project)
        job.signals.done.connect(self._intensity_done)
        self._intensity_job = job
        QThreadPool.globalInstance().start(job)

    def _intensity_done(self, text):
        self._intensity_button.setEnabled(True)
        self._intensity_job = None
        self._intensity_line.setText(
            text.replace("\n", "<br>") if text else "")

    def _refresh_control(self):
        """The control comparison, on its own line under the pickers.

        Separate from the four panels, because the ruling on this tab was that it stays
        slim: four panels, extended "later on the go". A control is one number and its
        caveat, so it is a line rather than a fifth panel.

        It does not cost the panels. A control that cannot be compared leaves this line
        saying why and the dashboard above untouched: they are independent measurements
        and neither is a precondition for the other, the same rule the layer-spread and
        density lines already follow.
        """
        control = self._control.currentData()
        if control is None:
            self._control_line.setText("")
            return
        media = {}
        condition = self._condition.currentText()
        if self._sample_medium.currentText() != "(not set)":
            media[condition] = self._sample_medium.currentText()
        if self._control_medium.currentText() != "(not set)":
            media[control] = self._control_medium.currentText()
        # THE COMPUTE IS IN THE CONTROLLER. This file may not import pandas at all, not
        # even inside a method, an AST test enforces it and caught the first attempt.
        text = ctrl.control_line(self._project, condition, control, media=media)
        self._control_line.setText(text.replace("\n", "<br><i>") +
                                   ("</i>" if "\n" in text else ""))

    def _changed(self, *_):
        from .widgets import QCDashboardWidget

        self._refresh_control()
        if self._widget is not None:
            self._widget.setParent(None)
            self._widget = None
        condition = self._condition.currentText()
        if not condition:
            return
        try:
            table = self._project.condition(condition)
        except Exception as exc:                       # noqa: BLE001 - named, not silent
            self._status.setText(
                f"{condition} could not be loaded: {type(exc).__name__}: {exc}")
            return
        try:
            self._widget = QCDashboardWidget(table, unit=self._unit.currentText())
        except Exception as exc:                       # noqa: BLE001
            self._status.setText(
                f"QC could not be formed for {condition}: {type(exc).__name__}: {exc}")
            return
        self._host.addWidget(self._widget)
        self._status.setText("")


class ComparisonBrowser(QWidget):
    """Comparison-first window: a condition multi-select drives every method view.

    The top bar (condition checkboxes, per-condition model dropdowns, a lock
    toggle, the mobility threshold, and the reference selector) defines a single
    ``ComparisonSpec`` that every tab reads. All of it round-trips through a
    ``SessionState`` so a selection survives tab switches and relaunches.
    """

    #: The N-condition views (the forest has its own dedicated tab, since it
    #: compares a reference + one target rather than all conditions at once).
    VIEWS = [v for v in cctrl.VIEWS if v[1] != "forest_view"]
    #: forest method rows a user may tick.
    FOREST_METHODS = ["SQD", "HMM", "appD", "saSPT", "confinement"]
    #: view attr -> glossary key(s), for hover tooltips on the tabs.
    TAB_GLOSSARY = {
        "heatmap_view": ["slow_fraction"],
        "sqd_view": ["slow_fraction", "bic"],
        "appd_view": ["D_fast", "D_slow"],
        "saspt_view": ["saspt_occupancy"],
        "kinetics_view": ["occupancy", "on_rate", "off_rate"],
        "fraction_view": ["slow_fraction", "bootstrap_ci"],
        "spatial_view": ["radial_enrichment"],
    }

    def __init__(self, project, *, n_boot=50, session=None, initial=None,
                 fast=False, cache=None, parent=None):
        super().__init__(parent)
        # The AS-LOADED project (derived σ). A manual-σ override (item D) rebuilds
        # ``self._project`` from this base, so toggling the override on/off is clean.
        self._base_project = project
        self._project = project
        self._n_boot = n_boot
        self._fast_initial = fast
        self._all = list(project.conditions)
        self._drill_children = []              # keep drilled MethodBrowser windows alive
        self._track_viewer = None              # only the CURRENT track viewer (B6)
        self._bubble_inspector = None          # single reused per-replicate panel (Item 3)
        self._replicate_payloads = {}          # condition -> payload, filled on SQD compute

        if cache is not None:
            self._cache = cache
        else:
            # Default to the committed warm cache (instant on the standard sample;
            # a miss falls through to compute). A different project simply misses.
            from .warmcache import WarmCache, default_warm_root

            self._cache = WarmCache(default_warm_root())

        from .session import SessionState

        self._session = session or SessionState()
        selected = (initial or self._session.selected_conditions
                    or list(self._all))
        selected = [c for c in selected if c in self._all] or [self._all[0]]

        self._build_header(selected)
        self._build_tabs()
        self._sync_reference()
        self._save_session()

    # -- header: pickers + model + threshold + lock + reference ----------- #

    def _build_header(self, selected):
        self._condition_boxes = {}
        self._model_combos = {}

        picker_group = QGroupBox("conditions (drive every view)")
        picker_layout = QGridLayout()
        for col, cond in enumerate(self._all):
            box = QCheckBox(cond)
            box.setChecked(cond in selected)
            box.stateChanged.connect(self._on_selection_changed)
            self._condition_boxes[cond] = box
            picker_layout.addWidget(box, 0, col)

            combo = QComboBox()
            combo.addItems(list(MODEL_CHOICES))
            combo.setCurrentText(self._session.per_condition_model.get(cond, "best"))
            combo.currentTextChanged.connect(self._on_control_changed)
            self._model_combos[cond] = combo
            picker_layout.addWidget(combo, 1, col)
        picker_group.setLayout(picker_layout)

        self._lock_combo = QComboBox()
        self._lock_combo.addItems(["(per-condition)", *MODEL_CHOICES])
        if self._session.lock_model:
            self._lock_combo.setCurrentText(self._session.lock_model)
        self._lock_combo.currentTextChanged.connect(self._on_lock_changed)

        # The mobility threshold: a live slider + a numeric field, kept in sync.
        # It is applied AFTER the fit (slow_fraction_below), so dragging it needs
        # no refit, on the SQD view it re-renders instantly off the cached fits.
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setDecimals(3)
        self._threshold_spin.setRange(_THRESH_MIN, _THRESH_MAX)
        self._threshold_spin.setSingleStep(0.01)
        self._threshold_spin.setValue(self._session.mobility_threshold_um2_s)
        self._threshold_spin.valueChanged.connect(self._on_threshold_spin)

        self._threshold_slider = QSlider(Qt.Horizontal)
        self._threshold_slider.setRange(_thresh_to_slider(_THRESH_MIN),
                                        _thresh_to_slider(_THRESH_MAX))
        self._threshold_slider.setValue(
            _thresh_to_slider(self._session.mobility_threshold_um2_s))
        self._threshold_slider.setFixedWidth(140)
        self._threshold_slider.valueChanged.connect(self._on_threshold_slider)

        self._live_box = QCheckBox("live")
        self._live_box.setChecked(True)
        self._live_box.setToolTip(
            "re-render the current view as you drag the threshold "
            "(instant on the SQD view; slower views update on release)")

        self._ecdf_box = QCheckBox("ECDF fast")
        self._ecdf_box.setChecked(bool(self._session.ecdf_subsample))
        self._ecdf_box.setToolTip(
            "fit the SQD point estimate to 2000 ECDF quantile points when a "
            "condition has more (validated identical to 3 s.f., about 3.4 times faster); "
            "thin conditions and the bootstrap stay exact")
        self._ecdf_box.stateChanged.connect(self._on_control_changed)

        self._ci_box = QCheckBox("SQD CIs")
        self._ci_box.setChecked(bool(self._session.sqd_cis))
        self._ci_box.setToolTip(
            "compute per-population D/fraction error bars on the SQD view: a cell "
            "bootstrap PER CONDITION, so slow; off by default for a fast first look")
        self._ci_box.stateChanged.connect(self._on_control_changed)

        # item C: D-axis Berglund correction, default OFF. On the SQD view only (the
        # per-state D display). Loud in the caption + provenance when on.
        self._dcorr_box = QCheckBox("D correction")
        self._dcorr_box.setChecked(bool(getattr(self._session, "d_correction", False)))
        self._dcorr_box.setToolTip(
            "Display D_true = (3/2)·(D_app − σ²/τ) on the SQD view: Berglund blur "
            "+ static-error removal, ALL states or none; states at/below the floor "
            "show as bounds. DISPLAY-only (fractions unchanged); fenced in the "
            "caption + provenance, like the fast-mode banner.")
        self._dcorr_box.stateChanged.connect(self._on_control_changed)

        # item D: manual localization σ. Default = derived (pixel/√12). Ticking
        # 'manual σ' rebuilds the acquisition params from the spin value so the
        # floor, the D-correction and the σ stamp all use it; because σ is
        # provenance key material, cached views recompute rather than serve stale.
        # tolerate a stub project (tests) with no acquisition params
        _params = getattr(self._base_project, "params", None)
        _sig_default = float(getattr(_params, "localization_sigma_nm", 30.6) or 30.6)
        _sig_source = getattr(_params, "localization_sigma_source", "derived_pixelation")
        self._sigma_manual_box = QCheckBox("manual σ")
        self._sigma_manual_box.setChecked(_sig_source == "manual")
        self._sigma_manual_box.setToolTip(
            "Substitute a MEASURED localization σ (nm) for the derived pixelation "
            "value (pixel/√12). Affects the floor, the D-correction and the σ stamp.")
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setDecimals(1)
        self._sigma_spin.setRange(1.0, 500.0)
        self._sigma_spin.setSingleStep(1.0)
        self._sigma_spin.setSuffix(" nm")
        self._sigma_spin.setValue(_sig_default)
        self._sigma_spin.setToolTip("localization σ in nm (used when 'manual σ' is ticked)")
        self._sigma_spin.setEnabled(self._sigma_manual_box.isChecked())
        self._sigma_manual_box.stateChanged.connect(self._on_sigma_changed)
        self._sigma_spin.valueChanged.connect(self._on_sigma_changed)

        # Fraction ESTIMAND for the simultaneous fit. Labelled by what it measures,
        # not by the mechanic: molecule fraction (track, DEFAULT, the pre-registered
        # endpoint) vs displacement/time fraction (step, the legacy consistency
        # check). The combo VALUE stays "track"/"step" for the spec.
        self._weighting_combo = QComboBox()
        self._weighting_combo.addItem("molecule fraction (track)", "track")
        self._weighting_combo.addItem("displacement / time fraction (step)", "step")
        _wv = getattr(self._session, "sqd_fraction_weighting", "track")
        self._weighting_combo.setCurrentIndex(
            max(0, self._weighting_combo.findData(_wv)))
        self._weighting_combo.setToolTip(
            "SQD fraction estimand: molecule fraction (track) = proportion of the "
            "population in a state, the pre-registered endpoint; displacement/time "
            "fraction (step) = fraction of observed time, matches the legacy fit")
        self._weighting_combo.currentTextChanged.connect(self._on_control_changed)

        # D-sharing scope, the simultaneous fit, finally on screen.
        #
        # `ComparisonSpec.d_sharing_scope` has always been a validated field driving a
        # real, tested fit (`condition_replicate_fit`), and `spec()` never passed it,
        # so every run took the default and no user could choose. Declared, documented,
        # plotted, and unreachable: this file's most-repeated defect shape.
        #
        # THE DEFAULT IS UNCHANGED and is what the experimenter asked for: D shared
        # across a condition's REPLICATE DAYS, fractions free per day, and the mean and
        # SD of those per-day fractions are what the comparison shows. The other two
        # scopes were already implemented; they were simply not offered.
        self._scope_combo = QComboBox()
        # FORCE SAME D ACROSS, in the Compute dialog's words and order (ruled 1 Nov 2026, `workspace_view`'s
        # `SHARED_D_PILLS`): Replicates · Conditions (one per factor of a folder design, 25 Oct 2026: the conditions
        # with one value of the factor share one D, `workspace_view.resolve_shared_d`) · Whole data set · Don't force
        # anywhere. Without a design "Conditions" would be the Whole data set's fit, so it is not offered.
        from .workspace_view import GROUP_PREFIX, SHARED_D_NAME, SHARED_D_PILLS, group_label

        factors = [str(f) for f in getattr(self._project, "factor_names", ()) or ()]
        self._scope_combo.addItem(f"{SHARED_D_PILLS['condition']} (simultaneous)", "condition")
        for factor in factors:
            self._scope_combo.addItem(f"{group_label(factor, len(factors))} (one D per {factor})",
                                      GROUP_PREFIX + factor)
        self._scope_combo.addItem(SHARED_D_PILLS["global"], "global")
        self._scope_combo.addItem(f"{SHARED_D_PILLS['replicate']} (each replicate its own D)", "replicate")
        _sv = getattr(self._session, "d_sharing_scope", DEFAULT_D_SHARING_SCOPE)
        self._scope_combo.setCurrentIndex(max(0, self._scope_combo.findData(_sv)))
        self._scope_combo.setToolTip(
            f"{SHARED_D_NAME}: which fit the per-replicate fractions come from.\n\n"
            f"{SHARED_D_PILLS['condition']} (DEFAULT): D is fitted once on the condition's "
            "pooled data and each replicate's fractions are fitted with that D "
            "held: the legacy simultaneous fit, and the only way per-replicate fractions "
            "mean the same thing.\n\n"
            + ("Conditions: one D per value of the folder design's factor (per protein), "
               "shared by that value's conditions.\n\n" if factors else "")
            + f"{SHARED_D_PILLS['global']}: one D for the whole comparison. Use when "
            "the conditions are one molecule and the D difference is not the claim.\n\n"
            f"{SHARED_D_PILLS['replicate']}: each replicate fitted independently, so each carries "
            "its OWN D. Diagnostic: fractions from fits with different D are NOT "
            "comparable, and there is then no single shared D to report.")
        self._scope_combo.currentTextChanged.connect(self._on_control_changed)

        # WHICH FITS THE BUBBLES OVERLAY, and which state counts.
        #
        # The scope combo above picks ONE arrangement, so the reader sees the fit that
        # was chosen and never what choosing it cost. Measured on a lab project
        # that cost is up to 9 percentage points, which is larger than
        # most effects these panels are read for. Drawing both is the fix.
        # The two named as the Compute dialog's Overlay names them (`workspace_view.ARRANGEMENT_NAMES`, 1 Nov 2026;
        # "independent" / "simultaneous" before)
        from .workspace_view import ARRANGEMENT_NAMES

        own, one = ARRANGEMENT_NAMES["independent"], ARRANGEMENT_NAMES["simultaneous"]
        self._bubble_modes_combo = QComboBox()
        self._bubble_modes_combo.addItem("both", ("independent", "simultaneous"))
        self._bubble_modes_combo.addItem(f"{own} only", ("independent",))
        self._bubble_modes_combo.addItem(f"{one} only", ("simultaneous",))
        self._bubble_modes_combo.setToolTip(
            "Which fits are drawn together, as separate series.\n\n"
            f"{own.upper()}: every condition keeps its own D. Honest about D, and leaves "
            "the fractions as percentages of different things.\n\n"
            f"{one.upper()}: D fitted once across the conditions, fractions free. The "
            "only arrangement in which the conditions' fractions mean ONE quantity, at "
            "the cost of forcing a D none of them has.\n\n"
            "Neither is right alone. A single series keeps the classic drawing, which "
            "carries the side table and the per-bubble click.")
        self._bubble_modes_combo.currentTextChanged.connect(self._on_control_changed)

        # STATE COUNTS, side by side. Each count is its own series and they are NEVER
        # averaged: 38 per cent of one of two states and 38 per cent of one of three are
        # different quantities, which is why `compare_fractions` refuses to tabulate
        # them together. Drawing them beside each other is a comparison of MODELS, and
        # it is only defensible while the count is visible on the figure -- so the
        # series label carries it.
        self._bubble_npops_combo = QComboBox()
        self._bubble_npops_combo.addItem("2 populations", (2,))
        self._bubble_npops_combo.addItem("3 populations", (3,))
        self._bubble_npops_combo.addItem("2 and 3, side by side", (2, 3))
        self._bubble_npops_combo.addItem("2, 3 and 4", (2, 3, 4))
        self._bubble_npops_combo.setToolTip(
            "State counts drawn together, each as its own series.\n\n"
            "BIC recommends a count, it does not decide one, and on this project's own "
            "data the recommendation moves with how well the fit converged. Seeing two "
            "counts beside each other is how that judgement gets made.\n\n"
            "4 populations needs about 100 optimiser starts and is slow; the expense is "
            "itself the surface reporting that it is barely constrained.")
        self._bubble_npops_combo.currentTextChanged.connect(self._on_control_changed)

        # Spatial-map label source: appD speed bins (per-track) or HMM (per-step).
        self._spatial_combo = QComboBox()
        self._spatial_combo.addItems(["appd", "hmm"])
        self._spatial_combo.setCurrentText(self._session.spatial_source)
        self._spatial_combo.setToolTip(
            "spatial map populations: appd = per-track speed bins (legacy speed "
            "map); hmm = per-step decoded states (binding map)")
        self._spatial_combo.currentTextChanged.connect(self._on_control_changed)
        # Spatial-map mode: enrichment (density/null, for claims) or raw density.
        self._spatial_mode_combo = QComboBox()
        self._spatial_mode_combo.addItems(["enrichment", "density"])
        self._spatial_mode_combo.setCurrentText(self._session.spatial_mode)
        self._spatial_mode_combo.setToolTip(
            "enrichment = density / matched null (for claims); density = raw "
            "projection density (orientation only, geometry-dominated)")
        self._spatial_mode_combo.currentTextChanged.connect(self._on_control_changed)

        # The uncertainty tier for the fraction view (see the tier policy).
        self._uncertainty_combo = QComboBox()
        self._uncertainty_combo.addItems(["analytic", "bootstrap", "none"])
        self._uncertainty_combo.setCurrentText(self._session.fraction_uncertainty)
        self._uncertainty_combo.setToolTip(
            "fraction-view interval: analytic = per-cell mean ± SD/√n (fast); "
            "bootstrap = cell resample (slow, opt-in); none = point only")
        self._uncertainty_combo.currentTextChanged.connect(self._on_control_changed)

        self._reference_combo = QComboBox()
        self._reference_combo.currentTextChanged.connect(self._on_control_changed)

        self._fast_box = QCheckBox("DEV FAST MODE")
        self._fast_box.setChecked(bool(self._fast_initial))
        self._fast_box.stateChanged.connect(self._on_fast_toggled)

        # Publication-styling: Export renders under the chosen journal's rcParams
        # at its column size (see figstyle.JOURNAL_PRESETS).
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(list(JOURNAL_PRESETS))
        self._preset_combo.setCurrentText(
            self._session.journal_preset if self._session.journal_preset
            in JOURNAL_PRESETS else "screen")
        self._preset_combo.currentTextChanged.connect(self._on_control_changed)
        self._width_combo = QComboBox()
        self._width_combo.addItems(["single", "double"])
        self._width_combo.setCurrentText(self._session.journal_width)
        self._width_combo.currentTextChanged.connect(self._on_control_changed)

        # THE DEFAULT BUTTON IS THE FAST PATH. `playback.TrackPlayer` is matplotlib and
        # blitted: 1530-2884 frames per second across 28 movies, 31 to 58 times the 50 fps
        # acquisition, and no GL context to build. The napari viewer's cost is the WINDOW
        # (Qt + OpenGL), not the data, preparing every layer for a whole condition is
        # 0.09 s. So the button labelled "track viewer" must not be the one that spends
        # seconds building a window this project's data does not need.
        self._track_views_btn = QPushButton("Open track viewer")
        self._track_views_btn.setToolTip(
            "the fast track views: pooled tracks, one cell, or blitted playback "
            "(matplotlib, no OpenGL window to build)")
        self._track_views_btn.clicked.connect(self._open_track_views)

        # NAPARI IS KEPT, and the reason is data-dependent: this project ships `phase/`
        # and `sum/` at 126x130, NOT the raw 1001-frame stack, so playback here animates
        # tracks over ONE image: the case matplotlib is good at. Hand it a real stack and
        # the per-frame cost becomes an image upload and the GPU path is right.
        self._track_viewer_btn = QPushButton("napari…")
        self._track_viewer_btn.setToolTip(
            "the napari single-track viewer: worth its startup cost for a real image "
            "STACK; for this project's single background image the fast views are the "
            "same picture without the window")
        self._track_viewer_btn.clicked.connect(self._open_track_viewer)

        # DRIVERS, the global state that changes every view's RESULT. These, plus
        # the conditions group above, are the only things in the top strip that
        # move the numbers. Per-view controls (SQD CIs, ECDF, frac CI, spatial
        # source/mode) are NOT here: they live in each view's own strip.
        drivers = QHBoxLayout()
        drivers.addWidget(QLabel("lock model:"))
        drivers.addWidget(self._lock_combo)
        # The mobility-threshold control is GONE (Amendment 4): it drove the
        # threshold-based cross-condition slow-fraction, which the tool no longer
        # computes. The slider/spin widgets still exist (so the spec carries a
        # default and the SQD bubble can draw the localization-floor guide line),
        # but they are NOT shown: there is no threshold endpoint to tune.
        drivers.addWidget(QLabel("   reference:"))
        drivers.addWidget(self._reference_combo)
        drivers.addStretch(1)

        # UTILITY: app/session knobs that do NOT change a view's numbers (redraw
        # behaviour, export styling, the dev toggles, the track-viewer action). A
        # separate, quieter row so the drivers above read as the load-bearing set.
        utility = QHBoxLayout()
        utility.addWidget(self._live_box)
        utility.addWidget(self._track_views_btn)
        utility.addWidget(self._track_viewer_btn)
        utility.addStretch(1)
        utility.addWidget(QLabel("export style:"))
        utility.addWidget(self._preset_combo)
        utility.addWidget(self._width_combo)
        utility.addWidget(self._fast_box)
        utility.addWidget(QLabel(f"n_boot={self._n_boot} (dev)"))
        self._utility_row = QWidget()
        self._utility_row.setLayout(utility)
        # "Quieter" via smaller type + the hairline + row position, NOT colour.
        # Muted-grey text is Qt's disabled convention; using it as a quiet signal
        # reads as "these controls are unavailable". Normal contrast, smaller size.
        self._utility_row.setStyleSheet("QLabel { font-size:11px; }")

        # The loud fast-mode banner (the GUI half of the honesty guard).
        self._fast_banner = QLabel(f"⚠  {BANNER}")
        self._fast_banner.setStyleSheet(
            "background:#c53030; color:white; font-weight:bold; padding:4px;")
        self._fast_banner.setVisible(bool(self._fast_initial))

        # Fit All (Item 8): compute every view in the background, cheapest first.
        self._fit_all_btn = QPushButton("Fit / Plot All")
        self._fit_all_btn.setToolTip(
            "compute every view in the background (cheapest first); cancellable, "
            "skips already-cached views, failures don't abort")
        self._fit_all_btn.clicked.connect(self._fit_all)
        self._fit_progress = QLabel("")
        self._fit_cancel_btn = QPushButton("Cancel all")
        self._fit_cancel_btn.clicked.connect(self._fit_all_cancel)
        self._fit_cancel_btn.setVisible(False)
        self._fit_cancelled = False
        self._fit_signals = _FitAllSignals()
        self._fit_signals.started.connect(self._on_fit_started)
        self._fit_signals.item_done.connect(self._on_fit_item_done)
        self._fit_signals.finished.connect(self._on_fit_finished)
        # Fit / Plot All is a DRIVER, not a utility: it is the verb that acts on
        # exactly the nouns in the drivers row (the selected conditions, the lock,
        # the threshold), so it sits at that row's right-hand end, adjacency
        # between what you set and what executes it. It is also the only
        # long-running, cancellable action, and it grows into progress + cancel IN
        # PLACE while running; a live progress bar could not sit in the quiet
        # utility row without becoming the loudest thing on screen.
        drivers.addWidget(self._fit_all_btn)
        drivers.addWidget(self._fit_cancel_btn)
        drivers.addWidget(self._fit_progress)

        # A hairline between the load-bearing drivers and the quieter utility row.
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("color:#2d3748;")

        self._header = QVBoxLayout()
        self._header.addWidget(self._fast_banner)
        self._header.addWidget(picker_group)
        self._header.addLayout(drivers)
        self._header.addWidget(separator)
        self._header.addWidget(self._utility_row)

    def _local_strip_for(self, attr):
        """This view's OWN controls, as a strip that sits under its canvas, the
        one consistent place per-view controls live. Returns a QWidget, or None
        for a view with no local control. The widgets are browser-owned (so
        ``spec()`` still reads them); placing them here just reparents them."""
        from .workspace_view import SHARED_D_NAME

        # the setting's one name (`workspace_view.SHARED_D_NAME`), lowered like the strip's other labels
        scope_label = SHARED_D_NAME[:1].lower() + SHARED_D_NAME[1:] + ":"
        items = {
            "sqd_view": [("", self._ci_box), ("", self._ecdf_box),
                         ("weighting:", self._weighting_combo),
                         (scope_label, self._scope_combo),
                         ("overlay:", self._bubble_modes_combo),
                         ("states:", self._bubble_npops_combo),
                         ("", self._dcorr_box)],
            # THE SCOPE BELONGS ON BOTH, because both read per-replicate fractions:
            # the SQD view shows the fit, the fraction view shows the per-day spread
            # that fit produces. A control on one and not the other would let the two
            # disagree about which fit they were drawing.
            "fraction_view": [("interval:", self._uncertainty_combo),
                              (scope_label, self._scope_combo)],
            "spatial_view": [("populations:", self._spatial_combo),
                             ("scale:", self._spatial_mode_combo)],
            "sigma_view": [("", self._sigma_manual_box), ("σ:", self._sigma_spin)],
        }.get(attr)
        if not items:
            return None
        row = QHBoxLayout()
        row.setContentsMargins(2, 0, 2, 0)
        row.addWidget(QLabel("this view:"))
        for label, widget in items:
            if label:
                row.addWidget(QLabel(label))
            row.addWidget(widget)
        row.addStretch(1)
        strip = QWidget()
        strip.setLayout(row)
        strip.setStyleSheet("QLabel { color:#718096; }")
        return strip

    def _build_tabs(self):
        self._tabs = QTabWidget()
        self._view_tabs = {}
        self._tab_attr = {}                    # id(tab) -> controller attr
        for label, attr in self.VIEWS:
            fn = getattr(cctrl, attr)
            # The SQD view routes a bubble click to the per-replicate INSPECTOR
            # (Item 3); every other view drills into the per-condition detail.
            on_click = (self._show_bubble_inspector if attr == "sqd_view"
                        else self._drill_into)
            tab = _ComparisonTab(
                label,
                self._compute_for(attr, fn),
                provenance=(lambda attr=attr: cctrl.provenance_for(
                    self._project, self.spec(), view=attr.replace("_view", ""))),
                on_drill=on_click,
                conditions=(lambda: self.spec().conditions),
                preset=self._current_preset,
                local=self._local_strip_for(attr),
                note="press Compute (uses cache when warm)",
            )
            self._view_tabs[attr] = tab
            self._tab_attr[id(tab)] = attr
            index = self._tabs.addTab(tab, label)
            self._tabs.setTabToolTip(index, _glossary_tooltip(self.TAB_GLOSSARY.get(attr)))

        # QC, in the window people actually open. Documented as "run BEFORE any condition
        # comparison" and reachable only from the drill-down window and `--qc`, i.e. only
        # AFTER the comparison it is supposed to precede.
        try:
            qc = _QCTab(self._project)
            self._qc_tab = qc
            index = self._tabs.addTab(qc, "QC: trust this data?")
            self._qc_index = index
            self._tabs.setTabToolTip(
                index, "Track-length filter, localizations per frame, N per group and "
                       "per-unit outliers. Flags are for inspection; nothing is dropped.")
        except Exception as exc:                      # noqa: BLE001
            print(f"note: QC tab unavailable ({exc})")

        # THE V2 CATALOGUE AND THE TRACK VIEWS, in the window people actually open.
        # `smtracker` with no arguments lands here, so a tool reachable only behind
        # `--analyses` is a tool nobody finds.
        try:
            tools = _ToolsTab(self._project)
            self._tools_tab = tools
            index = self._tabs.addTab(tools, "Tools (v2 + views)")
            self._tools_index = index
            self._tabs.setTabToolTip(
                index, "Every v2 tool with the .m file it reproduces, plus the track "
                       "views. Per condition, chosen inside the tab.")
        except Exception as exc:                      # noqa: BLE001
            print(f"note: tools tab unavailable ({exc})")

        # The cross-method forest was a cross-condition Δ endpoint, REMOVED
        # (Amendment 4). No dedicated forest tab.
        layout = QVBoxLayout()
        layout.addLayout(self._header)
        layout.addWidget(self._tabs, stretch=1)
        self.setLayout(layout)

    def _spec_cancel(self, cancel):
        """The live spec with a cooperative cancel hook injected (checked between
        conditions in the assembly). ``cancel`` is None for the synchronous path."""
        s = self.spec()
        s.should_cancel = cancel
        return s

    def _compute_for(self, attr, fn):
        """Compute callable for a view tab; accepts an optional cancel hook. SQD
        gets the CI toggle, the fraction view the uncertainty tier."""
        if attr == "sqd_view":
            def _run_sqd(cancel=None):
                # A recompute makes the old per-replicate payloads stale, drop
                # them so a click can never render against a superseded fit; the
                # compute refills the map keyed by condition (Item 3, instant click).
                self._replicate_payloads.clear()
                return fn(self._project, self._spec_cancel(cancel),
                          cache=self._cache, with_ci=self._ci_box.isChecked(),
                          payload_sink=self._replicate_payloads)
            return _run_sqd
        if attr == "fraction_view":
            return lambda cancel=None: fn(
                self._project, self._spec_cancel(cancel), cache=self._cache,
                uncertainty=self._uncertainty_combo.currentText())
        if attr == "spatial_view":
            return lambda cancel=None: fn(
                self._project, self._spec_cancel(cancel), cache=self._cache,
                source=self._spatial_combo.currentText(),
                mode=self._spatial_mode_combo.currentText())
        return lambda cancel=None: fn(self._project, self._spec_cancel(cancel),
                                      cache=self._cache)

    # The cross-method forest (a cross-condition Δ endpoint) was REMOVED
    # (Amendment 4), comparison belongs in the manuscript, not the tool.

    # -- Fit / Plot All (Item 8) ----------------------------------------- #

    def _view_extras(self, attr):
        """The per-view kwargs that enter its cache key (for the skip check)."""
        if attr == "sqd_view":
            return {"with_ci": self._ci_box.isChecked()}
        if attr == "fraction_view":
            return {"uncertainty": self._uncertainty_combo.currentText()}
        if attr == "spatial_view":
            return {"spatial_source": self._spatial_combo.currentText(),
                    "spatial_mode": self._spatial_mode_combo.currentText()}
        return {}

    def _fit_all_jobs(self):
        """(attr, compute, is_cached) for every view, cheapest first."""
        cancel = lambda: self._fit_cancelled
        jobs = []
        for attr in FIT_ALL_ORDER:                       # per-condition views only
            short = attr.replace("_view", "")
            tab = self._view_tabs[attr]
            compute = (lambda t=tab, c=cancel: t._compute(c))
            kw = self._view_extras(attr)
            # translate spatial_source/mode kwargs for is_view_cached
            is_cached = (lambda short=short, kw=kw: cctrl.is_view_cached(
                self._cache, self._project, self.spec(), short, **kw))
            jobs.append((attr, compute, is_cached))
        return jobs

    def _fit_all(self):
        if not self.selected_conditions():
            self._fit_progress.setText("select at least one condition first")
            return
        self._fit_cancelled = False
        self._fit_all_btn.setEnabled(False)
        self._fit_cancel_btn.setVisible(True)
        self._fit_progress.setText("starting…")
        QThreadPool.globalInstance().start(
            _FitAllRunnable(self._fit_all_jobs(), self._fit_signals,
                            lambda: self._fit_cancelled))

    def _fit_all_cancel(self):
        self._fit_cancelled = True
        self._fit_progress.setText("cancelling…")

    def _on_fit_started(self, i, total, attr):
        label = attr.replace("_view", "").replace("_", " ")
        self._fit_progress.setText(f"{i}/{total}: currently: {label}")

    def _on_fit_item_done(self, attr, result, error, skipped):
        if skipped or error is not None or result is None:
            return
        tab = self._view_tabs.get(attr)
        if tab is not None:
            draw, status = result
            tab.apply_result(draw, status)

    def _on_fit_finished(self, summary):
        self._fit_all_btn.setEnabled(True)
        self._fit_cancel_btn.setVisible(False)
        done = len(summary["done"])
        skipped = len(summary["skipped"])
        failed = summary["failed"]
        parts = [f"done {done}", f"skipped {skipped}"]
        if failed:
            parts.append(f"failed {len(failed)}: "
                         + ", ".join(a for a, _ in failed))
        if summary["cancelled"]:
            parts.insert(0, "CANCELLED: ")
        self._fit_progress.setText("Fit All: " + ", ".join(parts))

    # -- the live spec ---------------------------------------------------- #

    def selected_conditions(self):
        return [c for c in self._all if self._condition_boxes[c].isChecked()]

    def spec(self) -> ComparisonSpec:
        lock = self._lock_combo.currentText()
        lock = None if lock == "(per-condition)" else lock
        reference = self._reference_combo.currentText() or None
        from .workspace_view import resolve_shared_d

        scope, groups, factor = resolve_shared_d(self._scope_combo.currentData() or DEFAULT_D_SHARING_SCOPE,
                                                 self.selected_conditions(),
                                                 getattr(self._project, "factors", None))
        return ComparisonSpec(
            conditions=self.selected_conditions(),
            models={c: self._model_combos[c].currentText() for c in self._all},
            lock_model=lock,
            threshold_um2_s=float(self._threshold_spin.value()),
            reference=reference,
            n_boot=self._n_boot,
            fast=FastProfile() if self._fast_box.isChecked() else None,
            # 0, not None: `_ecdf_budget` reads None as the DEFAULT 2000-point
            # budget, so an unticked box used to run the fast fit anyway.
            ecdf_subsample=2000 if self._ecdf_box.isChecked() else 0,
            fraction_weighting=(self._weighting_combo.currentData() or "track"),
            d_sharing_scope=scope, d_sharing_groups=groups, d_sharing_factor=factor,
            # READ HERE OR THE CONTROL DOES NOT EXIST. `d_sharing_scope` sat validated,
            # tested and plotted for months while `spec()` never passed it, so every run
            # took the default and no user could choose. `tests/test_bubble_controls.py`
            # pins both of these against exactly that.
            bubble_fit_modes=tuple(self._bubble_modes_combo.currentData()
                                   or ("independent", "simultaneous")),
            bubble_n_pops=tuple(self._bubble_npops_combo.currentData() or (2,)),
            d_correction=self._dcorr_box.isChecked(),
        )

    # -- events ----------------------------------------------------------- #

    def _on_selection_changed(self, *_):
        self._sync_reference()
        self._on_control_changed()

    def _on_lock_changed(self, *_):
        locked = self._lock_combo.currentText() != "(per-condition)"
        for combo in self._model_combos.values():
            combo.setEnabled(not locked)
        self._on_control_changed()

    def _on_control_changed(self, *_):
        # Lazy: tabs recompute on their next Compute press (they read spec() live).
        self._save_session()

    def _on_fast_toggled(self, *_):
        self._fast_banner.setVisible(self._fast_box.isChecked())
        self._on_control_changed()

    def _on_sigma_changed(self, *_):
        """Manual-σ override (item D): rebuild ``self._project`` from the base with
        the chosen σ (or the derived σ when unticked). All compute reads
        ``self._project`` live, and σ is provenance key material, so cached views
        recompute on their next press rather than serving a stale σ."""
        from .project import with_localization_sigma_nm

        manual = self._sigma_manual_box.isChecked()
        self._sigma_spin.setEnabled(manual)
        sigma = float(self._sigma_spin.value()) if manual else None
        self._project = with_localization_sigma_nm(self._base_project, sigma)
        p = self._project.params
        note = (f"σ = {p.localization_sigma_nm:.1f} nm ({p.localization_sigma_source}); "
                "press Compute to refit affected views")
        if hasattr(self, "_fit_progress"):
            self._fit_progress.setText(note)
        self._on_control_changed()

    # -- the live mobility-threshold slider ------------------------------- #

    def _on_threshold_spin(self, value):
        """The numeric field moved: mirror the slider, save, live-redraw."""
        slider_value = _thresh_to_slider(value)
        if self._threshold_slider.value() != slider_value:
            self._threshold_slider.blockSignals(True)
            self._threshold_slider.setValue(slider_value)
            self._threshold_slider.blockSignals(False)
        self._threshold_changed()

    def _on_threshold_slider(self, slider_value):
        """The slider moved: mirror the numeric field, save, live-redraw."""
        value = _slider_to_thresh(slider_value)
        if abs(self._threshold_spin.value() - value) > 1e-9:
            self._threshold_spin.blockSignals(True)
            self._threshold_spin.setValue(value)
            self._threshold_spin.blockSignals(False)
        self._threshold_changed()

    def _threshold_changed(self):
        self._save_session()
        if self._live_box.isChecked():
            self._redraw_current_if_threshold_dependent()

    def _redraw_current_if_threshold_dependent(self):
        """Live-redraw the visible view when the threshold moves.

        The threshold is a POST-FIT quantity (``slow_fraction_below``), so on the
        SQD view the fits are reused (warm-cache hits) and only the below-threshold
        mass and the threshold line move: genuinely instant. Views that do not
        depend on the threshold (apparent-D, kinetics) are skipped; the ones that
        do but need a bootstrap (fraction, forest) still redraw, just not as fast.
        """
        tab = self._current_tab()
        if tab is None:
            return
        attr = self._tab_attr.get(id(tab))
        if attr in _THRESHOLD_DEPENDENT and tab.has_drawn():
            tab.run()

    def _current_preset(self):
        """The chosen export style: (journal preset, column width)."""
        return (self._preset_combo.currentText(), self._width_combo.currentText())

    def _current_tab(self):
        """The visible ``_ComparisonTab``."""
        widget = self._tabs.currentWidget()
        if isinstance(widget, _ComparisonTab):
            return widget
        return None

    def _sync_reference(self):
        selected = self.selected_conditions()
        current = self._reference_combo.currentText()
        self._reference_combo.blockSignals(True)
        self._reference_combo.clear()
        self._reference_combo.addItems(selected)
        if current in selected:
            self._reference_combo.setCurrentText(current)
        self._reference_combo.blockSignals(False)

    def _show_bubble_inspector(self, selection):
        """A bubble click → the condition's per-replicate simultaneous fit, in a
        single reused non-modal panel. The payload is a cache read (peek); a miss
        shows 'not computed: run Fit / Plot All', never a synchronous fit."""
        condition = getattr(selection, "condition", selection)
        if condition is None:
            return
        if self._bubble_inspector is None:
            self._bubble_inspector = BubbleInspector(self)
        # INSTANT path: the in-memory map filled during the SQD compute. A hit is a
        # dict lookup + render (no provenance hashing), well under the 100 ms budget.
        payload = self._replicate_payloads.get(condition)
        if payload is None:
            # Fall back to the on-disk cache (a warm cross-session hit) as a PEEK
            # only, never a synchronous fit.
            payload = cctrl.replicate_fit_payload(
                self._project, self.spec(), condition, cache=self._cache,
                compute=False)
        if payload is None:
            self._bubble_inspector.show_message(
                f"{condition}: replicate fit not computed: \nrun Fit / Plot All "
                "(or Compute the SQD view) first.")
        else:
            self._bubble_inspector.show_payload(payload)

    def _drill_into(self, selection):
        """A canvas click → drill into the clicked condition's per-cell detail.

        ``selection`` is a :class:`hittest.Selection` (the view's hit-test resolved
        it from the plotted geometry); a bare condition string is still accepted
        for back-compat. Routed through the shared ``navigation.drill`` state
        machine (conditions → replicates → cells → …) so the breadcrumb is the one
        used everywhere, then the resolved condition opens in its per-cell
        ``MethodBrowser``. The finer identity the click carried (method, target,
        population) rides the window title so the drill says WHAT was clicked.
        """
        from .navigation import Breadcrumb

        condition = getattr(selection, "condition", selection)
        if condition is None:
            return None

        if not hasattr(self, "_breadcrumb"):
            self._breadcrumb = Breadcrumb()
        else:
            self._breadcrumb.to_root()
        view = self._breadcrumb.drill_to({"condition": condition})

        # Pass the full selection so the drilled window opens ON the clicked
        # method's tab with a dismissable filter chip, the mark's identity
        # survives the click instead of collapsing to the whole condition. A bare
        # string carries condition only (selection=None → the whole condition).
        drill_selection = selection if not isinstance(selection, str) else None
        browser = MethodBrowser(self._project, condition=condition,
                                n_boot=self._n_boot, selection=drill_selection)
        browser.setWindowTitle(
            f"SMTracker · {view.label}{_drill_detail(selection)} (drill-down)")
        browser.show()
        self._drill_children.append(browser)
        return view

    def _open_track_views(self):
        """Show the FAST track views: the Qt-free player, not a napari window.

        Why this is a jump and not a new window. The track views already live in the
        Tools tab with their own condition / movie / cell pickers, exactly as designed.
        Opening a second copy in its own window would be a parallel path to the same
        widget, and a parallel path beside the one in use is the defect that put the whole
        catalogue behind `--analyses` where nobody found it.

        Returns the sub-tab index it selected, or None if there is nothing to select, so a
        test can assert the button LANDS on the views rather than merely firing.
        No modal dialog on the empty path, and this is a correctness point rather than a
        style one: a `QMessageBox` here blocks the event loop until someone clicks it,
        which headless is forever -- it hung the suite for the full 900 s timeout. It is
        also the wrong answer for a user, because the Tools tab already names which
        conditions loaded, so switching to it SHOWS the reason rather than describing it.
        """
        tools = getattr(self, "_tools_tab", None)
        index = getattr(self, "_tools_index", None)
        if tools is None or index is None:
            return None
        self._tabs.setCurrentIndex(index)
        groups = tools._groups
        for i in range(groups.count()):
            if groups.tabText(i) == "Track views":
                groups.setCurrentIndex(i)
                return i
        return None

    def _open_track_viewer(self):
        """Open the single-track viewer, CHOOSE condition/movie first, then show.

        A plain-Qt picker (no GL) chooses the condition/movie before any napari
        window exists, so a mis-click never spins one up. If a viewer is already
        open, the choice updates it IN PLACE via its session, never a second
        napari window, which is what churned the GL context and crashed
        (GL_INVALID_OPERATION). Failure is reported, never a traceback."""
        if not self.selected_conditions():
            return
        from .trackviewer import (TrackViewerError, open_for_condition,
                                  pick_condition_movie)

        choice = pick_condition_movie(
            self._project, condition=self.spec().reference_condition())
        if choice is None:
            return                                   # cancelled: nothing opened
        condition, movie = choice

        existing = getattr(self, "_track_viewer", None)
        session = getattr(existing, "_smtracker_session", None)
        if session is not None:
            try:                                     # reuse the open window
                session.switch(condition, movie)
                return
            except Exception:                        # pragma: no cover - closed window
                self._track_viewer = None            # stale → open a fresh one below

        try:
            viewer = open_for_condition(self._project, condition, movie=movie)
        except Exception as exc:                       # noqa: BLE001 - see below
            # NOT JUST TrackViewerError. Everything this call reaches -- grouping, mesh
            # reading, canonicalisation, the HMM -- raises its OWN exception type, and
            # only TrackViewerError was caught, so a GroupingError from the day guard
            # escaped as a raw traceback in the console while the window sat there
            # looking like nothing had happened. Observed on a real project.
            #
            # This is a REPORT, not a swallow: the type and the message are both shown,
            # and the traceback still goes to the console for a developer.
            import traceback

            from qtpy.QtWidgets import QMessageBox

            traceback.print_exc()
            QMessageBox.warning(
                self, "Track viewer",
                f"Could not open {condition!r}"
                + (f" / {movie}" if movie else "")
                + f":\n\n{type(exc).__name__}: {exc}")
            return
        self._track_viewer = viewer

    def _save_session(self):
        from .session import save_session

        self._session.selected_conditions = self.selected_conditions()
        self._session.per_condition_model = {
            c: self._model_combos[c].currentText() for c in self._all}
        lock = self._lock_combo.currentText()
        self._session.lock_model = None if lock == "(per-condition)" else lock
        self._session.mobility_threshold_um2_s = float(self._threshold_spin.value())
        ref = self._reference_combo.currentText()
        self._session.reference_condition = ref or None
        self._session.forest_methods = [
            m for m, b in getattr(self, "_forest_boxes", {}).items() if b.isChecked()]
        if hasattr(self, "_preset_combo"):
            self._session.journal_preset = self._preset_combo.currentText()
            self._session.journal_width = self._width_combo.currentText()
        if hasattr(self, "_spatial_combo"):
            self._session.spatial_source = self._spatial_combo.currentText()
            self._session.spatial_mode = self._spatial_mode_combo.currentText()
            self._session.fraction_uncertainty = self._uncertainty_combo.currentText()
            self._session.sqd_cis = self._ci_box.isChecked()
            self._session.sqd_fraction_weighting = (
                self._weighting_combo.currentData() or "track")
            self._session.ecdf_subsample = self._ecdf_box.isChecked()
        save_session(self._session)

    def compute_all(self) -> dict:
        """Run every view once: used by the launch smoke test."""
        return {attr: tab.run() for attr, tab in self._view_tabs.items()}
