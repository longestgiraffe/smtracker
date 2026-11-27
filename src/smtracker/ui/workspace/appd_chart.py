"""One condition's cross-checks page (`ApparentDDetail`): the SQD SuperPlots and the Dynamics evidence (HMM, saSPT,
clustering) of the condition open, each behind its own Run.

It is reached from the Apparent Diffusion section's replicate-day page (Cross-checks; `appd_section.AppDSection.
show_crosschecks`). This module used to hold the section's first version too -- a violin chart, its card and a Bubble
Plot dialog -- which the rebuilt section (`appd_section`, 16 Oct 2026) replaced; they were removed the same night
(asked: "remove old appd code"). What remains draws nothing of its own: the two grids are `fractions_chart`'s and
`dynamics_chart`'s.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QStackedWidget, QVBoxLayout, QWidget

from ..theme import typography as ty
from ..widgets.controls import Button
from ..widgets.primitives import label
from .bubble_chart import CARD_GAP, CARD_PAD
from .compute import _LineTabs
from .dynamics_chart import DynamicsEvidenceGrid
from .fractions_chart import FractionsChecksGrid
from .section_placeholder import _Spinner

__all__ = ["ApparentDDetail"]

#: `ApparentDDetail.visibility_section`/`visibility_states` -- see those
#: methods' own docstrings. Display title (the `EyeRow`/`toolbar.FIGURES`
#: key) -> each grid's own internal short key (`fractions_chart.
#: _fractions_placement`/`dynamics_chart._dynamics_placement`). Kept at
#: module level here, not inside the grids, so a grid stays free of any
#: UI-label dependency -- a title wording change touches only this page.
_VIS_SECTION = {"fractions": "appd-detail-fractions",
               "dynamics": "appd-detail-dynamics"}
_FRACTIONS_KEYS = {"Fraction (per-cell)": "fraction",
                   "Distribution (per-track appD)": "distribution"}
_DYNAMICS_KEYS = {"State count (saSPT)": "saspt",
                  "Kinetics (HMM)": "kinetics",
                  "Clustering": "clustering"}


class ApparentDDetail(QWidget):
    """One condition's cross-checks page, opened from the Apparent Diffusion day page: a PLAIN page
    (no enclosing card -- asked for directly, after a first draft put the
    tab strip and the four chart cards all inside one big outer `Card`,
    which is not "just a tab and four cards" but a tab and four cards
    wrapped in a fifth). A tab strip (`compute._LineTabs`, the real
    shadcn `Tabs`, `variant="line"` -- reused directly rather than a
    second one built here, the same cross-module reuse `_DimOverlay`
    already has) sits directly on the page background, above whichever
    tab's own content is selected (`self._tab_pages`). No back button --
    the breadcrumb IS the back-navigation, matching every other drill-in
    page in this workspace.

    **Nothing from `appd_points`'s own per-condition entry (a violin, a
    per-cell table) lives on this page any more** -- also asked for
    directly. `set_payload` therefore draws nothing itself; it only
    tracks WHICH condition is open and refreshes whichever tab is
    showing.

    See `docs/method_browser_tab_groups.md`: this page is where each of
    that document's proposed GROUPS lives, one tab each, for THIS
    condition -- not a fourth sidebar section, since the entry point is
    always "I clicked a condition", never "I want cross-checks in the
    abstract". Two tabs exist so far: "SuperPlots"
    (`fractions_chart.FractionsChecksGrid`) and "Dynamics"
    (`dynamics_chart.DynamicsEvidenceGrid`, Group 2 -- Kinetics/State
    count/Clustering) -- Spatial/QC are documented, not built.

    **Each tab is a PARALLEL, independent copy of the same idle/running/
    computed/cache shape**, not a shared abstraction -- this project's
    own stated rule (used identically when the workspace went from one
    section to two, SQD -> SQD + Apparent-D): a shared "detail tab"
    controller becomes the right call once a THIRD tab (Spatial) shows up
    to justify it, not at two. Neither tab is auto-computed on drill-in:
    both are real fits/bootstraps, not cheap, so each waits for its OWN
    "Run analysis" press, scoped to the condition on screen, cached per
    condition (`self._fractions_cache`/`self._dynamics_cache`) so
    revisiting a condition already computed never re-fits it. The actual
    off-GUI-thread dispatch lives in `window.py`
    (`_run_fractions_checks`/`_run_dynamics_evidence`), matching this
    file's own "every Run lives in the window" convention --
    `fractions_run_requested`/`dynamics_run_requested` are the signals
    that ask for it.
    """

    #: The condition to compute the SuperPlots FOR -- emitted only while
    #: a condition is on screen (see `_request_fractions_run`).
    fractions_run_requested = Signal(str)
    #: The condition to compute Dynamics (Kinetics/State count/Clustering)
    #: FOR -- see `_request_dynamics_run`.
    dynamics_run_requested = Signal(str)

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        self.t = t
        outer = QVBoxLayout(self)
        # No card padding -- this page has no enclosing card to keep its
        # content off the border of; the window's own outer margin
        # already gives it room.
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(CARD_GAP)

        # Internal ids stay stable ("fractions"/"dynamics") even as visible
        # labels change -- matching this project's "keep the key, change
        # the label" convention (the "appd" section key surviving its own
        # "Apparent Diffusion" rename).
        self.tabs = _LineTabs(
            t, [("fractions", "SuperPlots", "layers"),
               ("dynamics", "Dynamics", "activity")],
            current="fractions")
        self.tabs.changed.connect(self._on_tab_changed)
        outer.addWidget(self.tabs)

        self._fractions_run_btn = Button(t, "Run analysis", variant="default",
                                         glyph="play")
        self._fractions_run_btn.clicked.connect(self._request_fractions_run)
        self._fractions_idle_message = label(
            "This runs an SQD fit with its cell bootstrap and an HMM fit for this condition, which takes a "
            "while. Press Run to compute it.",
            ty.ui("sm"), t.muted_foreground)
        self._fractions_idle_message.setWordWrap(True)
        self._fractions_idle_message.setAlignment(Qt.AlignHCenter)
        #: `section_placeholder._Spinner`, reused directly rather than a
        #: second one built here -- the SAME cross-module reuse `_DimOverlay`
        #: already has, and this file's own module docstring already names
        #: the reason: a running fit should read as ONE visual language
        #: everywhere it is shown, not a different affordance per page.
        #: Hidden until `set_fractions_running(True)` -- shown IN PLACE of
        #: the idle message, matching `SectionPlaceholder`'s own idle/
        #: running swap, not stacked alongside it.
        self._fractions_spinner = _Spinner(t)
        self._fractions_spinner.hide()

        fractions_idle = QWidget()
        fractions_idle.setStyleSheet("background:transparent;")
        idle_lay = QVBoxLayout(fractions_idle)
        idle_lay.setContentsMargins(0, CARD_PAD, 0, CARD_PAD)
        idle_lay.setSpacing(CARD_GAP)
        idle_lay.addStretch(1)
        idle_lay.addWidget(self._fractions_idle_message)
        idle_lay.addWidget(self._fractions_spinner, alignment=Qt.AlignHCenter)
        idle_lay.addWidget(self._fractions_run_btn, alignment=Qt.AlignHCenter)
        idle_lay.addStretch(1)

        self._fractions_grid = FractionsChecksGrid(t)

        self._fractions_pages = QStackedWidget()
        self._fractions_pages.addWidget(fractions_idle)
        self._fractions_pages.addWidget(self._fractions_grid)

        #: condition -> `compute_fractions_checks`'s own payload, cached so
        #: revisiting a condition already computed never re-fits it.
        self._fractions_cache: dict[str, dict] = {}

        # -- Dynamics tab: a PARALLEL copy of the SuperPlots block above,
        # same shape, see the class docstring's own note on why this is
        # not generalised yet. -- #
        self._dynamics_run_btn = Button(t, "Run analysis", variant="default",
                                        glyph="play")
        self._dynamics_run_btn.clicked.connect(self._request_dynamics_run)
        self._dynamics_idle_message = label(
            "This runs an HMM fit with bootstrap, a saSPT spectrum and a clustering null for this condition, "
            "which takes a while. Press Run to compute it.",
            ty.ui("sm"), t.muted_foreground)
        self._dynamics_idle_message.setWordWrap(True)
        self._dynamics_idle_message.setAlignment(Qt.AlignHCenter)
        self._dynamics_spinner = _Spinner(t)
        self._dynamics_spinner.hide()

        dynamics_idle = QWidget()
        dynamics_idle.setStyleSheet("background:transparent;")
        dyn_idle_lay = QVBoxLayout(dynamics_idle)
        dyn_idle_lay.setContentsMargins(0, CARD_PAD, 0, CARD_PAD)
        dyn_idle_lay.setSpacing(CARD_GAP)
        dyn_idle_lay.addStretch(1)
        dyn_idle_lay.addWidget(self._dynamics_idle_message)
        dyn_idle_lay.addWidget(self._dynamics_spinner, alignment=Qt.AlignHCenter)
        dyn_idle_lay.addWidget(self._dynamics_run_btn, alignment=Qt.AlignHCenter)
        dyn_idle_lay.addStretch(1)

        self._dynamics_grid = DynamicsEvidenceGrid(t)

        self._dynamics_pages = QStackedWidget()
        self._dynamics_pages.addWidget(dynamics_idle)
        self._dynamics_pages.addWidget(self._dynamics_grid)

        self._dynamics_cache: dict[str, dict] = {}

        #: The outer switch between the two TABS' own content -- moved by
        #: `_on_tab_changed` only, never anything inside either tab.
        self._tab_pages = QStackedWidget()
        self._tab_pages.addWidget(self._fractions_pages)
        self._tab_pages.addWidget(self._dynamics_pages)
        outer.addWidget(self._tab_pages, 1)

        #: The condition currently on screen -- `_request_fractions_run`/
        #: `_request_dynamics_run` need this to know what they are asking
        #: for, and `set_payload` (called on every drill-in click) needs
        #: it to refresh whichever tab is showing.
        self._condition: str | None = None

    def _on_tab_changed(self, key: str) -> None:
        self._tab_pages.setCurrentIndex(0 if key == "fractions" else 1)

    def visibility_section(self) -> str:
        """Which `toolbar.FIGURES` section describes the ACTIVE tab's own
        charts -- `window.py` reads this to keep the Visibility popover in
        step with `self.tabs`."""
        return _VIS_SECTION[self.tabs.current()]

    def visibility_states(self) -> dict:
        """The active tab's own per-chart visibility, keyed by DISPLAY
        TITLE (matching what `VisibilityBody.set_section`'s `states` param
        expects -- the same strings `toolbar.FIGURES` and `EyeRow.key`
        use)."""
        if self.tabs.current() == "fractions":
            states = self._fractions_grid.visibility_states()
            keys = _FRACTIONS_KEYS
        else:
            states = self._dynamics_grid.visibility_states()
            keys = _DYNAMICS_KEYS
        return {title: states[short] for title, short in keys.items()}

    def set_fractions_chart_visible(self, key: str, on: bool) -> None:
        self._fractions_grid.set_card_visible(_FRACTIONS_KEYS[key], on)

    def set_dynamics_chart_visible(self, key: str, on: bool) -> None:
        self._dynamics_grid.set_card_visible(_DYNAMICS_KEYS[key], on)

    def _request_fractions_run(self) -> None:
        if self._condition is not None:
            self.fractions_run_requested.emit(self._condition)

    def set_fractions_running(self, on: bool) -> None:
        self._fractions_run_btn.setEnabled(not on)
        self._fractions_run_btn.set_text("Computing…" if on else "Run analysis")
        # `_Spinner.showEvent`/`hideEvent` start/stop its own animation, so
        # toggling visibility here is the whole mechanism -- no separate
        # start()/stop() call needed.
        self._fractions_idle_message.setVisible(not on)
        self._fractions_spinner.setVisible(on)

    def set_fractions_payload(self, condition: str, payload: dict) -> None:
        """A `window._run_fractions_checks` result landed -- cache it
        regardless of what is currently on screen (a Run started on one
        condition may finish after the user has already clicked to
        another), but only touch the Run button / show it if that
        condition is STILL the one open -- two different conditions can
        each have their own job in flight at once, and this page has only
        ONE Run button/running-state, shared by whichever condition is
        currently displayed."""
        self._fractions_cache[condition] = payload
        if condition == self._condition:
            self.set_fractions_running(False)
            self._show_fractions_for(condition)

    def set_fractions_error(self, condition: str, message: str) -> None:
        if condition == self._condition:
            self.set_fractions_running(False)
            self._fractions_grid.set_message(message)
            self._fractions_pages.setCurrentIndex(1)

    def _show_fractions_for(self, condition: str) -> None:
        cached = self._fractions_cache.get(condition)
        if cached is None:
            self._fractions_pages.setCurrentIndex(0)
        else:
            self._fractions_grid.set_payload(cached)
            self._fractions_pages.setCurrentIndex(1)

    # -- Dynamics tab -- line-for-line mirror of the SuperPlots methods
    # above; see the class docstring's own note on why this stays a
    # parallel copy rather than a shared method. -- #
    def _request_dynamics_run(self) -> None:
        if self._condition is not None:
            self.dynamics_run_requested.emit(self._condition)

    def set_dynamics_running(self, on: bool) -> None:
        self._dynamics_run_btn.setEnabled(not on)
        self._dynamics_run_btn.set_text("Computing…" if on else "Run analysis")
        self._dynamics_idle_message.setVisible(not on)
        self._dynamics_spinner.setVisible(on)

    def set_dynamics_payload(self, condition: str, payload: dict) -> None:
        self._dynamics_cache[condition] = payload
        if condition == self._condition:
            self.set_dynamics_running(False)
            self._show_dynamics_for(condition)

    def set_dynamics_error(self, condition: str, message: str) -> None:
        if condition == self._condition:
            self.set_dynamics_running(False)
            self._dynamics_grid.set_message(message)
            self._dynamics_pages.setCurrentIndex(1)

    def _show_dynamics_for(self, condition: str) -> None:
        cached = self._dynamics_cache.get(condition)
        if cached is None:
            self._dynamics_pages.setCurrentIndex(0)
        else:
            self._dynamics_grid.set_payload(cached)
            self._dynamics_pages.setCurrentIndex(1)

    def set_colour(self, colour: str | None) -> None:
        """The open condition's colour: the SuperPlots draw its replicates as shades."""
        self._fractions_grid.set_colour(colour)

    def condition(self) -> str | None:
        return self._condition

    def set_payload(self, condition: str, _payload: dict | None = None) -> None:
        """Which condition is now open, and refresh BOTH tabs for it
        (whichever is showing is the one the user sees; the other's cache
        state is still refreshed so switching tabs afterward is instant).
        `_payload` is accepted for call-site compatibility and unused --
        nothing on this page draws it, see the class docstring."""
        self._condition = condition
        self._show_fractions_for(condition)
        self._show_dynamics_for(condition)
