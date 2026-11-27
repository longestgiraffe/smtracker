"""The workspace window: `Sidebar` + a content area + `Toolbar`, assembled.

`open_workspace(project)` reads a few plain facts off an ALREADY LOADED
`smtracker.project.Project` -- its conditions, a title, a track count -- to
fill the sidebar header, the Conditions tool and the Compute dialog's own
per-condition rows. The PROJECT ITSELF is kept (`self.project`), because
`_run_pressed` now runs the real SQD fit through it -- see "RUN IS WIRED"
below.

COMPUTE IS AN IN-WINDOW OVERLAY, NOT A CONTENT-AREA PAGE OR AN OS WINDOW --
see `compute.py`'s module docstring for the popover -> page -> dialog ->
overlay history. It covers this window's own client area and disables
`self.centralWidget()` for as long as it is up (`ComputeDialog.open`), so
this window carries none of the page-swap machinery a page would have needed
(no stacked pages, no breadcrumb extension, nothing to return FROM): opening
the overlay does not touch `self.sidebar.current` or `self.breadcrumb` at
all, and dismissing it (`Cancel`, the corner `X`, or `Run`) is the overlay's
own business -- see `ComputeDialog._dismiss`.

RUN IS WIRED, through `bubble_chart.compute_bubble_view` -- which calls the
SAME assembly functions (`comparison_controller.assemble_view`,
`._overlay_table`) that `comparison_controller.sqd_view` itself calls, the
function `browser.ComparisonBrowser`'s own SQD tab uses. So this window and
the older browser fit the SAME numbers from the SAME spec; only the picture
drawn from them differs -- see `bubble_chart.py`'s own module docstring for
why the drawing is native (`BubbleChart`, `QPainter`) rather than matplotlib,
which is what this used the first time and was corrected.
`workspace_view.spec_from_values` is the ONE place the dialog's staged values
(`ComputePanel.values()` / `.condition_values()`, read by `_spec_from_dialog`)
become a real `comparison.ComparisonSpec` -- mirroring
`ComparisonBrowser.spec()`'s own field-by-field mapping exactly, since that
mapping is already validated (`tests/test_bubble_controls.py` exists
precisely because a spec field once sat wired to nothing).

**OFF THE GUI THREAD**, the same shape `browser._ComparisonTab.run_async`
already proved: a `QRunnable` on `QThreadPool.globalInstance()`, a
generation-token stale-result guard (`_sqd_gen`/`_on_compute_done`) so a
result from a superseded Run can never land on the canvas, and
`parallel.set_serial_only(True)` for the worker's duration so a bootstrap
CI's own process pool cannot be spawned from a non-main thread (Item 7e in
the browser's own history). `_ComputeSignals`/`_ComputeRunnable` are a
SEPARATE, structurally-identical pair here rather than an import of
`browser`'s -- those classes are UI plumbing, not a calculation, and this UI
stack otherwise has no dependency on the old browser module; duplicating a
thread-runner is not the "one source of truth" a fraction number needs.

**OFF THE GUI THREAD IS NOT ENOUGH FOR A REAL FIT -- SQD'S RUN IS OFF THE
GUI *PROCESS* TOO.** Reported directly: a real multistart `fit_sqd`
(no bootstrap even required -- this project's own Compute dialog defaults
of full ECDF + both overlay modes already make one) is a long, tight,
Python-heavy loop -- thousands of small calls, each re-acquiring the GIL
moments after releasing it -- and MEASURED DIRECTLY: a real ~5s fit pegged
one core (`effective_cores=1.03`, cpu time ~= wall time) while the whole
app's own event processing stalled for that same ~5s, spinner included.
Two THREADS in one process share ONE GIL, and CPython's handoff between
them is not fair enough to stop a loop shaped like this from starving the
main thread of it -- "the fit runs on a worker thread" is true and still
insufficient, since the bottleneck was never spare CPU cores (there were
idle ones the whole time) but a SOFTWARE lock inside the process. A
SEPARATE OS PROCESS has its own, independent GIL, so nothing it does can
ever contend for the main thread's; past that point it is purely an OS
scheduling question, and the OS's own preemptive scheduler is far fairer
than CPython's. `self._process_pool` (a persistent `ProcessPoolExecutor`,
see its own comment) is where `_run_pressed` submits the real fit; the
`QThreadPool` worker thread above still exists for it, but its whole job
is now "submit, then block on the result" -- that block is an OS-level
read, which releases the GIL for its ENTIRE duration, so
`parallel.set_serial_only(True)` around it is harmless but does no real
work for THIS path any more (it still matters for `_run_fractions_checks`,
which is NOT on this project pool and still runs its own compute directly on a
`QThreadPool` thread).
`compute_bubble_view`'s own arguments (`project`, `spec`) cross the
process boundary by pickling -- verified directly against a real
`Project`, not assumed.

While a fit is running, `ComputeMark.set_running(True)` turns the round
Compute glyph into its own spinner -- a control this rail already had for
exactly this, built and wired to nothing until now (`toolbar.py`'s own
`ComputeMark` docstring already describes the hover-to-stop-square
affordance for a running job; only actual cancellation is still unbuilt).

**THE FIRST RUN AND EVERY RUN AFTER IT ARE TWO DIFFERENT MOMENTS** -- see
`section_placeholder.SectionPlaceholder`'s own module docstring for the
reasoning. Before this section has EVER produced a real result,
`self._pages` shows `self._placeholder` instead of `self._card` -- one big
"nothing computed yet" surface owning its own idle/running sub-states,
rather than the chart card showing a bare text message the way it used to.
`self._ever_succeeded` flips once, on the first successful
`_on_compute_done`, and this window never flips it back and never shows the
placeholder again: a LATER failed or superseded Run dims `self._card` in
place (`BubbleCard.set_running`) instead of bouncing back to it, which is
also what keeps the previous, still-valid result on screen while a re-run
is in flight.

**EACH SECTION RUNS ON ITS OWN -- a section computes and renders only from its OWN Run.** An earlier version had
the SQD dialog's Run recompute the Apparent-D view too; reported as surprising, so `_run_pressed` computes ONLY the
SQD bubble view and every other section keeps its own Run, its own generation counter and its own messages, so one
section's completion can never supersede (and drop) another's in-flight result.

**APPARENT DIFFUSION, REBUILT 16 OCT 2026** (`appd_section`, `appd_view.appd_condition`): one apparent D per track
over all its steps, four pages behind a strip (Populations, Distributions with the track gallery, Per cell, Checks)
that the sidebar's sub-views follow, its own Run dialog (`AppDRunDialog`: populations, uncertainty, shortest track)
and one process-pool job per condition (`_run_appd`, the HMM section's shape). The old two displays (a violin
Distribution with no dialog and a 4-step-piece Bubble Plot with its own) are gone; the SQD/HMM cross-checks of one
condition (`appd_chart.ApparentDDetail`) are kept, one button from the apparent-D day page.
"""
from __future__ import annotations

import multiprocessing
import os
import threading
import time
from contextlib import contextmanager, nullcontext as _nullcontext
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

from qtpy.QtCore import (
    QEasingCurve,
    QObject,
    QPointF,
    QRunnable,
    QThreadPool,
    QTimer,
    QVariantAnimation,
    Qt,
    Signal,
    Slot,
)
from qtpy.QtWidgets import (
    QApplication,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QMainWindow,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ... import theme_view
from ... import condition_colours
from ... import subpages as sp
from ... import condition_order, factor_layout
from ... import options as opts
from ..widgets.toast import Toaster
from ...appd_view import appd_condition
from ...bubble_view import compute_bubble_view, progress_text, warm_condition
from ...comparison import ComparisonSpec
from ...dynamics_view import compute_dynamics_evidence
from ...fractions_view import compute_fractions_checks
from ...clustering_compare import compute_shared_clustering
from ...clustering_view import compute_clustering_view, compute_kwargs, wants_per_spot
from ...induction_view import (MOBILITY_CHANGES, POPULATION_READOUTS, movie_mobility, attach_reference, find_reference, movie_extents, compute_induction, condition_days, condition_movies,
                               reference_curves, run_permutation)
from ...kinetics_designs import load_additions, load_culture_times, save_additions, save_culture_times
from ...kinetics_mock_view import attach_mock, attach_no_mock, compute_mock
from ...kinetics_view import MOCK_READOUTS, compute_kinetics
from ...msd_view import msd_condition
from ...hmm_view import compute_hmm_view
from ...hmm_angles import compute_hmm_angles
from ...jd_view import compute_jd_view
from ...jumps_view import compute_jump_view
from ...parallel import bulk_slots, lower_priority, n_workers, stop_pool
from ...sectioncache import cached_group_job, cached_job
from ...qc_view import compute_qc_overview, track_cap_of
from ...run_progress import sidebar_job
from ... import map_fit_source
from ...map_fit_source import MAP_THIS_FIT, map_this_fit_tip
from ..widgets.controls import Button
from ...spatial_map_view import (
    compute_spatial_map,
    default_cmap_for,
    planned_counts,
    planned_n_pops,
    measure_cell_frame,
    project_pixel,
    split_preview,
    track_census,
)
from ...warmcache import WarmCache, default_warm_root
from ...workspace_view import (RunSettingsError, compute_toast, precision_off, spec_from_values,
                                workspace_subtitle)
from .. import theme
from ..theme import typography as ty
from ..widgets.help import show_help_menu
from ..widgets.options_dialog import open_options
from ..widgets.primitives import ShadowHost, fit_label, scrollbar_css
from ..widgets.screenfit import ScreenFit, show_fitted
from .appd_chart import ApparentDDetail
from .appd_section import PAGES as APPD_PAGES
from ... import rail_view
from ...rail_view import REPLICATE_PAGE, compute_job, compute_words, rail_section
from .appd_section import AppDRunDialog, AppDSection
from .bubble_chart import BubbleCard, ReplicateDetail
from .compute import ComputeDialog
from .composer_page import ComposerSection
from .images_page import ImagesSection
from .clustering_dialog import ClusteringConfigDialog
from .clustering_section import SHARED_KEY as SHARED_CLUSTERS_KEY
from .clustering_section import ClusteringSection
from .kinetics_chart import KineticsConfigDialog, KineticsSection
from .msd_chart import PAGES as MSD_PAGES

from .hmm_chart import PAGES as HMM_SECTION_PAGES

#: The SQD section's pages, in the strip's order (the sidebar's ``sqd.<page>`` sub-views).
SQD_PAGES = ("populations", "jumps")
from .msd_chart import MSDRunDialog, MSDSection
from .hmm_chart import HMMConfigDialog, HMMSection
from .jd_chart import JDSection
from .jumps_chart import JumpsConfigDialog, JumpsSection
from .qc_chart import _FLAG as _FIT_WARNING_INK
from .qc_chart import (
    QCConditionDetail,
    QCCellLengthCard,
    QCFairTable,
    QCNothingCard,
    QCOverviewTable,
)
from .section_placeholder import SectionPlaceholder
from .sidebar import GLYPH, SECTION_TITLE, SUBS, TRIGGER, Breadcrumb, Sidebar, SidebarTrigger
from .spatial_map_chart import SpatialMapDialog, SpatialMapSection
from .toolbar import Toolbar

#: The Visibility tool's sections for the Spatial Map, one per computational method: the
#: grid's columns differ by method, so each has its own list. ONE tuple, read wherever a
#: pick is routed -- it was written out four times, which a fourth method would have missed.
SPATIAL_SECTIONS = ("spatial", "spatial-appd", "spatial-hmm", "spatial-sla", "spatial-sda")

__all__ = ["WorkspaceWindow", "open_workspace"]


class _WidthBlind(QWidget):
    """A wrapper in the window's own column (the header row, the page wrapper) whose height NEVER follows its width.

    A word-wrapped label anywhere below makes every layout above it height-for-width, and Qt then re-solves the
    whole chain on each width change (29-46 ms a frame of the rail's slide, `tests/test_qc_widgets.py`). The labels
    are pinned (`primitives.FitLabel`), but a box layout CACHES its answer: found 17 Oct 2026, the page wrapper still
    said yes after every child said no, in the full run only. Nothing in the column needs its height from its width
    (every page scrolls), so the column's two wrappers say no, structurally."""

    def hasHeightForWidth(self) -> bool:
        return False


#: Why a session's Molecules results are not restored: the section was removed (19 Oct 2026). On SynR movies in
#: which every molecule is one fluorophore it still found three components and called 29 % of tracks "two or more"
#: (`docs/claude/intensity_and_controls.md`, "Molecule counting removed").
REMOVED_MOLECULES = ("this section was removed: its count of fluorophores per spot depended on the method, not on "
                     "the data")


class _ComputeSignals(QObject):
    """A worker's result, delivered back to the GUI thread.

    `(generation, result_or_None, error_or_None)` -- same shape as
    `browser._ComputeSignals`, so the two stay easy to compare, but a
    separate class (see the module docstring on why this isn't an import).
    """

    done = Signal(int, object, object)
    #: `(generation, done, total, finished_names)` -- SQD's per-condition progress.
    progress = Signal(int, int, int, object)


class _ComputeRunnable(QRunnable):
    """Runs one `compute()` call OFF the GUI thread, then signals the result.

    `parallel.set_serial_only(True)` for the duration: a bootstrap CI or a
    per-condition fit can otherwise reach for its own process pool, and a
    pool spawned from a worker thread (rather than the main thread) is
    exactly the nested-pool hazard `browser._ComputeRunnable` already
    exists to avoid.
    """

    def __init__(self, compute, gen: int, signals: _ComputeSignals):
        super().__init__()
        self._compute = compute
        self._gen = gen
        self._signals = signals

    def run(self):                                      # pragma: no cover - thread
        from ... import parallel

        parallel.set_serial_only(True)
        try:
            result, error = self._compute(), None
        except Exception as exc:                        # noqa: BLE001 - surfaced
            result, error = None, exc
        try:
            # The window this was FOR may have closed (or, at shutdown, the
            # whole interpreter may be tearing down) by the time a slow fit
            # finally finishes -- `_signals` is then a deleted C++ object on
            # the Python side. Same shape as `tooltip.show_for`'s own
            # `except RuntimeError: pass  # its widget went away first`:
            # nothing is left to update, so a late result is dropped rather
            # than raised inside a QRunnable, where Qt reports it as an
            # opaque "Error calling Python override of QRunnable::run()".
            self._signals.done.emit(self._gen, result, error)
        except RuntimeError:
            pass
        finally:
            parallel.set_serial_only(False)


def _spec_from_dialog(compute, factors=None, project=None) -> ComparisonSpec:
    """The Compute dialog's staged values, as a real `ComparisonSpec`: reads the panel's two
    plain dicts and hands them to `workspace_view.spec_from_values`, where the rules live, with the folder design's
    factors (Shared D within a group) and, under Localisation error Per spot, the Run's calibration
    (`workspace_view.run_precision`). Raises `RunSettingsError` when the settings cannot make a Run (the overlay
    names no one state count; a condition has no value of the factor D is shared within; no calibration)."""
    from ...workspace_view import PRECISION_PER_SPOT, precision_off, run_precision

    values, conditions = compute.values(), compute.condition_values()
    precision, off = None, ()
    if values.get("precision") == PRECISION_PER_SPOT and project is not None:
        chosen = [c for c, (on, _m) in conditions.items() if on]
        precision = run_precision(project, chosen)
        off = precision_off(project, chosen)          # a truth condition, a track table: fitted without the model
    return spec_from_values(values, conditions, factors=factors, precision=precision, precision_off=off)


# -- SUBPAGES: the window's page slots (`smtracker.subpages`) ------------------------------------------------------ #
#
# A section that offers the "+" (`subpages.ENABLED`) is not one set of widgets and counters but a LIST OF PAGE SLOTS:
# the section's own page and every added copy. A `_PageSlot` is a bag holding what ONE page keeps -- its widgets, its
# Compute dialog, its generation counters, its results. The window's attributes for those things (`_card`, `_jumps`,
# `compute_dialog`, `_sqd_gen` ... the lists below) are `_PerPage` descriptors that read and write the slot that is
# CURRENT for their section (`WorkspaceWindow._cur`), so the code written before subpages existed -- and every test
# that reads `window._card` -- keeps meaning "the page in hand". Which slot is in hand is decided in three places only:
#
# * the page on show: `_on_selected` makes the selected page current for its section;
# * a signal of a page's own widget: connected through `WorkspaceWindow._bind`, which runs the handler with that page
#   current;
# * a worker's result: delivered by a `_PageRelay`, a QObject (so the call is queued to the GUI thread) that runs the
#   handler with the page the job was started for current.
#
# Never read a per-page attribute from a worker thread: capture the slot (or the value) on the GUI thread first.

#: What the SQD section keeps per page: the strip, the three pages and everything their Runs count.
SQD_PAGE_ATTRS = (
    "_card", "_detail", "_pages", "_placeholder", "status", "fit_warning", "_sqd_section", "_sqd_tabs", "_sqd_stack",
    "_jumps", "_jumps_status", "_jd", "_jd_status", "compute_dialog", "_ever_succeeded", "_replicate_payloads",
    "_signals", "_sqd_progress", "_sqd_clock", "_last_sqd_run", "_sqd_gen", "_sqd_pending", "_sqd_total",
    "_sqd_plot", "_sqd_computed_at", "_detail_condition",
    "_jumps_signals", "_jumps_gen", "_jumps_run_id", "_jumps_by_gen", "_jumps_full", "_jumps_pending",
    "_jumps_total", "_jumps_failed",
    "_jd_signals", "_jd_gen", "_jd_run_id", "_jd_by_gen", "_jd_pending", "_jd_total", "_jd_failed",
    "_sqd_choice", "_sqd_brackets", "_sqd_copy_note",
)
#: What the Spatial Map keeps per page: the grid, its dialog and its Run's bookkeeping.
SPATIAL_PAGE_ATTRS = (
    "_spatial", "_spatial_status", "_spatial_section", "map_dialog", "_spatial_signals", "_map_preview_signals",
    "_map_preview_gen", "_map_preview_by_gen", "_map_hidden", "_hide_signals", "_hide_gen", "_hide_by_gen",
    "_hide_latest", "_spatial_gen", "_spatial_run_id", "_spatial_by_gen", "_spatial_pending", "_spatial_total",
    "_spatial_waiting", "_spatial_failed", "_spatial_last_config", "_spatial_computed_at", "_spatial_copy_note",
)
#: What the MSD section keeps per page (phase 2): the section, its status line, its Run's counters and signals.
CLUSTERS_PAGE_ATTRS = (
    "_clusters", "_clusters_status", "_clusters_section", "_clusters_copy_note", "_clusters_signals", "_clusters_gen",
    "_clusters_run_id", "_clusters_by_gen", "_clusters_pending", "_clusters_total", "_clusters_failed",
    "_clusters_cancel", "_clusters_futures",
)
KINETICS_PAGE_ATTRS = (
    "_kinetics", "_kinetics_status", "_kinetics_section", "_kinetics_copy_note", "_kinetics_signals", "_kinetics_gen",
    "_kinetics_run_id", "_kinetics_by_gen", "_kinetics_pending", "_kinetics_total", "_kinetics_failed",
    "_kinetics_cancel", "_kinetics_work",
)
HMM_PAGE_ATTRS = (
    "_hmm", "_hmm_status", "_hmm_section", "_hmm_copy_note", "_hmm_signals", "_hmm_gen", "_hmm_run_id", "_hmm_by_gen",
    "_hmm_pending", "_hmm_total", "_hmm_failed", "_hmm_config", "_hmm_joint", "_hmm_cancel",
    "_hmm_choice", "_hmm_brackets",
    # THE JUMP ANGLES TAB (23 Nov 2026, a sidebar section of its own from 22 Nov): its page, the fits asked for, its
    # jobs -- read from THIS page's fits
    "_angles_page", "_angles_asked", "_angles_signals", "_angles_gen", "_angles_by_gen", "_angles_pending",
    "_angles_total", "_angles_failed",
)
APPD_PAGE_ATTRS = (
    "_appd", "_appd_detail", "_appd_status", "_appd_section", "_appd_copy_note", "_appd_signals", "_appd_gen",
    "_appd_run_id", "_appd_by_gen", "_appd_pending", "_appd_total", "_appd_failed", "_appd_config",
    "_fractions_signals", "_fractions_gen", "_fractions_condition_by_gen",
    "_dynamics_signals", "_dynamics_gen", "_dynamics_condition_by_gen",
    "_appd_choice", "_appd_brackets",
)
MSD_PAGE_ATTRS = (
    "_msd", "_msd_status", "_msd_section", "_msd_copy_note", "_msd_signals", "_msd_gen", "_msd_run_id", "_msd_by_gen",
    "_msd_pending", "_msd_failed", "_msd_total",
)
#: The bubble sections whose brackets' choice and warnings are a PAGE's own (`_<section>_choice`, `_<section>_brackets`).
PAGED_CHOICE = ("sqd", "appd", "hmm")
PAGE_ATTRS = {"sqd": SQD_PAGE_ATTRS, "spatial": SPATIAL_PAGE_ATTRS, "msd": MSD_PAGE_ATTRS, "appd": APPD_PAGE_ATTRS, "hmm": HMM_PAGE_ATTRS, "kinetics": KINETICS_PAGE_ATTRS,
              "clusters": CLUSTERS_PAGE_ATTRS}


class _PageSlot:
    """One page of a section: the section's own (``number`` None) or an added copy. Holds the per-page attributes."""

    def __init__(self, section: str, number=None):
        self.section, self.number = section, number

    @property
    def key(self) -> str:
        return sp.page_key(self.section, self.number)


class _PerPage:
    """A window attribute that lives on the CURRENT page slot of its section (see the block comment above)."""

    def __init__(self, section: str, name: str):
        self.section, self.name = section, name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        try:
            return getattr(obj._cur[self.section], self.name)
        except AttributeError:
            raise AttributeError(self.name) from None

    def __set__(self, obj, value):
        setattr(obj._cur[self.section], self.name, value)


class _PageRelay(QObject):
    """Delivers a page's worker results to the window with that page current. A QObject on the GUI thread, so the
    signal that reaches it is queued there (a bare lambda would run on the emitting thread)."""

    def __init__(self, window, slot, done: str, progress: str | None = None):
        super().__init__(window)
        self._window, self._slot, self._done, self._progress = window, slot, done, progress

    @Slot(int, object, object)
    def done(self, gen, result, error):
        self._window._call_in(self._slot, getattr(self._window, self._done), gen, result, error)

    @Slot(int, int, int, object)
    def progress(self, gen, done, total, finished):
        if self._progress:
            self._window._call_in(self._slot, getattr(self._window, self._progress), gen, done, total, finished)



class _ExportAllSignals(QObject):
    #: index, total, name
    progress = Signal(int, int, str)
    #: rows (`figure_export.export_many`), folder, error
    done = Signal(list, str, str)


class _ExportAllJob(QRunnable):
    """`figure_export.export_many` off the GUI thread: twenty figures are tens of seconds."""

    def __init__(self, jobs, folder, stamp):
        super().__init__()
        self.jobs, self.folder, self.stamp = jobs, str(folder), stamp
        self.signals = _ExportAllSignals()

    def run(self):
        from ... import figure_export as fx

        try:
            rows = fx.export_many(self.jobs, self.folder, stamp=self.stamp,
                                  progress=lambda i, n, name: self.signals.progress.emit(
                                      i, n, name))
            self.signals.done.emit(rows, self.folder, "")
        except Exception as exc:                            # noqa: BLE001 - shown AND printed
            import traceback

            traceback.print_exc()
            self.signals.done.emit([], self.folder, f"{type(exc).__name__}: {exc}")

#: Threads the window's jobs may hold on Qt's shared pool, each WAITING on its worker (`WorkspaceWindow.__init__`).
JOB_THREADS = 64


class WorkspaceWindow(QMainWindow):
    """The shell. The Populations content is the SQD bubble card Run
    paints into; the Compute dialog beside it is real -- see the module
    docstring."""

    #: An added page was created, renamed, moved or deleted (`subpages`): the section's key. A map's source drop-down
    #: and anything else that lists the pages follows it.
    subpages_changed = Signal(str)
    #: A page was deleted: ``(section, its name)``, for whoever held it (a map that took its D from it).
    subpage_removed = Signal(str, str)

    def __init__(self, t, *, project, project_title: str, project_subtitle: str,
                 conditions, parent=None, colours=None):
        super().__init__(parent)
        # THE PAGE SLOTS (`_PageSlot`): before anything is assigned that lives on a page
        self._slots = {section: [_PageSlot(section)] for section in PAGE_ATTRS}
        self._cur = {section: slots[0] for section, slots in self._slots.items()}
        self._subpage_set = sp.SubpageSet(SECTION_TITLE)
        #: Deleted pages the Undo toast can still bring back (`_delete_subpage`).
        self._trash: list = []
        #: Track-view windows opened from a spot, HELD: a parentless window nobody
        #: references is destroyed at once (the segmentation door's lesson).
        self._track_windows: list = []
        self.t = t
        #: The loaded project Run actually fits against -- see the module
        #: docstring's "RUN IS WIRED". Kept, not just read for display: the
        #: sidebar/toolbar constructors below only need the derived facts,
        #: but `_run_pressed` needs the object itself.
        self.project = project
        #: The project's name as the sidebar shows it -- the folder holding the condition
        #: folders ("protein A") -- and the start of every figure's automatic title
        #: (`_titled`, `figure_export.auto_title`).
        self._project_title = project_title
        #: The project with every track, whatever cap `project` carries: a cap is applied
        #: to THIS (`_apply_track_cap`), so changing or removing it never compounds.
        self._full_project = project
        #: A condition whose QC Detail page opens when the next QC result lands (the
        #: rebuild after a cap was set from that page).
        self._qc_reopen: str | None = None
        #: Every loaded condition's key, in order -- what the sections that fit every condition (Apparent
        #: Diffusion, HMM, Kinetics, ...) offer, read once here rather than re-derived from the SQD Compute
        #: dialog's per-condition rows, which those Runs never open.
        # THE FOLDER DESIGN'S FACTORS (`folderdesign`) and the nesting chosen for them (`factor_layout`, 24 Oct
        # 2026): with two or more factors every list of conditions -- the Compute dialog, the Conditions tool, every
        # Run -- is in nesting order, outer factor first.
        self._factors = dict(getattr(project, "factors", {}) or {})
        self._factor_names = tuple(getattr(project, "factor_names", ()) or ())
        self._factor_order = factor_layout.recall(self._project_key(), self._factor_names)
        # THE ORDER SET IN THE CONDITIONS TOOL (`condition_order`, 3 Nov 2026), remembered for the project, then the
        # nesting: it ranks each factor's values by first appearance, so the set order decides the groups' order too
        shown = dict(conditions)
        conditions = [(c, shown[c]) for c in condition_order.arranged(
            condition_order.recall(self._project_key(), [c for c, _label in conditions]), None,
            self._factors, self._factor_order)]
        self._all_condition_keys = [c for c, _label in conditions]
        #: key -> the name to SHOW for it. The Spatial Map grid labels its
        #: rows from this; every other view gets its names from the chart it
        #: was handed, which is why this is the first place it is kept.
        self._condition_labels = {c: lbl for c, lbl in conditions}
        from ...buildstamp import build_stamp

        self.setWindowTitle(f"SMTracker · {project_title}  ·  {build_stamp()}")
        # `scrollbar_css`, set here so every scroll area in this window --
        # the bubble table's included -- inherits shadcn's own bar instead of
        # Qt's native one (`widgets.primitives.scrollbar_css`'s own docstring
        # on the incident this already happened once from). WRAPPED in a
        # `QMainWindow { }` selector, not left as the old bare
        # `"background:...;"` -- a selector-less rule is fine alone (Qt takes
        # it as this widget's own inline style) but concatenating braced
        # `QScrollBar` blocks after it is invalid CSS Qt silently refuses to
        # parse, which is exactly what a bare-property version of this line
        # did the first time (logged as "Could not parse stylesheet").
        self.setStyleSheet(
            f"QMainWindow {{ background:{t.background}; }}" + scrollbar_css(t))

        central = QWidget()
        #: What a page's dialogs are built over (`_build_sqd_plumbing`) and with (the conditions' rows).
        self._central = central
        self._compute_conditions = conditions
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.sidebar = Sidebar(t, project_title=project_title,
                               project_subtitle=project_subtitle)
        #: The project's name as the sidebar shows it (the legacy window's title).
        self._project_title = project_title
        #: The legacy window, once opened (one per workspace; held, CLAUDE.md: hold the windows you open).
        self.legacy_window = None
        outer.addWidget(self.sidebar)

        # -- the content column: a trigger + breadcrumb, then the page --
        # A `ShadowHost`, not a bare `QWidget`: the card below is a real
        # `widgets.primitives.Card`, and a Card's `shadow-sm` is painted by its
        # nearest `ShadowHost` ancestor, not by the card itself -- the same
        # pattern every other card in this UI stack sits inside.
        column = ShadowHost(t)
        column.setStyleSheet(f"background:{t.background};")
        col_lay = QVBoxLayout(column)
        col_lay.setContentsMargins(20, 16, 20, 16)
        col_lay.setSpacing(12)

        head = _WidthBlind()
        head.setStyleSheet("background:transparent;")
        head_lay = QHBoxLayout(head)
        head_lay.setContentsMargins(0, 0, 0, 0)
        head_lay.setSpacing(8)
        self.trigger = SidebarTrigger(t)
        self.trigger.clicked.connect(
            lambda: self.sidebar.set_collapsed(not self.sidebar.collapsed))
        head_lay.addWidget(self.trigger)
        self.breadcrumb = Breadcrumb(t)
        self.breadcrumb.set_trail(self.sidebar.trail(self.sidebar.current))
        self.breadcrumb.crumb_clicked.connect(self._on_crumb_clicked)
        # `_on_dropdown_picked` dispatches to whichever section is CURRENT
        # -- the breadcrumb itself has no notion of sections, so the same
        # dropdown serves "another SQD condition" and "another APPD
        # condition" depending on where it was opened from.
        self.breadcrumb.dropdown_picked.connect(self._on_dropdown_picked)
        head_lay.addWidget(self.breadcrumb)
        head_lay.addStretch(1)
        col_lay.addWidget(head)

        self._build_sqd_widgets()

        self._build_appd_widgets()

        self._build_spatial_widgets()

        self._build_kinetics_widgets()

        self._build_clusters_widgets()
        self._build_msd_widgets()
        # The Composer: charts added from their Export dialogs, put together as one figure.
        self._composer = ComposerSection(t)
        self._composer.export_all_requested.connect(self.export_every_figure)
        self._export_all_running = None
        # Images & tracks: a condition's movies, their outlines and tracks, one cell zoomed.
        self._images = ImagesSection(t)
        self._images.set_project(project, list(conditions))
        self._images.export_requested.connect(self.open_images_export)
        self._images.table_export_requested.connect(self.open_sample_size_export)
        self._images.condition_changed.connect(self._images_condition_changed)
        # THE EXPERIMENTER'S OWN CHANNELS, remembered per project (17 Oct 2026)
        self._images.channels_changed.connect(self._remember_image_channels)
        self._restore_image_channels()
        # MARKED FOCI, remembered per project (17 Oct 2026, v2's distances), and Distances... in the side panel
        self._images.foci_changed.connect(self._remember_foci)
        self._images.distances_requested.connect(self.open_distances)
        self._restore_foci()
        self._images.track_requested.connect(
            lambda c, uid, viewer: self._open_track(c, uid, viewer,
                                                    self._images.show_track_message))
        self._build_hmm_widgets()
        # The QC section -- the SAME idle -> content shape as the three
        # above, a fourth parallel copy (see the module docstring's "EACH
        # SECTION RUNS ON ITS OWN"). TWO content pages, not one: General
        # (every loaded condition, whether the comparison is fair, and what
        # contributes nothing -- the Comparator was retired 13 Oct 2026) and Detail (one
        # condition's five panels), reached by clicking a General-page row
        # -- see `qc_chart.py`'s own module docstring for why this
        # General/Detail split replaces the original app's one dense tab.
        self._qc_placeholder = SectionPlaceholder(
            t, title="QC", icon_name="check",
            description="Check every loaded condition before comparing "
                        "them, then run to see track counts, density and "
                        "outliers here.")
        self._qc_placeholder.run_requested.connect(self._run_qc)
        self._qc_placeholder.cancel_requested.connect(self._cancel_qc_pressed)
        self._qc_overview_table = QCOverviewTable(t)
        self._qc_overview_table.row_clicked.connect(self._on_qc_condition_clicked)
        self._qc_fair = QCFairTable(t)
        # DO THE REPLICATES DIFFER? (24 Oct 2026): the same table, its own payload (`qc_view.replicates_table`)
        self._qc_replicates = QCFairTable(t, glyph="shuffle", title="Do the replicates differ?",
                                          subtitle="Replicate values against the movies dealt out at random")
        self._qc_nothing = QCNothingCard(t)
        self._qc_nothing.exclusions_requested.connect(self._open_exclusions)
        self._qc_cells = QCCellLengthCard(t)
        qc_general_body = QWidget()
        qc_general_body.setStyleSheet("background:transparent;")
        qc_general_lay = QVBoxLayout(qc_general_body)
        qc_general_lay.setContentsMargins(0, 0, 0, 0)
        qc_general_lay.setSpacing(20)
        qc_general_lay.addWidget(self._qc_overview_table)
        qc_general_lay.addWidget(self._qc_fair)
        qc_general_lay.addWidget(self._qc_replicates)
        qc_general_lay.addWidget(self._qc_cells)
        qc_general_lay.addWidget(self._qc_nothing)
        qc_general_lay.addStretch(1)
        # THE GENERAL PAGE SCROLLS (found by `scripts/gui_sweep.py`, 22 Oct 2026): its four cards stacked without a
        # scroll area asked for 794 px, so once QC had run the WINDOW could not be shorter than 866 px -- taller than
        # a 768 px laptop screen, whatever section was on show. As every other section's pages (`appd_section._scroll`).
        self._qc_general = QScrollArea()
        self._qc_general.setWidgetResizable(True)
        self._qc_general.setFrameShape(QScrollArea.NoFrame)
        self._qc_general.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._qc_general.setStyleSheet("QScrollArea { background:transparent; }")
        self._qc_general.setWidget(qc_general_body)
        self._qc_detail = QCConditionDetail(t)
        self._qc_detail.export_requested.connect(self.open_qc_export)
        self._qc_detail.cap_requested.connect(self._open_track_cap)
        self._qc_pages = QStackedWidget()
        self._qc_pages.addWidget(self._qc_placeholder)
        self._qc_pages.addWidget(self._qc_general)
        self._qc_pages.addWidget(self._qc_detail)
        self._qc_ever_succeeded = False
        #: condition -> `compute_qc_overview`'s own per-condition entry --
        #: reused as-is for the Detail drill-in, so opening it never
        #: triggers a second compute.
        self._qc_payloads: dict = {}
        self._qc_status = fit_label(
            "Press Compute on the right, then Run.",
            ty.ui("sm"), t.muted_foreground)
        self._qc_status.hide()
        self._qc_section = QWidget()
        self._qc_section.setStyleSheet("background:transparent;")
        qc_lay = QVBoxLayout(self._qc_section)
        qc_lay.setContentsMargins(0, 0, 0, 0)
        qc_lay.setSpacing(12)
        qc_lay.addWidget(self._qc_pages, 1)
        qc_lay.addWidget(self._qc_status)

        #: The outer switch between the sections' own content --
        #: `_on_selected` moves this, never anything inside any section.
        #: `_qc_section` is added FIRST: nothing calls `_on_selected` at
        #: startup, so `QStackedWidget` shows whichever page was added
        #: first, and `sidebar.current` (SECTIONS[0]) now defaults to QC
        #: -- the two have to agree, or the window opens on QC in the
        #: rail while showing SQD's content underneath it. Found by
        #: testing, not by reading.
        self._section_pages = QStackedWidget()
        self._section_pages.addWidget(self._qc_section)
        self._section_pages.addWidget(self._sqd_section)
        self._section_pages.addWidget(self._appd_section)
        self._section_pages.addWidget(self._spatial_section)
        self._section_pages.addWidget(self._kinetics_section)
        self._section_pages.addWidget(self._clusters_section)
        self._section_pages.addWidget(self._msd_section)
        self._section_pages.addWidget(self._hmm_section)
        self._section_pages.addWidget(self._composer)
        self._section_pages.addWidget(self._images)
        # A plain wrapper, not `col_lay.addWidget` directly: `self.trigger`'s
        # own glyph sits CENTRED inside its `TRIGGER`-sized hit box (a
        # smaller `GLYPH`-sized icon in a bigger clickable square, the
        # right call for a button), so the glyph's own visible ink starts
        # `(TRIGGER - GLYPH) / 2` past the box's left edge -- while every
        # page here sits flush with that same box edge, with no such
        # inset. Reported directly: a card/tab's left edge read as not
        # lining up with the icon above it. This wrapper gives every page
        # (SQD, Apparent-D, whichever it is) that SAME inset, so the two
        # align on the icon's own visible glyph rather than its box.
        #
        # Mirrored on the RIGHT too -- asked for directly, once the left
        # inset alone left the card's two gutters unequal: sidebar-right to
        # card-left was `col_lay`'s 20px plus this 6px inset (26px), while
        # card-right to toolbar-left stayed at `col_lay`'s bare 20px, since
        # only the left side had gained the extra margin. Adding the SAME
        # inset on the right restores one gutter width on both sides.
        _inset = (TRIGGER - GLYPH) // 2
        pages_wrap = _WidthBlind()
        pages_wrap.setStyleSheet("background:transparent;")
        pages_wrap_lay = QVBoxLayout(pages_wrap)
        pages_wrap_lay.setContentsMargins(_inset, 0, _inset, 0)
        pages_wrap_lay.setSpacing(0)
        pages_wrap_lay.addWidget(self._section_pages)
        col_lay.addWidget(pages_wrap, 1)
        outer.addWidget(column, 1)

        self._build_sqd_plumbing()

        self._build_spatial_plumbing()
        self._build_msd_plumbing()
        self._build_appd_plumbing()
        self._build_hmm_plumbing()
        self._build_kinetics_plumbing()
        self._build_clusters_plumbing()

        # -- the async compute plumbing -- see the module docstring's
        # "OFF THE GUI THREAD" and "EACH SECTION RUNS ON ITS OWN". Each
        # section keeps its OWN generation counter (`_sqd_gen`/`_appd_gen`),
        # bumped on every Run press FOR THAT SECTION; a result whose
        # generation no longer matches is from a superseded press and is
        # dropped rather than drawn. Two separate counters, not one shared
        # one -- a shared counter is exactly what let pressing either
        # section's Run silently supersede (and drop) the OTHER section's
        # still-in-flight result, which is not "superseded", it is a
        # different section entirely.
        self._pool = QThreadPool.globalInstance()
        # THE JOB THREADS ONLY WAIT (3 Nov 2026): each section's job sits on a thread of this pool until its worker
        # answers, so Qt's eight (one per logical core) were all waiting during a Run of several sections, and the
        # figure composer, whose preview is drawn on the same pool, waited for one to free up ("assembling the figures
        # takes quite long"). A waiting thread costs next to nothing, so there are enough of them.
        if self._pool.maxThreadCount() < JOB_THREADS:
            self._pool.setMaxThreadCount(JOB_THREADS)
        #: WHERE ANY REAL FIT RUNS, not merely off the GUI thread but off
        #: the GUI PROCESS -- `_run_pressed` (SQD) and
        #: `_run_fractions_checks` (the SuperPlots tab) both
        #: submit their compute here rather than running it inside their
        #: own `QThreadPool` worker thread directly; that worker thread's
        #: whole job becomes "submit, then block on the `Future`". See
        #: `_run_pressed`'s own docstring for why: a real multistart fit
        #: is a long, tight, Python-heavy loop (thousands of small calls,
        #: each re-acquiring the GIL right after a brief release), and
        #: CPython's GIL handoff between THREADS in the SAME process is
        #: not fair enough to stop that from starving the main thread's
        #: own event loop for the fit's whole duration -- measured
        #: directly: a real ~5s fit pegged one core at
        #: `effective_cores=1.03` (CPU time ~= wall time) while the app's
        #: own responsiveness stalled for that same ~5s. A SEPARATE
        #: PROCESS has its OWN, independent GIL, so nothing it does can
        #: ever contend with the main thread's for this lock -- at that
        #: point it is purely an OS-scheduling question, and the OS's own
        #: preemptive scheduler is far fairer than CPython's.
        #:
        #: PERSISTENT, not spawned-and-torn-down per Run: `spawn` (the
        #: only safe start method inside a multi-threaded Qt app) re-
        #: imports the whole package, measured elsewhere in this project
        #: at ~1-2s -- paying that on every press would trade one
        #: annoyance for a worse one. Paid once here instead, amortised
        #: over the window's whole lifetime. ALL CORES BUT ONE
        #: (`parallel.n_workers`, 2 Nov 2026; it was 3, chosen when SQD's
        #: one Run was all that came here): every section now sends one job
        #: per condition (SQD, MSD, Jump distances and its fit-quality
        #: folds, the Spatial Map, HMM, apparent D), so four sections on
        #: five conditions queued 20-odd jobs behind three workers while
        #: five of an 8-thread laptop's threads idled -- "now it takes 20
        #: min to compute msd, sqd, sqd map, jump distance". The one core
        #: left is the window's. A job beyond the workers still just
        #: queues -- safe, only slower, never broken.
        #: `multiprocessing.get_context("spawn")` explicitly, rather than
        #: relying on the platform default -- Linux's own default,
        #: `fork`, is documented unsafe inside an already-threaded
        #: process (which a Qt app always is), and this project's own
        #: frozen build already calls `multiprocessing.freeze_support()`
        #: at its entry point, so `spawn` is proven to work there too.
        #: BELOW NORMAL PRIORITY (`parallel.lower_priority`, 3 Nov 2026): the workers still take every core
        #: nobody else wants, and the window comes first.
        workers = n_workers(None)
        self._process_pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=lower_priority)
        #: TWO LANES ON ONE POOL (3 Nov 2026): the long jobs, every section's one job per condition, hold at most
        #: `parallel.bulk_slots` workers (`_submit_bulk`), so one worker is free for a job somebody is waiting on --
        #: the SQD figure, a split preview, a redraw without a hidden track, a significance test, an exclusion's
        #: preview -- submitted straight to the pool, which then runs it next. The SQD figure had waited minutes
        #: behind the Spatial Map's and Jump distances' jobs, every one of them queued ahead of it.
        self._bulk_slots = threading.BoundedSemaphore(bulk_slots(workers))
        #: WHERE RESULTS ARE KEPT BETWEEN RUNS: the SQD fits, and since 3 Nov 2026 every
        #: section's per-condition job (`sectioncache.cached_job`: Spatial Map, MSD, Jump
        #: distances, Signed jumps, Apparent diffusion, HMM), which computed from scratch on
        #: every Run before. The workspace passed no cache at
        #: all, so every Run refitted from scratch and nothing survived a restart.
        #: PER-USER, not beside the data: the old browser's cache writes a hidden
        #: folder into each condition folder, and this tool must not write into the
        #: data drives. Entries are keyed on content, data fingerprint included, so
        #: one store serves every project without cross-talk.
        self._results_cache = WarmCache(default_warm_root() / "workspace")
        #: QC's own signal object -- see the comment above for why each
        #: section gets one rather than sharing.
        self._qc_signals = _ComputeSignals()
        self._qc_signals.done.connect(self._on_qc_only_done)
        self._qc_gen = 0
        #: THE PER-SPOT CALIBRATION (6 Nov 2026, the header menu's "Make per-spot calibration...",
        #: `calibration_view`): its measurement off the GUI thread, its dialog, the folder last chosen to save into
        self._calibration_signals = _ComputeSignals()
        self._calibration_signals.done.connect(self._on_calibration_done)
        self._calibration_gen = 0
        self.calibration_dialog = None
        self._calibration_target: str | None = None

        # THE COLOURS ARE CONFIRM PROJECT'S (`colours`, handed on by `app.main`),
        # else the ones remembered for this project, else the default -- one model,
        # `condition_colours.ConditionColours`, for both windows.
        self.toolbar = Toolbar(
            t, project=project_title, conditions=conditions, colours=colours,
            colour_key=condition_colours.project_key(
                list(getattr(project, "roots", {}).values())))
        self.toolbar.compute_pressed.connect(self._on_compute_mark_pressed)
        self.toolbar.rail.theme_pressed.connect(self._toggle_theme)
        self.toolbar.rail.options_pressed.connect(self.open_options)
        self.toolbar.rail.legacy_pressed.connect(self.open_legacy_window)
        self.toolbar.rail.help_pressed.connect(lambda: show_help_menu("analysis", self, self.toolbar.rail.help_button, "left"))
        # A palette or colour change re-colours EVERY section's already-drawn
        # figures -- see `bubble_chart.py`'s module docstring on why colour is
        # read from here rather than baked in by the fit.
        self.toolbar.conditions.colours_changed.connect(self._on_colours_changed)
        self.toolbar.conditions.set_nesting(factor_layout.orders(self._factor_names), self._factor_order)
        self.toolbar.conditions.set_factor_colouring(self._factors, self._factor_names, self._factor_order)
        self._group_keys()
        self.toolbar.conditions.nesting_changed.connect(self._on_nesting)
        self.toolbar.conditions.order_requested.connect(self.set_condition_order)
        self.toolbar.visibility.toggled.connect(self._on_visibility_toggled)
        self.toolbar.visibility.colormap_changed.connect(
            self._on_visibility_colormap)
        self.toolbar.visibility.rendering_changed.connect(
            self._on_visibility_rendering)
        self.toolbar.visibility.scale_changed.connect(self._on_visibility_scale)
        self.toolbar.visibility.custom_ramp_requested.connect(self._open_ramp_dialog)
        self._load_custom_ramps()
        # SIGNIFICANCE ON THE CHART (14 Oct 2026): one choice per bubble section, set in the
        # Visibility popover, painted on the chart and exported as shown.
        from ... import significance as sig

        #: each section's brackets choice; the SQD bubbles' is a page's own (`_sqd_choice`, `_choice_of`)
        self._sig_choices = {section: sig.ViewChoice() for section in sig.SECTION_KINDS if section not in PAGED_CHOICE}
        #: each section's brackets standing on a population "Check the fit" flags (`significance.bracket_warnings`)
        self._bracket_warnings: dict = {section: [] for section in sig.SECTION_KINDS if section not in PAGED_CHOICE}
        self._sig_gen = 0
        self._sig_jobs: dict = {}
        self._sig_signals = _ComputeSignals()
        self._sig_signals.done.connect(self._on_mixed_done)
        self.toolbar.significance.significance_changed.connect(self._on_significance)
        self.toolbar.section_set.connect(lambda _section: self._sync_significance_block())
        outer.addWidget(self.toolbar)
        # THE METADATA PANEL lies in the content area, beside the page where the window has its width to spare and
        # over the page's edge where it has not (`metadata.MetadataDock`, 30 Oct 2026). Attached before the toasts
        # are made, so they stay above it.
        self.toolbar.metadata_dock.attach(central)
        #: When the last Spatial Map Run finished, for the receipt's stamp: a page's own (`_spatial_computed_at`).

        self.sidebar.selected.connect(self._on_selected)
        self.sidebar.subpage_add_requested.connect(self._add_subpage)
        self.sidebar.subpage_renamed.connect(self._rename_subpage)
        self.sidebar.subpage_deleted.connect(self._delete_subpage)
        self.sidebar.subpage_moved.connect(self._move_subpage)
        self.sidebar.subpage_dropped.connect(self._drop_subpage)
        self.sidebar.header_centre.connect(self._follow_head)
        self.sidebar.opened.connect(self._open_project_menu)
        self.setCentralWidget(central)
        self.resize(1280, 800)
        # TOASTS (`widgets.toast`, the UI designer's prototype ported): a Run that finishes or
        # fails says so, wherever the reader has gone since pressing it; a second Run
        # in a section still computing ASKS before replacing it. Sections run side by
        # side, so the question is per section, never across the window.
        # clear of the tool rail: a toast lay over its Help, Options and theme buttons (4 Nov 2026, rail review)
        self.toasts = Toaster(t, central, right_inset=lambda: self.toolbar.rail.width())
        #: job key -> when it started (`time.monotonic`). A job is a section's Run.
        self._running: dict[str, float] = {}
        #: THE JOBS RUNNING IN THE BACKGROUND, listed at the sidebar's foot (`jobs_panel`, asked 2 Nov 2026: "on the
        #: left bar like an info that something (what and roughly percent) is running"): said again every second
        #: while any runs, so its clock moves and a condition that lands shows within the second.
        self._jobs_clock = QTimer(self)
        self._jobs_clock.setInterval(1000)
        self._jobs_clock.timeout.connect(self._sync_jobs)
        #: job key -> (the prompt toast, the start it is holding back).
        self._prompts: dict[str, tuple] = {}
        # every condition key built above grouped by the factors (some are built after the nesting is set)
        self._group_keys()
        # the mark's hover and state for the page opened first (the sidebar chose it before `_on_selected` listened)
        self._sync_compute_mark()

    # -- SUBPAGES: the page slots and the "+" (`smtracker.subpages`, the block comment above `_PageSlot`) -------- #
    @contextmanager
    def _in(self, slot):
        """Run a block with ``slot`` the current page of its section, restoring the one that was current."""
        section = slot.section
        before = self._cur[section]
        self._cur[section] = slot
        try:
            yield slot
        finally:
            if self._known_slot(before):
                self._cur[section] = before
            elif self._cur[section] is slot and slot in self._slots[section]:
                pass
            else:
                self._cur[section] = self._slots[section][0]

    def _call_in(self, slot, fn, *args):
        """``fn(*args)`` with ``slot`` current. A page that has been deleted since (a worker's late result, a signal
        of a widget being torn down) is ignored: nothing is left to draw it on."""
        if not self._known_slot(slot):
            return None
        with self._in(slot):
            return fn(*args)

    def _known_slot(self, slot) -> bool:
        """Whether ``slot`` is a page of its section now: its own or an added one."""
        return slot in self._slots.get(slot.section, ())

    def _bind(self, section: str, fn):
        """``fn`` as a signal handler of the page now current for ``section``: it runs with THAT page current,
        whichever page is on show when the signal comes. Takes only as many of the signal's arguments as ``fn`` does
        (Qt drops the extra ones for a plain callable; this wrapper must do the same)."""
        slot = self._cur[section]
        code = getattr(fn, "__code__", None)
        count = None
        if code is not None and not code.co_flags & 0x04:
            count = code.co_argcount - (1 if hasattr(fn, "__self__") else 0)

        def bound(*args):
            return self._call_in(slot, fn, *(args if count is None else args[:count]))

        return bound

    def _slot(self, section: str, number=None):
        """The page slot of ``section`` numbered ``number`` (None: the section's own), or None."""
        return next((s for s in self._slots.get(section, ()) if s.number == number), None)

    def _slot_of_key(self, key: str):
        section, number, _page = sp.split_view(key)
        return self._slot(section, number)

    def _ordered_slots(self, section: str) -> list:
        """The section's pages in the sidebar's order: its own first, then the added ones as listed."""
        out = [self._slot(section, None)]
        out += [s for s in (self._slot(section, n) for n in self._subpage_set.numbers(section)) if s is not None]
        return [s for s in out if s is not None]

    def _each(self, section: str):
        """Iterate ``section``'s pages, each current for the body of the loop (``for _ in self._each("sqd"):``)."""
        for slot in self._ordered_slots(section):
            with self._in(slot):
                yield slot

    def _jk(self, base: str) -> str:
        """The job key of ``base`` ("sqd", "jumps", "jd", "spatial") for the page now current: ``"jumps"`` on the
        section's own page, ``"jumps~2"`` on an added one."""
        return sp.job_key(base, self._cur[self._JOB_SECTION[base]].number)

    def _page_name(self, slot) -> str:
        return self._subpage_set.name(slot.section, slot.number)

    def _page_widget(self, section: str):
        return {"sqd": lambda: self._sqd_section, "spatial": lambda: self._spatial_section,
                "msd": lambda: self._msd_section, "appd": lambda: self._appd_section,
                "hmm": lambda: self._hmm_section, "kinetics": lambda: self._kinetics_section,
                "clusters": lambda: self._clusters_section}[section]()

    def _make_slot(self, section: str, number: int):
        """Build a new page of ``section`` (its widgets and plumbing, the same code the section's own page is built
        by) and put its widget in the page stack. Returns the slot."""
        slot = _PageSlot(section, number)
        self._slots[section].append(slot)
        with self._in(slot):
            getattr(self, f"_build_{section}_widgets")()
            getattr(self, f"_build_{section}_plumbing")()
            widget = self._page_widget(section)
        self._section_pages.addWidget(widget)
        return slot

    def _compute_settings(self, dialog) -> dict:
        """A Compute dialog's settings as plain data (what `_copy_settings` and the theme switch carry over)."""
        panel = dialog.compute
        return {"values": panel.values(), "conditions": panel.condition_values(),
                "own_model": dict(getattr(panel, "_own_model", {})),
                # each row's own Force same D across, those kept unshown under Whole data set included
                # (`ComputePanel.apply_values`)
                "own_shared_d": dict(getattr(panel, "_own_shared_d", {}))}

    def _apply_compute_settings(self, dialog, settings: dict) -> None:
        dialog.compute.apply_values(settings["values"], settings["conditions"], settings["own_model"],
                                    own_shared_d=settings.get("own_shared_d"))

    def _copy_settings(self, section: str, source, target) -> None:
        """A new page starts as a COPY OF THE SETTINGS of ``source`` and none of its results. SQD: the Compute dialog,
        the Jump distances Run's last choices, the brackets' choice; Spatial Map: the map dialog, every control."""
        if section == "sqd":
            self._apply_compute_settings(target.compute_dialog, self._compute_settings(source.compute_dialog))
            target._jumps.last_config = dict(source._jumps.last_config or {})
            target._sqd_choice = source._sqd_choice
        elif section == "spatial":
            target.map_dialog.set_state(source.map_dialog.state())
        elif section == "msd":
            target._msd.last_config = dict(source._msd.last_config or {})
        elif section == "appd":
            target._appd.last_config = dict(source._appd.last_config or {})
            target._appd_choice = source._appd_choice
        elif section == "hmm":
            target._hmm.last_config = dict(source._hmm.last_config or {})
            # the Jump angles tab's reading (Against chance), as the page stands
            if source._angles_page.chance_on() != target._angles_page.chance_on():
                target._angles_page.chance_box.set_checked(source._angles_page.chance_on(), animate=False)
        elif section == "kinetics":
            target._kinetics.last_config = dict(source._kinetics.last_config or {})
        elif section == "clusters":
            # the page's switches as they stand (its Run reads them), not only as the last Run had them
            target._clusters.set_config(source._clusters.config())

    def _sync_copy_note(self) -> None:
        """The note on each section's page now current: that it is a copy of another page's settings, shown until the
        page has a result of its own (an added page only)."""
        for section in self._slots:
            slot = self._cur[section]
            note = getattr(slot, f"_{section}_copy_note", None)
            if note is None:
                continue
            if slot.number is None:
                text, has = "", True
            else:
                origin = getattr(slot, "copied_from", "")
                text = (f"{self._page_name(slot)} is a copy of {origin}'s settings, with none of its results. It keeps "
                        f"its own settings and results: press Run to compute it." if origin else "")
                if section == "sqd":
                    has = bool(slot._ever_succeeded or slot._jumps.has_result() or slot._jd.has_result())
                else:
                    has = bool(getattr(slot, f"_{section}").has_result())
            note.setText(text)
            note.setVisible(bool(text) and not has)

    def _subpage_entries(self, section: str) -> list:
        return [(p.number, p.name) for p in self._subpage_set.pages(section)]

    def _sync_subpages(self, section: str) -> None:
        """Redraw the sidebar's list of ``section``'s added pages from the model (and the pages' own names)."""
        self.sidebar.set_subpages(section, self._subpage_entries(section))
        for slot in self._slots[section]:
            with self._in(slot):
                self._sync_copy_note()
        self.subpages_changed.emit(section)
        self._sync_map_pages(section)

    def _first_view(self, section: str, number) -> str:
        return sp.join_view(section, number, SUBS[section][0][0].partition(".")[2])

    def _add_subpage(self, section: str):
        """The sidebar's "+": a new page of ``section`` starting as a copy of the settings of the page in hand, shown
        at once (its empty state, which says so)."""
        source = self._cur[section]
        page = self._subpage_set.add(section)
        slot = self._make_slot(section, page.number)
        with self._in(slot):
            self._copy_settings(section, source, slot)
            slot.copied_from = self._page_name(source)
        self._sync_subpages(section)
        self.sidebar.select(self._first_view(section, page.number))
        return slot

    def _rename_subpage(self, section: str, number: int, text: str) -> None:
        try:
            self._subpage_set.rename(section, number, text)
        except (KeyError, ValueError):
            return
        self._sync_subpages(section)
        self._sync_jobs()

    def _delete_subpage(self, section: str, number: int):
        """Delete an added page: its running jobs stop, it leaves the sidebar, and an Undo toast offers it back with
        its settings and results (the slot is kept, out of sight, until the toast is gone)."""
        slot = self._slot(section, number)
        if slot is None:
            return None
        for key in [k for k in self._running if sp.split_job(k)[1] == number
                    and self._JOB_SECTION.get(sp.split_job(k)[0]) == section]:
            self._cancel_job(key)
            self._running.pop(key, None)
        self._sync_jobs()
        name = self._subpage_set.name(section, number)
        page, index = self._subpage_set.delete(section, number)
        self._slots[section].remove(slot)
        if self._cur[section] is slot:
            self._cur[section] = self._slots[section][0]
        with self._in(slot):
            widget = self._page_widget(section)
        self._section_pages.removeWidget(widget)
        widget.hide()
        self._sync_subpages(section)
        entry = {"section": section, "page": page, "index": index, "slot": slot, "widget": widget, "restored": False}
        for old in list(self._trash):
            self._empty_trash(old)
        self._trash.append(entry)
        toast = self.toasts.post(f"{name} deleted", description="Its settings and results go with it.",
                                 action="Undo", duration=8000)
        toast.on_action(lambda e=entry: self._undo_delete(e))
        toast.gone.connect(lambda e=entry: self._empty_trash(e))
        self.last_delete_toast = toast
        self.subpage_removed.emit(section, name)
        return entry

    def _undo_delete(self, entry: dict) -> None:
        """The Undo of `_delete_subpage`: the page back where it was, as it was."""
        if entry["restored"] or entry not in self._trash:
            return
        entry["restored"] = True
        self._trash.remove(entry)
        section, slot = entry["section"], entry["slot"]
        self._subpage_set.restore(section, entry["page"], entry["index"])
        self._slots[section].append(slot)
        self._section_pages.addWidget(entry["widget"])
        self._sync_subpages(section)
        self.sidebar.select(self._first_view(section, slot.number))

    def _empty_trash(self, entry: dict) -> None:
        """Free a deleted page that was not brought back: its widgets and its dialog."""
        if entry["restored"]:
            return
        if entry in self._trash:
            self._trash.remove(entry)
        elif not entry.get("_force"):
            return
        slot = entry["slot"]
        from ..widgets.primitives import discard

        for name in ("_sqd_clock",):
            timer = getattr(slot, name, None)
            if timer is not None:
                timer.stop()
        for name in ("compute_dialog", "map_dialog"):
            dialog = getattr(slot, name, None)
            if dialog is not None:
                dialog.deleteLater()
        discard(entry["widget"])
        slot.relays = []

    def _move_subpage(self, section: str, number: int, step: int) -> None:
        if self._subpage_set.move(section, number, step):
            self._sync_subpages(section)

    def _drop_subpage(self, section: str, number: int, index: int) -> None:
        if self._subpage_set.move_to(section, number, index):
            self._sync_subpages(section)

    def _subpage_names(self, section: str) -> list:
        """``[(page key, name)]`` of the section's own page and every added one, the order the sidebar lists them
        (what a drop-down choosing a page offers)."""
        return self._subpage_set.names(section)


    def _build_sqd_widgets(self) -> None:
        """The SQD section's widgets for the page in hand (`self._cur["sqd"]`): its placeholder, the bubble card and the
        replicate page, the strip over Populations, Jump distances and Signed jumps. Called once for the section's own
        page and once for every added copy (subpages), so what a page is has one definition. Every signal of a page's
        own widget is connected through `_bind`: the handler runs with THIS page current."""
        t = self.t
        # Populations content: the SQD bubble chart in its own card -- see
        # `bubble_chart.BubbleCard`. Replaced fresh on every Run
        # (`_run_pressed`), never accumulated onto. A `QStackedWidget` sits
        # above it rather than adding the card directly, so a bubble click
        # can drill into `ReplicateDetail` (page 1) as an IN-APP page --
        # `SQD > Populations > <condition>` -- instead of the legacy
        # floating `BubbleInspector` window; see `bubble_chart.py`'s own
        # "population detail page" section. `self._placeholder` is the
        # THIRD page, and the one shown FIRST -- see the module docstring's
        # "THE FIRST RUN AND EVERY RUN AFTER IT ARE TWO DIFFERENT MOMENTS".
        self._placeholder = SectionPlaceholder(
            t, title="Squared Displacement Analysis",
            description="Choose which conditions to compare, then run the "
                        "fit to see their populations here.")
        self._placeholder.run_requested.connect(self._bind("sqd", self._open_compute_dialog))
        self._placeholder.cancel_requested.connect(self._bind("sqd", self._cancel_sqd_pressed))
        self._card = BubbleCard(t, crossed=True)
        #: The condition drill-in: `ReplicateDetail`, the per-replicate fit. It once sat
        #: as the first of two tabs beside a per-condition SQD map; the map lives in the
        #: Spatial Map section, so the tab and its tab bar were removed.
        self._detail = ReplicateDetail(t)
        self._pages = QStackedWidget()
        self._pages.addWidget(self._placeholder)
        self._pages.addWidget(self._card)
        self._pages.addWidget(self._detail)
        self._card.chart.population_clicked.connect(self._bind("sqd", self._on_population_clicked))
        self._card.export_requested.connect(self._bind("sqd", self.open_bubble_export))
        self._detail.export_requested.connect(self._bind("sqd", self.open_replicate_export))
        self._card.table_export_requested.connect(self._bind("sqd", self.open_sqd_table_export))
        self._detail.table_export_requested.connect(self._bind("sqd", self.open_replicate_table_export))
        #: Flips to `True` on the FIRST successful `_on_compute_done` and
        #: never back -- `self._placeholder` is a ONE-WAY door, see the
        #: module docstring.
        self._ever_succeeded = False
        #: condition -> `comparison_controller.replicate_fit_payload`,
        #: precomputed for EVERY condition during `_run_pressed` (off the
        #: GUI thread, inside `compute_bubble_view`) so a bubble click reads
        #: this dict rather than fitting anything synchronously -- see
        #: `compute_bubble_view`'s own docstring on why that would risk a
        #: visible freeze under `d_sharing_scope == "replicate"`.
        self._replicate_payloads: dict = {}
        self.status = fit_label(
            "Press Compute on the right, then Run.",
            ty.ui("sm"), t.muted_foreground)
        # Hidden while `self._placeholder` is up: that surface already says
        # everything this line would repeat underneath it. Shown again the
        # moment the first Run succeeds -- see `_on_compute_done`.
        self.status.hide()
        self._sqd_section = QWidget()
        self._sqd_section.setStyleSheet("background:transparent;")
        sqd_lay = QVBoxLayout(self._sqd_section)
        sqd_lay.setContentsMargins(0, 0, 0, 0)
        sqd_lay.setSpacing(12)
        # TWO PAGES, one strip (asked 13 Oct 2026: "the sqd tab needs the jump distance
        # histograms"): the bubbles, and the jump histogram against the 1-3 population fits
        # with their goodness of fit. The strip moves the sidebar's sub-view, so the
        # breadcrumb and the Compute mark follow it.
        from ..widgets.controls import Segmented

        self._sqd_tabs = Segmented(t, ["Populations", "Jump distances"], current=0, size="sm")
        self._sqd_tabs.changed.connect(self._bind("sqd", self._on_sqd_tab))
        tabs_row = QHBoxLayout()
        tabs_row.setContentsMargins(0, 0, 0, 0)
        tabs_row.addWidget(self._sqd_tabs)
        tabs_row.addStretch(1)
        # MAP THIS FIT (23 Nov 2026): the Spatial Map's settings for this page's fit (`_map_this_fit`)
        self._sqd_map_button = Button(t, MAP_THIS_FIT, size="small", variant="outline", glyph="grid")
        self._sqd_map_button.setToolTip(map_this_fit_tip("sqd"))
        self._sqd_map_button.clicked.connect(self._bind("sqd", lambda: self._map_this_fit("sqd")))
        tabs_row.addWidget(self._sqd_map_button)
        sqd_lay.addLayout(tabs_row)
        populations = QWidget()
        populations.setStyleSheet("background:transparent;")
        pop_lay = QVBoxLayout(populations)
        pop_lay.setContentsMargins(0, 0, 0, 0)
        pop_lay.setSpacing(12)
        pop_lay.addWidget(self._pages, 1)
        #: CHECK THE FIT (asked 17 Oct 2026): a population under 5 %, or two within twofold in D
        #: (`bubble_view.fit_warnings`) -- in the flag colour, above the caption, hidden when there is none.
        self.fit_warning = fit_label("", ty.ui("sm", "medium"), _FIT_WARNING_INK)
        pop_lay.addWidget(self.fit_warning)
        self.fit_warning.hide()
        pop_lay.addWidget(self.status)
        self._jumps = JumpsSection(t)
        self._jumps.run_requested.connect(self._bind("sqd", self._open_jumps_dialog))
        self._jumps.cancel_requested.connect(self._bind("sqd", self._cancel_jumps_pressed))
        self._jumps.export_requested.connect(self._bind("sqd", self.open_jumps_export))
        self._jumps.export_all_requested.connect(self._bind("sqd", self.open_jumps_overview_export))
        self._jumps_status = fit_label(
            "Press Run to configure and compute.", ty.ui("sm"), t.muted_foreground)
        self._jumps_status.hide()
        jumps_page = QWidget()
        jumps_page.setStyleSheet("background:transparent;")
        jumps_lay = QVBoxLayout(jumps_page)
        jumps_lay.setContentsMargins(0, 0, 0, 0)
        jumps_lay.setSpacing(12)
        jumps_lay.addWidget(self._jumps, 1)
        jumps_lay.addWidget(self._jumps_status)
        # SIGNED JUMPS (17 Oct 2026): the jumps along and across the cell and on the camera's axes against the
        # 1-3 population normal mixtures (`jd_view`), a cross-check of the fractions on another observable.
        self._jd = JDSection(t)
        self._jd.run_requested.connect(self._bind("sqd", lambda: self._start_or_ask(self._jk("jd"), self._run_jd)))
        self._jd.cancel_requested.connect(self._bind("sqd", self._cancel_jd_pressed))
        self._jd.export_requested.connect(self._bind("sqd", self.open_jd_export))
        self._jd.table_export_requested.connect(self._bind("sqd", self.open_jd_table_export))
        self._jd_status = fit_label("", ty.ui("sm"), t.muted_foreground)
        self._jd_status.hide()
        jd_page = QWidget()
        jd_page.setStyleSheet("background:transparent;")
        jd_lay = QVBoxLayout(jd_page)
        jd_lay.setContentsMargins(0, 0, 0, 0)
        jd_lay.setSpacing(12)
        jd_lay.addWidget(self._jd, 1)
        jd_lay.addWidget(self._jd_status)
        self._sqd_stack = QStackedWidget()
        self._sqd_stack.addWidget(populations)
        self._sqd_stack.addWidget(jumps_page)
        # SIGNED JUMPS IS NO LONGER A PAGE (22 Nov 2026: "i dont think we need signed jumps btw"): its widgets are
        # built, so nothing that reads a page's attributes breaks, and kept out of sight with no door to them
        jd_page.setParent(self._sqd_section)
        jd_page.hide()
        sqd_lay.addWidget(self._sqd_stack, 1)
        #: Said on an added page until its first result: it starts as a copy of settings, not of results.
        self._sqd_copy_note = fit_label("", ty.ui("sm"), t.muted_foreground)
        self._sqd_copy_note.setWordWrap(True)
        self._sqd_copy_note.hide()
        sqd_lay.insertWidget(0, self._sqd_copy_note)

    def _build_spatial_widgets(self) -> None:
        """The Spatial Map section's widgets for the page in hand (`self._cur["spatial"]`), as `_build_sqd_widgets`."""
        t = self.t
        # The Spatial Map section -- the SAME idle -> content shape as the
        # two above, a third parallel copy rather than a shared abstraction,
        # for the reason the module docstring gives ("EACH SECTION RUNS ON
        # ITS OWN"). It has no card/detail split: the grid IS the overview,
        # which is the whole point of laying every condition out at once.
        #
        # SQD ONLY for now, by instruction. `comparison.spatial_maps` already
        # draws this grid from appD and HMM behind one `source=` argument, so
        # adding them later is routing rather than analysis -- and when they
        # arrive the method belongs INSIDE `SpatialMapDialog` beside the other
        # compute settings, not as a picker on the page: it is an input to the
        # computation, and half the settings beside it (jump position,
        # assignment) do not even apply to a per-TRACK method like appD.
        self._spatial = SpatialMapSection(t)
        self._spatial.run_requested.connect(self._bind("spatial", self._open_spatial_dialog))
        self._spatial.cancel_requested.connect(self._bind("spatial", self._cancel_spatial_pressed))
        self._spatial.condition_clicked.connect(self._bind("spatial", self._on_spatial_condition_clicked))
        self._spatial.track_requested.connect(self._bind("spatial", self._on_spatial_track_requested))
        self._spatial.track_hidden.connect(self._bind("spatial", self._on_map_track_hidden))
        self._spatial.export_requested.connect(self._bind("spatial", self.open_spatial_export))
        #: Track-view windows opened from a spot, HELD: a parentless window nobody
        #: references is destroyed at once (the segmentation door's lesson).
        self._spatial_status = fit_label(
            "Press Run to configure and compute the maps.",
            ty.ui("sm"), t.muted_foreground)
        self._spatial_status.hide()
        self._spatial_section = QWidget()
        self._spatial_section.setStyleSheet("background:transparent;")
        spatial_lay = QVBoxLayout(self._spatial_section)
        spatial_lay.setContentsMargins(0, 0, 0, 0)
        spatial_lay.setSpacing(12)
        spatial_lay.addWidget(self._spatial, 1)
        spatial_lay.addWidget(self._spatial_status)
        self._spatial.days.set_project(self.project)
        self._spatial.matrix.set_grouping(self._factors, self._factor_order)
        # THE METADATA RECEIPT follows whichever Spatial Map grid is on screen: both grids rebuild their settings
        # rows on every redraw and say so.
        self._spatial.matrix.caption_changed.connect(self._bind("spatial", self._sync_metadata))
        self._spatial.days.matrix.caption_changed.connect(self._bind("spatial", self._sync_metadata))
        self._spatial.notice_run_requested.connect(self._bind("spatial", self._rerun_spatial_maps))
        self._spatial_copy_note = fit_label("", ty.ui("sm"), t.muted_foreground)
        self._spatial_copy_note.setWordWrap(True)
        self._spatial_copy_note.hide()
        spatial_lay.insertWidget(0, self._spatial_copy_note)

    def _build_sqd_plumbing(self) -> None:
        """The SQD page in hand's Compute dialog, its worker signals and its Run's counters (a page's own, so two pages
        never drop each other's results). Results come back through a `_PageRelay`, which runs the handler with this
        page current."""
        slot = self._cur["sqd"]
        slot.relays = []
        self.compute_dialog = ComputeDialog(self.t, self._compute_conditions, content=self._central,
                                            parent=self, factor_names=self._factor_names)
        self.compute_dialog.run.connect(self._bind("sqd", self._run_pressed))
        # The Compute dialog starts from the user's standards (the options screen,
        # `smtracker.options`); a theme switch then carries the session's own values.
        self._apply_compute_standard(opts.compute_values(theme.remembered(opts.OPTIONS_KEY)))
        relay = _PageRelay(self, slot, "_on_compute_done", "_on_sqd_progress")
        slot.relays.append(relay)
        self._signals = _ComputeSignals()
        self._signals.done.connect(relay.done)
        self._signals.progress.connect(relay.progress)
        #: The running Run's progress, re-rendered every second so the clock moves
        #: even while one long condition is still fitting.
        self._sqd_progress = {"done": 0, "total": 0, "finished": [],
                              "started": 0.0, "slow": False}
        self._sqd_clock = QTimer(self)
        self._sqd_clock.setInterval(1000)
        self._sqd_clock.timeout.connect(self._bind("sqd", self._render_sqd_progress))
        #: The spec and Uncertainty of the last SQD Run that SUCCEEDED -- handed to
        #: the rebuilt window by the theme switch, which re-shows it from the cache.
        self._last_sqd_run = None
        #: Jump distances: one job per condition per press, each tagged with the press it belongs to.
        relay = _PageRelay(self, slot, "_on_jumps_done")
        slot.relays.append(relay)
        self._jumps_signals = _ComputeSignals()
        self._jumps_signals.done.connect(relay.done)
        self._jumps_gen = 0
        self._jumps_run_id = 0
        self._jumps_by_gen: dict[int, tuple[int, str, bool]] = {}
        #: Conditions of the current run whose FULL view has arrived: a quick pass that
        #: lands later must not replace it.
        self._jumps_full: set = set()
        self._jumps_pending = 0
        self._jumps_total = 0
        self._jumps_failed: list = []
        #: Signed jumps: one job per condition per press.
        relay = _PageRelay(self, slot, "_on_jd_done")
        slot.relays.append(relay)
        self._jd_signals = _ComputeSignals()
        self._jd_signals.done.connect(relay.done)
        self._jd_gen = 0
        self._jd_run_id = 0
        self._jd_by_gen: dict[int, tuple[int, str]] = {}
        self._jd_pending = 0
        self._jd_total = 0
        self._jd_failed: list = []
        self._sqd_gen = 0
        # this page's brackets on the bubbles (`significance`): its choice and the populations they stand on
        from ... import significance as sig

        self._sqd_choice = sig.ViewChoice()
        self._sqd_brackets: list = []

    def _build_spatial_plumbing(self) -> None:
        """The Spatial Map page in hand's Run dialog, worker signals and counters, as `_build_sqd_plumbing`."""
        slot = self._cur["spatial"]
        slot.relays = []
        # The map's own Run configuration. Built ONCE here and reopened,
        # exactly as `ComputeDialog` is -- `content=central` is the widget
        # it blur-snapshots and disables while it is up, which
        # `OverlayDialog` requires explicitly rather than inferring.
        self.map_dialog = SpatialMapDialog(self.t, content=self._central, parent=self)
        self.map_dialog.run.connect(self._bind("spatial", self._map_page_kept))
        self.map_dialog.run.connect(self._bind("spatial", self._run_spatial_section))
        self.map_dialog.cancelled.connect(self._bind("spatial", self._map_dialog_cancelled))
        self.map_dialog.preview_wanted.connect(self._bind("spatial", self._on_map_preview_wanted))
        #: The Spatial Map SECTION's own Run -- one `_ComputeRunnable` PER CONDITION, all sharing these signals.
        #: Keyed by (run id, condition) rather than by condition alone: a second press while the first is still
        #: finishing would otherwise let a stale result for Malate land in a grid drawn at different settings, which
        #: is exactly the pooled-colour-scale failure `_settings_key` exists to prevent, arriving by the back door.
        relay = _PageRelay(self, slot, "_on_spatial_section_done")
        slot.relays.append(relay)
        self._spatial_signals = _ComputeSignals()
        self._spatial_signals.done.connect(relay.done)
        # the map dialog's Split page: one condition's histogram at a time (`_on_map_preview_wanted`)
        relay = _PageRelay(self, slot, "_on_map_preview_done")
        slot.relays.append(relay)
        self._map_preview_signals = _ComputeSignals()
        self._map_preview_signals.done.connect(relay.done)
        self._map_preview_gen = 0
        self._map_preview_by_gen: dict = {}
        #: TRACKS HIDDEN FROM THE MAP by hand, per condition (the day page's eyes, asked 16 Oct 2026): every
        #: map Run leaves them out, and a click redraws that one condition (`_on_map_track_hidden`). The map
        #: only -- every other section keeps them -- and for this session, like the track cap.
        self._map_hidden: dict[str, set] = {}
        relay = _PageRelay(self, slot, "_on_map_redrawn")
        slot.relays.append(relay)
        self._hide_signals = _ComputeSignals()
        self._hide_signals.done.connect(relay.done)
        self._hide_gen = 0
        self._hide_by_gen: dict[int, tuple[int, str]] = {}
        #: condition -> the newest redraw asked for it: an older one landing late is dropped.
        self._hide_latest: dict[str, int] = {}
        self._spatial_gen = 0
        self._spatial_run_id = 0
        self._spatial_by_gen: dict[int, tuple[int, str]] = {}
        #: How many of this run's conditions are still in flight, so the
        #: status line can count down and the spinner can stop on the LAST
        #: one rather than the first.
        self._spatial_pending = 0
        self._spatial_total = 0
        self._spatial_failed: list = []
        #: A map Run WAITING for the section whose D it takes (`map_fit_source.plan`: "wait" or "start"):
        #: ``{"section": job key, "method", "config", "run_id"}``, None while none waits. It draws when that
        #: section's Run lands (`_map_section_ended`).
        self._spatial_waiting = None
        self._spatial_last_config = None
        #: When the last Spatial Map Run finished, for the receipt's stamp.
        self._spatial_computed_at: str | None = None
        # the pages SQD has, for the drop-down "Use the same D as the SQD fit"
        for method in self._fit_methods():
            self.map_dialog.set_fit_pages(method, self._subpage_names(method))

    def _build_clusters_widgets(self) -> None:
        """The Clustering section's widgets for the page in hand (`self._cur["clusters"]`): v2's Clustering tab (17 Oct
        2026), k-means on each track's MSD curve against free Brownian tracks of the same lengths
        (`clustering_view`), one card per condition; the switches are on the page. Built once for the section's own
        page and once per added copy."""
        t = self.t
        self._clusters = ClusteringSection(t)
        self._clusters.run_requested.connect(self._bind("clusters", self._open_clusters_dialog))
        self._clusters.export_requested.connect(self._bind("clusters", self.open_clusters_export))
        self._clusters.cancel_requested.connect(self._bind("clusters", self._cancel_clusters_pressed))
        self._clusters_status = fit_label("", ty.ui("sm"), t.muted_foreground)
        self._clusters_status.hide()
        self._clusters_section = QWidget()
        self._clusters_section.setStyleSheet("background:transparent;")
        clusters_lay = QVBoxLayout(self._clusters_section)
        clusters_lay.setContentsMargins(0, 0, 0, 0)
        clusters_lay.setSpacing(12)
        clusters_lay.addWidget(self._clusters, 1)
        clusters_lay.addWidget(self._clusters_status)
        self._clusters_copy_note = fit_label("", ty.ui("sm"), t.muted_foreground)
        self._clusters_copy_note.setWordWrap(True)
        self._clusters_copy_note.hide()
        clusters_lay.insertWidget(0, self._clusters_copy_note)

    def _build_clusters_plumbing(self) -> None:
        """The Clustering page in hand's worker signals and Run counters: one job per condition per press, as Signed
        jumps."""
        slot = self._cur["clusters"]
        slot.relays = []
        relay = _PageRelay(self, slot, "_on_clusters_done")
        slot.relays.append(relay)
        self._clusters_signals = _ComputeSignals()
        self._clusters_signals.done.connect(relay.done)
        self._clusters_gen = 0
        self._clusters_run_id = 0
        self._clusters_by_gen: dict[int, tuple[int, str]] = {}
        self._clusters_pending = 0
        self._clusters_total = 0
        self._clusters_failed: list = []
        #: A Run's cancel: set, a job still waiting for a worker is never submitted (`_submit_bulk_unless`), and the
        #: futures this Run submitted (`_clusters_futures`) that have not started are cancelled with it
        self._clusters_cancel = threading.Event()
        self._clusters_futures: list = []

    def _build_kinetics_widgets(self) -> None:
        """The Kinetics section's widgets for the page in hand (`self._cur["kinetics"]`): the triggered-kinetics tool, its
        own Run (a configuration dialog, because the trigger frame and the bin width have no default), one card per
        condition, filling in as each lands. Built once for the section's own page and once per added copy."""
        t = self.t
        self._kinetics = KineticsSection(t)
        self._kinetics.run_requested.connect(self._bind("kinetics", self._open_kinetics_dialog))
        self._kinetics.cancel_requested.connect(self._bind("kinetics", self._cancel_kinetics_pressed))
        self._kinetics.export_requested.connect(self._bind("kinetics", self.open_kinetics_export))
        self._kinetics.overview_export_requested.connect(self._bind("kinetics", self.open_kinetics_overview_export))
        self._kinetics.mock_export_requested.connect(self._bind("kinetics", self.open_kinetics_mock_export))
        self._kinetics_status = fit_label(
            "Press Run to configure and compute.", ty.ui("sm"), t.muted_foreground)
        self._kinetics_status.hide()
        self._kinetics_section = QWidget()
        self._kinetics_section.setStyleSheet("background:transparent;")
        kinetics_lay = QVBoxLayout(self._kinetics_section)
        kinetics_lay.setContentsMargins(0, 0, 0, 0)
        kinetics_lay.setSpacing(12)
        kinetics_lay.addWidget(self._kinetics, 1)
        kinetics_lay.addWidget(self._kinetics_status)
        self._kinetics_copy_note = fit_label("", ty.ui("sm"), t.muted_foreground)
        self._kinetics_copy_note.setWordWrap(True)
        self._kinetics_copy_note.hide()
        kinetics_lay.insertWidget(0, self._kinetics_copy_note)

    def _build_kinetics_plumbing(self) -> None:
        """The Kinetics page in hand's worker signals and Run counters: one job per condition, each carrying its press's
        run id, so a result from a superseded press is dropped by name."""
        slot = self._cur["kinetics"]
        slot.relays = []
        relay = _PageRelay(self, slot, "_on_kinetics_done", "_on_kinetics_progress")
        slot.relays.append(relay)
        self._kinetics_signals = _ComputeSignals()
        self._kinetics_signals.done.connect(relay.done)
        self._kinetics_signals.progress.connect(relay.progress)
        #: The event that stops this page's Run (`induction_view.run_permutation`'s ``cancelled``): set by Cancel, a
        #: newer Run and the window's close. ``_kinetics_work``: per condition ``[jobs done, jobs in all]`` for the
        #: Running list (the model's job, then the permutation test's chunks).
        self._kinetics_cancel = threading.Event()
        self._kinetics_work: dict = {}
        self._kinetics_gen = 0
        self._kinetics_run_id = 0
        self._kinetics_by_gen: dict[int, tuple[int, str]] = {}
        self._kinetics_pending = 0
        self._kinetics_total = 0
        self._kinetics_failed: list = []

    def _build_hmm_widgets(self) -> None:
        """The HMM section's widgets for the page in hand (`self._cur["hmm"]`): the diffusive-state HMM, MSD's shape --
        its own Run dialog, one card per condition filling in as each lands, and the Test page. Built once for the
        section's own page and once per added copy (subpages), as `_build_sqd_widgets`."""
        t = self.t
        slot = self._cur["hmm"]
        self._hmm = HMMSection(t)
        # the strip's pages move the sidebar's sub-view, as SQD's do (22 Nov 2026)
        self._hmm.page_changed.connect(
            lambda i, s=slot: self.sidebar.select(sp.join_view("hmm", s.number, HMM_SECTION_PAGES[i])))
        self._hmm.run_requested.connect(self._bind("hmm", self._open_hmm_dialog))
        self._hmm.cancel_requested.connect(self._bind("hmm", self._cancel_hmm_pressed))
        self._hmm.export_requested.connect(self._bind("hmm", self.open_hmm_export))
        self._hmm.comparison_export_requested.connect(self._bind("hmm", self.open_hmm_comparison_export))
        self._hmm.overview_export_requested.connect(self._bind("hmm", self.open_hmm_overview_export))
        self._hmm.map_requested.connect(self._bind("hmm", lambda: self._map_this_fit("hmm")))
        self._hmm_status = fit_label(
            "Press Run to configure and compute.", ty.ui("sm"), t.muted_foreground)
        self._hmm_status.hide()
        self._hmm_section = QWidget()
        self._hmm_section.setStyleSheet("background:transparent;")
        hmm_lay = QVBoxLayout(self._hmm_section)
        hmm_lay.setContentsMargins(0, 0, 0, 0)
        hmm_lay.setSpacing(12)
        hmm_lay.addWidget(self._hmm, 1)
        hmm_lay.addWidget(self._hmm_status)
        self._hmm_copy_note = fit_label("", ty.ui("sm"), t.muted_foreground)
        self._hmm_copy_note.setWordWrap(True)
        self._hmm_copy_note.hide()
        hmm_lay.insertWidget(0, self._hmm_copy_note)
        self._hmm.set_grouping(self._factors, self._factor_order)
        self._hmm.set_pairs(self._replicate_pairs())
        # the comparison drawn again (a result, an eye, the order): its brackets follow (`_apply_significance`)
        self._hmm.comparison_changed.connect(self._bind("hmm", lambda: self._apply_significance("hmm")))
        # THE JUMP ANGLES TAB (23 Nov 2026; a sidebar section of its own from 22 Nov): this page's fits, asked for
        # while the tab is on show
        self._angles_page = self._hmm.angles
        self._angles_page.export_requested.connect(self._bind("hmm", self.open_hmm_angles_export))
        self._angles_page.chance_changed.connect(self._bind("hmm", self._angles_reading_changed))
        self._angles_asked = set()

    def _build_hmm_plumbing(self) -> None:
        """The HMM page in hand's worker signals and Run counters, its shared fit's progress and the event that stops
        it. Results and progress come back through a `_PageRelay`."""
        slot = self._cur["hmm"]
        slot.relays = []
        relay = _PageRelay(self, slot, "_on_hmm_done", "_on_hmm_progress")
        slot.relays.append(relay)
        self._hmm_signals = _ComputeSignals()
        self._hmm_signals.done.connect(relay.done)
        self._hmm_signals.progress.connect(relay.progress)
        self._hmm_gen = 0
        self._hmm_run_id = 0
        self._hmm_by_gen: dict[int, tuple[int, str]] = {}
        self._hmm_pending = 0
        self._hmm_total = 0
        self._hmm_failed: list = []
        self._hmm_config = None
        #: THE SHARED FIT'S PROGRESS while it runs (`hmm_run.joint_progress`), None outside it; and the event that
        #: stops it (`hmm_run.joint_D`'s ``cancelled``), set by Cancel, a newer Run and the window's close.
        self._hmm_joint: dict | None = None
        self._hmm_cancel = threading.Event()
        # THE BRACKETS ON THE COMPARISON (5 Nov 2026, off by default): this page's own choice, a panel's values compared
        from ... import significance as sig

        self._hmm_choice = sig.ViewChoice(population="occupancy")
        self._hmm_brackets: list = []
        # the Jump angles tab's jobs, one a condition (`_run_hmm_angles`)
        angles_relay = _PageRelay(self, slot, "_on_hmm_angles_done")
        slot.relays.append(angles_relay)
        self._angles_signals = _ComputeSignals()
        self._angles_signals.done.connect(angles_relay.done)
        self._angles_gen = 0
        self._angles_by_gen: dict = {}
        self._angles_pending = 0
        self._angles_total = 0
        self._angles_failed: list = []

    def _build_appd_widgets(self) -> None:
        """The Apparent Diffusion section's widgets for the page in hand (`self._cur["appd"]`), built once for the
        section's own page and once per added copy (subpages), as `_build_sqd_widgets`.

        APPARENT DIFFUSION, rebuilt 16 Oct 2026 (`appd_section`, `appd_view.appd_condition`): one apparent D per
        track over all its steps, four pages behind a strip (Populations, Distributions with the track gallery,
        Per cell, Checks), one job per condition."""
        t = self.t
        slot = self._cur["appd"]
        self._appd = AppDSection(t)
        self._appd.run_requested.connect(self._bind("appd", self._open_appd_dialog))
        self._appd.cancel_requested.connect(self._bind("appd", self._cancel_appd_pressed))
        self._appd.map_requested.connect(self._bind("appd", lambda: self._map_this_fit("appd")))
        self._appd.page_changed.connect(
            lambda i, s=slot: self.sidebar.select(sp.join_view("appd", s.number, APPD_PAGES[i])))
        self._appd.export_requested.connect(self._bind("appd", self.open_appd_export))
        self._appd.table_export_requested.connect(self._bind("appd", self.open_appd_table_export))
        self._appd.distribution_export_requested.connect(self._bind("appd", self.open_appd_distribution_export))
        self._appd.checks_export_requested.connect(self._bind("appd", self.open_appd_checks_export))
        self._appd.cells_export_requested.connect(self._bind("appd", self.open_appd_cells_export))
        self._appd.cell_length_export_requested.connect(self._bind("appd", self.open_appd_cell_length_export))
        self._appd.overview_export_requested.connect(self._bind("appd", self.open_appd_overview_export))
        self._appd.track_requested.connect(self._bind("appd", self._on_appd_track_requested))
        self._appd.crosschecks_requested.connect(self._bind("appd", self._on_appd_condition_clicked))
        # its replicate page replaces the bubbles: the rail follows it (after the section's own switch, connected first)
        self._appd.populations.chart.population_clicked.connect(
            self._bind("appd", lambda _column: self._sync_appd_rail()))
        self._appd.days.back_requested.connect(self._bind("appd", self._sync_appd_rail))
        # THE SQD / HMM CROSS-CHECKS of one condition (SuperPlots, Dynamics): not apparent D, kept reachable from
        # the apparent-D day page so nothing that existed is lost.
        self._appd_detail = ApparentDDetail(t)
        self._appd_detail.fractions_run_requested.connect(self._bind("appd", self._run_fractions_checks))
        self._appd_detail.dynamics_run_requested.connect(self._bind("appd", self._run_dynamics_evidence))
        self._appd_detail.tabs.changed.connect(self._bind("appd", lambda _k: self._sync_appd_visibility_section()))
        self._appd_status = fit_label("", ty.ui("sm"), t.muted_foreground)
        self._appd_status.hide()
        self._appd_section = QWidget()
        self._appd_section.setStyleSheet("background:transparent;")
        appd_lay = QVBoxLayout(self._appd_section)
        appd_lay.setContentsMargins(0, 0, 0, 0)
        appd_lay.setSpacing(12)
        appd_lay.addWidget(self._appd, 1)
        appd_lay.addWidget(self._appd_status)
        self._appd_copy_note = fit_label("", ty.ui("sm"), t.muted_foreground)
        self._appd_copy_note.setWordWrap(True)
        self._appd_copy_note.hide()
        appd_lay.insertWidget(0, self._appd_copy_note)
        self._appd.set_grouping(self._factors, self._factor_order)

    def _build_appd_plumbing(self) -> None:
        """The Apparent Diffusion page in hand's worker signals, Run counters and brackets' choice; the cross-checks'
        two per-condition jobs (SuperPlots, Dynamics) are the page's too. Results come back through `_PageRelay`s."""
        slot = self._cur["appd"]
        slot.relays = []

        def signals(done: str):
            relay = _PageRelay(self, slot, done)
            slot.relays.append(relay)
            out = _ComputeSignals()
            out.done.connect(relay.done)
            return out

        #: Apparent Diffusion: one job per condition (`_run_appd`), each tagged with the press it belongs to.
        self._appd_signals = signals("_on_appd_done")
        self._appd_gen = 0
        self._appd_run_id = 0
        self._appd_by_gen: dict[int, tuple[int, str]] = {}
        self._appd_pending = 0
        self._appd_total = 0
        self._appd_failed: list = []
        self._appd_config = None
        #: SuperPlots (`ApparentDDetail`'s own tab) is PER-CONDITION and independent of everything else here -- two
        #: different conditions' own Runs never supersede each other, so it shares no `_appd_gen`.
        #: `_fractions_condition_by_gen` is its OWN staleness bookkeeping: which condition a press was FOR, looked up
        #: by the done handler (which needs it even on error). Dynamics (the second tab) is the same shape, its own trio.
        self._fractions_signals = signals("_on_fractions_checks_done")
        self._fractions_gen = 0
        self._fractions_condition_by_gen: dict[int, str] = {}
        self._dynamics_signals = signals("_on_dynamics_evidence_done")
        self._dynamics_gen = 0
        self._dynamics_condition_by_gen: dict[int, str] = {}
        # this page's brackets on the bubbles (`significance`): its choice and the populations they stand on
        from ... import significance as sig

        self._appd_choice = sig.ViewChoice()
        self._appd_brackets: list = []

    def _build_msd_widgets(self) -> None:
        """The MSD section's widgets for the page in hand (`self._cur["msd"]`): v2's MSD tab, its own Run (a
        configuration dialog), the shared chart above one card per condition, filling in as each lands. Called once for
        the section's own page and once per added copy (subpages), as `_build_sqd_widgets`."""
        t = self.t
        slot = self._cur["msd"]
        self._msd = MSDSection(t)
        self._msd.run_requested.connect(self._bind("msd", self._open_msd_dialog))
        self._msd.cancel_requested.connect(self._bind("msd", self._cancel_msd_pressed))
        self._msd.export_requested.connect(self._bind("msd", self.open_msd_export))
        self._msd.table_export_requested.connect(self._bind("msd", self.open_msd_table_export))
        self._msd.checks_export_requested.connect(
            self._bind("msd", lambda k: self.open_msd_page_export(k, "checks")))
        self._msd.v2_export_requested.connect(self._bind("msd", lambda k: self.open_msd_page_export(k, "v2")))
        self._msd.cells_csv_requested.connect(self._bind("msd", self.save_msd_per_cell))
        self._msd.page_changed.connect(
            lambda i, s=slot: self.sidebar.select(sp.join_view("msd", s.number, MSD_PAGES[i])))
        self._msd_status = fit_label(
            "Press Run to configure and compute.", ty.ui("sm"), t.muted_foreground)
        self._msd_status.hide()
        self._msd_section = QWidget()
        self._msd_section.setStyleSheet("background:transparent;")
        msd_lay = QVBoxLayout(self._msd_section)
        msd_lay.setContentsMargins(0, 0, 0, 0)
        msd_lay.setSpacing(12)
        msd_lay.addWidget(self._msd, 1)
        msd_lay.addWidget(self._msd_status)
        self._msd_copy_note = fit_label("", ty.ui("sm"), t.muted_foreground)
        self._msd_copy_note.setWordWrap(True)
        self._msd_copy_note.hide()
        msd_lay.insertWidget(0, self._msd_copy_note)

    def _build_msd_plumbing(self) -> None:
        """The MSD page in hand's worker signals and Run counters (the same per-condition, per-press scheme as
        Kinetics), results delivered through a `_PageRelay`."""
        slot = self._cur["msd"]
        slot.relays = []
        relay = _PageRelay(self, slot, "_on_msd_done")
        slot.relays.append(relay)
        self._msd_signals = _ComputeSignals()
        self._msd_signals.done.connect(relay.done)
        self._msd_gen = 0
        self._msd_run_id = 0
        self._msd_by_gen: dict[int, tuple[int, str]] = {}
        self._msd_pending = 0
        self._msd_failed: list = []
        self._msd_total = 0

    def _on_selected(self, view: str) -> None:
        # AN ADDED PAGE ("sqd~2.jumps"): it becomes the section's current page, and everything below reads the view as
        # the section's own page would name it ("sqd.jumps"); only the breadcrumb keeps the page's name
        full = view
        number = sp.split_view(view)[1]
        view = sp.base_view(view)
        section, _, _sub = view.partition(".")
        if section in self._slots:
            self._cur[section] = self._slot(section, number) or self._slots[section][0]
        # the PAGE's: Jump distances never offers the Populations chart's brackets (`rail_view.rail_section`)
        self.toolbar.set_section(rail_section(view))
        self._sync_compute_mark()
        self.breadcrumb.set_trail(self.sidebar.trail(full))
        # A drill-in only ever means something WITHIN the view that opened
        # it -- leaving via the sidebar (rather than a crumb click) still
        # has to drop the detail page for the same reason, in WHICHEVER
        # section it was open in. Never lands on either placeholder here:
        # once a section has graduated, leaving via the sidebar is not a
        # reason to show it again -- see the module docstring.
        if self._ever_succeeded:
            self._pages.setCurrentWidget(self._card)
        self._appd.show_overview()
        if self._qc_ever_succeeded:
            self._qc_pages.setCurrentWidget(self._qc_general)
        self._spatial.show_overview()
        self._section_pages.setCurrentWidget(
            {"appd": self._appd_section,
             "spatial": self._spatial_section,
             "kinetics": self._kinetics_section,
             "clusters": self._clusters_section,
             "msd": self._msd_section,
             "hmm": self._hmm_section,
             "composer": self._composer,
             "images": self._images,
             "qc": self._qc_section}.get(section, self._sqd_section))
        if section == "appd":
            self._appd.show_page(APPD_PAGES.index(_sub) if _sub in APPD_PAGES else 0)
        if section == "msd":
            self._msd.show_page(MSD_PAGES.index(_sub) if _sub in MSD_PAGES else 0)
        if section == "hmm":
            self._hmm.show_page(_sub or HMM_SECTION_PAGES[0])
            if _sub == "angles":
                self._ask_angles()             # the Jump angles of this page's fits, asked for while on show
        self._view_before = view
        if section == "sqd":
            index = SQD_PAGES.index(_sub) if _sub in SQD_PAGES else 0
            self._sqd_stack.setCurrentIndex(index)
            if self._sqd_tabs.current() != index:
                self._sqd_tabs.blockSignals(True)
                self._sqd_tabs.select(index)
                self._sqd_tabs.blockSignals(False)
        self._sync_metadata()
        if section in self._slots:
            self._sync_copy_note()
        if section == "images":
            self._images.set_colours(self._condition_colours(self._images.conditions()))
            self._images.ensure_shown()
            self._images_trail()
        if section == "spatial":
            # `toolbar.set_section` above rebuilt the Visibility list from
            # `FIGURES["spatial"]` with every row back ON and no ramp
            # selected. This re-asserts what the grid actually shows, so
            # leaving the section and coming back does not silently unhide a
            # column the user hid or reset the colormap they chose.
            self._sync_spatial_visibility_section()

    # -- jobs and their toasts ---------------------------------------------- #

    #: A job's section, and the words that name it on a toast.
    _JOB_SECTION = {"sqd": "sqd", "appd": "appd", "qc": "qc",
                    "spatial": "spatial", "kinetics": "kinetics", "msd": "msd",
                    "hmm": "hmm", "jumps": "sqd", "jd": "sqd", "clusters": "clusters", "hmm_angles": "hmm"}
    #: A job that belongs to ONE page of its section, not the section: Jump distances is
    #: SQD's second page, so its mark, "here" and Show are that page's, never Populations'.
    _JOB_VIEW = {"jumps": "sqd.jumps", "jd": "sqd.signed", "hmm_angles": "hmm.angles"}
    _JOB_SUFFIX = {"jumps": "Jump distances", "jd": "Signed jumps", "hmm_angles": "Jump angles"}

    def _job_title(self, key: str) -> str:
        """A job's name on a toast and in the Running list: its section's title, or an added page's own name, then the
        page of the section it is ("SQD 2 \u00b7 Jump distances")."""
        base, number = sp.split_job(key)
        section = self._JOB_SECTION[base]
        title = SECTION_TITLE.get(section, key)
        if number is not None:
            try:
                title = self._subpage_set.name(section, number)
            except KeyError:
                pass
        suffix = self._JOB_SUFFIX.get(base)
        return f"{title} \u00b7 {suffix}" if suffix else title

    def _job_view(self, key: str) -> str:
        """The view a job's row and toast open: its own page where it has one, else its section's first, on the
        page the job belongs to (``sqd~2.jumps``)."""
        base, number = sp.split_job(key)
        section = self._JOB_SECTION[base]
        return sp.join_view(section, number, self._JOB_VIEW.get(base, SUBS[section][0][0]).partition(".")[2])

    def _job_here(self, key: str) -> bool:
        """Is ``key``'s page the one on screen: its own page where it has one, else its section."""
        base, number = sp.split_job(key)
        section = self._JOB_SECTION[base]
        current, cur_number, _page = sp.split_view(self.sidebar.current)
        if current != section or cur_number != number:
            return False
        if base in self._JOB_VIEW:
            return self.sidebar.current_base == self._JOB_VIEW[base]
        return self.sidebar.current_base not in self._JOB_VIEW.values()      # a page with its own job shows only it

    def _sync_compute_mark(self) -> None:
        """The rail's Compute mark spins while the section ON SCREEN is computing.

        Sections run side by side, and every one used to switch the ONE mark on and
        off: two running, the first to finish stopped the spinner while the other was
        still going. The mark answers "is THIS page's job running" -- the prototype's
        own rule -- and follows the sidebar.

        Its HOVER says what a press does on this page (`rail_view.compute_words`: "Run SQD\u2026", "Stop HMM"), and
        on a page with nothing to compute it is dimmed (4 Nov 2026, rail review: it said "Compute" everywhere)."""
        mark = self.toolbar.rail.compute
        running = self._job_running_here()
        mark.set_running(running is not None)
        job = compute_job(self.sidebar.current_base)
        mark.set_available(job is not None)
        # an added page names itself in the hover ("Run SQD 2...")
        key = sp.job_key(job, sp.split_view(self.sidebar.current)[1]) if job else None
        self.toolbar.rail.set_compute_tip(compute_words(job, self._job_title(running or key) if job else "",
                                                        running is not None))

    def _job_running_here(self):
        """The key of the job running for the page on screen, or None."""
        return next((k for k in self._running if self._job_here(k)), None)

    def _job_started(self, key: str) -> None:
        self._running[key] = time.monotonic()
        self._sync_compute_mark()
        self._sync_jobs()

    def _job_cancelled(self, key: str) -> None:
        """A cancel says nothing: the reader just pressed it (the prototype's rule). A map waiting for this
        section's fit hears of it after the event loop turns, so a Run that replaces the cancelled one
        (`_replace_job`) is waited for instead (`_map_section_ended`)."""
        was = self._running.pop(key, None)
        self._sync_compute_mark()
        self._sync_jobs()
        if was is not None and sp.split_job(key)[0] in map_fit_source.SECTION_KEYS.values():
            QTimer.singleShot(0, self, lambda k=key: self._map_section_ended(k, False))

    def _job_counts(self, key: str) -> tuple:
        """``(done, total)`` conditions of ``key``'s Run: SQD's from its progress (`_on_sqd_progress`); every Run made
        of one job per condition keeps ``_<key>_total`` and ``_<key>_pending``; QC is one job of no known size (0).
        An added page's job is read on that page."""
        base, number = sp.split_job(key)
        section = self._JOB_SECTION[base]
        slot = self._slot(section, number) if section in self._slots else None
        if slot is None and section in self._slots:
            return 0, 0
        with (self._in(slot) if slot is not None else _nullcontext()):
            if base == "sqd":
                return int(self._sqd_progress.get("done", 0)), int(self._sqd_progress.get("total", 0))
            if base == "hmm" and self._hmm_joint is not None:     # the shared fit counts its starts (`_job_words`)
                return int(self._hmm_joint.get("done", 0)), int(self._hmm_joint.get("total", 0))
            if base == "kinetics" and self._kinetics_work:        # the model's job, then the test's chunks
                return (sum(d for d, _t in self._kinetics_work.values()),
                        sum(t for _d, t in self._kinetics_work.values()))
            total, pending = getattr(self, f"_{base}_total", None), getattr(self, f"_{base}_pending", None)
            if not total or pending is None:
                return 0, 0
            return int(total) - int(pending), int(total)

    def _job_words(self, key: str) -> dict:
        """What ``key``'s counts count, when not conditions (`run_progress.sidebar_job`'s ``unit`` and ``stage``): the
        HMM's shared fit counts its starts, "shared states: 7 of 12 fits"."""
        base, number = sp.split_job(key)
        slot = self._slot("hmm", number) if base == "hmm" else None
        joint = getattr(slot, "_hmm_joint", None)
        if joint is not None:
            return {"unit": str(joint.get("unit", "fit")), "stage": str(joint.get("stage", ""))}
        work = getattr(self._slot("kinetics", number), "_kinetics_work", None) if base == "kinetics" else None
        if work and any(t > 1 for _d, t in work.values()):    # a permutation test is under way: it counts its jobs
            return {"unit": "job"}
        return {}

    def _sync_jobs(self) -> None:
        """The sidebar's Running list from the job register (`run_progress.sidebar_job`), each row opening its job's
        page (`_JOB_VIEW`, else its section's first); the clock runs while any job does."""
        now = time.monotonic()
        jobs = []
        for key, started in self._running.items():
            done, total = self._job_counts(key)
            job = sidebar_job(title=self._job_title(key), done=done, total=total, started=started, now=now,
                              **self._job_words(key))
            job.update(key=key, view=self._job_view(key))
            jobs.append(job)
        self.sidebar.set_jobs(jobs)
        if jobs and not self._jobs_clock.isActive():
            self._jobs_clock.start()
        elif not jobs:
            self._jobs_clock.stop()

    def _group_keys(self) -> None:
        """Every condition key on screen grouped by the factors (`ConditionKey.set_grouping`): at the start, at every
        nesting change and after every Run (a Run can build new cards with keys)."""
        from .condition_key import ConditionKey

        if len(self._factor_order) < 2 or not self._factors:
            return
        for key in self.findChildren(ConditionKey):
            key.set_grouping(self._factors, self._factor_order)

    def _job_finished(self, key: str, *, error=None, note: str | None = None) -> None:
        """A job concluded: the toast, then any start a prompt was holding back.

        Called only from a CURRENT generation's result -- a superseded or cancelled
        one returns before it gets here -- and only once per job: a key that is not
        running posts nothing."""
        self._group_keys()
        started = self._running.pop(key, None)
        self._sync_compute_mark()
        self._sync_jobs()
        if started is None:
            return
        self._post_job_toast(key, "failed" if error is not None else "done",
                             seconds=time.monotonic() - started, error=error, note=note)
        # A SECTION A MAP TAKES ITS D FROM landed (`map_fit_source`): a map waiting for it draws, maps drawn from an
        # earlier fit of it say so -- before a held prompt starts anything new
        if sp.split_job(key)[0] in map_fit_source.SECTION_KEYS.values():
            self._map_section_ended(key, error is None)
        # A question whose premise expired: nothing is left to replace, so the
        # prompt is withdrawn and what was asked for starts (the prototype's rule).
        held = self._prompts.pop(key, None)
        if held is not None:
            prompt, start = held
            prompt.dismiss()
            start()

    def _finish_per_condition(self, key: str, failed: list, total: int) -> None:
        """Spatial Map and Kinetics run one job per condition and keep going past a
        failing one. Every condition failing is a failed Run; some failing is a
        finished one whose toast names how many did not."""
        if failed and len(failed) == total:
            self._job_finished(key, error=failed[0][1])
            return
        self._job_finished(key, note=(f"{len(failed)} of {total} conditions failed"
                                      if failed else None))

    def _post_job_toast(self, key: str, outcome: str, *, seconds=None, error=None,
                        note: str | None = None):
        words = compute_toast(self._job_title(key), outcome, seconds=seconds,
                              error=error, here=self._job_here(key), note=note)
        duration = None if outcome == "ask" else 4000
        toast = self.toasts.post(words["title"], description=words["description"],
                                 action=words["action"], duration=duration)
        if words["action"] == "Show":
            toast.on_action(lambda v=self._job_view(key): self.sidebar.select(v))
        return toast

    def _start_or_ask(self, key: str, start) -> None:
        """Start ``key``'s Run, or ASK first when that section is still computing.

        A second Run in a running section once replaced the first in silence, which
        can throw away minutes of bootstrap. The prompt waits (no timer) and has two
        exits: Proceed cancels the running job (the section's own soft cancel) and
        starts this one; the close keeps what is running. One prompt per section: a
        newer press replaces the question rather than stacking a second."""
        # a start held back runs on the page it was asked for, whichever page is on show by then
        base, number = sp.split_job(key)
        section = self._JOB_SECTION.get(base)
        slot = self._slot(section, number) if section in self._slots else None
        if slot is not None:
            asked = start
            start = lambda: self._call_in(slot, asked)          # noqa: E731
        if key not in self._running:
            start()
            return
        old = self._prompts.pop(key, None)
        if old is not None:
            old[0].dismiss()
        prompt = self._post_job_toast(key, "ask")
        self._prompts[key] = (prompt, start)
        prompt.on_action(lambda k=key: self._replace_job(k))
        prompt.on_close(lambda k=key: self._prompts.pop(k, None))

    def _replace_job(self, key: str) -> None:
        held = self._prompts.pop(key, None)
        if held is None:
            return
        _prompt, start = held
        self._cancel_job(key)
        start()

    def _follow_head(self, centre: float) -> None:
        """Keep the trigger level with the sidebar header through the
        collapse -- same idea as the prototype's `WorkspacePanel._follow_head`."""
        self.trigger.move(self.trigger.x(), round(centre - self.trigger.height() / 2))

    # -- the Compute dialog ---------------------------------------------------

    def _open_compute_dialog(self) -> None:
        """The mark was pressed. `ComputeDialog.open` covers this window's
        whole client area, disables `self.centralWidget()`, and centres its
        own card within itself -- nothing left for this method to compute."""
        self.compute_dialog.open()

    def _on_compute_mark_pressed(self) -> None:
        """The toolbar rail's Compute glyph -- the SAME action either
        section's own placeholder Run button offers, reached from wherever
        the user happens to be looking. Section-aware: this window owns one Compute
        dialog and it is entirely SQD settings, so every other section's mark opens that
        section's own dialog."""
        section = self.sidebar.current_base.partition(".")[0]
        running = self._job_running_here()
        if running is not None:
            # THE STOP SQUARE MEANS STOP: a press while this page's job runs cancels it, as the placeholder's Cancel
            # does (a soft cancel: its results are dropped on arrival). It opened the settings again (rail review).
            self._cancel_job(running)
            return
        if section == "appd":
            self._open_appd_dialog()
        elif section == "qc":
            self._run_qc()
        elif section == "spatial":
            # The map's own configuration dialog, not the SQD Compute form.
            # Same argument as the `appd` branch: this window owns one
            # Compute dialog and it is entirely SQD fit settings, so the
            # glyph must reach whatever the CURRENT section calls a Run.
            self._open_spatial_dialog()
        elif section == "kinetics":
            # Its own configuration dialog, never the SQD Compute form: the trigger frame
            # and the bin width have no default and live there.
            self._open_kinetics_dialog()
        elif section == "msd":
            self._open_msd_dialog()
        elif section == "hmm":
            self._open_hmm_dialog()
        elif section == "clusters":
            self._open_clusters_dialog()
        elif self.sidebar.current_base == "sqd.jumps":
            self._open_jumps_dialog()
        elif self.sidebar.current_base == "sqd.signed":
            # Signed jumps has no settings of its own to ask: its Run, never the SQD Compute form (found 3 Nov 2026
            # by the overnight tester: the mark opened "Configure SQD", whose Run would refit the bubbles)
            self._start_or_ask(self._jk("jd"), self._run_jd)
        elif section in ("composer", "images"):
            # Nothing to compute: these show what is loaded or what the sections made.
            return
        else:
            self._open_compute_dialog()

    def _cancel_job(self, key: str) -> None:
        """Stop ``key``'s Run the way its own Cancel does (on the page the job belongs to)."""
        base, number = sp.split_job(key)
        cancel = {"sqd": self._cancel_sqd_pressed, "jumps": self._cancel_jumps_pressed, "jd": self._cancel_jd_pressed,
                  "appd": self._cancel_appd_pressed, "qc": self._cancel_qc_pressed,
                  "spatial": self._cancel_spatial_pressed, "kinetics": self._cancel_kinetics_pressed,
                  "clusters": self._cancel_clusters_pressed, "msd": self._cancel_msd_pressed,
                  "hmm": self._cancel_hmm_pressed}.get(base)
        if cancel is None:
            return
        section = self._JOB_SECTION[base]
        slot = self._slot(section, number) if section in self._slots else None
        if slot is not None:
            self._call_in(slot, cancel)
        else:
            cancel()

    def _show_sqd_message(self, text: str) -> None:
        """Render TEXT on the SQD section's own chart ONLY -- never leave a
        blank or a stale plot up, the same rule `browser._ComparisonTab
        ._message` follows. Used for SQD's own no-conditions refusal and a
        failed SQD fit alike -- whichever is current is always what is ON
        the chart, not just in a status line. Also drops any open SQD
        detail page -- a stale drill-in beside a freshly-blanked chart is
        the same staleness this method exists to avoid.

        SQD ONLY -- see the module docstring's "EACH SECTION RUNS ON ITS
        OWN". A refusal or a failed fit in this section says nothing about
        whether Apparent-D Distribution's own last result is still good, so
        it must not touch `self._appd_*` at all.

        BEFORE this section's first success, there is no chart to blank --
        its own card is still empty and not even the visible page. Its
        placeholder is a ONE-WAY door (see `section_placeholder`'s own
        module docstring), so a refusal or a failed FIRST attempt returns it
        to its own idle state rather than jumping to the card early; the
        status line underneath still carries the message (unhidden just for
        this line, since the placeholder's own idle state is otherwise mute
        about why the Run it just tried did not land).
        """
        clean = text.replace("\n", " ")
        if not self._ever_succeeded:
            self._placeholder.set_running(False)
            self.status.show()
            self.status.setText(clean)
        else:
            # Undimmed even though this OVERWRITES the previous result with
            # plain text -- an error redesign of its own is out of scope
            # here (this project's existing behaviour, untouched); a card
            # left dimmed forever because nothing ever turned it back off
            # would be a new, worse defect layered on top of an old, known
            # one.
            self._card.set_running(False)
            self._card.set_message(text)
            self._pages.setCurrentWidget(self._card)
            self.status.setText(clean)
        self.breadcrumb.set_trail(self.sidebar.trail(self.sidebar.current))

    def _show_appd_message(self, text: str) -> None:
        """A word for the Apparent Diffusion section, under it (no conditions, a refusal)."""
        self._appd_status.setText(text.replace("\n", " "))
        self._appd_status.show()

    def _run_pressed(self) -> None:
        """Fit and draw the SQD bubble view ONLY, through the same assembly
        functions the browser's own SQD tab calls -- see the module
        docstring's "RUN IS WIRED" and "EACH SECTION RUNS ON ITS OWN". The
        dialog has already closed itself (`ComputeDialog._on_run`) by the
        time this fires. NEVER touches Apparent-D Distribution's own state
        -- that section computes and renders only from its own Run press
        (`_run_appd`), asked for directly after an earlier version
        fired both from this one press.

        Dispatched OFF THE GUI PROCESS, not merely off the GUI thread -- see
        `self._process_pool`'s own comment for why a thread is not enough
        here: a real multistart fit is a long, tight, Python-heavy loop, and
        two THREADS in one process share one GIL, which is not fair enough
        to stop that loop from starving the main thread's own event
        processing for its whole duration. `compute_bubble_view` and its
        arguments (`project`, `spec`) must stay PICKLABLE for this to work --
        verified directly, not assumed (`project.py`'s `Project` is a plain
        dataclass over a DataFrame/dict/tuples, nothing exotic). The
        `QThreadPool` worker thread below still exists, but its OWN job is
        now just "submit to the process pool and block on the result" --
        that block is a blocking OS-level read, which releases the GIL for
        its ENTIRE duration, so the worker thread sits there costing this
        process's GIL nothing while the real work happens elsewhere.

        Building `spec` and reading `values()` stays here, synchronous: both
        are just widget reads, not a fit, and the worker needs a spec object
        already built rather than a live dialog it would be touching from
        the wrong thread.
        """
        try:
            spec = _spec_from_dialog(self.compute_dialog.compute, self._factors, self.project)
        except RunSettingsError as exc:
            self._show_sqd_message(str(exc))
            return
        if not spec.conditions:
            self._show_sqd_message("Select at least one condition to compare.")
            return
        uncertainty = self.compute_dialog.compute.values()["uncertainty"]
        self._start_or_ask(self._jk("sqd"), lambda: self._start_sqd_run(spec, uncertainty))

    def _start_sqd_run(self, spec, uncertainty: str) -> None:
        """Start an SQD Run for `spec` -- what `_run_pressed` does once it has read
        the dialog, split out so the theme switch can re-show the last figure from
        the cache without a dialog in between."""
        self._sqd_pending = (spec, uncertainty)
        project = self.project
        process_pool = self._process_pool
        cache = self._results_cache
        signals = self._signals
        slot = self._cur["sqd"]               # the page this Run is of: the worker below reads ITS counter, never
        gen = self._sqd_gen + 1               # whichever page is current when the worker looks
        # ONE JOB PER CONDITION, THEN THE FIGURE. Each condition's fit runs as its own
        # process-pool job, so conditions fit side by side and each completion can be
        # reported; the figure is then assembled from the cache they filled. The joint
        # fit (Shared D across conditions, or within a group) needs a group's conditions at once, so
        # those scopes stay one job.
        per_condition = spec.d_sharing_scope not in ("global", "group")
        conditions = list(spec.conditions)
        # THE CORES ARE SHARED OUT, not multiplied: each condition's job starts its
        # own pool for the BIC selection's starts, so every job gets its share of the
        # machine rather than all of it (2 conditions on 8 logical cores: 4 each).
        cores = os.cpu_count() or 2
        job_spec = replace(spec, n_jobs=max(1, cores // max(1, len(conditions)))
                           if per_condition else None)

        def compute():
            if per_condition:
                futures = {self._submit_bulk(
                    warm_condition, project, job_spec, c, uncertainty=uncertainty,
                    cache=cache): c for c in conditions}
                finished = []
                for future in as_completed(futures):
                    if gen != slot._sqd_gen:            # superseded or cancelled
                        for other in futures:
                            other.cancel()
                        return None
                    try:
                        finished.append(future.result())
                    except Exception:
                        for other in futures:           # stop the rest, then report
                            other.cancel()
                        raise
                    signals.progress.emit(gen, len(finished), len(conditions),
                                          list(finished))
            future = process_pool.submit(
                compute_bubble_view, project, spec, uncertainty=uncertainty,
                cache=cache)
            return future.result()

        self._sqd_gen = gen
        self._sqd_total = len(conditions) if per_condition else 0
        self._job_started(self._jk("sqd"))
        # Two different moments -- see the module docstring. Pre-success,
        # this section's own placeholder (already the visible page, if this
        # section is) owns the ENTIRE "running" story on its own, no status
        # line needed underneath it. After it, the previous result stays
        # up, dimmed, and this section's own status line is what says
        # something is happening.
        if self._ever_succeeded:
            self._card.set_running(True)
            self.status.setText("computing…")
        else:
            self._placeholder.set_running(True)
        self._start_sqd_progress(spec, self._sqd_total)
        self._pool.start(_ComputeRunnable(compute, self._sqd_gen, self._signals))

    def _cancel_sqd_pressed(self) -> None:
        """SQD's own placeholder Cancel, pre-first-success only -- see its
        own module docstring's "Cancel is a SOFT cancel". Bumping
        `_sqd_gen` is the whole mechanism: the worker keeps running to
        completion on its own thread, and `_on_compute_done` drops its
        result the moment it lands, exactly as it already would for any
        ordinary superseded Run. Touches SQD's own placeholder only -- see
        the module docstring's "EACH SECTION RUNS ON ITS OWN"."""
        self._sqd_gen += 1
        self._sqd_clock.stop()
        self._job_cancelled(self._jk("sqd"))
        self._placeholder.set_running(False)
        if self._ever_succeeded:
            # after a first result only the rail's mark reaches this: the line said "computing..." for good
            self.status.setText("Run stopped: the chart is the last finished Run's.")

    def _start_sqd_progress(self, spec, total: int) -> None:
        """Reset and start the Run's progress line -- see `progress_text`."""
        models = {spec.model_for(c) for c in spec.conditions}
        self._sqd_progress = {"done": 0, "total": total, "finished": [],
                              "started": time.monotonic(),
                              # fits side by side choose a count only where Best or saSPT is ticked
                              "slow": bool(set(getattr(spec, "side_by_side", ()) or models) & {"best", "saspt"})}
        self._render_sqd_progress()
        self._sqd_clock.start()
        # the sidebar's line was said when the job started, from the last Run's counts: said again before it paints
        self._sync_jobs()

    def _on_sqd_progress(self, gen: int, done: int, total: int, finished) -> None:
        """A condition finished fitting. Dropped if a later Run superseded this one."""
        if gen != self._sqd_gen:
            return
        self._sqd_progress.update(done=done, total=total, finished=list(finished))
        self._render_sqd_progress()

    def _render_sqd_progress(self) -> None:
        """Where the line goes depends on the moment, as "computing…" did: in the
        placeholder before the section's first result, on the status line after."""
        state = self._sqd_progress
        text = progress_text(state["done"], state["total"], state["finished"],
                             time.monotonic() - state["started"],
                             choosing_count=state["slow"])
        if self._ever_succeeded:
            self.status.setText(f"computing… {text}")
        else:
            self._placeholder.set_progress(text)

    def _cancel_appd_pressed(self) -> None:
        """The placeholder's Cancel: a soft cancel; results of the cancelled press are dropped on arrival."""
        self._appd_run_id += 1
        self._appd_pending = 0
        self._appd.set_running(False)
        self._appd_status.hide()
        self._job_cancelled(self._jk("appd"))

    def _on_compute_done(self, gen: int, result, error) -> None:
        """The SQD worker's result, back on the GUI thread. Dropped outright
        if a LATER SQD Run press has since bumped `_sqd_gen` -- see
        `_run_pressed`. Never touches Apparent-D Distribution's own state --
        see the module docstring's "EACH SECTION RUNS ON ITS OWN".
        """
        if gen != self._sqd_gen:
            return
        self._sqd_clock.stop()
        if error is not None:
            self._show_sqd_message(f"Could not compute this view:\n{error}")
            self._job_finished(self._jk("sqd"), error=error)
            return
        self._last_sqd_run = getattr(self, "_sqd_pending", None)

        graduating = not self._ever_succeeded
        self._ever_succeeded = True
        plot_data, caption, payloads = result
        plot_data = self._grouped(plot_data)            # grouped by the factors, as the nesting says
        self._sqd_plot = plot_data
        from datetime import datetime

        self._sqd_computed_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._card.set_running(False)
        self._card.set_data(plot_data, self._column_colours(plot_data))
        self._apply_significance("sqd")
        self._sync_fit_warning("sqd")
        self.status.show()
        self.status.setText(caption)
        self._replicate_payloads = payloads
        self._sync_metadata()
        # A fresh Run invalidates whatever detail page was open -- its
        # payload belongs to the PREVIOUS Run's fit and would otherwise sit
        # on screen looking current. Back to the chart, same as leaving via
        # the sidebar (`_on_selected`).
        self._pages.setCurrentWidget(self._card)
        if graduating:
            self._fade_in(self._card)
        self.breadcrumb.set_trail(self.sidebar.trail(self.sidebar.current))
        self._job_finished(self._jk("sqd"))

    def _open_appd_dialog(self) -> None:
        """The section's Run: its three settings (Populations, Uncertainty, Shortest track), seeded from the last
        run. Every loaded condition is fitted."""
        if not self._all_condition_keys:
            self._show_appd_message("No conditions loaded.")
            return
        dialog = AppDRunDialog(self.t, conditions=list(self._all_condition_keys),
                               labels=dict(self._condition_labels), factors=dict(self._factors or {}),
                               factor_names=self._factor_names,
                               seed=self._appd.last_config, content=self.centralWidget(), parent=self)
        dialog.run.connect(self._bind("appd", lambda config: self._start_or_ask(
            self._jk("appd"), lambda: self._run_appd(config))))
        dialog.closed.connect(dialog.deleteLater)
        dialog.open()

    def _run_appd(self, config: dict) -> None:
        """One process-pool job per condition (`appd_view.appd_condition`, given `appd_fits.condition_settings`: only
        what moves that condition's numbers, so its cache key too), so each condition's cards fill in as it lands; the
        bubbles and the Per cell chart, which compare conditions, when the last one has -- under a set-wide Force same
        D across after one more job, the joint fit over conditions (`_on_appd_done`)."""
        from ...appd_fits import AppDRunError, condition_settings, with_precision

        conditions = list(self._all_condition_keys)
        if not conditions:
            return                      # pragma: no cover - the dialog is only offered with conditions
        try:
            config = with_precision(config, self.project, conditions)     # Per spot: one calibration for the Run
        except AppDRunError as exc:
            self._appd_status.setText(str(exc))
            self._appd_status.show()
            return
        project, process_pool = self.project, self._process_pool
        self._appd_run_id += 1
        run_id = self._appd_run_id
        self._appd_pending = self._appd_total = len(conditions)
        self._appd_failed = []
        self._appd.begin_run(conditions, dict(self._condition_labels),
                             self._condition_colours(conditions), config)
        self._job_started(self._jk("appd"))
        self._appd_status.setText(
            f"computing {len(conditions)} condition{'' if len(conditions) == 1 else 's'}\u2026")
        self._appd_status.show()
        self._appd_config = dict(config)
        for condition in conditions:
            self._appd_gen += 1
            gen = self._appd_gen
            self._appd_by_gen[gen] = (run_id, condition)
            settings = condition_settings(config, condition)

            def compute(condition=condition, settings=settings, cache=self._results_cache):
                return self._submit_bulk(cached_job, cache, "appd", appd_condition, project, condition,
                                           settings).result()

            self._pool.start(_ComputeRunnable(compute, gen, self._appd_signals))

    def _on_appd_done(self, gen: int, result, error) -> None:
        """One condition landed; a failure is reported in its own cards."""
        entry = self._appd_by_gen.pop(gen, None)
        if entry is None:
            return                      # pragma: no cover - defensive
        run_id, condition = entry
        if run_id != self._appd_run_id:
            return                      # a superseded or cancelled press: dropped by name
        if condition is None:
            # THE JOINT FIT over conditions (a set-wide Force same D across) landed: the bubbles now
            if error is not None:
                self._appd_failed.append(("the joint fit", error))
            self._finish_appd(None if error is not None else result)
            return
        if error is not None:
            self._appd_failed.append((condition, error))
            self._appd.set_error(condition, f"failed: {error}".replace("\n", " "))
        else:
            self._appd.set_result(condition, result)
        self._appd_pending = max(0, self._appd_pending - 1)
        if self._appd_pending:
            done = self._appd_total - self._appd_pending
            self._appd_status.setText(f"computing\u2026 {done} of {self._appd_total} done")
            return
        config = self._appd_config or self._appd.last_config
        from ...appd_fits import SET_SCOPES, joint_designs, joint_inputs

        if (config or {}).get("scope") in SET_SCOPES:
            # ONE D ACROSS CONDITIONS: the joint fit needs every condition's tracks, so it is one more pool job, made
            # from the results' own per-track values (`appd_fits.joint_inputs`)
            inputs = joint_inputs(self._appd.results(), self._appd.conditions())
            self._appd_gen += 1
            gen = self._appd_gen
            self._appd_by_gen[gen] = (run_id, None)
            self._appd_status.setText("computing\u2026 the fit with one D across conditions")

            def compute(inputs=inputs, config=dict(config)):
                return self._submit_bulk(joint_designs, inputs, config).result()

            self._pool.start(_ComputeRunnable(compute, gen, self._appd_signals))
            return
        self._finish_appd(None)

    def _finish_appd(self, joint) -> None:
        """Every condition landed (and the joint fit, ``joint``, under a set-wide Force same D across)."""
        self._appd.finish(joint)
        self._apply_significance("appd")
        self._appd_status.hide()
        self.breadcrumb.set_trail(self.sidebar.trail(self.sidebar.current))
        self._finish_per_condition(self._jk("appd"), self._appd_failed, self._appd_total)

    def _on_appd_track_requested(self, condition: str, uid: str) -> None:
        """A gallery track's Show: that track in Images & tracks, on its own cell."""
        self._open_track(condition, uid, "fast", self._show_appd_message)

    def _show_qc_message(self, text: str) -> None:
        """`_show_appd_message`'s own shape, for the QC section."""
        clean = text.replace("\n", " ")
        if not self._qc_ever_succeeded:
            self._qc_placeholder.set_running(False)
        self._qc_status.show()
        self._qc_status.setText(clean)
        self.breadcrumb.set_trail(self.sidebar.trail(self.sidebar.current))

    def _run_qc(self) -> None:
        """QC's Run: asks first when one is still computing, then `_start_qc`."""
        self._start_or_ask("qc", self._start_qc)

    def _start_qc(self) -> None:
        """QC's own Run — no settings dialog, every loaded condition, the
        same shape as the other per-section Runs but
        simpler: QC has no sibling chart to coordinate a first-run reveal
        with, and every quantity it computes is a cheap pandas
        aggregation, never a multistart fit — the async plumbing here is
        for UI responsiveness across many movies, not because any one
        step is slow.
        """
        spec = ComparisonSpec(conditions=list(self._all_condition_keys))
        if not spec.conditions:
            self._show_qc_message("No conditions loaded to compare.")
            return
        project = self.project

        def compute():
            return compute_qc_overview(project, spec)

        self._qc_gen += 1
        self._job_started("qc")
        if self._qc_ever_succeeded:
            self._qc_status.setText("computing…")
        else:
            self._qc_placeholder.set_running(True)
        self._pool.start(_ComputeRunnable(compute, self._qc_gen, self._qc_signals))

    def _cancel_qc_pressed(self) -> None:
        """`_cancel_appd_pressed`'s twin, for QC's own placeholder Cancel."""
        self._qc_gen += 1
        self._job_cancelled("qc")
        self._qc_placeholder.set_running(False)

    def _on_qc_only_done(self, gen: int, result, error) -> None:
        """The `_run_qc` worker's result. Dropped outright if a LATER QC
        Run has since bumped `_qc_gen`."""
        if gen != self._qc_gen:
            return
        if error is not None:
            self._show_qc_message(f"Could not compute this view:\n{error}")
            self._job_finished("qc", error=error)
            return

        qc_graduating = not self._qc_ever_succeeded
        self._qc_ever_succeeded = True
        self._show_qc_cards(result)
        # The condition count is already on screen — the Overview card's
        # own header badge and its rows — so this line only says anything
        # when a condition genuinely failed to compute; otherwise it stays
        # hidden rather than repeating a number that's already there.
        n_errors = sum(1 for v in self._qc_payloads.values() if "error" in v)
        if n_errors:
            self._qc_status.show()
            self._qc_status.setText(
                f"{n_errors} condition(s) could not be computed.")
        else:
            self._qc_status.hide()
        self._qc_pages.setCurrentWidget(self._qc_general)
        reopen, self._qc_reopen = self._qc_reopen, None
        if reopen in self._qc_payloads:
            self._on_qc_condition_clicked(reopen)
            return
        if qc_graduating:
            self._fade_in(self._qc_general.widget())      # the cards, not the scroll area around them
        self.breadcrumb.set_trail(self.sidebar.trail(self.sidebar.current))
        self._job_finished("qc")

    def _show_qc_cards(self, result) -> None:
        """The QC General page's cards from ``result``, in the set condition order (`qc_view.reordered`)."""
        from ...qc_view import reordered

        result = reordered(result, self._all_condition_keys)
        self._qc_payloads = result["per_condition"]
        self._qc_result = result                       # the whole of it, for a saved session
        colours = self._condition_colours(self._qc_payloads.keys())
        self._qc_overview_table.set_payload(self._qc_payloads, colours)
        self._qc_fair.set_payload(result["fair"], colours)
        # .get: a session saved before 24 Oct 2026 has no such part, and the card then stays hidden
        self._qc_replicates.set_payload(result.get("replicates"), colours)
        self._qc_cells.set_payload(result.get("cells"), colours)
        self._qc_nothing.set_payload(result["nothing"], colours)

    def _on_qc_condition_clicked(self, condition: str) -> None:
        """A General-page row was clicked — drill into `QCConditionDetail`
        for that condition, the same shape `_on_appd_condition_clicked`
        gives Apparent-D Distribution."""
        self._qc_detail_condition = condition
        self._qc_detail.set_payload(
            condition, self._qc_payloads.get(condition, {}),
            colour=self._condition_colours([condition]).get(condition))
        self._qc_pages.setCurrentWidget(self._qc_detail)
        self.breadcrumb.set_trail(
            self.sidebar.trail(self.sidebar.current) + [condition],
            dropdown_options=list(self._qc_payloads.keys()))

    def _run_fractions_checks(self, condition: str) -> None:
        """`ApparentDDetail.fractions_run_requested` -- see that class's
        own docstring and its module (`fractions_chart.py`)'s. ONE
        condition, off the GUI thread, the ordinary `_ComputeRunnable`
        path -- but NOT `self._appd_gen`-guarded (see `_fractions_signals`'s
        own comment): a second press for a DIFFERENT condition must not
        drop this one's result.

        Off the GUI PROCESS too, same as SQD's `_run_pressed` -- see
        `self._process_pool`'s own comment for why a thread is not enough.
        `compute_fractions_checks` is the SAME shape of risk as SQD's own
        fit: a real SQD fit (its own multistart loop), its cell
        bootstrap, and an HMM fit, all for one condition -- not measured
        to stutter yet, but the mechanism is identical, so it gets the
        same fix pre-emptively rather than waiting for a report.
        `condition` is a plain string, trivially picklable alongside
        `project` (already verified for the SQD path).
        """
        project = self.project
        self._fractions_gen += 1
        gen = self._fractions_gen
        self._fractions_condition_by_gen[gen] = condition
        process_pool = self._process_pool

        def compute():
            future = self._submit_bulk(compute_fractions_checks, project, condition)
            return future.result()

        self._appd_detail.set_fractions_running(True)
        self._pool.start(_ComputeRunnable(compute, gen, self._fractions_signals))

    def _on_fractions_checks_done(self, gen: int, result, error) -> None:
        """The `_run_fractions_checks` worker's result. `condition` is
        looked up by `gen` rather than unpacked from `result`, because on
        an ERROR `result` is `None` -- `compute()` never got to return
        anything -- and the condition is still needed to know whether
        THIS page still cares."""
        condition = self._fractions_condition_by_gen.pop(gen, None)
        if condition is None:
            return                      # pragma: no cover - defensive
        if error is not None:
            self._appd_detail.set_fractions_error(
                condition, f"Could not compute this:\n{error}".replace("\n", " "))
            return
        self._appd_detail.set_fractions_payload(condition, result)

    def _run_dynamics_evidence(self, condition: str) -> None:
        """`ApparentDDetail.dynamics_run_requested` -- a line-for-line
        mirror of `_run_fractions_checks` above (own gen/condition-by-gen,
        submitted through the SAME `self._process_pool`, off the GUI
        process for the same reason: an HMM fit + bootstrap, a saSPT
        spectrum and a clustering null, all for one condition, is the
        SAME shape of risk as SQD's own fit."""
        project = self.project
        self._dynamics_gen += 1
        gen = self._dynamics_gen
        self._dynamics_condition_by_gen[gen] = condition
        process_pool = self._process_pool

        def compute():
            future = self._submit_bulk(compute_dynamics_evidence, project, condition)
            return future.result()

        self._appd_detail.set_dynamics_running(True)
        self._pool.start(_ComputeRunnable(compute, gen, self._dynamics_signals))

    def _on_dynamics_evidence_done(self, gen: int, result, error) -> None:
        """The `_run_dynamics_evidence` worker's result -- mirrors
        `_on_fractions_checks_done` exactly."""
        condition = self._dynamics_condition_by_gen.pop(gen, None)
        if condition is None:
            return                      # pragma: no cover - defensive
        if error is not None:
            self._appd_detail.set_dynamics_error(
                condition, f"Could not compute this:\n{error}".replace("\n", " "))
            return
        self._appd_detail.set_dynamics_payload(condition, result)

    # -- the Spatial Map section ---------------------------------------------

    def _open_spatial_dialog(self) -> None:
        """The section's "Run analysis" -- opens the map's configuration dialog,
        told it is running for the whole grid.

        **One cell frame for every condition, measured over all of them
        together.** `measure_cell_frame` takes the list for that. Measuring
        each condition separately would be the more obvious thing and would
        quietly break the grid: the frame is part of `_settings_key`, which
        `_panel_limits` uses to decide what may share a colour scale, so
        per-condition frames would give every row its own scale and the one
        bar per column would stop being true. `compare_plots.draw_spatial`
        makes the same call, and the spread travelling with the median is what
        keeps it honest rather than silent.
        """
        conditions = list(self._all_condition_keys)
        if not conditions:
            self._spatial_status.setText("No conditions loaded to map.")
            self._spatial_status.show()
            return
        # THE PROJECT'S ONE PIXEL SIZE (and a note when its conditions' differ within 1 %): the
        # Bin size page's standard, "1 px (106 nm)", reads it (ruled 30 Oct 2026).
        pixel_nm, pixel_note = project_pixel(self.project)
        self.map_dialog.set_context(
            measured=measure_cell_frame(self.project, conditions),
            pixel_size_nm=pixel_nm, pixel_note=pixel_note,
            # No condition NAME: this run is every condition, and the
            # dialog's own readout says "in <condition>" when given one.
            condition="",
            census=self._map_census(conditions),
            # the conditions the Split page's histogram can show, by name
            conditions=[(c, self._condition_labels.get(c, c)) for c in conditions])
        self.map_dialog.open()

    def _map_this_fit(self, method: str) -> None:
        """A fitted section's "Map this fit" (`map_fit_source.MAP_THIS_FIT`), run with its page current: a NEW
        Spatial Map page named after this page's fit ("HMM map 1", `map_fit_source.map_page_name`), its settings a
        copy of the Spatial Map page in hand's with ``method`` chosen, "Use the same D as the ... fit" ticked and THIS
        page chosen for the D, opened. Nothing runs until Run is pressed there; closed without a Run, the page goes
        again (`_map_dialog_cancelled`) and the view it was made from comes back."""
        source = self._cur[method]
        back = self.sidebar.current
        name = map_fit_source.map_page_name(self._page_name(source),
                                            [p.name for p in self._subpage_set.pages("spatial")])
        slot = self._add_subpage("spatial")
        self._subpage_set.rename("spatial", slot.number, name)
        slot.copied_from = ""                   # its name says where it comes from; no "a copy of" note
        slot.made_for_fit = back                # until its first Run: closed without one, it goes again
        self._sync_subpages("spatial")
        with self._in(slot):
            self.map_dialog.preset_fit(method, source.key)
            self._open_spatial_dialog()

    def _map_page_kept(self) -> None:
        """A Run on a page "Map this fit" made: it stays, whatever its dialog does later."""
        self._cur["spatial"].made_for_fit = None

    def _map_dialog_cancelled(self) -> None:
        """The map dialog closed without a Run. A page "Map this fit" made and never run goes again, once the dialog
        has finished closing, and the view it was made from comes back."""
        slot = self._cur["spatial"]
        back = getattr(slot, "made_for_fit", None)
        if not back:
            return
        from qtpy.QtCore import QTimer

        QTimer.singleShot(0, lambda: self._drop_unrun_map(slot, back))

    def _drop_unrun_map(self, slot, back: str) -> None:
        if slot not in self._slots["spatial"] or not getattr(slot, "made_for_fit", None):
            return
        self._remove_slot(slot)
        self._subpage_set.delete("spatial", slot.number)
        self._sync_subpages("spatial")
        try:
            self.sidebar.select(back)
        except KeyError:                    # pragma: no cover - the view it came from is gone
            self.sidebar.select("spatial.overview")

    def _on_map_preview_wanted(self, request: dict) -> None:
        """The map dialog's Split page wants one condition's histogram (`spatial_map_view.split_preview`): computed in
        the process pool, off the GUI thread, and handed back with `SpatialMapDialog.set_preview`."""
        project, process_pool = self.project, self._process_pool
        request = dict(request)
        self._map_preview_gen += 1
        gen = self._map_preview_gen
        self._map_preview_by_gen[gen] = request

        page_key = request.get("fit_page") if request["method"] in self._fit_methods() else None
        gradient = self._sqd_gradient(page_key if request["method"] == "sqd" else None)
        # THE BOX TICKED: the histogram's populations at the section fit's D, where the section has a fit of this
        # condition (its last finished Run; none, and the page draws the map's own as before)
        fit = None
        if request.get("same_D"):
            source = self._section_source(request["method"], page_key)
            fit = ((source or {}).get("conditions") or {}).get(str(request["condition"]))
            if fit is not None and map_fit_source.refusal(request["method"], fit):
                fit = None

        def compute(request=request, fit=fit):
            future = process_pool.submit(split_preview, project, request["condition"], method=request["method"],
                                         n_pop=request["n_pop"], lag_frames=request["lag_frames"],
                                         weighting=request["weighting"], gradient=gradient, section_fit=fit)
            return future.result()

        self._pool.start(_ComputeRunnable(compute, gen, self._map_preview_signals))

    def _on_map_preview_done(self, gen: int, result, error) -> None:
        request = self._map_preview_by_gen.pop(gen, None)
        if request is None:
            return                      # pragma: no cover - defensive
        self.map_dialog.set_preview(request, result, None if error is None else str(error).split("\n")[0])

    def _map_census(self, conditions) -> list:
        """The track lengths Map Configuration's cutoff chart and cost line show.

        From a finished map when there is one (`track_census` of exactly the
        table it drew from); otherwise from the canonical table, which is cheap
        -- the dialog must open without projecting, the cost it configures --
        and holds the same tracks before projection drops the few rows a cell
        without a centreline cannot place. One list over every condition the
        run covers, so the chart shows what the cutoff will act on.
        """
        drawn = self._spatial.matrix.payloads()
        census: list = []
        for condition in conditions:
            payload = drawn.get(condition)
            if payload and payload.get("track_census") is not None:
                census.extend(payload["track_census"])
            else:
                census.extend(track_census(self.project.condition(condition)))
        return census

    def _run_spatial_section(self) -> None:
        """Run the map for EVERY loaded condition, one job per condition.

        **One runnable per condition rather than one for all of them**, so the
        grid fills in as results land instead of staying blank for the length
        of the slowest. Each is the same `compute_spatial_map` the drill-in
        submits; nothing new is computed here, it is dispatched more than once.

        Every job of one press carries that press's `_spatial_run_id`. A
        second press bumps it, so a result from the first can still arrive and
        be DROPPED BY NAME rather than landing in a grid drawn at different
        settings -- the pooled-scale failure `_settings_key` exists to stop.
        """
        self._start_or_ask(self._jk("spatial"), self._start_spatial_section)

    # -- the D a fitted map takes from its section (4 Nov 2026, `map_fit_source`) ---------------------------------- #

    def _fit_methods(self) -> list:
        """The map methods whose section has pages (the sections a map can take its D from, `map_fit_source`)."""
        return [m for m in map_fit_source.SECTION_KEYS if m in self._slots]

    def _sqd_gradient(self, page_key=None) -> str:
        """The Optimiser of the SQD page ``page_key`` (the section's own by default; a page that is gone: the
        section's own): the fits inside the maps read it as Jump distances does."""
        slot = self._slot_of_key(page_key or "sqd") or self._slot("sqd")
        return slot.compute_dialog.compute.values().get("gradient", "numeric")

    def _section_source(self, method: str, page_key: str | None = None):
        """``method``'s section's last FINISHED Run as `map_fit_source` reads it (its D per condition), or None when
        it has none. A Run still computing is not read: the map waits for it (`map_fit_source.plan`). SQD has
        pages (``page_key``: ``"sqd"``, ``"sqd~2"``; the section's own by default): the map takes the one it was
        told to."""
        if method == "sqd":
            slot = self._slot_of_key(page_key or "sqd")
            if slot is None:
                return None
            run, plot = getattr(slot, "_last_sqd_run", None), getattr(slot, "_sqd_plot", None)
            if run is None or plot is None:
                return None
            return map_fit_source.sqd_source(plot, getattr(slot, "_replicate_payloads", {}) or {}, run[0], run[1],
                                             page=self._page_name(slot), page_key=slot.key)
        if method == "appd":
            slot = self._slot_of_key(page_key or "appd")
            if slot is None or not slot._appd.has_result() or not slot._appd.results():
                return None
            return map_fit_source.appd_source(slot._appd.results(), slot._appd.joint(),
                                              dict(slot._appd.last_config or {}),
                                              page=self._page_name(slot), page_key=slot.key)
        if method == "hmm":
            slot = self._slot_of_key(page_key or "hmm")
            if slot is None or not slot._hmm.has_result():
                return None
            return map_fit_source.hmm_source({k: card.result() for k, card in slot._hmm.cards().items()},
                                             dict(slot._hmm.last_config or {}),
                                             page=self._page_name(slot), page_key=slot.key)
        return None

    def _start_fit_section(self, method: str, page_key: str | None = None) -> bool:
        """Start ``method``'s section Run with its CURRENT settings, as its dialog shows them (the map's box ticked
        and the section never run): SQD's Compute dialog (of the page the map takes D from), the apparent-D and HMM
        sections' last config (their standards before any). True when it is now running; where it refused, its own
        section says why."""
        key = map_fit_source.SECTION_KEYS[method]
        if method == "sqd":
            slot = self._slot_of_key(page_key or "sqd")
            if slot is None:
                return False
            self._call_in(slot, self._run_pressed)
            key = slot.key
        elif method == "appd":
            from ...appd_fits import AppDRunError, run_config

            slot = self._slot_of_key(page_key or "appd")
            if slot is None:
                return False
            conditions, factors = list(self._all_condition_keys), dict(self._factors or {})
            try:
                config = run_config(slot._appd.last_config or {}, conditions=conditions, factors=factors)
            except AppDRunError:
                config = run_config({}, conditions=conditions, factors=factors)
            self._call_in(slot, self._start_or_ask, slot.key, lambda: self._run_appd(config))
            key = slot.key
        elif method == "hmm":
            slot = self._slot_of_key(page_key or "hmm")
            if slot is None:
                return False
            self._call_in(slot, self._request_hmm, dict(slot._hmm.last_config or {}))
            key = slot.key
        return key in self._running

    def _start_spatial_section(self, config: dict | None = None) -> None:
        """The map Run itself -- see `_run_spatial_section`. ``config``: the settings to run at (the notice's Run:
        the maps on screen again), else the map dialog's.

        THE D (`map_fit_source.plan`): with the dialog's box "Use the same D as the ... fit" ticked, the maps take
        that section's last Run -- WAITING for it while it computes, STARTING it (its current settings) when it has
        never run -- and a condition it has no usable fit for is refused in its own row; unticked, every map fits
        its own, as before."""
        conditions = list(self._all_condition_keys)
        if not conditions:
            return                      # pragma: no cover - guarded at open
        # THE SQD SECTION'S OPTIMISER for every SQD fit inside the maps, as Jump distances reads it (3 Nov 2026: a
        # Best (BIC) map ran at the numeric one, about fourfold slower: "crazy that bic spatial map runs for 30 min")
        config = dict(config if config is not None else self.map_dialog.config())
        method = config.get("method", "sqd")
        # WHICH PAGE of the section the D comes from (subpages): SQD's drop-down; a page that is gone is said, not
        # replaced in silence
        page_key = map_fit_source.fit_page_key(config, method) if method in self._fit_methods() else None
        config["gradient"] = self._sqd_gradient(page_key if method == "sqd" else None)
        self._spatial.set_notice(None)
        self._spatial_waiting = None
        section = map_fit_source.SECTION_KEYS.get(method)
        if page_key is not None:
            slot = self._slot_of_key(page_key)
            if slot is None and map_fit_source.uses_section(config):
                self._spatial_status.setText(map_fit_source.gone_text(
                    method, config.get("fit_page_name") or page_key))
                self._spatial_status.show()
                return
            if slot is not None:
                section = slot.key
                config["fit_page_name"] = self._page_name(slot)
        plan = map_fit_source.plan(config, conditions=conditions, source=self._section_source(method, page_key),
                                   running=section in self._running if section else False)
        if plan["action"] in ("wait", "start"):
            self._spatial_run_id += 1          # anything still in flight from an earlier map Run is dropped
            started = plan["action"] == "wait" or self._start_fit_section(method, page_key)
            if not started:
                self._spatial_status.setText(map_fit_source.not_finished_text(method, config.get("fit_page_name")))
                self._spatial_status.show()
                return
            self._spatial_waiting = {"section": section, "method": method, "config": dict(config),
                                     "run_id": self._spatial_run_id, "page_key": page_key}
            self._spatial_pending = self._spatial_total = len(conditions)
            self._spatial_failed = []
            self._spatial.set_running(True)
            self._job_started(self._jk("spatial"))
            self._spatial_status.setText(plan["text"])
            self._spatial_status.show()
            return
        self._draw_spatial_maps(config, plan.get("fits") or {}, plan.get("refused") or {})

    def _map_section_ended(self, key: str, ok: bool) -> None:
        """A section's Run ended (``key``: its job key, a page's for SQD). EVERY Spatial Map page hears of it: each
        decides for itself whether it was waiting for that fit (`_map_section_ended_on_page`)."""
        for _ in self._each("spatial"):
            self._map_section_ended_on_page(key, ok)

    def _map_section_ended_on_page(self, key: str, ok: bool) -> None:
        """A section's Run ended (``ok``: it finished; else it failed or was stopped). A map waiting for it draws
        now, or says why it cannot; maps drawn from an EARLIER fit of it say so with a Run (`map_fit_source`), never
        redrawn in silence."""
        waiting = self._spatial_waiting
        if waiting is not None and waiting["section"] == key:
            if waiting["run_id"] != self._spatial_run_id:
                self._spatial_waiting = None
                return                  # cancelled or replaced while it waited
            if not ok:
                if key in self._running:
                    return              # stopped for a new Run of it (`_replace_job`): wait for that one
                self._spatial_waiting = None
                self._spatial_pending = 0
                self._spatial.set_running(False)
                self._job_cancelled(self._jk("spatial"))
                self._spatial_status.setText(map_fit_source.not_finished_text(
                    waiting["method"], waiting["config"].get("fit_page_name")))
                self._spatial_status.show()
                return
            self._spatial_waiting = None
            config = waiting["config"]
            plan = map_fit_source.plan(config, conditions=list(self._all_condition_keys),
                                       source=self._section_source(waiting["method"], waiting.get("page_key")),
                                       running=False)
            if plan["action"] != "draw":
                self._spatial_pending = 0
                self._spatial.set_running(False)
                self._job_cancelled(self._jk("spatial"))
                self._spatial_status.setText(map_fit_source.not_finished_text(
                    waiting["method"], config.get("fit_page_name")))
                self._spatial_status.show()
                return
            self._draw_spatial_maps(config, plan["fits"], plan["refused"], waited=True)
            return
        if not ok:
            return
        method = next((m for m, k in map_fit_source.SECTION_KEYS.items() if k == sp.split_job(key)[0]), None)
        source = self._section_source(method, key if method in self._fit_methods() else None)
        stale = map_fit_source.stale_conditions(self._spatial.matrix.payloads(), source)
        if stale:
            self._spatial.set_notice(map_fit_source.stale_text(method, stale, self._condition_labels,
                                                               page=(source or {}).get("page")))

    def _rerun_spatial_maps(self) -> None:
        """The notice's Run: the maps on screen again at their own settings, taking the section's D anew."""
        config = {k: v for k, v in (getattr(self, "_spatial_last_config", None) or {}).items() if k != "gradient"}
        if not config:
            self._open_spatial_dialog()
            return
        method = config.get("method")
        if method in self._fit_methods() and self._slot_of_key(map_fit_source.fit_page_key(config, method)) is None:
            # THE PAGE THE D CAME FROM IS GONE: the maps' Run takes the section's own page, and the dialog follows
            config["fit_page"] = method
            config.pop("fit_page_name", None)
            self.map_dialog.set_fit_page(method, method)
        self._start_or_ask(self._jk("spatial"), lambda: self._start_spatial_section(config))

    def _draw_spatial_maps(self, config: dict, fits: dict, refused: dict, *, waited: bool = False) -> None:
        """Dispatch the map Run: one job per condition at its own config (`map_fit_source.job_config`: its count,
        or the section's fit ``fits[condition]``); a condition in ``refused`` is said in its own row. ``waited``: the
        Run already counts as started (it waited for its section)."""
        conditions = list(self._all_condition_keys)
        project = self.project
        # THE MAP'S OWN FIT PER SPOT (6 Nov 2026, `spatial_map_view.OWN_PRECISION_KEY`): the Run's calibration for
        # each condition whose tracks carry precision, none for one that carries none; decided before anything starts
        from ...spatial_map_view import OWN_PRECISION_KEY
        from ...workspace_view import RunSettingsError

        drawn = [c for c in conditions if c not in refused]
        try:
            per_spot = self._run_precision_for(drawn, bool(config.get(OWN_PRECISION_KEY)) and bool(drawn))
        except RunSettingsError as exc:
            self._spatial_status.setText(str(exc))
            self._spatial_status.show()
            return
        # what a redraw after hiding a track repeats: this Run's settings, not whatever the dialog holds later
        self._spatial_last_config = dict(config)

        if not waited:
            self._spatial_run_id += 1
        run_id = self._spatial_run_id
        jobs = [c for c in conditions if c not in refused]
        self._spatial_pending = len(jobs)
        self._spatial_total = len(conditions)
        self._spatial_failed: list = []
        # each condition's own count (`condition_config`); the grid's columns cover every fixed one, and a Best
        # condition's join when its maps land. A section's fit gives each condition its count.
        planning = (dict(config, n_pop_by_condition={**(config.get("n_pop_by_condition") or {}),
                                                     **{c: int(f["n_pop"]) for c, f in fits.items()}})
                    if fits else config)
        counts = planned_counts(planning, conditions)
        self._spatial.begin_run(conditions, dict(self._condition_labels),
                                method=config.get("method", "sqd"),
                                n_pop=counts[0] if counts else 2,
                                colours=self._condition_colours(conditions),
                                ranges=config.get("ranges"), counts=counts,
                                planned=planned_n_pops(planning, conditions))
        for condition, why in refused.items():
            self._spatial.set_error(condition, f"not drawn: {why}")
            self._spatial_failed.append((condition, why))
        # `begin_run` has left any replicate-day page; the trail follows. The Visibility
        # popover follows the run's METHOD too: its rows are that method's maps and its
        # ramp that method's own (v2's for SDA), not the previous run's.
        if self.sidebar.current_base.partition(".")[0] == "spatial":
            self.breadcrumb.set_trail(self.sidebar.trail(self.sidebar.current))
            self._sync_spatial_visibility_section()
        if waited:
            self._sync_jobs()
        else:
            self._job_started(self._jk("spatial"))
        if not jobs:
            # every condition refused (the section fit has none it can be drawn at): said in each row, nothing runs
            self._spatial.set_running(False)
            self._spatial_status.hide()
            self._sync_metadata()
            self._finish_per_condition(self._jk("spatial"), self._spatial_failed, self._spatial_total)
            return
        self._spatial_status.setText(
            f"computing {len(jobs)} condition"
            f"{'' if len(jobs) == 1 else 's'}…")
        self._spatial_status.show()

        for condition in jobs:
            self._spatial_gen += 1
            gen = self._spatial_gen
            self._spatial_by_gen[gen] = (run_id, condition)
            # its count, or the section's fit (`map_fit_source.job_config`): every setting it is handed is in the
            # result cache's key, the section fit's D included
            one = map_fit_source.job_config(config, condition, fits.get(condition))
            if per_spot.get(condition) is not None:
                one["precision"] = list(per_spot[condition])

            def compute(condition=condition, hidden=sorted(self._map_hidden.get(condition, ())),
                        cache=self._results_cache, one=one):
                future = self._submit_bulk(cached_job, cache, "spatial", compute_spatial_map, project, condition,
                                             {"hidden": hidden, **one})
                return future.result()

            self._pool.start(
                _ComputeRunnable(compute, gen, self._spatial_signals))

    def _on_spatial_section_done(self, gen: int, result, error) -> None:
        """One condition of a section Run landed.

        A failure is reported IN ITS OWN ROW and the rest of the run carries
        on -- `catalogue_view`'s rule, that a condition which raises is kept
        and named rather than left as an absence, because a missing panel
        reads as "no data" where the truth is "this one failed".
        """
        entry = self._spatial_by_gen.pop(gen, None)
        if entry is None:
            return                      # pragma: no cover - defensive
        run_id, condition = entry
        if run_id != self._spatial_run_id:
            # A superseded press. Dropped by name rather than drawn: its
            # settings are not this grid's.
            return
        if error is not None:
            self._spatial.set_error(
                condition, f"failed: {error}".replace("\n", " "))
            self._spatial_failed.append((condition, error))
        else:
            columns, drawn = self._spatial.matrix.kinds(), self._spatial.has_result()
            self._spatial.set_payload(condition, result)
            # a Best row's count, known only now, may add a column, and the first map drawn gives the list its rows:
            # the Visibility list follows both
            if ((self._spatial.matrix.kinds() != columns or self._spatial.has_result() != drawn)
                    and self.sidebar.current_base.partition(".")[0] == "spatial"):
                self._sync_spatial_visibility_section()
        self._spatial_pending = max(0, self._spatial_pending - 1)
        if self._spatial_pending:
            done = self._spatial_total - self._spatial_pending
            self._spatial_status.setText(
                f"computing… {done} of {self._spatial_total} done")
            return
        self._spatial.set_running(False)
        from datetime import datetime
        self._spatial_computed_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._sync_metadata()
        self._finish_per_condition(self._jk("spatial"), self._spatial_failed, self._spatial_total)
        # NOTHING on finishing. "2 conditions mapped" restated what the grid
        # already shows -- two rows, both drawn -- and a status line that is
        # only ever true is not a status. It carries the COUNTDOWN, which the
        # grid cannot show, and then gets out of the way.
        self._spatial_status.hide()

    def _sync_metadata(self) -> None:
        """Hand the toolbar's Metadata tool the receipt for what is on screen.

        Only the Spatial Map prints one in this build (ruled: "Spatial Map now"); the
        grid it reads is the page on screen -- the overview, or the day page -- since
        the two carry different rows (the day page adds the shared-fit line and scales
        over days). Every other section says it has no metadata yet."""
        from ...metadata_view import NO_SECTION_TEXT, NOT_RUN_TEXT

        sidebar = getattr(self, "sidebar", None)
        section = sidebar.current_base.partition(".")[0] if sidebar is not None else ""
        if section in self._slots and sidebar.current_page != self._cur[section].key:
            return                      # a result of a page that is not the one on show: its receipt is not wanted
        if section == "sqd":
            # THE SQD RECEIPT: the settings and every replicate day's fractions (15 Oct 2026)
            from ...bubble_view import sqd_receipt

            run = getattr(self, "_last_sqd_run", None)
            receipt = (sqd_receipt(getattr(self, "_sqd_plot", None), getattr(self, "_replicate_payloads", {}),
                                   run[0], run[1], computed_at=getattr(self, "_sqd_computed_at", None))
                       if run is not None else None)
            self.toolbar.metadata.set_receipt(receipt, "Run SQD to print its receipt.")
            return
        if section != "spatial":
            self.toolbar.metadata.set_receipt(None, NO_SECTION_TEXT)
            return
        matrix = (self._spatial.days.matrix if self._spatial.showing_days()
                  else self._spatial.matrix)
        receipt = matrix.receipt(computed_at=self._spatial_computed_at)
        self.toolbar.metadata.set_receipt(receipt, NOT_RUN_TEXT)

    def _cancel_spatial_pressed(self) -> None:
        """The placeholder's Cancel. A SOFT cancel, the same one both other
        sections offer: the run id is bumped so every in-flight result is
        dropped on arrival, and the work itself is left to finish rather than
        killed -- see `_cancel_sqd_pressed`. A map waiting for its section's fit stops waiting; the section's own
        Run goes on (it is that section's)."""
        self._spatial_run_id += 1
        self._spatial_waiting = None
        self._spatial_pending = 0
        self._spatial.set_running(False)
        self._job_cancelled(self._jk("spatial"))
        self._spatial_status.hide()

    # -- Kinetics ------------------------------------------------------------- #

    def _open_kinetics_dialog(self) -> None:
        """The section's Run: the configuration dialog, built fresh and seeded from the
        last committed settings so a re-run does not retype them. Nothing runs until the
        dialog's own Run."""
        conditions = [(key, self._condition_labels.get(key, key))
                      for key in self._all_condition_keys]
        if not conditions:
            self._kinetics_status.setText("No conditions loaded.")
            self._kinetics_status.show()
            return
        # each condition's replicate days, for the times of the addition (`induction_view.condition_days`)
        days = {key: condition_days(self.project, key) for key, _name in conditions}
        # the per-sample table of the design "Added to each sample": each condition's movies, and what was saved
        # beside its day map (READ here; written only by a Run, `_start_kinetics`)
        movies = {key: condition_movies(self.project, key) for key, _name in conditions}
        saved, saved_times = {}, {}
        for key, _name in conditions:
            try:
                table = load_additions(self.project.roots[key])
            except ValueError as exc:
                self._kinetics_status.setText(str(exc))
                self._kinetics_status.show()
                return
            if table is not None:
                saved[key] = table
            try:
                culture = load_culture_times(self.project.roots[key])
            except ValueError as exc:
                self._kinetics_status.setText(str(exc))
                self._kinetics_status.show()
                return
            if culture is not None:
                saved_times[key] = culture
        dialog = KineticsConfigDialog(
            self.t, conditions, seed=self._kinetics.last_config,
            content=self.centralWidget(), parent=self, days=days, movies=movies, saved=saved,
            saved_times=saved_times, found_reference=find_reference(self.project.roots.values()),
            extents_for=lambda key, camera: movie_extents(self.project, key, camera=camera),
            mobility_for=lambda key: movie_mobility(self.project, key))
        dialog.run.connect(self._bind("kinetics", self._run_kinetics))
        dialog.closed.connect(dialog.deleteLater)
        dialog.open()

    def _run_kinetics(self, config: dict) -> None:
        """The dialog's Run: asks first when Kinetics is still computing."""
        self._start_or_ask(self._jk("kinetics"), lambda: self._start_kinetics(config))

    def _start_kinetics(self, config: dict) -> None:
        """One job per included condition, on the process pool, so each card fills in as
        its result lands. The HMM fit is what makes it slow enough to belong there."""
        frames = config["conditions"]
        conditions = [key for key in self._all_condition_keys if key in frames]
        if not conditions:
            return                      # pragma: no cover - the dialog refuses an empty run
        # PER SPOT (6 Nov 2026): the populations readout with each step's own error, the Run's calibration; a
        # condition whose tracks carry none fitted without it. Decided before anything starts.
        from ...workspace_view import RunSettingsError

        try:
            per_condition = self._run_precision_for(conditions, bool(config.get("precision")))
        except RunSettingsError as exc:
            self._kinetics_status.setText(str(exc))
            self._kinetics_status.show()
            return
        project = self.project
        populations = config.get("readout") in POPULATION_READOUTS
        mobility = config.get("readout") == "mobility"
        # THE PER-SAMPLE TABLE is saved beside each condition's day map on a Run, never on opening the dialog; a
        # table that cannot be written is refused by name before anything starts
        if populations and config.get("clock") == "sample":
            try:
                for condition in conditions:
                    save_additions(project.roots[condition], config["samples"][condition])
            except (OSError, ValueError) as exc:
                self._kinetics_status.setText(f"The addition times could not be saved beside the condition: {exc}")
                self._kinetics_status.show()
                return
        # THE ADDITION TIMES typed per replicate are remembered the same way (`save_culture_times`), on a Run only
        if populations and config.get("clock") == "addition":
            try:
                for condition in conditions:
                    save_culture_times(project.roots[condition], config["additions"][condition])
            except (OSError, ValueError) as exc:
                self._kinetics_status.setText(f"The addition times could not be saved beside the condition: {exc}")
                self._kinetics_status.show()
                return
        # THE REFERENCE FILE is read now, on this thread, and refused by name before anything runs (a file that does
        # not read must not cost a Run); each condition takes its own rows (`induction_view.reference_curves`)
        references: dict = {}
        if config.get("readout") == "populations" and config.get("reference"):
            try:
                references = {c: reference_curves(config["reference"], c) for c in conditions}
            except ValueError as exc:
                self._kinetics_status.setText(str(exc))
                self._kinetics_status.show()
                return
        # A MOCK ADDITION (8 Nov 2026, `kinetics_mock_view`): a ticked condition whose addition is a mock; every other
        # ticked condition is read against it after its own model. Each condition's own clock goes with the job.
        mock = config.get("mock") if populations and config.get("readout") in MOCK_READOUTS else None
        if mock and (mock not in conditions or len(conditions) < 2):
            self._kinetics_status.setText(f"The mock addition {mock} must be ticked, with another condition to "
                                          "compare with it.")
            self._kinetics_status.show()
            return

        def clock_of(key: str) -> dict:
            """What `induction_view.clock_steps_for` needs of one condition: the clock the Run set, its per-spot noise."""
            clock = {"trigger_frame": frames[key]}
            if config.get("clock") == "addition":
                clock.update(trigger_frame=None, additions=config["additions"][key])
            if config.get("clock") == "sample":
                clock.update(trigger_frame=None, samples=config["samples"][key])
            if per_condition.get(key) is not None:
                clock["precision"] = per_condition[key]
            return clock

        self._kinetics_cancel.set()                     # a Run still going for an older press stops
        self._kinetics_cancel = cancel = threading.Event()
        self._kinetics_run_id += 1
        run_id = self._kinetics_run_id
        self._kinetics_pending = self._kinetics_total = len(conditions)
        self._kinetics_failed: list = []
        self._kinetics_work = {c: [0, 1] for c in conditions}
        self._kinetics.begin_run(
            conditions, dict(self._condition_labels),
            self._condition_colours(conditions), config)
        self._job_started(self._jk("kinetics"))
        self._kinetics_status.setText(
            f"computing {len(conditions)} condition{'' if len(conditions) == 1 else 's'}…")
        self._kinetics_status.show()
        for condition in conditions:
            self._kinetics_gen += 1
            gen = self._kinetics_gen
            self._kinetics_by_gen[gen] = (run_id, condition)
            if populations:
                # the D's and shares over time (`induction_view`): one model per condition, on the pool
                function = compute_induction
                settings = {"trigger_frame": frames[condition], "unit": config["unit"],
                            "bin_seconds": config["bin_seconds"], "n_pop": config["n_pop"],
                            "spread": config["spread"], "permutation": bool(config.get("permutation"))}
                if mobility:                            # ONE population, a change in D only (`MOBILITY_CHANGES`)
                    settings["changes"] = MOBILITY_CHANGES
                if config.get("clock") == "addition":   # each movie's own time since its day's addition
                    settings.update(trigger_frame=None, additions=config["additions"][condition])
                if config.get("clock") == "sample":     # each movie's own time since its sample's addition
                    settings.update(trigger_frame=None, samples=config["samples"][condition])
                if config.get("design"):                # an old config names none: `kinetics_designs.design_of`
                    settings["design"] = config["design"]
                if per_condition.get(condition) is not None:
                    settings["precision"] = per_condition[condition]
            else:
                function = compute_kinetics
                settings = {"trigger_frame": frames[condition], "unit": config["unit"],
                            "bin_seconds": config["bin_seconds"], "align": config["align"]}

            against_mock = None
            if mock and condition != mock:
                against_mock = {"conditions": [condition, mock],
                                "settings": {"readout": MOCK_READOUTS[config["readout"]],
                                             "treated_clock": clock_of(condition), "mock_clock": clock_of(mock),
                                             "after_from_s": config.get("mock_after_s"),
                                             # the bound share at the Run's own count (review, 8 Nov 2026)
                                             "n_pop": int(config.get("n_pop") or 2)
                                             if MOCK_READOUTS[config["readout"]] == "bound_share" else 2}}

            def compute(condition=condition, settings=settings, function=function, gen=gen,
                        cache=self._results_cache, reference=references.get(condition),
                        signals=self._kinetics_signals, permutation=populations and bool(config.get("permutation")),
                        against_mock=against_mock, mock_chosen=bool(mock)):
                return self._compute_kinetics_condition(project, condition, function, settings, cache, cancel,
                                                        reference, permutation, signals, gen, against_mock,
                                                        mock_chosen)

            self._pool.start(_ComputeRunnable(compute, gen, self._kinetics_signals))
        self._sync_jobs()

    def _compute_kinetics_condition(self, project, condition, function, settings, cache, cancel, reference,
                                    permutation, signals, gen, against_mock=None, mock_chosen=False):
        """One condition's whole job, on a job thread: the model through the section cache (one pool job), then the
        permutation test's chunks (each a pool job through the cache, each landing as progress), then the change against
        a mock addition (``against_mock`` ``{"conditions": [this, the mock], "settings": ...}``: one more pool job
        through the section cache, `kinetics_mock_view.compute_mock`; a refusal is the card's words, never a failed
        Run), then the reference. Says how many jobs there are and how many have landed (`_on_kinetics_progress`). A set ``cancel`` stops the
        submitting and the chunks not started (`induction.run_permutation_chunks`)."""
        from functools import partial

        def said(done: int, total: int) -> None:
            try:
                signals.progress.emit(gen, int(done), int(total), {"condition": condition})
            except RuntimeError:
                pass                                    # the window went away first: nothing is left to tell

        result = self._submit_bulk_unless(cancel, cached_job, cache, "kinetics", function, project, condition,
                                          settings).result()
        if "error" in result:
            return result
        if permutation:
            chunks_said = lambda done, total: said(1 + done, 1 + total)           # noqa: E731
            result = run_permutation(project, condition, result, submit=partial(self._submit_bulk_unless, cancel),
                                     cache=cache, progress=chunks_said, cancelled=cancel)
        if against_mock is not None:
            block = self._submit_bulk_unless(
                cancel, cached_group_job, cache, "kinetics", compute_mock, project, against_mock["conditions"],
                dict(against_mock["settings"], response=result.get("response"))).result()
            result = attach_mock(result, block)
        elif not mock_chosen:                       # no mock in this Run: a drift over the session reads as a response
            result = attach_no_mock(result)
        if reference is not None:
            try:
                result = attach_reference(result, reference)
            except ValueError as exc:
                return {"error": f"ValueError: {exc}"}
        return result

    def _on_kinetics_done(self, gen: int, result, error) -> None:
        """One condition landed. A failure is reported IN ITS OWN CARD and the rest of
        the run carries on: a condition that raises is kept and named, never an absence."""
        entry = self._kinetics_by_gen.pop(gen, None)
        if entry is None:
            return                      # pragma: no cover - defensive
        run_id, condition = entry
        if run_id != self._kinetics_run_id:
            return                      # a superseded press: dropped by name
        if condition in self._kinetics_work:
            self._kinetics_work[condition][0] = self._kinetics_work[condition][1]      # however it ended, it is over
        if error is not None:
            self._kinetics.set_error(condition, f"failed: {error}".replace("\n", " "))
            self._kinetics_failed.append((condition, error))
        else:
            self._kinetics.set_payload(condition, result)
        self._kinetics_pending = max(0, self._kinetics_pending - 1)
        self._sync_jobs()
        if self._kinetics_pending:
            done = self._kinetics_total - self._kinetics_pending
            self._kinetics_status.setText(f"computing… {done} of {self._kinetics_total} done")
            return
        self._kinetics.set_running(False)
        self._finish_per_condition(self._jk("kinetics"), self._kinetics_failed, self._kinetics_total)
        self._kinetics_status.hide()

    def _on_kinetics_progress(self, gen: int, done: int, total: int, payload) -> None:
        """A condition's job said how many of its jobs have landed (the model, then the permutation test's chunks):
        the Running list's count and the status line. A superseded or cancelled Run's word is dropped by name."""
        entry = self._kinetics_by_gen.get(gen)
        if entry is None or entry[0] != self._kinetics_run_id or entry[1] not in self._kinetics_work:
            return
        self._kinetics_work[entry[1]] = [min(int(done), int(total)), int(total)]
        left = sum(1 for d, t in self._kinetics_work.values() if d < t)
        self._kinetics_status.setText(f"computing… {sum(d for d, _t in self._kinetics_work.values())} of "
                                      f"{sum(t for _d, t in self._kinetics_work.values())} jobs done"
                                      f" ({left} condition{'' if left == 1 else 's'} still running)")
        self._sync_jobs()

    def _cancel_kinetics_pressed(self) -> None:
        """The placeholder's Cancel: stops the Run for real -- no job is submitted any more, the permutation chunks
        not yet started are dropped (`_kinetics_cancel`), and a job already on a worker (a chunk is a few seconds, the
        model's fit a minute or two) ends unread. Nothing the older press finishes reaches a card (`_kinetics_run_id`)."""
        self._kinetics_cancel.set()
        self._kinetics_run_id += 1
        self._kinetics_pending = 0
        self._kinetics_work = {}
        self._kinetics.set_running(False)
        self._job_cancelled(self._jk("kinetics"))
        self._kinetics_status.hide()

    # -- MSD ------------------------------------------------------------------ #

    def _open_msd_dialog(self) -> None:
        """The section's Run: its five settings (`msd_view.SECTION_HELP`), seeded from the last run. Every loaded
        condition is fitted (the MSD section's rebuild, 16 Oct 2026)."""
        if not self._all_condition_keys:
            self._msd_status.setText("No conditions loaded.")
            self._msd_status.show()
            return
        dialog = MSDRunDialog(self.t, seed=self._msd.last_config, content=self.centralWidget(), parent=self)
        dialog.run.connect(self._bind("msd", self._request_msd))
        dialog.closed.connect(dialog.deleteLater)
        dialog.open()

    def _request_msd(self, config: dict) -> None:
        """The MSD dialog's Run: start it, or ask first when an MSD Run is still computing
        (`_start_or_ask`), as every other section does."""
        self._start_or_ask(self._jk("msd"), lambda: self._run_msd(config))

    def _run_msd(self, config: dict) -> None:
        """One job per condition on the process pool (`msd_view.msd_condition`), so each condition's cards fill in
        as its result lands (the cell bootstrap and v2's per-track fits are what take time)."""
        conditions = list(self._all_condition_keys)
        if not conditions:
            return                      # pragma: no cover - the dialog is only offered with conditions
        project, process_pool = self.project, self._process_pool
        self._msd_run_id += 1
        run_id = self._msd_run_id
        self._msd_pending = self._msd_total = len(conditions)
        self._msd.begin_run(conditions, dict(self._condition_labels),
                            self._condition_colours(conditions), config)
        # On the job record, not the rail's mark directly: the mark follows the section
        # ON SCREEN (`_sync_compute_mark`), and the Run gets its finished/failed toast.
        self._msd_failed: list = []
        self._job_started(self._jk("msd"))
        self._msd_status.setText(
            f"computing {len(conditions)} condition{'' if len(conditions) == 1 else 's'}\u2026")
        self._msd_status.show()
        settings = {k: config[k] for k in ("n_lags", "max_lag", "weighting", "direction", "uncertainty")}
        for condition in conditions:
            self._msd_gen += 1
            gen = self._msd_gen
            self._msd_by_gen[gen] = (run_id, condition)

            def compute(condition=condition, cache=self._results_cache):
                future = self._submit_bulk(cached_job, cache, "msd", msd_condition, project, condition, settings)
                return future.result()

            self._pool.start(_ComputeRunnable(compute, gen, self._msd_signals))

    def _on_msd_done(self, gen: int, result, error) -> None:
        """One condition landed; a failure is reported in its own card."""
        entry = self._msd_by_gen.pop(gen, None)
        if entry is None:
            return                      # pragma: no cover - defensive
        run_id, condition = entry
        if run_id != self._msd_run_id:
            return                      # a superseded press: dropped by name
        if error is not None:
            self._msd.set_error(condition, f"failed: {error}".replace("\n", " "))
            self._msd_failed.append((condition, error))
        else:
            self._msd.set_payload(condition, result)
        self._msd_pending = max(0, self._msd_pending - 1)
        if self._msd_pending:
            done = self._msd_total - self._msd_pending
            self._msd_status.setText(f"computing\u2026 {done} of {self._msd_total} done")
            return
        self._msd.finish()
        self._finish_per_condition(self._jk("msd"), self._msd_failed, self._msd_total)
        self._msd_status.hide()

    def _cancel_msd_pressed(self) -> None:
        """The placeholder's Cancel: a soft cancel, like every other section's."""
        self._msd_run_id += 1
        self._msd_pending = 0
        self._msd.set_running(False)
        self._job_cancelled(self._jk("msd"))
        self._msd_status.hide()

    # -- SQD: jump distances --------------------------------------------------- #

    def _on_sqd_tab(self, index: int) -> None:
        self.sidebar.select(sp.join_view("sqd", self._cur["sqd"].number, SQD_PAGES[index]))

    def _open_jumps_dialog(self) -> None:
        conditions = [(key, self._condition_labels.get(key, key))
                      for key in self._all_condition_keys]
        if not conditions:
            self._jumps_status.setText("No conditions loaded.")
            self._jumps_status.show()
            return
        dialog = JumpsConfigDialog(self.t, conditions, seed=self._jumps.last_config,
                                   content=self.centralWidget(), parent=self)
        dialog.run.connect(self._request_jumps)
        dialog.closed.connect(dialog.deleteLater)
        dialog.open()

    def _request_jumps(self, config: dict) -> None:
        """The dialog's Run: asks first while a Jump distances Run is still computing
        (`_start_or_ask`), as every other section does."""
        self._start_or_ask(self._jk("jumps"), lambda: self._run_jumps(config))

    def _run_jumps(self, config: dict) -> None:
        """Two jobs per condition on the process pool: every condition's quick pass first
        (histogram and curves, seconds), then every full one (the likelihood fits and the
        folds, minutes on a large condition), which replaces it (`jumps_view`)."""
        chosen = set(config["conditions"])
        conditions = [key for key in self._all_condition_keys if key in chosen]
        if not conditions:
            return                      # pragma: no cover - the dialog refuses an empty run
        project, process_pool = self.project, self._process_pool
        # THE SQD RUN'S LOCALISATION ERROR (6 Nov 2026), as its Optimiser and lag below. Decided before anything
        # starts, so a refusal leaves nothing half begun.
        from ...workspace_view import RunSettingsError

        try:
            per_condition = self._sqd_page_precision(conditions)
        except RunSettingsError as exc:
            self._jumps_status.setText(str(exc))
            self._jumps_status.show()
            return
        self._jumps_run_id += 1
        run_id = self._jumps_run_id
        self._jumps_pending = 2 * len(conditions)
        self._jumps_total = len(conditions)
        self._jumps_full = set()
        self._jumps_failed = []
        self._jumps.begin_run(conditions, dict(self._condition_labels),
                              self._condition_colours(conditions), config)
        # A job like every section's (main's page drove the mark by hand): the mark is
        # this page's, and the Run gets its finished / failed toast.
        self._job_started(self._jk("jumps"))
        max_pop = int(config.get("max_pop", 3))
        self._jumps_status.setText(
            f"fitting 1-{max_pop} populations to {len(conditions)} condition"
            f"{'' if len(conditions) == 1 else 's'}\u2026")
        self._jumps_status.show()
        cv_unit = config["cv_unit"]
        # the SQD Compute dialog's Optimiser: this page's fits are the section's fits
        gradient = self.compute_dialog.compute.values().get("gradient", "numeric")
        lag = self.compute_dialog.compute.values().get("lag", "1")
        if lag == "1-4":
            # one histogram is one lag: under lags 1-4 together the page draws lag 1, and says so
            self._jumps_status.setText(self._jumps_status.text()
                                       + "  (lags 1-4 together: the jump histogram is drawn at lag 1)")
            lag = "1"
        lag = int(lag)
        for quality in (False, True):
            for condition in conditions:
                self._jumps_gen += 1
                gen = self._jumps_gen
                self._jumps_by_gen[gen] = (run_id, condition, quality)
                settings = {"cv_unit": cv_unit, "quality": quality, "gradient": gradient, "lag": lag,
                            "max_pop": max_pop}
                if per_condition.get(condition) is not None:
                    settings["precision"] = per_condition[condition]

                def compute(condition=condition, settings=settings, cache=self._results_cache):
                    future = self._submit_bulk(cached_job, cache, "jumps", compute_jump_view, project, condition,
                                                 settings)
                    return future.result()

                self._pool.start(_ComputeRunnable(compute, gen, self._jumps_signals))

    def _on_jumps_done(self, gen: int, result, error) -> None:
        entry = self._jumps_by_gen.pop(gen, None)
        if entry is None:
            return                      # pragma: no cover - defensive
        run_id, condition, full = entry
        if run_id != self._jumps_run_id:
            return
        if full:
            self._jumps_full.add(condition)
        if full and error is not None:
            self._jumps_failed.append((condition, error))
        if full or condition not in self._jumps_full:
            if error is not None:
                self._jumps.set_error(condition, f"failed: {error}".replace("\n", " "))
            else:
                self._jumps.set_payload(condition, result)
        self._jumps_pending = max(0, self._jumps_pending - 1)
        if self._jumps_pending:
            self._jumps_status.setText(
                f"BIC and held-out fits\u2026 {len(self._jumps_full)} of "
                f"{self._jumps_total} done")
            return
        self._jumps.set_running(False)
        self._jumps_status.hide()
        self._finish_per_condition(self._jk("jumps"), self._jumps_failed, self._jumps_total)

    def _cancel_jumps_pressed(self) -> None:
        self._jumps_run_id += 1
        self._jumps_pending = 0
        self._jumps.set_running(False)
        self._jumps_status.hide()
        self._job_cancelled(self._jk("jumps"))

    # -- Signed jumps ----------------------------------------------------------- #

    def _sqd_page_precision(self, conditions) -> dict:
        """``{condition: (c_nm, p, source) or None}``: the SQD Compute dialog's Localisation error as the section's
        pages read it (Jump distances, Signed jumps; 6 Nov 2026) -- under Per spot the Run's calibration
        (`workspace_view.run_precision`) for every condition whose tracks carry precision, None for one whose tracks
        carry none (a truth condition: fitted as before); all None under Not modelled. RAISES `RunSettingsError` when
        Per spot finds no calibration."""
        return self._run_precision_for(conditions, self.compute_dialog.compute.values().get("precision") == "per_spot")

    def _run_precision_for(self, conditions, per_spot: bool) -> dict:
        """``{condition: (c_nm, p, source) or None}`` for a Run with per-spot precision on (``per_spot``): the Run's
        calibration (`workspace_view.run_precision`) for each condition whose tracks carry precision, None for one
        that carries none (a truth condition, fitted without it); all None when off. RAISES `RunSettingsError` when
        no calibration can be found."""
        from ...workspace_view import precision_off, run_precision

        if not per_spot:
            return {c: None for c in conditions}
        precision = tuple(run_precision(self.project, conditions))
        off = set(precision_off(self.project, conditions))
        return {c: None if c in off else precision for c in conditions}

    def _run_jd(self) -> None:
        """One job per loaded condition on the process pool (`jd_view.compute_jd_view`), at the SQD Compute
        dialog's Time lag (lag 1 under lags 1-4 together: a histogram is one lag) and its Localisation error."""
        from ...workspace_view import RunSettingsError

        conditions = list(self._all_condition_keys)
        if not conditions:
            self._jd_status.setText("No conditions loaded.")
            self._jd_status.show()
            return
        try:
            per_condition = self._sqd_page_precision(conditions)
        except RunSettingsError as exc:
            self._jd_status.setText(str(exc))
            self._jd_status.show()
            return
        project, process_pool = self.project, self._process_pool
        self._jd_run_id += 1
        run_id = self._jd_run_id
        self._jd_pending = self._jd_total = len(conditions)
        self._jd_failed = []
        self._jd.begin_run(conditions, dict(self._condition_labels), self._condition_colours(conditions))
        self._job_started(self._jk("jd"))
        lag = self.compute_dialog.compute.values().get("lag", "1")
        note = ""
        if lag == "1-4":
            note, lag = "  (lags 1-4 together: the jumps are drawn at lag 1)", "1"
        lag = int(lag)
        self._jd_status.setText(f"fitting signed jumps of {len(conditions)} condition"
                                f"{'' if len(conditions) == 1 else 's'}\u2026" + note)
        self._jd_status.show()
        for condition in conditions:
            self._jd_gen += 1
            gen = self._jd_gen
            self._jd_by_gen[gen] = (run_id, condition)

            settings = {"lag": lag}
            if per_condition.get(condition) is not None:
                settings["precision"] = per_condition[condition]

            def compute(condition=condition, settings=settings, cache=self._results_cache):
                return self._submit_bulk(cached_job, cache, "jd", compute_jd_view, project, condition,
                                           settings).result()

            self._pool.start(_ComputeRunnable(compute, gen, self._jd_signals))

    def _on_jd_done(self, gen: int, result, error) -> None:
        entry = self._jd_by_gen.pop(gen, None)
        if entry is None:
            return                      # pragma: no cover - defensive
        run_id, condition = entry
        if run_id != self._jd_run_id:
            return
        if error is not None:
            self._jd_failed.append((condition, error))
            self._jd.set_error(condition, f"failed: {error}".replace("\n", " "))
        else:
            if result.get("error"):
                self._jd_failed.append((condition, result["error"]))
            self._jd.set_payload(condition, result)
        self._jd_pending = max(0, self._jd_pending - 1)
        if self._jd_pending:
            self._jd_status.setText(f"signed jumps\u2026 {self._jd_total - self._jd_pending} of "
                                    f"{self._jd_total} done")
            return
        self._jd.set_running(False)
        self._jd_status.hide()
        self._finish_per_condition(self._jk("jd"), self._jd_failed, self._jd_total)

    # -- Clustering ------------------------------------------------------------------------------------------- #
    def _open_clusters_dialog(self) -> None:
        """The Clustering Run: its configuration dialog (`clustering_dialog`), seeded from the last Run."""
        conditions = list(self._all_condition_keys)
        if not conditions:
            self._clusters_status.setText("No conditions loaded.")
            self._clusters_status.show()
            return
        dialog = ClusteringConfigDialog(self.t, len(conditions), seed=self._clusters.config(),
                                        content=self.centralWidget(), parent=self)
        dialog.run.connect(self._bind("clusters", self._request_clusters))
        dialog.closed.connect(dialog.deleteLater)
        dialog.open()

    def _request_clusters(self, config: dict) -> None:
        """The dialog's Run: asks first while a Clustering Run is still computing (`_start_or_ask`)."""
        self._start_or_ask(self._jk("clusters"), lambda: self._run_clusters(config))

    def _run_clusters(self, config: dict | None = None) -> None:
        """Each condition alone: one job per loaded condition on the process pool
        (`clustering_view.compute_clustering_view`); each card fills in as its condition lands. Shared clusters: ONE
        job over all of them (`clustering_compare.compute_shared_clustering`), one card. Both go through the section
        cache (`cached_job`, `cached_group_job`: a repeat Run at the same settings on the same data is read, not
        computed) and a Cancel ends what waits for a worker (`_clusters_cancel`)."""
        conditions = list(self._all_condition_keys)
        if not conditions:
            self._clusters_status.setText("No conditions loaded.")
            self._clusters_status.show()
            return
        config = self._clusters.config() if config is None else config
        try:
            settings = compute_kwargs(config)
        except ValueError as exc:
            self._clusters_status.setText(str(exc))
            self._clusters_status.show()
            return
        shared = config.get("mode") == "shared"
        if shared and len(conditions) < 2:
            self._clusters_status.setText("Shared clusters compare conditions: load at least two.")
            self._clusters_status.show()
            return
        # PER SPOT (7 Nov 2026): the SQD-mixture null fitted and drawn with each spot's own calibrated error, the
        # project's calibration as every per-spot Run resolves it; a condition carrying no precision takes the
        # acquisition's, and a project where none carries any runs on as before, never refused for it
        precision = {c: None for c in conditions}
        if wants_per_spot(config) and set(conditions) - set(precision_off(self.project, conditions)):
            try:
                precision = self._run_precision_for(conditions, True)
            except RunSettingsError as exc:
                self._clusters_status.setText(str(exc))
                self._clusters_status.show()
                return
        self._clusters_cancel.set()                    # a Run still submitting for an older press stops
        self._clusters_cancel = cancel = threading.Event()
        self._clusters_futures = futures = []
        self._clusters_run_id += 1
        run_id = self._clusters_run_id
        self._clusters_pending = self._clusters_total = 1 if shared else len(conditions)
        self._clusters_failed = []
        self._clusters.begin_run(conditions, dict(self._condition_labels), config)
        self._clusters.set_colours(self._condition_colours(conditions))
        self._job_started(self._jk("clusters"))
        project, cache = self.project, self._results_cache
        if shared:
            self._clusters_status.setText(f"clustering the pooled tracks of {len(conditions)} conditions\u2026")
            self._clusters_status.show()
            self._clusters_gen += 1
            gen = self._clusters_gen
            self._clusters_by_gen[gen] = (run_id, SHARED_CLUSTERS_KEY)

            def compute_shared():
                future = self._submit_bulk_unless(cancel, cached_group_job, cache, "clusters",
                                                  compute_shared_clustering, project, conditions,
                                                  dict(settings, precision=precision))
                futures.append(future)
                return future.result()

            self._pool.start(_ComputeRunnable(compute_shared, gen, self._clusters_signals))
            return
        self._clusters_status.setText(f"clustering the tracks of {len(conditions)} condition"
                                      f"{'s' * (len(conditions) != 1)}\u2026")
        self._clusters_status.show()
        for condition in conditions:
            self._clusters_gen += 1
            gen = self._clusters_gen
            self._clusters_by_gen[gen] = (run_id, condition)

            def compute(c=condition):
                future = self._submit_bulk_unless(cancel, cached_job, cache, "clusters", compute_clustering_view,
                                                  project, c, dict(settings, precision=precision[c]))
                futures.append(future)
                return future.result()

            self._pool.start(_ComputeRunnable(compute, gen, self._clusters_signals))

    def _on_clusters_done(self, gen: int, result, error) -> None:
        entry = self._clusters_by_gen.pop(gen, None)
        if entry is None:
            return
        run_id, condition = entry
        if run_id != self._clusters_run_id:
            return
        if error is not None:
            self._clusters_failed.append((condition, error))
            self._clusters.set_error(condition, f"failed: {error}".replace("\n", " "))
        else:
            if "error" in result:
                self._clusters_failed.append((condition, result["error"]))
            self._clusters.set_payload(condition, result)
        self._clusters_pending = max(0, self._clusters_pending - 1)
        if self._clusters_pending:
            self._clusters_status.setText(f"clustering\u2026 {self._clusters_total - self._clusters_pending} of "
                                          f"{self._clusters_total} done")
            return
        self._clusters.set_running(False)
        self._clusters_status.hide()
        self._finish_per_condition(self._jk("clusters"), self._clusters_failed, self._clusters_total)

    def _cancel_clusters_pressed(self) -> None:
        """The placeholder's Cancel: a job still waiting for a worker is never submitted and one queued is cancelled
        (`_clusters_cancel`); the Run's results are dropped on arrival. A job already on a worker cannot be told to
        stop, so where NOTHING ELSE is running the pool itself is stopped and made again (`parallel.stop_pool`, as the
        window's close does): the workers end now instead of finishing for nobody (2 Nov 2026: three ran on 40
        minutes). While another section computes, only its own work is left alone."""
        self._clusters_cancel.set()
        for future in list(self._clusters_futures):
            future.cancel()
        self._clusters_run_id += 1
        self._clusters_pending = 0
        self._clusters.set_running(False)
        self._clusters_status.hide()
        self._job_cancelled(self._jk("clusters"))
        self._restart_pool_if_idle()

    def _restart_pool_if_idle(self) -> bool:
        """Stop the process pool's workers and start a new pool, when no job of this window is running (the register
        `_running` is empty) and the pool is a real process pool (the tests' thread pools cannot be killed). Returns
        whether it did. The bulk slots stay: a stopped worker's future ends and gives its slot back."""
        if self._running or not isinstance(self._process_pool, ProcessPoolExecutor):
            return False
        old = self._process_pool
        self._process_pool = ProcessPoolExecutor(
            max_workers=old._max_workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=lower_priority)
        stop_pool(old)
        return True

    def _cancel_jd_pressed(self) -> None:
        self._jd_run_id += 1
        self._jd_pending = 0
        self._jd.set_running(False)
        self._jd_status.hide()
        self._job_cancelled(self._jk("jd"))

    # -- HMM ------------------------------------------------------------------ #

    def _open_hmm_dialog(self) -> None:
        """The HMM section's Run: its configuration dialog, seeded from the last run."""
        conditions = [(key, self._condition_labels.get(key, key))
                      for key in self._all_condition_keys]
        if not conditions:
            self._hmm_status.setText("No conditions loaded.")
            self._hmm_status.show()
            return
        from ...hmm_view import sigma_note

        dialog = HMMConfigDialog(self.t, conditions, seed=self._hmm.last_config,
                                 factors=dict(self._factors or {}), factor_names=self._factor_names,
                                 sigma_note=sigma_note(self.project), content=self.centralWidget(), parent=self)
        dialog.run.connect(self._bind("hmm", self._request_hmm))
        dialog.closed.connect(dialog.deleteLater)
        dialog.open()

    def _request_hmm(self, config: dict) -> None:
        """The dialog's Run: asks first while an HMM Run is still computing
        (`_start_or_ask`), as every other section does."""
        self._start_or_ask(self._jk("hmm"), lambda: self._run_hmm(config))

    def _run_hmm(self, config: dict) -> None:
        """One job per included condition on the process pool, so each card fills in as
        its result lands (the cell bootstrap is what takes time). Each condition is given
        `hmm_run.condition_settings` (its count, its own Force same D across -- its cache key
        too); under a set-wide Force same D across the group's D come first, from the pooled
        tracks (`hmm_run.joint_D`: a pool job per EM start, each through a bulk slot, saying
        how far it is as each lands, `_on_hmm_progress`), and every condition is then fitted
        at them (`_on_hmm_done`)."""
        from functools import partial

        from ...hmm_run import groups_of, run_config, with_precision

        try:
            config = run_config(config, conditions=list(self._all_condition_keys),
                                factors=dict(self._factors or {}))
            config = with_precision(config, self.project)       # Per spot: one calibration for the whole Run
        except ValueError as exc:
            self._hmm_status.setText(str(exc))
            self._hmm_status.show()
            return
        chosen = set(config["conditions"])
        conditions = [key for key in self._all_condition_keys if key in chosen]
        if not conditions:
            return                      # pragma: no cover - the dialog refuses an empty run
        project, process_pool = self.project, self._process_pool
        self._hmm_run_id += 1
        run_id = self._hmm_run_id
        self._hmm_pending = self._hmm_total = len(conditions)
        self._hmm_failed = []
        self._hmm_cancel.set()          # a shared fit still submitting for an older Run stops
        self._hmm_cancel = cancel = threading.Event()
        self._hmm_joint = None
        self._hmm.begin_run(conditions, dict(self._condition_labels),
                            self._condition_colours(conditions), config)
        self._angles_follow_run()
        self._job_started(self._jk("hmm"))
        self._hmm_status.setText(
            f"computing {len(conditions)} condition{'' if len(conditions) == 1 else 's'}\u2026")
        self._hmm_status.show()
        self._hmm_config = config
        groups = groups_of(config)
        if groups:
            # ONE D ACROSS CONDITIONS: the group's D from its pooled tracks first, then every condition at them
            from ...hmm_run import joint_D

            self._hmm_gen += 1
            gen = self._hmm_gen
            self._hmm_by_gen[gen] = (run_id, None)
            self._hmm_joint = {"done": 0, "total": 0, "stage": "shared states", "unit": "fit"}
            self._hmm_status.setText("computing\u2026 the states shared across conditions")
            signals = self._hmm_signals

            def said(payload, gen=gen):
                try:
                    signals.progress.emit(gen, int(payload["done"]), int(payload["total"]), payload)
                except RuntimeError:
                    pass                # the window went away first: nothing is left to tell

            def joint(groups=groups, states=config["states"], sigma=config["sigma"], best_max=config["best_max"],
                      precision=config.get("precision")):
                return joint_D(project, groups, states=states, sigma=sigma, best_max=best_max,
                               submit=partial(self._submit_bulk_unless, cancel), progress=said, cancelled=cancel,
                               precision=precision)

            self._pool.start(_ComputeRunnable(joint, gen, self._hmm_signals))
            self._sync_jobs()
            return
        self._submit_hmm_conditions(run_id, conditions, config, None)

    def _submit_hmm_conditions(self, run_id: int, conditions: list, config: dict, joint) -> None:
        """Every condition's own job (`compute_hmm_view` through the section cache); ``joint`` is `hmm_run.joint_D`'s
        result under a set-wide Force same D across (each condition at its group's D), else None."""
        from ...hmm_run import condition_settings

        project = self.project
        where = {c: name for name, g in (joint or {}).items() for c in g["members"]}
        for condition in conditions:
            self._hmm_gen += 1
            gen = self._hmm_gen
            self._hmm_by_gen[gen] = (run_id, condition)
            fixed = (joint or {}).get(where.get(condition), {}).get("D") if joint else None
            settings = condition_settings(config, condition, fixed_D=fixed)

            def compute(condition=condition, settings=settings, cache=self._results_cache):
                future = self._submit_bulk(cached_job, cache, "hmm", compute_hmm_view, project, condition, settings)
                return future.result()

            self._pool.start(_ComputeRunnable(compute, gen, self._hmm_signals))

    def _on_hmm_done(self, gen: int, result, error) -> None:
        """One condition landed; a failure is reported in its own card."""
        entry = self._hmm_by_gen.pop(gen, None)
        if entry is None:
            return                      # pragma: no cover - defensive
        run_id, condition = entry
        if run_id != self._hmm_run_id:
            return                      # a superseded press: dropped by name
        if condition is None:
            # THE SHARED STATES landed (`hmm_run.joint_D`): every condition at them, or every card says why not
            self._hmm_joint = None
            conditions = [k for k in self._all_condition_keys if k in set(self._hmm_config["conditions"])]
            if error is not None:
                for c in conditions:
                    self._hmm_failed.append((c, error))
                    self._hmm.set_error(c, f"failed: the shared states could not be fitted ({error})"
                                        .replace("\n", " "))
                    self._angles_heard(c, "the shared states could not be fitted")
                self._hmm_pending = 0
                self._hmm.set_running(False)
                self._hmm_status.hide()
                self._finish_per_condition(self._jk("hmm"), self._hmm_failed, self._hmm_total)
                return
            self._hmm_status.setText(f"computing {len(conditions)} condition{'' if len(conditions) == 1 else 's'}"
                                     f"\u2026 at the shared states")
            self._submit_hmm_conditions(run_id, conditions, self._hmm_config, result)
            return
        if error is not None:
            self._hmm_failed.append((condition, error))
            self._hmm.set_error(condition, f"failed: {error}".replace("\n", " "))
            self._angles_heard(condition, f"failed: {error}".replace("\n", " "))
        else:
            self._hmm.set_payload(condition, result)
            message = result.get("error") if isinstance(result, dict) else None
            self._angles_heard(condition, message)
        self._hmm_pending = max(0, self._hmm_pending - 1)
        if self._hmm_pending:
            done = self._hmm_total - self._hmm_pending
            self._hmm_status.setText(f"computing\u2026 {done} of {self._hmm_total} done")
            return
        self._hmm.set_running(False)
        self._hmm_status.hide()
        self._finish_per_condition(self._jk("hmm"), self._hmm_failed, self._hmm_total)

    def _on_hmm_progress(self, gen: int, done: int, total: int, payload) -> None:
        """The shared fit said how far it is (`hmm_run.joint_progress`, as each start lands): the section's status
        line and the sidebar's Running list. A superseded or cancelled Run's word is dropped by name."""
        entry = self._hmm_by_gen.get(gen)
        if entry is None or entry[0] != self._hmm_run_id or entry[1] is not None or self._hmm_joint is None:
            return
        self._hmm_joint = dict(payload)
        self._hmm_status.setText(str(payload["status"]))
        self._sync_jobs()

    # -- THE JUMP ANGLES TAB: each HMM page's angles, read from that page's own fits (23 Nov 2026) ----------------- #
    def _angles_shown(self) -> bool:
        """Whether the HMM page now current is on screen, on its Jump angles tab."""
        return (self.sidebar.current_base == "hmm.angles"
                and sp.split_view(self.sidebar.current)[1] == self._cur["hmm"].number)

    def _angles_follow_run(self) -> None:
        """The HMM page now current began a Run (or its tab starts again): its Jump angles tab starts from that Run --
        its conditions, waiting for the angles of every fit already in, asked for when on show."""
        self._angles_asked = set()
        conditions, labels, colours = self._hmm.run_conditions()
        self._angles_page.begin(conditions, labels, colours)
        for condition in self._hmm.results():
            self._angles_page.set_waiting(condition, True)
        if self._angles_shown():
            self._ask_angles()

    def _angles_heard(self, condition: str, message: str | None = None) -> None:
        """A condition's HMM fit landed on the page now current (``message``: why it did not): its Jump angles tab
        waits for the angles, asked for at once where the tab is on show, or says why not."""
        if message is not None:
            self._angles_page.set_result(condition, {"condition": condition, "error": f"HMM: {message}"})
            return
        self._angles_page.set_waiting(condition, True)
        if self._angles_shown():
            self._ask_angles([condition])

    def _ask_angles(self, only=None) -> None:
        """The angles of every fit of this page's Run not yet asked for (``only``: of those)."""
        results = self._hmm.results()
        keys = [k for k in (only if only is not None else results) if k in results and k not in self._angles_asked]
        if keys:
            self._angles_asked.update(keys)
            self._run_hmm_angles(keys)

    def _angles_reading_changed(self, _on: bool) -> None:
        """Against chance ticked or unticked: every fit's angles are asked for again, in the new reading."""
        self._angles_asked = set()
        if self._angles_shown():
            self._ask_angles()

    def _run_hmm_angles(self, conditions) -> None:
        """The Jump angles tab asked for ``conditions``' angles: one pool job each (`hmm_angles.compute_hmm_angles`
        through the section cache, keyed on the condition's data and its HMM result), read from this page's Run. A
        newer Run, or the other reading, drops what lands for the old one."""
        project = self.project
        key = self._jk("hmm_angles")
        # the tab's reading (Against chance, off by default): the simulation is computed only when asked
        chance = self._angles_page.chance_on()
        submitted = 0
        for condition in conditions:
            result = self._hmm.hmm_result(condition)
            if result is None:
                continue
            self._angles_gen += 1
            gen = self._angles_gen
            self._angles_by_gen[gen] = (self._hmm_run_id, condition, chance)

            def compute(condition=condition, result=result, cache=self._results_cache):
                future = self._submit_bulk(cached_job, cache, "hmm_angles", compute_hmm_angles, project, condition,
                                           {"hmm": result, "chance": chance})
                return future.result()

            self._pool.start(_ComputeRunnable(compute, gen, self._angles_signals))
            submitted += 1
        if not submitted:
            return
        if key not in self._running:
            self._angles_total, self._angles_pending, self._angles_failed = 0, 0, []
            self._job_started(key)
        self._angles_total += submitted
        self._angles_pending += submitted
        self._sync_jobs()

    def _on_hmm_angles_done(self, gen: int, result, error) -> None:
        """One condition's angles landed (or failed, said in its card); the job ends with the last."""
        entry = self._angles_by_gen.pop(gen, None)
        if entry is None:
            return                      # pragma: no cover - defensive
        run_id, condition, chance = entry
        self._angles_pending = max(0, self._angles_pending - 1)
        # a newer Run of this page, or the other reading than the tab now shows: dropped
        if run_id == self._hmm_run_id and chance == self._angles_page.chance_on():
            if error is not None:
                self._angles_failed.append((condition, error))
                self._angles_asked.discard(condition)
                self._angles_page.set_result(condition, {"condition": condition,
                                                         "error": f"failed: {error}".replace("\n", " ")})
            else:
                self._angles_page.set_result(condition, result)
        if self._angles_pending:
            self._sync_jobs()
            return
        self._finish_per_condition(self._jk("hmm_angles"), self._angles_failed, self._angles_total)

    def _cancel_hmm_pressed(self) -> None:
        """The placeholder's Cancel: a soft cancel, like every other section's -- and the shared fit stops submitting
        its starts and drops those not started (`_hmm_cancel`); a start already on a worker runs to its end."""
        self._hmm_cancel.set()
        self._hmm_joint = None
        self._hmm_run_id += 1
        self._hmm_pending = 0
        self._hmm.set_running(False)
        self._hmm_status.hide()
        self._job_cancelled(self._jk("hmm"))

    def _sync_spatial_visibility_section(self) -> None:
        """Point the Visibility popover at the Spatial Map section.

        Directly on `.visibility` for the same reason
        `_sync_sqd_visibility_section` gives -- `_on_selected`'s own
        `toolbar.set_section` has already pointed Visibility at `"spatial"`;
        this adds the remembered per-column state and the colormap, which that
        call does not carry.

        `"spatial"` vs `"spatial-appd"`/`"spatial-hmm"`: the grid's own
        CURRENT computational method (`SpatialMapSection.method`), not the
        dialog's -- the dialog can be reconfigured before the next Run, but
        the popover must describe what is actually ON SCREEN right now.
        """
        method = self._spatial.method()
        section = f"spatial-{method}" if f"spatial-{method}" in SPATIAL_SECTIONS else "spatial"
        # THE GRID'S OWN COLUMNS, never a fixed list: All jumps, Middle, a split by hand's ranges (3 Nov 2026). And
        # only once a map is DRAWN: before a Run the rows named maps that do not exist, and the next Run may draw
        # others (4 Nov 2026, rail review B5).
        drawn = self._spatial.has_result()
        self.toolbar.visibility.set_section(
            section, self._spatial.visibility_states() if drawn else {},
            names=list(self._spatial.matrix.panel_keys()) if drawn else [])
        self.toolbar.visibility.set_figures_note("" if drawn else rail_view.NO_MAP_YET)
        self.toolbar.visibility.set_colormap(self._spatial.colormap())
        offered = self.toolbar.visibility.rendering_keys()
        if self._spatial.rendering() not in offered:
            # Tracks belongs to Confinement (SLA) alone: a run of another method leaves it
            # for the first rendering offered (Bins), rather than a page of empty capsules.
            self._spatial.set_rendering(offered[0])
        self.toolbar.visibility.set_rendering(self._spatial.rendering())
        self.toolbar.visibility.set_scale(self._spatial.scale())

    def _sync_sqd_visibility_section(self) -> None:
        """Point the rail at SQD's replicate page on a drill-in: it draws no bubble chart, so it offers no brackets
        (`rail_view.REPLICATE_PAGE`). It offered the hidden chart's Significance block until 4 Nov 2026 (rail
        review); an earlier crumb gives the section back (`_on_crumb_clicked`)."""
        self.toolbar.set_section(f"sqd.{REPLICATE_PAGE}")

    def _sync_appd_rail(self) -> None:
        """Apparent Diffusion's replicate page (a bubble's click, the crumb's drop-down) is drawn IN PLACE of the
        bubbles, on the Populations page: the rail follows it as SQD's, and gives the page back on its way out."""
        if self.sidebar.current_base.partition(".")[0] != "appd":
            return
        stack = self._appd.days.parentWidget()          # the page stack the days page is one page of
        on_days = getattr(stack, "currentWidget", lambda: None)() is self._appd.days
        self.toolbar.set_section(f"appd.{REPLICATE_PAGE}" if on_days else rail_section(self.sidebar.current_base))

    def _on_visibility_colormap(self, key: str) -> None:
        """A colormap pick. Routed by the popover's CURRENT section, exactly
        as `_on_visibility_toggled` routes a toggle -- today the SQD map and
        the Spatial Map section, which are the only two views with a ramp."""
        section = self.toolbar.visibility.section
        if section in SPATIAL_SECTIONS:
            self._spatial.set_colormap(key)

    def _on_visibility_rendering(self, key: str) -> None:
        """A Bins/Smooth pick, routed by the popover's CURRENT section exactly
        as `_on_visibility_colormap` is. DISPLAY only: nothing is recomputed."""
        section = self.toolbar.visibility.section
        if section in SPATIAL_SECTIONS:
            self._spatial.set_rendering(key)

    def _load_custom_ramps(self) -> list:
        """The user's custom ramps from `ui.json`, registered so the maps can paint them.
        A store this build cannot read is reported in the console and ignored, never fatal."""
        from ...colour_ramps import RampError, ramps_from_json
        from .spatial_map_chart import register_custom_ramps

        try:
            ramps = ramps_from_json(theme.remembered("custom_ramps") or [])
        except RampError as exc:
            print(f"custom colour ramps in ui.json ignored: {exc}")
            ramps = []
        register_custom_ramps(ramps)
        return ramps

    def _open_ramp_dialog(self) -> None:
        from .ramp_dialog import RampDialog

        dialog = RampDialog(self.t, ramps=self._load_custom_ramps(),
                            current=self._spatial.colormap(),
                            content=self.centralWidget(), parent=self)
        dialog.saved.connect(self._custom_ramps_saved)
        dialog.used.connect(self._custom_ramp_used)
        dialog.closed.connect(dialog.deleteLater)
        dialog.open()

    def _custom_ramps_saved(self, ramps: list) -> None:
        from ...colour_ramps import ramps_to_json
        from .spatial_map_chart import ramp_for, register_custom_ramps

        theme.remember("custom_ramps", ramps_to_json(ramps))
        register_custom_ramps(ramps)
        if ramp_for(self._spatial.colormap()) is None:      # the ramp on screen was deleted
            fallback = default_cmap_for(self._spatial.method())
            self._spatial.set_colormap(fallback)
            self.toolbar.visibility.set_colormap(fallback)
        else:
            self._spatial.refresh_colours()

    def _custom_ramp_used(self, key: str) -> None:
        self._spatial.set_colormap(key)
        self.toolbar.visibility.set_colormap(key)

    def _on_visibility_scale(self, key: str) -> None:
        """A colour-scale pick for the Spatial Map grid, routed like the rendering.
        DISPLAY only: nothing is recomputed."""
        if self.toolbar.visibility.section in SPATIAL_SECTIONS:
            self._spatial.set_scale(key)

    def _fade_in(self, card) -> None:
        """The ONE animated transition either section has: its placeholder
        replacing itself with `card` on that section's FIRST successful
        Run, softened with a fade rather than the jump cut a plain
        page-swap would otherwise be. `card` itself opacity 0 -> 1 over the
        `widgets.dialogs.OverlayDialog` curve/duration this app's other big
        state-change animation already uses (`_ANIM_MS` = 200, the same
        Bezier ease) -- one visual language for "something just settled
        in", not a second animation invented for this one moment. ONE
        method for both sections rather than two near-identical copies --
        unlike the page-stack shape above, an animation helper has no
        section-specific state to keep separate, so generalising it costs
        nothing the "parallel copies" module-docstring note is arguing
        against.

        A plain fade, not the full scale-from-a-point zoom once discussed:
        that needs a snapshot-and-resize renderer (`menus.Panel` built
        exactly that for ITS OWN open animation, at real engineering cost)
        to scale a widget Qt otherwise cannot, and a placeholder is never
        shown again for the rest of this window's life to earn it. The
        fade covers the one real problem (a chart popping in with no
        transition at all); the zoom stays a documented future refinement
        if it turns out to still matter once this is on screen.
        """
        effect = QGraphicsOpacityEffect(card)
        effect.setOpacity(0.0)
        card.setGraphicsEffect(effect)
        anim = QVariantAnimation(card)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(200)
        curve = QEasingCurve(QEasingCurve.BezierSpline)
        curve.addCubicBezierSegment(QPointF(0.25, 0.1), QPointF(0.25, 1.0),
                                    QPointF(1.0, 1.0))
        anim.setEasingCurve(curve)
        anim.valueChanged.connect(effect.setOpacity)
        anim.finished.connect(lambda: card.setGraphicsEffect(None))
        # Kept alive on `self`, in a LIST rather than one named attribute --
        # `BubbleCard._anim`/`OverlayDialog._anim` are this codebase's own
        # precedent for why a RUNNING `QVariantAnimation` needs a live
        # Python reference of its own (Qt's C++ parenting, to `card`, keeps
        # the object alive, but a collected Python wrapper mid-flight is a
        # real, separate risk this project already avoids everywhere else
        # one is actually started rather than merely built); a list because
        # this method now has two possible callers and both animations may
        # be in flight at once (a project's first Run graduates both
        # sections together). Never cleared -- at most two entries ever
        # exist, one per section, for this window's whole life.
        self._fade_anims = getattr(self, "_fade_anims", [])
        self._fade_anims.append(anim)
        anim.start()

    def _on_population_clicked(self, condition: str) -> None:
        """A bubble was clicked -- drill into `ReplicateDetail` for its
        condition, in place, per the module docstring: `SQD > Populations >
        <condition>` rather than the legacy floating `BubbleInspector`.

        Also reached from `_on_dropdown_picked` when the SQD section is
        current: switching FROM one condition's detail page TO another's is
        the exact same transition as arriving at the first one, so picking
        "Malate" from the last crumb's dropdown while already on Glucose's
        page re-enters here rather than needing its own method.
        `dropdown_options=` re-offers every loaded condition on the new
        trail's last crumb -- without it, switching once would leave the
        dropdown behind, since `set_trail` rebuilds the crumbs from nothing
        every time.
        """
        self._detail_condition = condition
        self._detail.set_payload(condition, self._replicate_payloads.get(condition),
                                 colour=self._column_colours(getattr(self, "_sqd_plot", None)).get(condition),
                                 pairs=self._replicate_pairs().get(condition))
        self._pages.setCurrentWidget(self._detail)
        self.breadcrumb.set_trail(self.sidebar.trail(self.sidebar.current) + [condition],
                                  dropdown_options=self._card.chart.conditions())
        self._sync_sqd_visibility_section()

    def _on_appd_condition_clicked(self, condition: str) -> None:
        """The day page's Cross-checks: the SQD/HMM/saSPT checks of that condition (`ApparentDDetail`), held in
        the Apparent Diffusion section so they stay reachable."""
        self._appd_detail.set_payload(condition, None)
        self._appd_detail.set_colour(self._condition_colours([condition]).get(condition))
        self._appd.show_crosschecks(self._appd_detail)
        self.breadcrumb.set_trail(self.sidebar.trail(self.sidebar.current) + [condition, "Cross-checks"],
                                  dropdown_options=self._appd.conditions())
        self._sync_appd_visibility_section()

    def _on_map_track_hidden(self, condition: str, uid: str, hidden: bool) -> None:
        """A track's eye on the day page: that condition's map again, without every track hidden from it, at the
        last Run's settings. Only that condition is redrawn; a redraw overtaken by a newer one, or by a new Run,
        is dropped when it lands."""
        mine = self._map_hidden.setdefault(condition, set())
        if hidden:
            mine.add(uid)
        else:
            mine.discard(uid)
        config = getattr(self, "_spatial_last_config", None)
        if config is None:
            return                      # pragma: no cover - a day page exists only after a Run
        project, process_pool = self.project, self._process_pool
        uids = sorted(mine)
        self._hide_gen += 1
        gen = self._hide_gen
        self._hide_latest[condition] = gen
        self._hide_by_gen[gen] = (self._spatial_run_id, condition)

        # at the count this condition was drawn with: a Best one keeps the count its Run chose; one drawn from a
        # section's fit keeps THAT fit (`map_fit_source.job_config`), whatever the section has run since
        shown = self._spatial.matrix.payloads().get(condition) or {}
        one = map_fit_source.job_config(config, condition, shown.get("section_fit"), chosen=shown.get("n_pop"))
        if shown.get("precision"):
            one["precision"] = list(shown["precision"])     # the map's own fit Per spot: the calibration it was drawn with

        def compute(cache=self._results_cache):
            return process_pool.submit(cached_job, cache, "spatial", compute_spatial_map, project, condition,
                                       {"hidden": uids, **one}).result()

        name = self._condition_labels.get(condition, condition)
        self._spatial_status.setText(
            f"redrawing {name} without {len(uids)} hidden track{'' if len(uids) == 1 else 's'}…"
            if uids else f"redrawing {name} with every track…")
        self._spatial_status.show()
        self._pool.start(_ComputeRunnable(compute, gen, self._hide_signals))

    def _on_map_redrawn(self, gen: int, result, error) -> None:
        entry = self._hide_by_gen.pop(gen, None)
        if entry is None:
            return                      # pragma: no cover - defensive
        run_id, condition = entry
        if run_id != self._spatial_run_id or self._hide_latest.get(condition) != gen:
            return                      # overtaken: a newer redraw or a new Run carries the hidden set
        if "spatial" not in self._running:
            self._spatial_status.hide()
        if error is not None:
            self._spatial.show_track_message(
                f"The map could not be redrawn: {error}".replace("\n", " "))
            return
        self._spatial.redrawn(condition, result)
        self._sync_metadata()

    def _on_spatial_track_requested(self, condition: str, uid: str, viewer: str) -> None:
        """A spot's track, handed to a viewer from the Spatial Map's day page."""
        self._open_track(condition, uid, viewer, self._spatial.show_track_message)

    def _open_track(self, condition: str, uid: str, viewer: str, say) -> None:
        """One track of the project (its PROJECT uid) in a viewer: the Spatial Map's spot
        card and the Images & tracks side card both ask here.

        Both viewers, by ruling: "fast" is the matplotlib track view on the track's
        OWN cell (no OpenGL, opens at once), "napari" loads the condition from its
        folder and opens at that track. What happened is said on the asking card
        (``say``), never in a modal dialog (a modal blocks the event loop, which
        headless is forever)."""
        from ...trackviewer import TrackViewerError, track_target

        try:
            target = track_target(self.project, condition, uid, projected=False)
        except TrackViewerError as exc:
            say(str(exc))
            return
        name = f"track {target['track_id']} · {target['short_movie']} · {condition}"
        if viewer == "fast":
            # THE TRACK VIEWER IS THE IMAGES & TRACKS SECTION now (asked: "i dont see
            # controls, i think we need to redesign the track viewer"): the track's movie on
            # its own image, its cell zoomed, the track picked and playing from just before
            # it starts. The old matplotlib window drew one still frame with no play control.
            cell = target["cell"]
            cell = (None if cell is None or cell != cell else
                    f"{target['short_movie']}:{str(cell).rsplit(':', 1)[-1]}")
            self.sidebar.select("images.overview")
            self._images.show_track(condition, target["short_movie"], cell,
                                    target["short_uid"])
            say(f"Opened {name} in Images & tracks, playing from just before it.")
            return
        from ...trackviewer import open_for_condition

        say(f"Opening {name} in napari (it loads the condition from its folder, which "
            "takes a moment)…")
        try:
            open_for_condition(self.project, condition, movie=target["short_movie"],
                               uid=target["short_uid"])
        except TrackViewerError as exc:
            say(f"napari could not open it: {exc}")
            return
        say(f"Opened {name} in napari.")

    def _on_spatial_condition_clicked(self, condition: str) -> None:
        """A condition's maps were clicked -- open its REPLICATE DAYS, the same
        shape `_on_population_clicked` gives the SQD section: `Spatial Map >
        <condition>`, one row per day, under that condition's own fit.

        Also reached from `_on_dropdown_picked` while the Spatial section is
        current, since switching from one condition's days to another's is the
        same transition. The dropdown offers only conditions that HAVE maps
        (`drillable_conditions`) -- one still computing, or one that failed,
        has no days to show.
        """
        self._spatial.show_days(condition)
        self.breadcrumb.set_trail(
            self.sidebar.trail(self.sidebar.current) + [condition],
            dropdown_options=self._spatial.drillable_conditions())
        self._sync_spatial_visibility_section()
        self._sync_metadata()

    def _sync_appd_visibility_section(self) -> None:
        """Point the Visibility popover at whichever of `ApparentDDetail`'s
        own two tabs is on screen -- called on drill-in and on every tab
        switch inside it. `states` carries the tab's own remembered
        toggle state back in, so switching tabs and back never resets a
        chart the user hid."""
        self.toolbar.visibility.set_section(
            self._appd_detail.visibility_section(),
            self._appd_detail.visibility_states())
        self.toolbar.drop_significance()        # the detail page draws no bubble chart: no brackets

    def _on_visibility_toggled(self, key: str, on: bool) -> None:
        """An `EyeRow` fired. Routed by the Visibility popover's CURRENT
        section: the two Apparent-D DETAIL sections and the Spatial Map hide or
        show charts; `"sqd"` and `"appd"` list no rows (`toolbar.FIGURES`, emptied
        because they hid nothing)."""
        section = self.toolbar.visibility.section
        if section == "appd-detail-fractions":
            self._appd_detail.set_fractions_chart_visible(key, on)
        elif section == "appd-detail-dynamics":
            self._appd_detail.set_dynamics_chart_visible(key, on)
        elif section in SPATIAL_SECTIONS:
            self._spatial.set_panel_visible(key, on)

    def _on_dropdown_picked(self, condition: str) -> None:
        """`Breadcrumb.dropdown_picked` -- routes to whichever section is
        CURRENT, since the breadcrumb itself has no notion of sections (see
        `Breadcrumb`'s own docstring)."""
        section = self.sidebar.current_base.partition(".")[0]
        if section == "images":
            self._images.show_condition(condition)
            return
        if section in ("kinetics", "msd", "hmm", "composer", "clusters"):
            # No per-condition page to pick into -- the cards already show
            # every condition. Guarded rather than left to the `else`, which
            # would drill the SQD section in behind the user's back.
            return
        if section == "spatial":
            self._on_spatial_condition_clicked(condition)
        elif section == "appd":
            self._appd.show_days(condition)
            self._sync_appd_rail()
        elif section == "qc":
            self._on_qc_condition_clicked(condition)
        else:
            self._on_population_clicked(condition)

    def _on_crumb_clicked(self, _index: int) -> None:
        """Any EARLIER crumb leads back to the same place, in WHICHEVER
        section is current: there are only two levels in either one (the
        chart, and one condition's detail page), so which non-last crumb
        was clicked does not change where clicking it goes.

        The Spatial Map section's drill-in is a condition's replicate days;
        an earlier crumb returns to its grid. Kinetics has none -- returned
        from BY NAME rather than left to fall through the `else`, which would
        swap the SQD section's pages underneath a user who is looking at it.
        """
        section = self.sidebar.current_base.partition(".")[0]
        if section == "images":
            # The Overview crumb: back to the sample size of every condition.
            self._images.show_overview()
            return
        if section in ("kinetics", "msd", "hmm", "composer", "clusters"):
            return
        if section == "spatial":
            self._spatial.show_overview()
            self.breadcrumb.set_trail(self.sidebar.trail(self.sidebar.current))
            self._sync_metadata()
            return
        if self.sidebar.current_base.partition(".")[0] == "appd":
            self._appd.show_overview()
            # Restores the overview-level Visibility (its Significance block;
            # no eye rows since 4 Nov 2026) -- without this it would keep
            # showing the detail page's own chart list after navigating back
            # out of it. `set_section` drives Visibility
            # alone now that the Visualization tool is gone. The PAGE's (`rail_view.rail_section`): a crumb on
            # Distributions never offers the Populations chart's brackets.
            self.toolbar.set_section(rail_section(self.sidebar.current_base))
        elif self.sidebar.current_base.partition(".")[0] == "qc":
            self._qc_pages.setCurrentWidget(self._qc_general)
            self.toolbar.set_section("qc")
        else:
            self._pages.setCurrentWidget(self._card)
            # The same restore the `appd` branch above makes, for the page on screen (Jump distances keeps its own).
            self.toolbar.set_section(rail_section(self.sidebar.current_base))
        self.breadcrumb.set_trail(self.sidebar.trail(self.sidebar.current))

    # -- significance on the chart (14 Oct 2026) ------------------------------------------ #
    def _sig_chart(self, section: str):
        """``(chart, data)`` of ``section``'s bubble chart (the HMM's: its comparison strip and payload); data None
        before a Run."""
        if section == "sqd":
            return self._card.chart, self._card.chart._data
        if section == "hmm":
            return self._hmm, self._hmm.bracket_data()
        return self._appd.populations.chart, self._appd.bubble_data()

    def _choice_of(self, section: str):
        """``section``'s brackets choice: a paged bubble section's is the current page's own, RESOLVED for the chart
        it is on (`significance.resolved`: on random chunks the minimum-effect test until one is picked by hand)."""
        from ... import significance as sig

        choice = getattr(self, f"_{section}_choice") if section in PAGED_CHOICE else self._sig_choices[section]
        _chart, data = self._sig_chart(section)
        return sig.resolved(choice, data) if data is not None else choice

    def _set_choice(self, section: str, choice) -> None:
        if section in PAGED_CHOICE:
            setattr(self, f"_{section}_choice", choice)
        else:
            self._sig_choices[section] = choice

    def _brackets_of(self, section: str) -> list:
        return list((getattr(self, f"_{section}_brackets") if section in PAGED_CHOICE
                     else self._bracket_warnings.get(section)) or ())

    def _apply_significance_everywhere(self, section: str) -> None:
        """`_apply_significance` for every page of ``section`` (the bubbles' conditions or colours changed)."""
        if section in self._slots:
            for _ in self._each(section):
                self._apply_significance(section)
        else:
            self._apply_significance(section)

    def _sig_view(self, section: str) -> dict:
        from ... import significance as sig

        _chart, data = self._sig_chart(section)
        if data is None:
            return {"marks": [], "lines": [], "pending": [],
                    "why": "Run the fit first; then brackets can be drawn on it."}
        return sig.chart_view(sig.SECTION_KINDS[section], {"data": data, "replicate_pairs": self._replicate_pairs()},
                              self._choice_of(section))

    def _replicate_pairs(self) -> dict:
        """The replicates declared as pairs on the Groups board (`project.Project.replicate_pairs`): what the paired
        significance test reads, put in every chart payload beside the data (`significance.declared_pairs`)."""
        return dict(getattr(self.project, "replicate_pairs", None) or {})

    def _apply_significance(self, section: str) -> None:
        """Paint ``section``'s brackets from its choice, refresh the popover's lines when it
        shows that section, and start any mixed-model p still missing (in the process pool)."""
        view = self._sig_view(section)
        chart, _data = self._sig_chart(section)
        if section == "hmm":                        # its caption names the correction; the bubbles' legend does
            chart.set_brackets(view["marks"], view.get("correction_name") or "Holm")
        else:
            chart.set_brackets(view["marks"])
        if section in PAGED_CHOICE:
            setattr(self, f"_{section}_brackets", list(view.get("warnings") or ()))
        else:
            self._bracket_warnings[section] = list(view.get("warnings") or ())
        self._sync_fit_warning(section)
        # the popover describes the page on show: another page's result does not rewrite it
        other_page_shown = (section in self._slots and sp.split_view(self.sidebar.current)[0] == section
                            and self.sidebar.current_page != self._cur[section].key)
        if self.toolbar.significance_section == section and not other_page_shown:
            self._sync_significance_block(view)
        if view["pending"]:
            self._compute_mixed(view["pending"])

    def _sync_fit_warning(self, section: str) -> None:
        """The flag line under ``section``'s bubbles (`bubble_view.warning_line`): SQD's fit warnings with the brackets
        standing on a flagged population beside them; the apparent-D section's fit warnings are in its caption, so its
        line carries the brackets' alone (ruled 4 Nov 2026: no mark on such a bracket, a warning that names it)."""
        from ...bubble_view import warning_line

        brackets = self._brackets_of(section)
        if section == "sqd":
            data = self._card.chart._data
            text = warning_line(list(getattr(data, "warnings", None) or ()) + brackets)
            self.fit_warning.setText(text)
            self.fit_warning.setVisible(bool(text))
        elif section == "appd":
            self._appd.set_bracket_warning(warning_line(brackets))

    def _sync_significance_block(self, view: dict | None = None) -> None:
        """The Significance tool's block for the section it now shows: the
        chart's conditions, its populations, its choice and lines; hidden elsewhere."""
        from ... import significance as sig

        tool = self.toolbar.significance
        section = self.toolbar.significance_section
        if section not in sig.SECTION_KINDS:
            tool.hide_significance()
            return
        _chart, data = self._sig_chart(section)
        view = view or self._sig_view(section)
        kind = sig.SECTION_KINDS[section]
        names = sig.conditions(kind, {"data": data}) if data is not None else []
        conditions = tuple((c, self._condition_labels.get(c, c)) for c in names)
        populations = sig.population_choices(kind, {"data": data}) if data is not None else ()
        tool.set_significance(conditions, populations, kind in sig.MIXED_OFFERED,
                              self._choice_of(section), view["lines"], view["why"],
                              paired=bool(self._replicate_pairs()), kind=kind)

    def _on_significance(self, choice) -> None:
        """The Significance block changed: kept for the section it describes, then shown."""
        from ... import significance as sig

        section = self.toolbar.significance_section
        if section not in sig.SECTION_KINDS:
            return
        self._set_choice(section, choice)
        self._apply_significance(section)

    def _compute_mixed(self, keys) -> None:
        """Mixed-model p values off the GUI process (`significance.compute_mixed` in the
        process pool, as every heavy fit here); each key is asked for once at a time."""
        from ... import significance as sig

        running = {k for job in self._sig_jobs.values() for k in job}
        keys = [k for k in keys if k not in running]
        if not keys:
            return
        self._sig_gen += 1
        self._sig_jobs[self._sig_gen] = keys
        pool = self._process_pool

        def compute():
            return pool.submit(sig.compute_mixed, keys).result()

        self._pool.start(_ComputeRunnable(compute, self._sig_gen, self._sig_signals))

    def _on_mixed_done(self, gen: int, result, error) -> None:
        from ... import significance as sig

        self._sig_jobs.pop(gen, None)
        if error is not None:
            self.status.show()
            self.status.setText(f"The cell test could not be computed: {error}")
            return
        sig.store_mixed(result or {})
        for section in sig.SECTION_KINDS:
            self._apply_significance_everywhere(section)

    # -- figure export: one TARGET per chart, one way to open it, and all at once ---------- #
    @staticmethod
    def _safe(text) -> str:
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(text))

    @staticmethod
    def _range_changes(axes, which=("x", "y")) -> dict:
        """A card's typed ranges (its Axes, `axes_view`) as the spec's range fields, AS SHOWN: an automatic end is
        None, so a range set back to Auto on screen is automatic in the export too (30 Oct 2026). ONE route: the
        figure reads them where it read the dialog's (`figure_export._apply_ranges`)."""
        out = {}
        for axis in which:
            lo, hi = (axes or {}).get(axis) or (None, None)
            out[f"{axis}_min"], out[f"{axis}_max"] = lo, hi
        return out

    def _bubble_target(self):
        data = self._card.chart._data
        if data is None:
            return None
        provenance = None
        if self._last_sqd_run is not None:
            try:
                from ...comparison_controller import provenance_for

                provenance = provenance_for(self.project, self._last_sqd_run[0],
                                            view="sqd_bubbles_figure",
                                            extra={"uncertainty": self._last_sqd_run[1]})
            except Exception as exc:                        # noqa: BLE001 - reported
                self.status.setText(f"provenance not written: {type(exc).__name__}: {exc}")
        if self._card.current_view() == "crossed":
            from ...bubble_view import crossed_payload

            crossed = crossed_payload(data)
            if not crossed.get("available"):
                return None
            hidden = self._card.crossed_hidden()
            if hidden:                                     # as shown: the eyes' hidden conditions left out
                crossed = {**crossed, "points": [p for p in crossed["points"] if p["condition"] not in hidden],
                           "legend_conditions": [e for e in crossed["legend_conditions"] if e[0] not in hidden]}
                if not crossed["points"]:
                    return None
            return {"chart": "sqd_crossed", "kind": "sqd_crossed", "provenance": provenance,
                    "payload": {"crossed": crossed, "data": data,
                                "colours": self._column_colours(data),
                                "file_name": "sqd_fraction_against_D"},
                    "spec_changes": self._range_changes(self._card.typed_axes("crossed"))}
        return {"chart": "sqd_bubbles", "kind": "sqd_bubbles", "provenance": provenance,
                "payload": {"data": data, "colours": self._column_colours(data),
                            "file_name": "sqd_populations"},
                "spec_changes": self._range_changes(self._card.typed_axes("bubbles"), ("y",))}

    def _replicate_target(self):
        fit = self._detail.chart._payload
        if not fit or not fit.get("replicates"):
            return None
        condition = getattr(self._detail, "condition", "")
        return {"chart": f"sqd_replicates:{condition}", "kind": "sqd_replicates",
                "payload": {"payload": fit, "colour": getattr(self._detail, "colour", None),
                            "condition": condition,
                            "file_name": f"sqd_replicates_{self._safe(condition)}"}}

    def _sqd_table_target(self):
        """The SQD data table, the rows its filters show."""
        from ...bubble_view import populations_table

        data = self._card.chart._data
        if data is None:
            return None
        conditions, series = self._card.toolbar_filters.selections()
        payload = populations_table(data, condition_filter=conditions, series_filter=series,
                                    labels=self._condition_labels)
        return {"chart": "table:sqd_populations", "kind": "table", "payload": payload}

    def _replicate_table_target(self):
        from ...bubble_view import replicate_table

        fit = self._detail.chart._payload
        if not fit or not fit.get("replicates"):
            return None
        condition = getattr(self._detail, "condition", "")
        payload = replicate_table(fit, condition, self._condition_labels.get(condition, condition),
                                  tags=self._replicate_pairs().get(condition))
        return {"chart": f"table:sqd_replicates:{condition}", "kind": "table",
                "payload": payload}

    def _sample_size_target(self):
        from ... import images_view as iv

        overview = self._images.overview
        payload = overview.payload()
        if payload is None:
            return None
        level = overview.level()
        return {"chart": f"table:sample_size:{level}", "kind": "table",
                "payload": iv.inventory_table(payload, level, self._condition_labels)}

    def _images_target(self):
        payload = self._images.export_payload()
        if payload is None:
            return None
        name = f"{payload['condition']}_{payload['movie']}"
        if payload.get("cell"):
            name += f"_cell{payload['cell'].rsplit(':', 1)[-1]}"
        payload["file_name"] = self._safe(name)
        return {"chart": f"cell_image:{name}", "kind": "cell_image", "payload": payload}

    def _spatial_target(self, which: str = "grid"):
        if which in ("profiles", "day_profiles"):
            day = which == "day_profiles"
            card = self._spatial.days.profiles if day else self._spatial.profiles
            payload = card.export_payload()
            if payload is None:
                return None
            condition = str(self._spatial.days.condition() or "") if day else ""
            payload["condition_label"] = self._condition_labels.get(condition, condition) if day else ""
            payload["file_name"] = (f"spatial_profiles_{self._safe(condition)}" if day else "spatial_profiles")
            return {"chart": f"spatial_profiles:{condition}" if day else "spatial_profiles",
                    "kind": "spatial_profiles", "payload": payload}
        days = which == "days"
        matrix = self._spatial.days.matrix if days else self._spatial.matrix
        payload = matrix.export_payload()
        if payload is None:
            return None
        condition = str(self._spatial.days.condition() or "") if days else ""
        payload["condition"] = condition
        payload["file_name"] = (f"spatial_map_days_{self._safe(condition)}" if days
                                else "spatial_map")
        return {"chart": f"spatial_maps:replicates:{condition}" if days else "spatial_maps",
                "kind": "spatial_maps", "payload": payload}

    def _qc_target(self, key: str, condition: str | None = None):
        if condition is None:
            got = self._qc_detail.export_payload(key)
        else:
            from .qc_chart import qc_panel_payload

            got = qc_panel_payload(self._qc_payloads.get(condition) or {}, condition, key,
                                   self._condition_colours([condition]).get(condition))
        if got is None:
            return None
        kind, payload = got
        if kind == "table":                          # N per group: named by what it is
            return {"chart": f"table:qc_groups:{payload['condition']}", "kind": kind,
                    "payload": payload}
        payload["file_name"] = f"{kind}_{self._safe(payload['condition'])}"
        return {"chart": f"{kind}:{payload['condition']}", "kind": kind, "payload": payload}

    def _msd_target(self, key: str = ""):
        entries = self._msd.export_entries(key)
        if not entries:
            return None
        kind = "msd_condition" if key else "msd_overlay"
        return {"chart": f"msd_condition:{key}" if key else "msd_overlay", "kind": kind,
                "payload": {"entries": entries, "file_name": f"msd_{self._safe(key)}"
                            if key else "msd_all_conditions"},
                "spec_changes": self._range_changes(self._msd.typed_axes(key))}

    def _msd_page_target(self, key: str, page: str):
        """An MSD Checks ("checks") or v2 ("v2") card's figure, or None before its result."""
        entries = self._msd.export_entries(key)
        if not entries:
            return None
        kind = "msd_checks" if page == "checks" else "msd_v2"
        return {"chart": f"{kind}:{key}", "kind": kind,
                "payload": {"entries": entries, "file_name": f"{kind}_{self._safe(key)}"}}

    def _msd_table_target(self):
        """The MSD Curves page's table, or None before a result."""
        payload = self._msd.table_export()
        if payload is None:
            return None
        return {"chart": "table:msd", "kind": "table", "payload": payload}

    def _jumps_target(self, key: str):
        entries = self._jumps.export_entries(key)
        if not entries:
            return None
        # the card's typed ranges, as shown (30 Oct 2026): the dialog's range fields start on them
        axes = entries[0].get("axes") or {}
        x, y = axes.get("x") or (None, None), axes.get("y") or (None, None)
        return {"chart": f"jumps_condition:{key}", "kind": "jumps_condition",
                "payload": {"entries": entries, "file_name": f"jumps_{self._safe(key)}"},
                "spec_changes": {"x_min": x[0], "x_max": x[1], "y_min": y[0], "y_max": y[1]}}

    def _jumps_overview_target(self):
        """Every condition's jump histogram on one figure, or None before the first result."""
        entries = self._jumps.export_all_entries()
        if not entries:
            return None
        settings = self._jumps.overview_settings()
        x = (settings.get("axes") or {}).get("x") or (None, None)
        # the page's Columns (one plot per condition) is where the dialog's Grid row starts; never more than are drawn
        columns = settings.pop("columns", None)
        if settings.get("layout") != "condition" or columns is None:
            columns = None
        else:
            columns = min(int(columns), len(entries))
        # the residual plots as the All conditions view shows them (12 Nov 2026; `FigureSpec.pearson_residuals`,
        # `.ecdf_residuals`, off by default here as there)
        residuals = self._jumps.residuals_shown()
        return {"chart": "jumps_overview", "kind": "jumps_overview",
                "payload": {"entries": entries, "file_name": "jumps_all_conditions", **settings},
                "spec_changes": {"x_min": x[0], "x_max": x[1], "grid_columns": columns,
                                 "pearson_residuals": residuals["pearson"], "ecdf_residuals": residuals["ecdf"]}}

    def _jd_target(self, key: str):
        """A Signed jumps card's figure, on the axis and count the page shows, or None."""
        entries = self._jd.export_entries(key)
        if not entries:
            return None
        return {"chart": f"jd_condition:{key}", "kind": "jd_condition",
                "payload": {"entries": entries,
                            "file_name": f"signed_jumps_{entries[0]['axis']}_{self._safe(key)}"}}

    def _jd_table_target(self):
        payload = self._jd.table_export()
        if payload is None:
            return None
        payload["file_name"] = "signed_jumps_table"
        return {"chart": "table:signed_jumps", "kind": "table", "payload": payload}

    def _hmm_target(self, key: str):
        entries = self._hmm.export_entries(key)
        if not entries:
            return None
        return {"chart": f"hmm_condition:{key}", "kind": "hmm_condition",
                "payload": {"entries": entries, "file_name": f"hmm_{self._safe(key)}"}}

    def _hmm_overview_target(self):
        """Every shown condition's HMM state diagram in one figure (5 Nov 2026, `figure_export`'s "hmm_overview"), or
        None before a fit."""
        entries = self._hmm.overview_entries()
        if not entries:
            return None
        # the conditions the comparison's eyes hide start unticked in the dialog (`FigureSpec.left_out`)
        return {"chart": "hmm_overview", "kind": "hmm_overview",
                "payload": {"entries": entries, "file_name": "hmm_all_conditions"},
                "spec_changes": {"left_out": self._hmm.overview_left_out()}}

    def _hmm_comparison_target(self):
        """The HMM section's "Across conditions" card: every condition's slow state, from
        the payload the card paints."""
        payload = self._hmm.comparison_export()
        if payload is None:
            return None
        payload["file_name"] = "hmm_across_conditions"
        # the pair lines and the dwell panel as the card shows them (`FigureSpec.pair_lines`, `hmm_dwell`, both off by
        # default)
        return {"chart": "hmm_comparison", "kind": "hmm_comparison", "payload": payload,
                "spec_changes": {"pair_lines": self._hmm.pair_lines_on(), "hmm_dwell": self._hmm.dwell_on()}}

    def _hmm_angles_target(self):
        """The Jump angles page's figure (kind "hmm_angles"), from what the page shows, or None before a result."""
        payload = self._angles_page.export_payload()
        if payload is None:
            return None
        payload["file_name"] = "hmm_jump_angles"
        return {"chart": "hmm_angles", "kind": "hmm_angles", "payload": payload}

    def _appd_target(self):
        """The Apparent-D populations figure (kind "appd_bubbles"), or None before a Run."""
        data = self._appd.bubble_data()
        if data is None:
            return None
        return {"chart": "appd_bubbles", "kind": "appd_bubbles",
                "payload": {"data": data, "colours": self._column_colours(data),
                            "file_name": "apparent_d_populations"},
                "spec_changes": self._range_changes(self._appd.populations.typed_axes("bubbles"), ("y",))}

    def _appd_page_target(self, key: str, page: str):
        """A Distributions ("distributions") or Checks ("checks") card's figure, or None before its result."""
        entries = self._appd.export_entries(key, page)
        if not entries:
            return None
        kind, stem = (("appd_histogram", "apparent_d") if page == "distributions"
                      else ("appd_checks", "apparent_d_checks"))
        target = {"chart": f"{kind}:{key}", "kind": kind,
                  "payload": {"entries": entries, "file_name": f"{stem}_{self._safe(key)}"}}
        if page == "distributions":
            target["spec_changes"] = self._range_changes(self._appd.dist_axes(key))
        return target

    def _appd_overview_target(self):
        """Every condition's apparent-D distribution in one figure (the Distributions page's All conditions), or None
        before a result (3 Nov 2026, `figure_export`'s "appd_overview")."""
        entries = self._appd.overview_entries()
        if not entries:
            return None
        return {"chart": "appd_overview", "kind": "appd_overview",
                "payload": {"entries": entries, "file_name": "apparent_d_all_conditions"}}

    def _appd_cell_length_target(self):
        """The cell-length card's figure, or None before a Run (or without measured lengths)."""
        payload = self._appd.cell_length_export()
        if payload is None:
            return None
        payload["file_name"] = f"apparent_d_cell_length_{payload.get('source', 'appd')}"
        return {"chart": "appd_cell_length", "kind": "appd_cell_length", "payload": payload}

    def _appd_cells_target(self):
        """The Per cell card's figure, or None before a Run has finished."""
        payload = self._appd.cells_export()
        if payload is None:
            return None
        payload["file_name"] = "apparent_d_per_cell"
        return {"chart": "appd_cells", "kind": "appd_cells", "payload": payload}

    def _appd_table_target(self):
        """The Apparent-D populations table, the rows its filters show (`_sqd_table_target`'s twin)."""
        from ...bubble_view import populations_table

        data = self._appd.bubble_data()
        if data is None:
            return None
        conditions, series = self._appd.populations.toolbar_filters.selections()
        payload = populations_table(data, condition_filter=conditions, series_filter=series,
                                    labels=self._condition_labels)
        return {"chart": "table:appd_populations", "kind": "table", "payload": payload}

    def _kinetics_target(self, key: str):
        entries = self._kinetics.export_entries(key)
        if not entries:
            return None
        # the card's readout decides the figure: the populations over time, or the HMM slow fraction
        kind = ("kinetics_populations" if entries[0]["result"].get("readout") == "populations"
                else "kinetics_condition")
        target = {"chart": f"{kind}:{key}", "kind": kind,
                  "payload": {"entries": entries, "file_name": f"kinetics_{self._safe(key)}"}}
        if kind == "kinetics_populations":
            # the shared time axis through the spec (the dialog's x); each panel's y rides in the entry ("axes")
            target["spec_changes"] = self._range_changes(entries[0].get("axes"), ("x",))
        return target

    def _kinetics_overview_target(self):
        """Every condition's Populations over time in one figure (7 Nov 2026, `figure_export`'s "kinetics_overview"):
        the conditions as columns on one time axis, one y range per row. None before two results."""
        entries = self._kinetics.overview_entries()
        if len(entries) < 2:
            return None
        return {"chart": "kinetics_overview", "kind": "kinetics_overview",
                "payload": {"entries": entries, "file_name": "kinetics_all_conditions"},
                "spec_changes": self._range_changes(entries[0].get("axes"), ("x",))}

    def _kinetics_mock_target(self):
        """Every treated condition against its mock addition in one figure (8 Nov 2026, `figure_export`'s
        "kinetics_mock"): a panel per treated card whose comparison could be made. None before one."""
        entries = self._kinetics.mock_entries()
        if not entries:
            return None
        return {"chart": "kinetics_mock", "kind": "kinetics_mock",
                "payload": {"entries": entries, "file_name": "kinetics_against_the_mock"}}

    def export_targets(self) -> list:
        """Every chart that has something to export right now, in sidebar order."""
        from .qc_chart import QCConditionDetail

        found = self._page_targets("sqd", lambda: [self._bubble_target(), self._sqd_table_target(),
                                                   self._replicate_target(), self._replicate_table_target()])
        found += [self._qc_target(key, condition) for condition in self._qc_payloads
                  for key in QCConditionDetail.EXPORT_KINDS]
        found += [self._sample_size_target(), self._images_target()]
        found += self._page_targets("appd", lambda: (
            [self._appd_target(), self._appd_table_target()]
            + [self._appd_page_target(key, "distributions") for key in self._appd.conditions()]
            + ([self._appd_overview_target()] if len(self._appd.overview_entries()) > 1 else [])
            + [self._appd_cells_target(), self._appd_cell_length_target()]
            + [self._appd_page_target(key, "checks") for key in self._appd.conditions()]))
        found += self._page_targets("spatial", lambda: [self._spatial_target("grid"),
                                                        self._spatial_target("profiles")])
        found += self._page_targets("kinetics", lambda: [self._kinetics_target(key) for key in self._kinetics.cards()]
                                    + [self._kinetics_overview_target(), self._kinetics_mock_target()])
        found += self._page_targets("msd", lambda: (
            [self._msd_target("")] + [self._msd_target(key) for key in self._msd.cards()]
            + [self._msd_table_target()]
            + [self._msd_page_target(key, page) for page in ("checks", "v2") for key in self._msd.cards()]))
        found += self._page_targets("sqd", lambda: [self._jumps_target(key) for key in self._jumps.cards()]
                                    + ([self._jumps_overview_target()]
                                       if len(self._jumps.export_all_entries()) > 1 else []))
        found += self._page_targets("sqd", lambda: [self._jd_target(key) for key in self._jd.cards()]
                                    + [self._jd_table_target()])
        found += self._page_targets("clusters", lambda: (
            [self._clusters_target(key) for key in self._clusters.cards()]
            + ([self._clusters_target(SHARED_CLUSTERS_KEY, curves=True)]
               if SHARED_CLUSTERS_KEY in self._clusters.cards() else [])))
        found += self._page_targets("hmm", lambda: [self._hmm_comparison_target()]
                                    + ([self._hmm_overview_target()] if len(self._hmm.overview_entries()) > 1 else [])
                                    + [self._hmm_target(key) for key in self._hmm.cards()])
        return [self._titled(target) for target in found if target is not None]

    def _page_target(self, target, slot):
        """``target`` of an ADDED page made distinct: its chart key (the remembered settings) and its file name carry the
        page, so two pages' figures never write over each other and are told apart on disk (the section's own page
        keeps its names)."""
        if target is None or slot.number is None:
            return target
        name = self._page_name(slot)
        payload = target.get("payload")
        if isinstance(payload, dict):
            payload = dict(payload, page_name=name,
                           file_name=f"{sp.safe_name(name)}_{payload.get('file_name', target['kind'])}")
        return dict(target, chart=f"{target['chart']}@{slot.key}", payload=payload)

    def _page_targets(self, section: str, getter) -> list:
        """``getter()``'s targets (a list) on every page of ``section``, each page current for its call."""
        out = []
        for slot in self._each(section):
            out += [self._page_target(t, slot) for t in getter()]
        return out

    def _titled(self, target: dict) -> dict:
        """``target`` with what its automatic title is made of (`figure_export.auto_title`):
        the project's name -- the folder holding the condition folders, as the sidebar
        shows it -- and the display name of the one condition it draws, if any."""
        payload = target.get("payload")
        if not isinstance(payload, dict):
            return target
        payload = dict(payload, project=getattr(self, "_project_title", "") or "")
        condition = payload.get("condition")
        if condition:
            payload["condition_label"] = self._condition_labels.get(condition, condition)
        # EVERY CONDITION COLOUR IN BOTH PAGE VARIANTS (4 Nov 2026): the figure draws the one for ITS page
        # (`figure_export.page_coloured`), so the dark theme never changes an exported figure's colours
        conditions = self.toolbar.conditions
        payload[condition_colours.PAGE_COLOURS] = conditions.model.page_variants(dark=bool(conditions.dark))
        # THE DECLARED PAIRS (5 Nov 2026): a bracket's paired test reads them from the payload, on screen and here
        pairs = self._replicate_pairs()
        if pairs:
            payload["replicate_pairs"] = pairs
        return dict(target, payload=payload)

    # -- each chart's own settings, remembered per project ----------------------------- #
    def _restore_image_channels(self) -> None:
        """Images & tracks' channels as last saved for this project; a set that no longer reads is dropped."""
        from ..theme import remembered

        stored = ((remembered("image_channels") or {}).get(self._project_key())) or []
        try:
            self._images.set_channels(list(stored))
        except (ValueError, TypeError, AttributeError, KeyError):
            self._images.set_channels([])

    def _remember_image_channels(self, channels: list) -> None:
        from ..theme import remember, remembered

        stored = dict(remembered("image_channels") or {})
        stored[self._project_key()] = [dict(c) for c in channels]
        remember("image_channels", stored)

    def _restore_foci(self) -> None:
        """The foci marked in this project before (Images & tracks' Distances to foci)."""
        from ..theme import remembered

        stored = ((remembered("distance_foci") or {}).get(self._project_key())) or {}
        try:
            self._images.set_foci(dict(stored))
        except (ValueError, TypeError, AttributeError):
            self._images.set_foci({})

    def _remember_foci(self, foci: dict) -> None:
        from ..theme import remember, remembered

        stored = dict(remembered("distance_foci") or {})
        stored[self._project_key()] = foci
        remember("distance_foci", stored)

    def open_distances(self):
        """v2's Distances measurements (17 Oct 2026): every marked cell's localisations to their nearest focus,
        beside chance (`distances_view.distance_payload`), in a dialog. Returns it, or None with a toast saying why."""
        from ...distances_view import DistancesError, distance_payload
        from .distances_dialog import DistancesDialog

        try:
            payload = distance_payload(self.project, self._images.foci(), labels=self._condition_labels)
        except DistancesError as exc:
            self.toasts.post("No distances to show", description=str(exc))
            return None
        keys = [c["key"] for c in payload["conditions"]]
        content = self.centralWidget() if hasattr(self, "centralWidget") else None
        dialog = DistancesDialog(self.t, payload, self._condition_colours(keys), content=content, parent=self)
        dialog.closed.connect(dialog.deleteLater)
        dialog.open()
        self._distances_dialog = dialog
        return dialog

    def _on_nesting(self, order) -> None:
        """Another nesting was chosen in the Conditions tool: every list of conditions follows it (the Compute
        dialog's rows, the Conditions tool, the next Run of every section) and it is remembered for the project.
        No number changes: a fit is per condition."""
        order = tuple(order)
        if order == self._factor_order:
            return
        self._factor_order = order
        factor_layout.remember(self._project_key(), order)
        # FROM THE ORDER LAST SET (remembered), not the one on screen: each group keeps its own order (D14), and a
        # nesting re-derived from another nesting's columns would lose it, so switching away and back gives it again
        self._all_condition_keys = condition_order.arranged(
            condition_order.recall(self._project_key(), self._all_condition_keys), None, self._factors, order)
        self.toolbar.conditions.set_conditions(
            [(c, self._condition_labels.get(c, c)) for c in self._all_condition_keys])
        self.compute_dialog.compute.reorder(self._all_condition_keys)
        # the colours follow the new inner factor unless one was chosen (emits colours_changed: every chart)
        self.toolbar.conditions.follow_nesting(order)
        # THE CHARTS ALREADY DRAWN regroup in place: the same points, moved (`bubble_view.regrouped`)
        for _ in self._each("sqd"):
            if getattr(self, "_sqd_plot", None) is not None:
                self._sqd_plot = self._grouped(self._sqd_plot)
                self._card.set_data(self._sqd_plot, self._column_colours(self._sqd_plot))
        for _ in self._each("appd"):
            self._appd.set_grouping(self._factors, self._factor_order)
        for _ in self._each("hmm"):
            self._hmm.set_grouping(self._factors, self._factor_order)
        for _ in self._each("spatial"):
            self._spatial.matrix.set_grouping(self._factors, self._factor_order)
        self._group_keys()
        for section in ("sqd", "appd"):
            self._apply_significance_everywhere(section)

    def set_condition_order(self, order) -> None:
        """Draw the conditions in ``order`` everywhere (3 Nov 2026, asked: "set order ... change order of how samples
        are arranged. not only sqd bubbles, but everywhere possible"): the Conditions tool, the Compute dialog's rows,
        every section's cards, rows, columns and tiles, and so every figure exported from them. DISPLAY ONLY: no fit is
        made again and no cache is keyed on the order. A condition ``order`` leaves out keeps its place after the
        rest; the folder design's nesting still groups (`condition_order.arranged`)."""
        keys = condition_order.arranged(condition_order.apply(self._all_condition_keys, order), None,
                                        self._factors, self._factor_order)
        if keys == self._all_condition_keys:
            return
        self._all_condition_keys = keys
        condition_order.remember(self._project_key(), keys)
        self.toolbar.conditions.set_conditions([(c, self._condition_labels.get(c, c)) for c in keys])
        self.compute_dialog.compute.reorder(keys)
        self._apply_condition_order()

    def _apply_condition_order(self) -> None:
        """Every section's drawn results in `_all_condition_keys`' order -- the same results, moved."""
        from ...bubble_view import reordered

        order = list(self._all_condition_keys)
        for _ in self._each("sqd"):
            if getattr(self, "_sqd_plot", None) is not None:
                self._sqd_plot = self._grouped(reordered(self._sqd_plot, order))
                self._card.set_data(self._sqd_plot, self._column_colours(self._sqd_plot))
            for section in (self._jumps, self._jd):
                section.set_order(order)
        for _ in self._each("msd"):
            self._msd.set_order(order)
        for _ in self._each("appd"):
            self._appd.set_order(order)
        for _ in self._each("hmm"):
            self._hmm.set_order(order)
        for _ in self._each("kinetics"):
            self._kinetics.set_order(order)
        for _ in self._each("clusters"):
            self._clusters.set_order(order)
        self._images.set_order(order)
        for _ in self._each("spatial"):
            matrix = self._spatial.matrix
            if matrix.conditions():
                matrix.set_conditions(condition_order.apply(matrix.conditions(), order), dict(matrix._labels))
        if getattr(self, "_qc_result", None):
            self._show_qc_cards(self._qc_result)
        self._group_keys()
        for section in ("sqd", "appd"):
            self._apply_significance_everywhere(section)

    def _grouped(self, data):
        """``data`` (`bubble_view.BubblePlotData`) in the nesting's order with its grouped layout, when the project
        has two or more factors; else unchanged."""
        from ...bubble_view import regrouped

        if data is None or len(self._factor_order) < 2 or not self._factors:
            return data
        return regrouped(data, self._factors, self._factor_order)

    def _project_key(self) -> str:
        roots = getattr(self.project, "roots", None) or {}
        # `condition_order.project_key`: the door keys an order set on its tabs the same way (6 Nov 2026)
        return (condition_order.project_key(list(roots.values())) if roots else None) or str(
            getattr(self, "_project_title", "project"))

    def _chart_spec(self, chart: str):
        """``chart``'s settings as last exported in this project (titles and names too)."""
        from ..theme import remembered

        from ... import figure_export as fx

        record = ((remembered("figure_charts") or {}).get(self._project_key()) or {}).get(chart)
        return fx.FigureSpec.from_dict(record) if record else None

    def _remember_chart_spec(self, chart: str, spec) -> None:
        from ..theme import remember, remembered

        store = dict(remembered("figure_charts") or {})
        mine = dict(store.get(self._project_key()) or {})
        mine[chart] = spec.record()
        store[self._project_key()] = mine
        try:
            remember("figure_charts", store)
        except OSError:
            pass

    def _target_spec(self, target):
        """The settings ``target`` starts from: those it was last exported with, else (for a
        table never exported) the general ones at a width the table fits."""
        spec = self._chart_spec(target["chart"])
        if spec is None and target["kind"] == "table":
            from ... import figure_export as fx
            from .export_dialog import remembered_spec

            spec = fx.roomy_table_spec(target["payload"], remembered_spec())
        return spec

    def _job_spec(self, target):
        """The settings "Export every figure" writes ``target`` with: those it was last exported with, and what the
        chart shows that the spec must carry (its typed ranges, "spec_changes") -- as its Export dialog would start."""
        from ... import figure_export as fx
        from .export_dialog import remembered_spec

        spec = self._target_spec(target) or remembered_spec()
        if target.get("spec_changes"):
            spec = fx.with_changes(spec, **target["spec_changes"])
        return spec

    def _open_export(self, target, missing: str, section: str | None = None):
        """The Export dialog for ``target``, starting from that chart's own settings. ``section``: the paged section
        the chart belongs to; on an added page the figure carries the page's name (`_page_target`)."""
        if target is not None and section in self._slots:
            target = self._page_target(target, self._cur[section])
        if target is None:
            if missing:
                self.status.show()
                self.status.setText(missing)
            return None
        from ...buildstamp import build_stamp
        from .export_dialog import ExportDialog

        target = self._titled(target)
        chart = target["chart"]
        spec = self._target_spec(target)
        # THE CHART'S SIGNIFICANCE, AS SHOWN (14 Oct 2026): a bubble chart exports the brackets
        # its Visibility options draw, whatever an earlier export of it carried.
        from ... import significance as sig

        section = {kind: s for s, kind in sig.SECTION_KINDS.items()}.get(target["kind"])
        if section is not None:
            from ... import figure_export as fx
            from .export_dialog import remembered_spec

            spec = fx.with_changes(spec or remembered_spec(),
                                   **self._choice_of(section).spec_changes())
        if target.get("spec_changes"):
            # what the chart shows and the spec must carry (the Jump distances cards' typed ranges)
            from ... import figure_export as fx
            from .export_dialog import remembered_spec

            spec = fx.with_changes(spec or remembered_spec(), **target["spec_changes"])
        dialog = ExportDialog(self.t, target["kind"], target["payload"],
                              provenance=target.get("provenance"), stamp=build_stamp(),
                              content=self.centralWidget(), parent=self, spec=spec)
        dialog.closed.connect(dialog.deleteLater)
        dialog.composer_requested.connect(
            lambda record, d=dialog: self._add_to_composer(record, d))
        dialog.exported.connect(
            lambda _paths, d=dialog, c=chart: self._remember_chart_spec(c, d.spec))
        dialog.open()
        return dialog

    def open_bubble_export(self):
        """The SQD card's Export button: this Run's bubbles at print size."""
        return self._open_export(self._bubble_target(),
                                 "Run the SQD fit first; then its figure can be exported.", section="sqd")

    def open_replicate_export(self):
        """The replicate-days page's Export button: that condition's per-replicate fit."""
        return self._open_export(self._replicate_target(),
                                 "This condition has no replicate fit to export.", section="sqd")

    def open_sqd_table_export(self):
        """The SQD card's table Export: the data table's rows as a table figure."""
        return self._open_export(self._sqd_table_target(),
                                 "Run the SQD fit first; then its table can be exported.", section="sqd")

    def open_replicate_table_export(self):
        """The replicate-days page's table Export."""
        return self._open_export(self._replicate_table_target(),
                                 "This condition has no replicate fit to export.", section="sqd")

    def open_sample_size_export(self):
        """The Images Overview's table Export: the sample size, by day or by movie."""
        return self._open_export(self._sample_size_target(),
                                 "The sample size is still being counted.")

    def open_images_export(self):
        """The Images page's Export: the movie, or the zoomed cell, as shown."""
        return self._open_export(self._images_target(),
                                 "Open a movie with an image first; then it can be exported.")

    def open_spatial_export(self, which: str = "grid"):
        """The Spatial Map's Export: the grid on screen (every condition), or -- from a
        condition's replicate-day page -- that condition's days."""
        return self._open_export(self._spatial_target(which),
                                 "Run the maps first; then they can be exported.", section="spatial")

    def open_qc_export(self, key: str):
        """A QC Detail card's Export: that panel, for the condition on screen."""
        return self._open_export(self._qc_target(key),
                                 "This panel has nothing to export for this condition.")

    def open_msd_export(self, key: str = ""):
        """An MSD Export button: "" the shared chart of every condition, else one."""
        return self._open_export(self._msd_target(key), "", section="msd")

    def open_clusters_export(self, key: str):
        """A Clustering card's Export: that condition's cluster curves, on the ranges its Axes shows."""
        return self._open_export(self._clusters_target(key), "This condition has no clustering result to export.",
                                 section="clusters")

    def _clusters_target(self, key: str, curves: bool = False):
        """A Clustering card's export target. The shared card's own is its shares figure; ``curves`` asks for the
        pooled clusters' curves instead (the same kind as a condition's)."""
        entries = self._clusters.export_entries(key)
        if not entries:
            return None
        if key == SHARED_CLUSTERS_KEY and not curves:
            shares = self._clusters.shares_export()
            return {"chart": "clusters_shares", "kind": "clusters_shares",
                    "payload": {**shares, "file_name": "clusters_shares"}, "spec_changes": {}}
        return {"chart": f"clusters_condition:{key}", "kind": "clusters_condition",
                "payload": {"entries": entries, "file_name": f"clusters_{self._safe(key)}"},
                "spec_changes": self._range_changes(self._clusters.typed_axes(key))}

    def open_msd_page_export(self, key: str, page: str):
        """An MSD Checks or v2 card's Export: that condition's checks, or v2's sigma histogram."""
        return self._open_export(self._msd_page_target(key, page), "This condition has no MSD result to export.",
                                 section="msd")

    def save_msd_per_cell(self, path: str | None = None) -> str | None:
        """The MSD section's "Save MSD per cell (CSV)": every cell's MSD into ``path`` (asked for when None). Returns
        the path written, or None."""
        text = self._msd.cells_csv_text()
        if text is None:
            self.toasts.post("No cell's MSD to save", description="Run MSD on conditions with cell outlines first.")
            return None
        if path is None:
            from qtpy.QtWidgets import QFileDialog

            slot = self._cur["msd"]
            stem = "msd_per_cell" if slot.number is None else f"{sp.safe_name(self._page_name(slot))}_msd_per_cell"
            path, _ = QFileDialog.getSaveFileName(self, "Save MSD per cell", f"{stem}.csv", "CSV (*.csv)")
            if not path:
                return None
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        self.toasts.post("MSD per cell saved", description=path)
        return path

    def open_msd_table_export(self):
        """The MSD Curves table's Export."""
        return self._open_export(self._msd_table_target(), "Run MSD first; then its table can be exported.",
                                 section="msd")

    def open_jumps_export(self, key: str):
        """A Jump distances card's Export: that condition's histogram, residuals and fits,
        the fits drawn the card's own (its Show) unless the dialog's Fit row sets others."""
        return self._open_export(self._jumps_target(key),
                                 "This condition has no jump-distance fits to export.", section="sqd")

    def open_jumps_overview_export(self):
        """The All conditions view's Export all: every condition's histogram in one figure, each as its card shows
        it (the dialog's Fit row can set them all)."""
        return self._open_export(self._jumps_overview_target(),
                                 "No condition has jump-distance fits to export yet.", section="sqd")

    def open_jd_export(self, key: str):
        """A Signed jumps card's Export: that condition's signed jumps on the axis shown, the count the card's unless
        the dialog picks another."""
        return self._open_export(self._jd_target(key), "This condition has no signed-jump fit on this axis.", section="sqd")

    def open_jd_table_export(self):
        """The Signed jumps table's Export."""
        return self._open_export(self._jd_table_target(), "Run Signed jumps first; then its table can be exported.", section="sqd")

    def open_kinetics_overview_export(self):
        """The Kinetics section's "Export all conditions…": every Populations over time card in one figure."""
        return self._open_export(self._kinetics_overview_target(),
                                 "Two conditions need a Populations over time result to be drawn together.",
                                 section="kinetics")

    def open_kinetics_mock_export(self):
        """The Kinetics section's "Export against the mock…": every treated condition against its mock in one figure."""
        return self._open_export(self._kinetics_mock_target(),
                                 "Run Kinetics with a mock addition chosen to draw a comparison.",
                                 section="kinetics")

    def open_hmm_export(self, key: str):
        """An HMM card's Export: that condition's state diagram."""
        return self._open_export(self._hmm_target(key),
                                 "This condition has no HMM fit to export.", section="hmm")

    def open_hmm_comparison_export(self):
        """The HMM comparison card's Export: every condition's slow state, four panels."""
        return self._open_export(self._hmm_comparison_target(),
                                 "No condition has an HMM fit to compare yet.", section="hmm")

    def open_hmm_angles_export(self):
        """The Jump angles tab's Export: the strip across conditions, each condition's histograms, the tables."""
        return self._open_export(self._hmm_angles_target(),
                                 "No condition has jump angles to export yet.", section="hmm")

    def open_hmm_overview_export(self):
        """The HMM section's "Export all diagrams…": every shown condition's state diagram in one figure."""
        return self._open_export(self._hmm_overview_target(),
                                 "No condition has an HMM fit to export yet.", section="hmm")

    def open_appd_export(self):
        """The Apparent-D populations card's Export."""
        return self._open_export(self._appd_target(),
                                 "Run Apparent Diffusion first; then its figure can be exported.", section="appd")

    def open_appd_distribution_export(self, key: str):
        """A Distributions card's Export: that condition's histogram with the fit the card shows."""
        return self._open_export(self._appd_page_target(key, "distributions"),
                                 "This condition has no apparent-D fit to export.", section="appd")

    def open_appd_overview_export(self):
        """The Distributions page's "Export all conditions…": every condition's histogram in one figure, each as its
        card shows it (the dialog's Fit row can set them all)."""
        return self._open_export(self._appd_overview_target(),
                                 "No condition has an apparent-D fit to export yet.", section="appd")

    def open_appd_checks_export(self, key: str):
        """A Checks card's Export: that condition's switching and track-length checks."""
        return self._open_export(self._appd_page_target(key, "checks"),
                                 "This condition has no apparent-D result to export.", section="appd")

    def open_appd_cell_length_export(self):
        """The cell-length card's Export: slow share against cell length, from the source picked."""
        return self._open_export(self._appd_cell_length_target(),
                                 "No cell has a measured length here (the outlines need a centreline).", section="appd")

    def open_appd_cells_export(self):
        """The Per cell card's Export: every condition's cells."""
        return self._open_export(self._appd_cells_target(),
                                 "Run Apparent Diffusion first; then its per-cell figure can be exported.", section="appd")

    def open_appd_table_export(self):
        """The Apparent-D populations card's table Export."""
        return self._open_export(self._appd_table_target(),
                                 "Run Apparent Diffusion first; then its table can be exported.", section="appd")

    def open_kinetics_export(self, key: str):
        """A Kinetics card's Export: that condition's response curve."""
        return self._open_export(self._kinetics_target(key), "", section="kinetics")

    def export_every_figure(self, folder=None):
        """The Composer's "Export every figure…": every chart computed here, each with the
        settings it was last exported with and the axis ranges its chart shows (`_job_spec`), plus the
        composition, into one folder -- the re-export after a re-run. Written on a worker; the Composer
        page reports it."""
        from ... import figure_export as fx
        from .export_dialog import REMEMBER_FOLDER

        jobs = [{"kind": t["kind"], "payload": t["payload"],
                 "spec": self._job_spec(t),
                 "name": t["payload"].get("file_name", t["kind"]),
                 "provenance": t.get("provenance")} for t in self.export_targets()]
        composed = self._composer.composed_job()
        if composed is not None:
            jobs.append(composed)
        if not jobs:
            self._composer.say("Nothing to export yet: run a section, then come back.")
            return None
        if folder is None:
            from qtpy.QtWidgets import QFileDialog

            from ..theme import remembered

            folder = QFileDialog.getExistingDirectory(
                self, "Export every figure into", remembered(REMEMBER_FOLDER) or str(Path.home()))
            if not folder:
                return None
        from ...buildstamp import build_stamp

        job = _ExportAllJob(jobs, folder, build_stamp())
        job.signals.progress.connect(lambda i, n, name: self._composer.say(
            f"Writing {i + 1} of {n}: {name}…"))
        job.signals.done.connect(self._export_all_done)
        self._export_all_running = job
        QThreadPool.globalInstance().start(job)
        return job

    def _export_all_done(self, rows: list, folder: str, error: str) -> None:
        self._export_all_running = None
        if error:
            self._composer.say(f"Export stopped: {error}")
            return
        written = [r for r in rows if not r["error"]]
        refused = [r for r in rows if r["error"]]
        flagged = sum(1 for r in written if r["checks"])
        text = f"Wrote {len(written)} figure{'s' if len(written) != 1 else ''} to {folder}"
        if flagged:
            text += f"; {flagged} with check findings"
        if refused:
            text += "; not written: " + ", ".join(r["name"] for r in refused)
        # a figure written without its composition file (`figure_export.export_many`'s "note"): said
        text += "".join(f"; {r['name']}: {r['note']}" for r in written if r.get("note"))
        self._composer.say(text + " (figures.txt lists them).")

    def _images_trail(self) -> None:
        """The breadcrumb: the Overview alone, or the condition shown with the others offered."""
        condition = self._images.condition()
        if condition is None:
            self.breadcrumb.set_trail(self.sidebar.trail(self.sidebar.current))
        else:
            self.breadcrumb.set_trail(self.sidebar.trail(self.sidebar.current) + [condition],
                                      dropdown_options=self._images.conditions())

    def _images_condition_changed(self, _condition: str) -> None:
        if self.sidebar.current_base.partition(".")[0] == "images":
            self._images_trail()

    def _add_to_composer(self, record: dict, dialog) -> None:
        """An Export dialog's Add to composer: the chart as set there becomes a panel."""
        letter = self._composer.add_panel(record)
        dialog.say(f"Added as panel {letter}. Arrange and export it in Composer, at the "
                   "foot of the sidebar.")

    def _condition_colours(self, conditions) -> dict:
        """Every drawn condition's colour, read from the workspace's own
        Conditions tool -- see `bubble_chart.py`'s module docstring on why
        this chart does not compute its own palette. GUI-thread only: the
        `ConditionsBody` this reads is a live widget, never touched from the
        worker (`compute_bubble_view` runs off-thread and returns no colour
        of its own for exactly that reason)."""
        return {c: (self.toolbar.conditions.colour_for(c) or self.t.muted_foreground)
                for c in conditions}

    def _column_colours(self, data) -> dict:
        """A bubble figure's colours by column name (SQD's and the apparent-D populations'): each condition's own, or
        under fits side by side its shade per count (`bubble_view.column_colours`), for the chart, its replicate page
        and its export alike."""
        from ...bubble_view import column_colours, fit_conditions

        return column_colours(data, self._condition_colours(fit_conditions(data)))

    def _on_colours_changed(self) -> None:
        """Re-colour everything already drawn, in every section -- no refit.

        It once re-coloured the SQD bubbles only, so a change made while looking
        at Apparent D, QC or Kinetics appeared to do nothing at all."""
        for _ in self._each("sqd"):
            if self._card.conditions():
                self._card.set_colours(self._column_colours(getattr(self, "_sqd_plot", None)))
            self._jd.set_colours(self._condition_colours(self._all_condition_keys))
            self._jumps.set_colours(self._condition_colours(self._all_condition_keys))
            detail = getattr(self, "_detail_condition", None)
            if detail is not None:
                self._detail.chart.set_colour(self._column_colours(getattr(self, "_sqd_plot", None)).get(detail))
        for _ in self._each("appd"):
            if self._appd.conditions():
                self._appd.set_colours(self._condition_colours(self._appd.conditions()))
        if getattr(self, "_qc_result", None) is not None:
            # EVERY QC card, not its overview table alone (found 3 Nov 2026 by the overnight tester: "Is the
            # comparison fair?", "Do the replicates differ?" and "Cell length" kept the old colour)
            self._show_qc_cards(self._qc_result)
        elif self._qc_payloads:
            self._qc_overview_table.set_payload(
                self._qc_payloads, self._condition_colours(self._qc_payloads.keys()))
        for _ in self._each("kinetics"):
            self._kinetics.set_colours(self._condition_colours(self._all_condition_keys))
        for _ in self._each("msd"):
            self._msd.set_colours(self._condition_colours(self._all_condition_keys))
        for _ in self._each("hmm"):
            self._hmm.set_colours(self._condition_colours(self._all_condition_keys))
        self._images.set_colours(self._condition_colours(self._images.conditions()))
        # The condition DETAIL pages, for the condition they show (SQD's, per page, above).
        for _ in self._each("appd"):
            appd_detail = self._appd_detail.condition()
            if appd_detail is not None:
                self._appd_detail.set_colour(self._condition_colours([appd_detail]).get(appd_detail))
        qc_detail = getattr(self, "_qc_detail_condition", None)
        if qc_detail is not None:
            self._qc_detail.set_colour(self._condition_colours([qc_detail]).get(qc_detail))

    # Older name, kept for callers that still use it.
    def _on_palette_changed(self, _palette: str = "") -> None:
        self._on_colours_changed()

    # -- the options screen ----------------------------------------------------------- #
    def _apply_compute_standard(self, changes: dict) -> None:
        """Put standards into the Compute dialog through `apply_values`. The standards name no model (no
        ``fit_models``, no ``shown_fits``), so every row keeps the models it ticks (`apply_values`)."""
        if not changes:
            return
        self.compute_dialog.compute.apply_values(changes, {})

    def open_options(self):
        """The options screen (`ui.widgets.options_dialog`); a save applies at once."""
        return open_options(self, self.t, "analysis", on_saved=self._options_saved)

    def _options_saved(self, old_record: dict, new_record: dict) -> None:
        """A Compute setting still at the OLD standard follows the new one; one changed
        in the dialog this session stays (`options.follows`). Nothing is re-run: the
        next Run uses them, and the dialog's stale note says so once figures exist."""
        old = opts.compute_values(old_record)
        new = opts.compute_values(new_record)
        for _ in self._each("sqd"):
            current = self.compute_dialog.compute.values()
            self._apply_compute_standard({key: new[key] for key in new
                                          if new[key] != old[key] and current[key] == old[key]})

    def _toggle_theme(self) -> None:
        """Light <-> dark: SAVED, then this window rebuilt in the other theme.

        REBUILT, NOT RECOLOURED IN PLACE. 180 `setStyleSheet` calls under `ui/`
        bake their colour in at construction, so an in-place switch would have to
        reach every one of them and would silently miss the next one written. The
        PROJECT carries over -- it is the object already in memory, nothing is
        re-read from disk -- but a figure on screen does not, so it needs Run again.

        THE NEW WINDOW IS HELD HERE. `app.py` holds only the first one, and a
        parentless window nobody holds is collected at once, taking its threads
        with it -- the exit-139 crash CLAUDE.md records for the segmentation door.
        """
        new = theme_view.other_scheme(getattr(self.t, "color_scheme", "light"))
        theme.save_scheme(new)
        window = self._rebuild(self.project, new)
        # THE SQD FIGURE COMES BACK FROM THE CACHE the Run filled -- a read, not a
        # refit, on every SQD page (the added ones too). The other sections' results are not carried and need Run
        # again.
        for slot in self._ordered_slots("sqd"):
            if slot._last_sqd_run is not None:
                window._call_in(window._slot("sqd", slot.number), window._start_sqd_run, *slot._last_sqd_run)
        self.close()

    def _carry_pages(self, window) -> None:
        """The added pages of this window made again in ``window`` (a rebuild: the theme, the track cap, the
        exclusions): the same numbers and names, each with its settings; results are not carried."""
        window._subpage_set = sp.SubpageSet.from_state(SECTION_TITLE, self._subpage_set.state())
        for section in self._slots:
            for old in self._ordered_slots(section)[1:]:
                new = window._make_slot(section, old.number)
                new.copied_from = getattr(old, "copied_from", "")
                with window._in(new):
                    window._copy_settings(section, old, new)
            window._sync_subpages(section)
        # the section's own HMM page: the new window built its own; its Jump angles reading comes along
        old, new = self._slot("hmm"), window._slot("hmm")
        if old._angles_page.chance_on() != new._angles_page.chance_on():
            new._angles_page.chance_box.set_checked(old._angles_page.chance_on(), animate=False)

    def _rebuild(self, project, scheme: str):
        """This window built again around ``project`` in ``scheme``, shown where this one
        is, with every setting a user made carried over; returns it (held in
        `_OPEN_WINDOWS`). The caller closes this one."""
        settings = self._compute_settings(self._slot("sqd").compute_dialog)
        # BUILT HIDDEN AND PLACED BEFORE IT IS SHOWN, so it appears exactly where
        # this one is instead of opening at a default size and jumping.
        _app, window = open_workspace(project, color_scheme=scheme, show=False,
                                      colours=self.toolbar.conditions.colours())
        window._full_project = self._full_project
        # `setGeometry`, not `restoreGeometry`: restoring onto a window that has never
        # been shown clamped it -- measured 1180 -> 989 px wide, and moved.
        window.setGeometry(self.geometry())
        # EVERYTHING A USER SET, carried over: without this the switch reset the
        # Compute settings, the sidebar and the section -- which is what made it
        # read as a new program rather than a new colour.
        window._apply_compute_settings(window._slot("sqd").compute_dialog, settings)
        self._carry_pages(window)
        window.sidebar.set_collapsed(self.sidebar.collapsed, animate=False)
        if window.sidebar.current != self.sidebar.current:
            window.sidebar.select(self.sidebar.current)
        _OPEN_WINDOWS.add(window)
        # the same screen as this one, its minimum held to it before it shows and from then on (`screenfit`)
        ScreenFit.install(window).cap()
        if self.isMaximized():
            window.showMaximized()
        else:
            window.show()
        return window

    # -- the cap on track length (QC's "Max length") ------------------------------------ #

    def _open_track_cap(self) -> None:
        from .track_cap_dialog import TrackCapDialog

        dialog = TrackCapDialog(self.t, canonical=self._full_project.canonical,
                                current=track_cap_of(self.project),
                                content=self.centralWidget(), parent=self,
                                order=self._all_condition_keys)
        dialog.applied.connect(self._apply_track_cap)
        dialog.closed.connect(dialog.deleteLater)
        dialog.open()

    def _apply_track_cap(self, max_steps) -> None:
        """Leave out every track longer than ``max_steps`` steps (None: put them back), in
        every condition, and REBUILD the workspace around that table, as the theme switch
        does: every section's result came from the old table, so none is carried over --
        QC is run again at once and reopens on the condition this was set from. Applied to
        the UNCAPPED project, so a second cap replaces the first."""
        from ...canonical import CanonicalError
        from ...project import with_exclusions, with_track_cap

        if max_steps == track_cap_of(self.project):
            return
        try:
            # the exclusions in force stay in force under the new cap
            project = with_exclusions(with_track_cap(self._full_project, max_steps), self._exclusions_in_force())
        except CanonicalError as exc:
            self._show_qc_message(str(exc))
            return
        window = self._rebuild(project, getattr(self.t, "color_scheme", "light"))
        if self.sidebar.current_base.partition(".")[0] == "qc":
            window._qc_reopen = getattr(self, "_qc_detail_condition", None)
            window._run_qc()
        self.close()

    # -- exclusions (QC's "Contributes nothing" card) ------------------------------------ #
    #: Asked 16 Oct 2026: "Exclusions in the workspace -- exclude a track, cell or movie with a reason ... the
    #: fractions shown with and without it". Applied like the track cap: to the project as loaded (capped as set),
    #: and the workspace rebuilt around it, so every section and every cache key sees one table. For the session,
    #: as the cap is; the last Apply is remembered per project and the dialog offers it again (`Restore`), never
    #: applied by itself on opening.

    def _exclusions_in_force(self) -> list:
        """The items applied now, as the table records them (`exclusions.apply_exclusions`)."""
        attrs = getattr(getattr(self.project, "canonical", None), "attrs", {}) or {}
        return [dict(i) for i in attrs.get("exclusions") or []]

    def _exclusion_base(self):
        """The project the exclusions are applied to: as loaded, capped as the user set it."""
        from ...project import with_track_cap

        return with_track_cap(self._full_project, track_cap_of(self.project))

    def _remembered_exclusions(self) -> list:
        from ..theme import remembered

        return list(((remembered("exclusions") or {}).get(self._project_key())) or [])

    def _open_exclusions(self) -> None:
        from .exclusions_dialog import ExclusionsDialog

        dialog = ExclusionsDialog(self.t, project=self._exclusion_base(), in_force=self._exclusions_in_force(),
                                  remembered=self._remembered_exclusions(), qc_payloads=self._qc_payloads,
                                  content=self.centralWidget(), parent=self)
        dialog.applied.connect(self._apply_exclusions)
        dialog.preview_requested.connect(lambda items, d=dialog: self._run_exclusion_preview(d, items))
        dialog.closed.connect(lambda d=dialog: setattr(d, "_gone", True))
        dialog.closed.connect(dialog.deleteLater)
        dialog.open()
        self._exclusions_dialog = dialog

    def _run_exclusion_preview(self, dialog, items: list) -> None:
        """The dialog's Preview: the with-and-without fit (`exclusions_view.preview`) on the process pool."""
        from ...exclusions_view import preview

        base, process_pool = self._exclusion_base(), self._process_pool
        signals = _ComputeSignals()
        #: Held: a signal object with no owner is collected before the job lands.
        self._exclusion_preview_signals = signals

        def landed(_gen, result, error):
            if getattr(dialog, "_gone", False):
                return                          # the dialog was closed while the fit ran
            if error is not None:
                dialog.set_preview_error(str(error))
            else:
                dialog.set_preview(result)

        signals.done.connect(landed)
        self._pool.start(_ComputeRunnable(lambda: process_pool.submit(preview, base, list(items)).result(), 0,
                                          signals))

    def _apply_exclusions(self, items: list) -> None:
        """Rebuild the workspace around the project without ``items`` (none: with everything back), and remember
        them for this project. Every section's result came from the old table, so none is carried over; QC runs
        again at once when it is on screen."""
        from ..theme import remember, remembered

        from ...exclusions import ExclusionError
        from ...project import with_exclusions

        items = [dict(i) for i in items]
        if items == self._exclusions_in_force():
            return
        try:
            project = with_exclusions(self._exclusion_base(), items)
        except ExclusionError as exc:
            self._show_qc_message(str(exc))
            return
        if items:
            stored = dict(remembered("exclusions") or {})
            stored[self._project_key()] = items
            remember("exclusions", stored)
        window = self._rebuild(project, getattr(self.t, "color_scheme", "light"))
        if self.sidebar.current_base.partition(".")[0] == "qc":
            window._run_qc()
        self.close()

    # -- the way back: the sidebar header's menu ------------------------------------- #
    def _open_project_menu(self) -> None:
        """The header's menu (asked 15 Oct 2026: "we need a go back button or something to open a new folder"):
        **Open another folder** -- the analysis start screen, with its Recent list and folder picker -- or **Back to
        menu**, the stage chooser. Either closes this window; `app.main` reads `take_after_close`."""
        from ..widgets.menus import Panel, menu_row, retriggered

        header = self.sidebar.header
        if retriggered(header):
            return
        menu = Panel(self.t, self)
        menu.set_trigger(header)
        self.project_menu_rows = {
            "vbspt": menu_row(self.t, menu, "Export tracks for vbSPT\u2026", glyph="download",
                              on_click=lambda: self._later(menu, self.export_vbspt)),
            "smmtrack": menu_row(self.t, menu, "Export tracks for SMMTrack\u2026", glyph="download",
                                 on_click=lambda: self._later(menu, self.export_smmtrack)),
            "dwell": menu_row(self.t, menu, "Import SMMTrack dwell times\u2026", glyph="folder-open",
                              on_click=lambda: self._later(menu, self.import_smmtrack_dwell)),
            "calibration": menu_row(self.t, menu, "Per-spot calibration\u2026", glyph="crosshair",
                                    on_click=lambda: self._later(menu, self.open_calibration)),
            # A SIMULATED SET's truth beside the fit (25 Nov 2026, `truth_view`): only where the generator wrote one
            **({"truth": menu_row(self.t, menu, "Compare with the simulation's truth\u2026", glyph="flask-conical",
                                  on_click=lambda: self._later(menu, self.open_truth))}
               if self._simulation_truth() is not None else {}),
            "save_session": menu_row(self.t, menu, "Save session\u2026", glyph="download",
                                     on_click=lambda: self._later(menu, self.save_session)),
            "open_session": menu_row(self.t, menu, "Open session\u2026", glyph="folder-open",
                                     on_click=lambda: self._later(menu, self.open_session)),
            "report": menu_row(self.t, menu, "Report a problem\u2026", glyph="file-text",
                               on_click=lambda: self._later(menu, self.report_problem)),
            "another": menu_row(self.t, menu, "Open another folder\u2026", glyph="folder-open",
                                on_click=lambda: self._leave_later(menu, "another")),
            "menu": menu_row(self.t, menu, "Back to menu", glyph="chevron-left",
                             on_click=lambda: self._leave_later(menu, "menu")),
        }
        menu.popup_under(header)
        self._project_menu = menu

    def _simulation_truth(self):
        """``(truth, path)`` when the loaded conditions are a generated set (`truth_view.find_truth`), else None."""
        from ...truth_view import find_truth

        try:
            return find_truth(self.project.roots.values())
        except (AttributeError, OSError):
            return None

    def open_legacy_window(self):
        """The right rail's "history" button (ruled 26 Nov 2026, `docs/legacy_window_plan.md`): SMTracker 2.0's tabs on
        these conditions, in a window of their own; a second press raises the one already open."""
        from ..legacy.window import open_legacy_window

        window = self.legacy_window
        if window is not None:
            try:
                if window.isVisible():
                    window.raise_()
                    window.activateWindow()
                    return window
            except RuntimeError:                        # its C++ side is gone
                pass
        self.legacy_window = open_legacy_window(
            self.project, colours=self._condition_colours(self.project.conditions), title=self._project_title,
            home=self,
            exporters={"vbspt": self.export_vbspt, "smmtrack": self.export_smmtrack})
        return self.legacy_window

    def open_truth(self):
        """The header menu's "Compare with the simulation's truth...": declared, counted and the last SQD Run's fit."""
        from ...truth_view import fitted_from_plot, truth_table
        from .truth_dialog import TruthDialog

        found = self._simulation_truth()
        if found is None:
            return None
        truth, path = found
        table = truth_table(truth, fitted_from_plot(getattr(self, "_sqd_plot", None)))
        dialog = TruthDialog(self.t, table, str(path), content=self.centralWidget(), parent=self)
        dialog.closed.connect(dialog.deleteLater)
        self._truth_dialog = dialog              # held: a dialog collected while open takes its children with it
        dialog.open()
        return dialog

    def _later(self, menu, action) -> None:
        """Run ``action`` after the menu has closed, never inside its own click."""
        menu.close_chain()
        QTimer.singleShot(0, action)

    def report_problem(self, dest: str | None = None):
        """The header menu's "Report a problem..." (10 Nov 2026): one zip to send when something went wrong -- the
        error log, the build, the package versions, the settings file; no tracks, images or results
        (`crashlog.report_bundle`). Asked where to save when ``dest`` is None; a notice says where it is and what
        it holds. Returns the zip's path, or None."""
        from ... import crashlog
        from ..widgets.dialogs import NoticeDialog
        from ..widgets.report import open_issue, save_report

        path, title, body = save_report(self, dest)
        if title:
            # "Report on GitHub" once the report is saved (27 Nov 2026): the repository's new-issue page, filled in
            dialog = NoticeDialog(self.t, title, body, content=self.centralWidget(), parent=self,
                                  extra=(crashlog.ISSUE_BUTTON, lambda: open_issue(self)) if path else None)
            dialog.closed.connect(dialog.deleteLater)
            self._report_notice = dialog          # held: a dialog collected while open takes its children with it
            dialog.open()
        return path

    # -- the per-spot calibration (6 Nov 2026; asked "i wonder where to do the calibraion ?", "ja build it", then
    # "people should choose (internal calibration this folder), own folder" and "yes to all") ------------------- #
    def open_calibration(self) -> None:
        """The header menu's "Per-spot calibration...": which calibration every Per spot Run of this project uses
        (built-in, the project's file, or one from the library; `calibrations`), and making one from a separate
        stationary measurement into the library (`calibration_view`, `CalibrationDialog`)."""
        from .calibration_dialog import CalibrationDialog

        dialog = CalibrationDialog(self.t, content=self.centralWidget(), parent=self)
        dialog.choice_made.connect(self._choose_calibration)
        dialog.remove_requested.connect(self._remove_calibration)
        dialog.folder_wanted.connect(self._choose_calibration_folder)
        dialog.measure_requested.connect(self._measure_calibration)
        dialog.add_requested.connect(self._add_calibration)
        dialog.save_requested.connect(self._save_calibration)
        dialog.closed.connect(self._calibration_closed)
        self.calibration_dialog = dialog
        self._calibration_target = None
        self._sync_calibration_options()
        dialog.open()

    def _calibration_trackers(self) -> set:
        from ...calibrations import project_trackers

        return project_trackers(self.project, self._all_condition_keys)

    def _sync_calibration_options(self, status: str | None = None) -> None:
        from ...calibrations import options

        dialog = self.calibration_dialog
        if dialog is None:
            return
        dialog.set_options(options(self.project, self._calibration_trackers()))
        if status is not None:
            dialog.set_choose_status(status)

    def _choose_calibration(self, choice: dict) -> None:
        from ...calibrations import choose

        try:
            choose(self.project, choice["kind"], choice.get("id"))
        except ValueError as exc:
            self._sync_calibration_options(f"Not chosen: {exc}")
            return
        self._sync_calibration_options("Chosen. Every Per spot Run of this project uses it. Results "
                                       "already drawn keep their own calibration, named in their captions.")

    def _remove_calibration(self, entry_id: str) -> None:
        from ...calibrations import entry, remove

        item = entry(entry_id)
        remove(entry_id)
        self._sync_calibration_options(f"Removed from the library: {item['name']}" if item else None)

    def _calibration_closed(self) -> None:
        dialog, self.calibration_dialog = self.calibration_dialog, None
        self._calibration_gen += 1                     # a measurement still running lands nowhere
        if dialog is not None:
            dialog.deleteLater()

    def _choose_calibration_folder(self) -> None:
        from qtpy.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(self, "The folder of the tracked stationary movies")
        if folder and self.calibration_dialog is not None:
            self.calibration_dialog.set_folder(folder)

    def _measure_calibration(self, root: str) -> None:
        """Measure a stationary measurement's folder off the GUI thread (`calibration_view.measure_folder`)."""
        from ...calibration_view import measure_folder

        dialog = self.calibration_dialog
        if dialog is None or not root:
            return
        self._calibration_gen += 1
        gen = self._calibration_gen

        def compute(root=str(root)):
            return measure_folder(root)

        dialog.set_busy("Measuring\u2026")
        self._pool.start(_ComputeRunnable(compute, gen, self._calibration_signals))

    def _on_calibration_done(self, gen: int, result, error) -> None:
        dialog = self.calibration_dialog
        if gen != self._calibration_gen or dialog is None:
            return                      # superseded, or the dialog was closed
        dialog.set_result(result, None if error is None else f"Not measured: {error}")

    def _add_calibration(self, name: str, use: bool) -> None:
        """The measurement into the library under ``name`` (`calibrations.add`); chosen for this project when
        ``use``; back to the choice."""
        from ...calibrations import add, choose

        dialog = self.calibration_dialog
        if dialog is None or not dialog.payload:
            return
        item = add(dialog.payload, name)
        if use:
            choose(self.project, "library", item["id"])
        self._sync_calibration_options(f"Added to the library: {item['name']}"
                                       + (". Every Per spot Run of this project uses it." if use else ""))
        dialog.show_choose()

    def _calibration_start_folder(self) -> str:
        """Where the folder picker opens: the folder above this project's conditions."""
        roots = list((getattr(self.project, "roots", None) or {}).values())
        return str(Path(roots[0]).parent) if roots else ""

    def _save_calibration(self, replace: bool) -> None:
        """Also save the measurement into a project folder (asked for), or Replace (the folder just chosen)."""
        folder = self._calibration_target if replace else None
        if folder is None:
            from qtpy.QtWidgets import QFileDialog

            folder = QFileDialog.getExistingDirectory(self, "Save the calibration into a project folder",
                                                      self._calibration_start_folder())
            if not folder:
                return
        self.save_calibration_to(folder, replace=replace)

    def save_calibration_to(self, folder, *, replace: bool = False) -> str | None:
        """Save the dialog's measurement in ``folder`` (`calibration_view.save`); where one is there already the
        dialog asks first (Replace / Keep it) unless ``replace``. Returns the file written, or None."""
        from ...calibration_view import existing, save

        dialog = self.calibration_dialog
        if dialog is None or not dialog.payload:
            return None
        self._calibration_target = str(folder)
        there = existing(folder)
        if there is not None and not replace:
            dialog.ask_replace(str(folder), there)
            return None
        try:
            path = save(dialog.payload, folder, replace=replace)
        except (OSError, ValueError) as exc:
            dialog.set_busy(f"Not saved: {exc}")
            return None
        dialog.saved(str(path))
        self._sync_calibration_options()
        self.toasts.post("Per-spot calibration saved",
                         description=f"{path}: a project there uses it where it chooses This project's file")
        return str(path)

    def export_vbspt(self, folder: str | None = None) -> list:
        """v2's File > Export > vbSPT (agreed 17 Oct 2026): every condition's tracks -- the analysis's, as the fits
        read them -- into ``folder`` (asked for when None), one ``vbSPT<condition>.mat`` and its note each
        (`trackexport.write_vbspt`). Returns what was written; a toast says where."""
        from ...trackexport import TrackExportError, write_vbspt

        if folder is None:
            from qtpy.QtWidgets import QFileDialog

            folder = QFileDialog.getExistingDirectory(self, "Export tracks for vbSPT into")
            if not folder:
                return []
        written, failed = [], []
        for key in self._all_condition_keys:
            try:
                written.append(write_vbspt(self.project.condition(key), folder, key,
                                           label=self._condition_labels.get(key, key)))
            except TrackExportError as exc:
                failed.append(f"{self._condition_labels.get(key, key)}: {exc}")
        n = sum(w["n_traj"] for w in written)
        if written:
            self.toasts.post(f"Tracks exported for vbSPT ({len(written)} file{'s' * (len(written) != 1)})",
                             description=f"{n:,} trajectories, x and y in nm, in {folder}"
                             + (f". Not exported: {'; '.join(failed)}" if failed else ""))
        else:
            self.toasts.post("No tracks exported for vbSPT", description="; ".join(failed) or None)
        return written

    def export_smmtrack(self, folder: str | None = None) -> list:
        """v2's File > Export > SMMTrack (17 Oct 2026): every condition's tracks -- the analysis's -- into ``folder``
        (asked for when None): v2's headerless ``SMMTrack_<project>_<condition>.csv`` (Nr, TrackID, PointID, x and
        y in µm, t in s), its per-track sidecar and a note (`trackexport.write_smmtrack`). A toast says where."""
        from ...trackexport import TrackExportError, write_smmtrack

        if folder is None:
            from qtpy.QtWidgets import QFileDialog

            folder = QFileDialog.getExistingDirectory(self, "Export tracks for SMMTrack into")
            if not folder:
                return []
        written, failed = [], []
        for key in self._all_condition_keys:
            try:
                written.append(write_smmtrack(self.project.condition(key), folder, key, protein=self._project_title,
                                              label=self._condition_labels.get(key, key)))
            except TrackExportError as exc:
                failed.append(f"{self._condition_labels.get(key, key)}: {exc}")
        n = sum(w["n_tracks"] for w in written)
        if written:
            self.toasts.post(f"Tracks exported for SMMTrack ({len(written)} file{'s' * (len(written) != 1)})",
                             description=f"{n:,} tracks, x and y in \u00b5m, t in s, in {folder}"
                             + (f". Not exported: {'; '.join(failed)}" if failed else ""))
        else:
            self.toasts.post("No tracks exported for SMMTrack", description="; ".join(failed) or None)
        return written

    def _frame_interval_s(self) -> float:
        """The project's one frame interval (`smmtrack_view.frame_interval_s`)."""
        from ...smmtrack_view import frame_interval_s

        return frame_interval_s(self.project, self._all_condition_keys)

    def import_smmtrack_dwell(self, folder: str | None = None):
        """v2's File > Import (17 Oct 2026): SMMTrack's dwell-time files from ``folder`` (asked for when None), one
        ``<condition>.txt`` each, matched by the condition's name or label (`smmtrack_view.match_dwell_files`), drawn
        as v2 drew them (`dwell_dialog.DwellDialog`). Returns the dialog, or None with a toast saying why."""
        from ...smmtrack_view import DwellImportError, dwell_payload, match_dwell_files, read_dwell_file
        from .dwell_dialog import DwellDialog

        if folder is None:
            from qtpy.QtWidgets import QFileDialog

            folder = QFileDialog.getExistingDirectory(self, "SMMTrack's dwell-time files (one <condition>.txt each)")
            if not folder:
                return None
        conditions = [(k, self._condition_labels.get(k, k)) for k in self._all_condition_keys]
        try:
            found = match_dwell_files(folder, conditions)
            if not found["matched"]:
                names = ", ".join(f"{label}.txt" for _k, label in conditions)
                raise DwellImportError(f"no file named for a condition in {folder} (looked for {names})")
            parsed = {k: read_dwell_file(path) for k, path in found["matched"].items()}
            payload = dwell_payload(parsed, self._frame_interval_s(), labels=self._condition_labels)
        except DwellImportError as exc:
            self.toasts.post("No dwell times imported", description=str(exc))
            return None
        missing = [self._condition_labels.get(k, k) for k in found["unmatched_conditions"]]
        unmatched = ("No file for " + ", ".join(missing) + "." if missing else "")
        content = self.centralWidget() if hasattr(self, "centralWidget") else None
        dialog = DwellDialog(self.t, payload, self._condition_colours(list(parsed)), unmatched=unmatched,
                             content=content, parent=self)
        dialog.closed.connect(dialog.deleteLater)
        dialog.open()
        self._dwell_dialog = dialog
        return dialog

    # -- sessions (v2's File > Save session / Load session; 17 Oct 2026) ------------------------------------------- #
    def _sqd_page_state(self) -> dict:
        """The SQD page now current as a session keeps it: its Compute settings, its figure and Run, its Jump distances
        and Signed jumps results, its typed axis ranges and its brackets' choice."""
        settings, sections = {}, {}
        panel = self.compute_dialog.compute
        settings["sqd"] = {"values": panel.values(), "conditions": panel.condition_values(),
                           "own_model": dict(getattr(panel, "_own_model", {})),
                           # each row's own Shared D (1 Nov 2026); a session from before has none: `.get` below
                           "own_shared_d": dict(getattr(panel, "_own_shared_d", {}))}
        if getattr(self, "_sqd_plot", None) is not None and getattr(self, "_last_sqd_run", None) is not None:
            sections["sqd"] = {"plot": self._sqd_plot, "caption": self.status.text(),
                               "payloads": dict(getattr(self, "_replicate_payloads", {}) or {}),
                               "run": self._last_sqd_run, "computed_at": getattr(self, "_sqd_computed_at", None)}

        def cards(section, getter):
            out = {}
            for key, card in section.cards().items():
                result = getter(card)
                if result is not None:
                    out[key] = result
            return out

        if self._jumps.has_result():
            sections["jumps"] = {"results": cards(self._jumps, lambda c: c.result()),
                                 "config": dict(self._jumps.last_config or {}),
                                 "legend_places": self._jumps.legend_places(),
                                 "view": self._jumps.view_state()}
        if self._jd.has_result():
            sections["jd"] = {"results": dict(self._jd._results), "conditions": list(self._jd._conditions)}
        return {"settings": settings, "sections": sections, "axes": self._card.axes_state(),
                "choice": self._sqd_choice}

    def _spatial_page_state(self) -> dict:
        """The Spatial Map page now current as a session keeps it: the map dialog's every control (B8, 4 Nov 2026: Open
        session brought the grid back and this dialog on its defaults, so Run drew another map; a split by hand's
        ranges were not kept at all) and the grid with its display."""
        settings, sections = {"spatial": {"dialog": self.map_dialog.state()}}, {}
        matrix = self._spatial.matrix
        if getattr(matrix, "_payloads", None):
            sections["spatial"] = {"payloads": dict(matrix._payloads), "errors": dict(getattr(matrix, "_errors", {})),
                                   "method": matrix._method, "n_pop": matrix._n_pop,
                                   "conditions": list(matrix._conditions), "labels": dict(matrix._labels),
                                   "config": getattr(self, "_spatial_last_config", None),
                                   "computed_at": getattr(self, "_spatial_computed_at", None),
                                   # A SPLIT BY HAND, the columns and the display (found 3 Nov 2026: a manual split
                                   # reopened with fitted columns, the colour map and hidden columns at defaults)
                                   "ranges": getattr(matrix, "_ranges", None),
                                   "counts": list(getattr(matrix, "_counts", None) or ()),
                                   "planned": dict(getattr(matrix, "_planned", None) or {}),
                                   "shown": dict(getattr(matrix, "_shown", None) or {}),
                                   "cmap": self._spatial.colormap(), "rendering": self._spatial.rendering(),
                                   "scale": self._spatial.scale()}
        return {"settings": settings, "sections": sections, "axes": None, "choice": None}

    def _msd_page_state(self) -> dict:
        """The MSD page now current as a session keeps it: its results and last Run's settings, its typed axes."""
        sections = {}
        if self._msd.has_result():
            sections["msd"] = {"results": dict(self._msd._results), "conditions": list(self._msd._conditions),
                               "config": dict(self._msd.last_config or {})}
        return {"settings": {}, "sections": sections, "axes": self._msd.axes_state(), "choice": None}

    def _appd_page_state(self) -> dict:
        """The Apparent Diffusion page now current as a session keeps it: its results, joint fit and last Run's
        settings, its typed axes and its brackets' choice."""
        sections = {}
        if self._appd.has_result():
            sections["appd"] = {"results": self._appd.results(), "conditions": list(self._appd.conditions()),
                                "config": dict(self._appd.last_config or {}), "joint": self._appd.joint()}
        return {"settings": {}, "sections": sections, "axes": self._appd.axes_state(), "choice": self._appd_choice}

    def _hmm_page_state(self) -> dict:
        """The HMM page now current as a session keeps it: each card's result and the last Run's settings, the
        comparison's typed axes."""
        sections = {}
        if self._hmm.has_result():
            results = {key: card.result() for key, card in self._hmm.cards().items() if card.result() is not None}
            sections["hmm"] = {"results": results, "config": dict(self._hmm.last_config or {})}
        # the Jump angles tab's reading (Against chance): its angles are not kept, they come back from the fits on show
        return {"settings": {"hmm_angles_chance": self._angles_page.chance_on()}, "sections": sections,
                "axes": self._hmm.comparison_axes(), "choice": self._hmm_choice}

    def _kinetics_page_state(self) -> dict:
        """The Kinetics page now current as a session keeps it: each card's result and the last Run's settings, the
        typed axes."""
        sections = {}
        if self._kinetics.has_result():
            results = {key: card.kinetics_body._result for key, card in self._kinetics.cards().items()
                       if card.kinetics_body._result is not None}
            sections["kinetics"] = {"results": results, "config": dict(self._kinetics.last_config or {})}
        return {"settings": {}, "sections": sections, "axes": self._kinetics.axes_state(), "choice": None}

    def _clusters_page_state(self) -> dict:
        """The Clustering page now current as a session keeps it: its results, conditions and the Run's switches, its
        typed axes."""
        sections = {}
        if self._clusters.has_result():
            sections["clusters"] = {"results": self._clusters.results(), "conditions": self._clusters.conditions(),
                                    "config": dict(self._clusters.last_config or {})}
        return {"settings": {}, "sections": sections, "axes": self._clusters.axes_state(), "choice": None}

    #: Each paged section's reader of the page now current (`session_state`); `_restore_page` is its mirror.
    _PAGE_STATE = {"sqd": "_sqd_page_state", "spatial": "_spatial_page_state", "msd": "_msd_page_state",
                   "appd": "_appd_page_state", "hmm": "_hmm_page_state",
                   "kinetics": "_kinetics_page_state", "clusters": "_clusters_page_state"}

    def session_state(self) -> dict:
        """Every section's settings and results as this window holds them, plain enough to pickle
        (`session_file`): SQD's figure and its Run, each per-condition section's results and its last Run's
        settings, QC's result, the significance choices, the foci and the view. The ADDED pages of SQD and the
        Spatial Map (subpages) are under ``"subpages"``: the model and, per page, what its section keeps."""
        sections, settings = {}, {}
        main = {section: self._call_in(self._slot(section), getattr(self, getter))
                for section, getter in self._PAGE_STATE.items()}
        for page in main.values():
            settings.update(page["settings"])
            sections.update(page["sections"])

        if getattr(self, "_qc_result", None) is not None:
            sections["qc"] = {"result": self._qc_result}
        # the charts' typed ranges (their Axes, 30 Oct 2026): restored after the results they range
        axes = {"sqd": main["sqd"]["axes"], "appd": main["appd"]["axes"], "msd": main["msd"]["axes"],
                "hmm": main["hmm"]["axes"], "kinetics": main["kinetics"]["axes"],
                "clusters": main["clusters"]["axes"]}
        significance = dict(self._sig_choices)
        significance.update({section: main[section]["choice"] for section in PAGED_CHOICE})
        state = {"sections": sections, "settings": settings, "significance": significance,
                 "foci": self._images.foci(), "view": self.sidebar.current, "axes": axes}
        # THE ADDED PAGES (format 2): the model, and each page's own settings and results
        pages = {}
        for section, getter in self._PAGE_STATE.items():
            for slot in self._ordered_slots(section):
                if slot.number is not None:
                    page = self._call_in(slot, getattr(self, getter))
                    page["copied_from"] = getattr(slot, "copied_from", "")
                    pages[slot.key] = page
        if pages:
            state["subpages"] = {"set": self._subpage_set.state(), "pages": pages}
        return state

    def save_session(self, path: str | None = None):
        """Write this window's session (`session_file.write_session`) to ``path`` (asked for when None). Returns the
        path written, or None; a toast says where, or why not."""
        from ...buildstamp import build_stamp
        from ...session_file import SUFFIX, SessionError, project_identity, write_session

        if path is None:
            from qtpy.QtWidgets import QFileDialog

            path, _ = QFileDialog.getSaveFileName(self, "Save session", f"{self._project_title}{SUFFIX}",
                                                  f"SMTracker session (*{SUFFIX})")
            if not path:
                return None
        try:
            written = write_session(path, identity=project_identity(self.project, self._all_condition_keys),
                                    state=self.session_state(), labels=dict(self._condition_labels),
                                    track_cap=track_cap_of(self.project), exclusions=self._exclusions_in_force(),
                                    view=self.sidebar.current, build=build_stamp())
        except (SessionError, OSError) as exc:
            self.toasts.post("Session not saved", description=str(exc))
            return None
        kept = sorted(self.session_state()["sections"])
        self.toasts.post("Session saved", description=f"{written} ({', '.join(kept) or 'settings only'})")
        return written

    def open_session(self, path: str | None = None):
        """Read a session (asked for when None) and show it here: the track cap and the exclusions it records are
        applied first (a rebuilt window), then the data are checked against its fingerprints and every stored
        result drawn without a refit. A session of another project leaves this window for that one
        (`app.main` opens it). Returns the window that shows it, or None with a toast saying why."""
        from ...session_file import SUFFIX, SessionError, check_project, project_identity, read_manifest

        if path is None:
            from qtpy.QtWidgets import QFileDialog

            path, _ = QFileDialog.getOpenFileName(self, "Open session", "", f"SMTracker session (*{SUFFIX})")
            if not path:
                return None
        try:
            manifest = read_manifest(path)
        except SessionError as exc:
            self.toasts.post("Session not opened", description=str(exc))
            return None
        roots = sorted((manifest.get("project") or {}).get("roots", {}).values())
        mine = sorted(project_identity(self.project, [])["roots"].values())
        if roots != mine:
            # ANOTHER PROJECT: this window closes and the app opens the session's folders, then the session
            AFTER_CLOSE["session"] = str(path)
            self.leave("session")
            return None
        window = self
        cap, exclusions = manifest.get("track_cap"), list(manifest.get("exclusions") or [])
        if cap != track_cap_of(self.project) or exclusions != self._exclusions_in_force():
            from ...project import with_exclusions, with_track_cap

            try:
                project = with_exclusions(with_track_cap(self._full_project, cap), exclusions)
            except Exception as exc:                        # noqa: BLE001 - named on the toast
                self.toasts.post("Session not opened", description=f"its track cap or exclusions could not be "
                                                                   f"applied: {exc}")
                return None
            window = self._rebuild(project, getattr(self.t, "color_scheme", "light"))
            self.close()
        problems = check_project(manifest, project_identity(window.project, list(
            (manifest.get("project") or {}).get("conditions", {}))))
        if problems:
            window.toasts.post("Session not opened: the data changed", description="; ".join(problems))
            return None
        window.apply_session(path)
        return window

    def _restore_sqd_page(self, settings: dict, sections: dict, axes, choice, attempt, labels: dict) -> None:
        """Draw a session's SQD page into the page now current: its Compute settings, its figure (from the stored
        result, nothing refitted), its Jump distances and Signed jumps results, its typed axes and brackets' choice."""
        def per_condition(widget, entry, *, config=True):
            conds = list(entry.get("conditions") or entry["results"].keys())
            args = [conds, labels, self._condition_colours(conds)]
            if config:
                args.append(entry.get("config") or {})
            widget.begin_run(*args)
            for c in conds:
                if c in entry["results"]:
                    widget.set_payload(c, entry["results"][c])
            widget.set_running(False)

        def sqd():
            sd = sections["sqd"]
            self._sqd_pending = sd["run"]
            self._sqd_gen += 1
            self._on_compute_done(self._sqd_gen, (sd["plot"], sd["caption"], sd["payloads"]), None)
            if sd.get("computed_at"):
                self._sqd_computed_at = sd["computed_at"]

        if settings.get("sqd"):
            sq = settings["sqd"]
            attempt("SQD settings", lambda: self.compute_dialog.compute.apply_values(
                sq["values"], sq["conditions"], sq["own_model"], own_shared_d=sq.get("own_shared_d")))
        if choice is not None:
            self._sqd_choice = choice
        if "sqd" in sections:
            attempt("sqd", sqd)
        if "jumps" in sections:
            attempt("jumps", lambda: (per_condition(self._jumps, sections["jumps"]),
                                      self._jumps.set_legend_places(sections["jumps"].get("legend_places") or {}),
                                      self._jumps.set_view_state(sections["jumps"].get("view") or {})))
            self._jumps_full = set(sections["jumps"]["results"])
        if "jd" in sections:
            attempt("jd", lambda: per_condition(self._jd, sections["jd"], config=False))
        if axes:
            attempt("typed axis ranges", lambda: self._card.set_axes_state(axes))

    def _restore_spatial_page(self, settings: dict, sections: dict, attempt) -> None:
        """Draw a session's Spatial Map page into the page now current."""
        # THE MAP DIALOG AS IT WAS (B8): every control from the session's dialog; a session from before keeps only
        # its last map Run's config, which names the Run's own settings
        if (settings.get("spatial") or {}).get("dialog"):
            attempt("Spatial Map settings", lambda: self.map_dialog.set_state(settings["spatial"]["dialog"]))
        elif ((sections.get("spatial") or {}).get("config")):
            attempt("Spatial Map settings", lambda: self.map_dialog.set_config(
                {k: v for k, v in sections["spatial"]["config"].items() if k != "gradient"}))
        if "spatial" not in sections:
            return

        def spatial():
            entry = sections["spatial"]
            self._spatial.begin_run(entry["conditions"], entry["labels"], method=entry["method"],
                                    n_pop=entry["n_pop"], colours=self._condition_colours(entry["conditions"]),
                                    ranges=entry.get("ranges"), counts=tuple(entry.get("counts") or ()) or None,
                                    planned=entry.get("planned") or None)
            for c, payload in entry["payloads"].items():
                self._spatial.set_payload(c, payload)
            for c, message in entry["errors"].items():
                self._spatial.set_error(c, message)
            self._spatial.set_running(False)
            # the display as it was saved (a session from before keeps the defaults)
            if entry.get("cmap"):
                self._spatial.set_colormap(entry["cmap"])
            if entry.get("rendering"):
                self._spatial.set_rendering(entry["rendering"])
            if entry.get("scale"):
                self._spatial.set_scale(entry["scale"])
            for kind, on in (entry.get("shown") or {}).items():
                self._spatial.matrix.set_panel_visible(kind, bool(on))
            self._spatial_last_config = entry.get("config")
            self._spatial_computed_at = entry.get("computed_at")
            self._sync_metadata()
            self._sync_spatial_visibility_section()

        attempt("spatial", spatial)

    def _restore_msd_page(self, sections: dict, axes, attempt, labels: dict) -> None:
        """Draw a session's MSD page into the page now current: its results, then its typed axis ranges."""
        if "msd" in sections:
            attempt("msd", lambda: self._restore_per_condition(self._msd, sections["msd"], labels,
                                                              finish=self._msd.finish))
        if axes:
            attempt("typed axis ranges", lambda: self._msd.set_axes_state(axes))

    def _restore_appd_page(self, sections: dict, axes, choice, attempt, labels: dict) -> None:
        """Draw a session's Apparent Diffusion page into the page now current: its brackets' choice, its results and
        joint fit (nothing refitted), its typed axis ranges."""
        if choice is not None:
            self._appd_choice = choice
        if "appd" in sections:
            def appd():
                entry = sections["appd"]
                conds = list(entry["conditions"])
                self._appd.begin_run(conds, labels, self._condition_colours(conds), entry.get("config") or {})
                for c in conds:
                    result = entry["results"].get(c)
                    if result is None:
                        continue
                    if "error" in result and len(result) <= 2:
                        self._appd.set_error(c, result["error"])
                    else:
                        self._appd.set_result(c, result)
                self._appd.finish(entry.get("joint"))
                self._apply_significance("appd")

            attempt("appd", appd)
        if axes:
            attempt("typed axis ranges", lambda: self._appd.set_axes_state(axes))

    def _restore_hmm_page(self, settings: dict, sections: dict, axes, choice, attempt, labels: dict) -> None:
        """Draw a session's HMM page into the page now current: its brackets' choice, its Jump angles tab's reading,
        its cards' results (the tab starting from them: its angles come back from the fits when it is shown), then the
        comparison's axes."""
        if choice is not None and getattr(choice, "population", None) is not None:
            self._hmm_choice = choice
        chance = (settings or {}).get("hmm_angles_chance")
        if chance is not None and bool(chance) != self._angles_page.chance_on():
            self._angles_page.chance_box.set_checked(bool(chance), animate=False)
        if "hmm" in sections:
            def restore():
                self._restore_per_condition(self._hmm, sections["hmm"], labels)
                self._angles_follow_run()

            attempt("hmm", restore)
        if axes:
            attempt("typed axis ranges", lambda: self._hmm.set_comparison_axes(axes))

    def _restore_kinetics_page(self, sections: dict, axes, attempt, labels: dict) -> None:
        """Draw a session's Kinetics page into the page now current: its cards' results, then its typed axes."""
        if "kinetics" in sections:
            attempt("kinetics", lambda: self._restore_per_condition(self._kinetics, sections["kinetics"], labels))
        if axes:
            attempt("typed axis ranges", lambda: self._kinetics.set_axes_state(axes))

    def _restore_clusters_page(self, sections: dict, axes, attempt, labels: dict) -> None:
        """Draw a session's Clustering page into the page now current: its switches and results, then its typed axes."""
        if "clusters" in sections:
            def clusters():
                entry = sections["clusters"]
                self._clusters.set_config(entry.get("config") or {})
                self._clusters.begin_run(list(entry["conditions"]), labels, entry.get("config") or {})
                for c, r in entry["results"].items():
                    self._clusters.set_payload(c, r)
                self._clusters.set_running(False)

            attempt("clusters", clusters)
        if axes:
            attempt("typed axis ranges", lambda: self._clusters.set_axes_state(axes))

    def _restore_page(self, slot, settings: dict, sections: dict, axes, choice, attempt, labels: dict) -> None:
        """A session's page drawn into ``slot`` (its section's own page or an added one): one door for every paged
        section, each reading what its `_<section>_page_state` wrote."""
        if slot.section == "sqd":
            self._call_in(slot, self._restore_sqd_page, settings, sections, axes, choice, attempt, labels)
        elif slot.section == "spatial":
            self._call_in(slot, self._restore_spatial_page, settings, sections, attempt)
        elif slot.section == "msd":
            self._call_in(slot, self._restore_msd_page, sections, axes, attempt, labels)
        elif slot.section == "appd":
            self._call_in(slot, self._restore_appd_page, sections, axes, choice, attempt, labels)
        elif slot.section == "hmm":
            self._call_in(slot, self._restore_hmm_page, settings, sections, axes, choice, attempt, labels)
        elif slot.section == "kinetics":
            self._call_in(slot, self._restore_kinetics_page, sections, axes, attempt, labels)
        elif slot.section == "clusters":
            self._call_in(slot, self._restore_clusters_page, sections, axes, attempt, labels)

    def _restore_per_condition(self, widget, entry: dict, labels: dict, *, colours: bool = True, config: bool = True,
                               finish=None) -> None:
        """A per-condition section drawn from a session's entry (``results``, ``conditions``, ``config``): its Run
        begun, each stored result set, then finished."""
        conds = list(entry.get("conditions") or entry["results"].keys())
        args = [conds, labels]
        if colours:
            args.append(self._condition_colours(conds))
        if config:
            args.append(entry.get("config") or {})
        widget.begin_run(*args)
        for c in conds:
            if c in entry["results"]:
                widget.set_payload(c, entry["results"][c])
        if finish is not None:
            finish()
        else:
            widget.set_running(False)

    def _restore_subpage_set(self, data) -> None:
        """Replace the added pages by a session's: every page now present is dropped without an Undo, the session's
        are built (empty: their results are drawn by the restorers) in its order."""
        for section in list(self._slots):
            for slot in [s for s in self._slots[section] if s.number is not None]:
                self._remove_slot(slot)
        self._subpage_set = sp.SubpageSet.from_state(SECTION_TITLE, data)
        for section in self._slots:
            for number in self._subpage_set.numbers(section):
                self._make_slot(section, number)

    def _subpages_synced(self, section: str) -> None:
        self._sync_subpages(section)

    def _sync_map_pages(self, section: str = "sqd") -> None:
        """The pages a fit section (SQD, Apparent Diffusion ...) has, offered by every map dialog's drop-down ("Use
        the same D as the ... fit"), and every grid drawn at a page that is gone saying so (`map_fit_source.gone_text`):
        never redrawn in silence."""
        if section not in self._fit_methods() or "spatial" not in self._slots:
            return
        pages = self._subpage_names(section)
        for _ in self._each("spatial"):
            self.map_dialog.set_fit_pages(section, pages)
            config = self._spatial_last_config or {}
            if config.get("method") != section or not map_fit_source.uses_section(config):
                continue
            key = map_fit_source.fit_page_key(config, section)
            slot = self._cur["spatial"]
            if self._slot_of_key(key) is None and self._spatial.has_result():
                name = config.get("fit_page_name") or key
                self._spatial.set_notice(map_fit_source.gone_text(section, name))
                slot.gone_notice = True
            elif self._slot_of_key(key) is not None and getattr(slot, "gone_notice", False):
                self._spatial.set_notice(None)
                slot.gone_notice = False

    def _remove_slot(self, slot) -> None:
        """Take an added page out at once, its jobs stopped and its widgets freed (no Undo: a session replaces it)."""
        section = slot.section
        for key in [k for k in self._running if sp.split_job(k)[1] == slot.number
                    and self._JOB_SECTION.get(sp.split_job(k)[0]) == section]:
            self._cancel_job(key)
            self._running.pop(key, None)
        with self._in(slot):
            widget = self._page_widget(section)
        self._slots[section].remove(slot)
        if self._cur[section] is slot:
            self._cur[section] = self._slots[section][0]
        self._section_pages.removeWidget(widget)
        self._empty_trash({"section": section, "slot": slot, "widget": widget, "restored": False,
                           "page": None, "index": 0, "_force": True})

    def apply_session(self, path) -> list:
        """Draw a session's stored results in this window (already checked to be its data). Returns what could not
        be restored, one line a section; a toast names it."""
        from ...session_file import SessionError, load_state

        try:
            state = load_state(path)
        except SessionError as exc:
            self.toasts.post("Session not opened", description=str(exc))
            return [str(exc)]
        failed, done = [], []
        settings = state.get("settings") or {}
        sections = state.get("sections") or {}
        labels = dict(self._condition_labels)

        def attempt(name, action):
            try:
                action()
                done.append(name)
            except Exception as exc:                        # noqa: BLE001 - named on the toast, the rest goes on
                failed.append(f"{name}: {type(exc).__name__}: {exc}")

        # THE ADDED PAGES first (they are slots the main restore does not touch): the model, then their widgets
        subpages = state.get("subpages") or {}
        try:
            self._restore_subpage_set(subpages.get("set"))
        except Exception as exc:                            # noqa: BLE001 - named on the toast, the rest goes on
            failed.append(f"subpages: {type(exc).__name__}: {exc}")
        # THE SECTIONS' OWN PAGES, then each added page the same way (`_restore_sqd_page`, `_restore_spatial_page`)
        saved_axes, saved_choice = state.get("axes") or {}, state.get("significance") or {}
        for section in self._PAGE_STATE:
            self._restore_page(self._slot(section), settings, sections, saved_axes.get(section),
                               saved_choice.get(section), attempt, labels)
        for key, page in (subpages.get("pages") or {}).items():
            slot = self._slot_of_key(key)
            if slot is None:
                continue
            slot.copied_from = page.get("copied_from", "")
            name = self._page_name(slot)
            sub = lambda what, action, name=name: attempt(f"{name} {what}", action)      # noqa: E731
            self._restore_page(slot, page["settings"], page["sections"], page.get("axes"), page.get("choice"), sub,
                               labels)
        for section, choice in (state.get("significance") or {}).items():
            if section in self._sig_choices:
                self._sig_choices[section] = choice
        if state.get("foci"):
            attempt("foci", lambda: (self._images.set_foci(state["foci"]), self._remember_foci(state["foci"])))

        def per_condition(widget, entry, **kwargs):
            self._restore_per_condition(widget, entry, labels, **kwargs)

        table = {
            "qc": lambda: self._on_qc_only_done(self._qc_gen, sections["qc"]["result"], None),
        }
        for name, restore in table.items():
            if name in sections:
                attempt(name, restore)
        typed = state.get("axes") or {}
        if any(typed.values()):
            attempt("typed axis ranges", lambda: (
                                                  self._clusters.set_axes_state(typed.get("clusters") or {})))
        if "molecules" in sections:
            # Saved before the Molecules section was removed (19 Oct 2026): named, never dropped in silence.
            failed.append(f"Molecules ({REMOVED_MOLECULES})")
        from ... import significance as sig

        for section in sig.SECTION_KINDS:
            try:
                self._apply_significance_everywhere(section)
            except Exception as exc:                        # noqa: BLE001 - named
                failed.append(f"significance on {section}: {exc}")
        for section in self._slots:
            self._subpages_synced(section)
        view = state.get("view")
        if view:
            try:
                self.sidebar.select(view)
            except Exception:                               # noqa: BLE001 - a view this build has not: stay
                pass
        if failed:
            self.toasts.post("Session opened, with gaps", description="Not restored: " + "; ".join(failed))
        else:
            shown = [d for d in done if d in table or d in ("sqd", "jumps", "jd", "spatial", "msd", "appd", "hmm", "kinetics", "clusters")]
            self.toasts.post("Session opened", description=(f"{len(shown)} section"
                                                            f"{'s' * (len(shown) != 1)} restored, nothing refitted"
                                                            if shown else "settings restored"))
        return failed

    def _leave_later(self, menu, where: str) -> None:
        """After the menu has closed, never inside its own click."""
        menu.close_chain()
        QTimer.singleShot(0, lambda: self.leave(where))

    def leave(self, where: str) -> bool:
        """Close this window to go ``where`` (``"another"`` / ``"menu"``); False when the close was refused."""
        AFTER_CLOSE["next"] = where
        if not self.close():
            AFTER_CLOSE["next"] = None
            return False
        return True

    def _submit_bulk(self, function, *args, **kwargs):
        """A LONG job -- a section's Run, one per condition -- on the process pool: it waits here, on its job
        thread, for one of the `_bulk_slots`, so the pool always has a worker free for a job somebody is waiting
        on (see `_bulk_slots`); the slot is given back when the job ends, however it ends. Read at call time, so a
        pool swapped in (the tests' thread pools) is the one used."""
        return self._submit_bulk_unless(None, function, *args, **kwargs)

    def _submit_bulk_unless(self, cancelled, function, *args, **kwargs):
        """`_submit_bulk`, its wait for a slot ended by ``cancelled`` (a `threading.Event`, or None to wait as long as
        it takes): set, the wait raises `concurrent.futures.CancelledError` and nothing is submitted. For a Run that
        submits many jobs from one thread (the HMM's shared fit, a job per start: `hmm_run.joint_D`), so a Cancel
        never waits for a worker to come free."""
        from concurrent.futures import CancelledError

        if cancelled is None:
            self._bulk_slots.acquire()
        else:
            while not self._bulk_slots.acquire(timeout=0.2):
                if cancelled.is_set():
                    raise CancelledError("stopped while waiting for a worker")
            if cancelled.is_set():
                self._bulk_slots.release()
                raise CancelledError("stopped while waiting for a worker")
        try:
            future = self._process_pool.submit(function, *args, **kwargs)
        except BaseException:
            self._bulk_slots.release()
            raise
        future.add_done_callback(lambda _future: self._bulk_slots.release())
        return future

    def closeEvent(self, event) -> None:
        """Tear down `self._process_pool`'s own worker processes -- left
        running, they are orphaned OS processes outliving the window that
        owned them. A Run still in flight when the window closes has nothing
        left to report its result TO, so what has not started is dropped and
        what has is STOPPED (`parallel.stop_pool`, 2 Nov 2026): the shutdown
        alone dropped only the queue, and a worker in the middle of a fit ran
        on for minutes, unseen, slowing whatever the laptop did next. The HMM's
        shared fit stops submitting its starts too (`_hmm_cancel`)."""
        for slot in self._slots["hmm"]:
            slot._hmm_cancel.set()
        for slot in self._slots["kinetics"]:
            slot._kinetics_cancel.set()
        # the legacy window shows this workspace's project: it goes with it
        if getattr(self, "legacy_window", None) is not None:
            try:
                self.legacy_window.close()
            except RuntimeError:
                pass
        stop_pool(self._process_pool)
        super().closeEvent(event)
        if event.isAccepted():
            _OPEN_WINDOWS.discard(self)


#: Workspace windows opened by the theme toggle -- see `_toggle_theme`. Released in
#: `closeEvent`, only once the close is actually accepted.
_OPEN_WINDOWS: set = set()

#: Where the analysis goes after its workspace closes, set by the header's menu: ``"another"`` (the start screen,
#: to open another folder) or ``"menu"`` (the stage chooser); None = leave. Module-level, not on a window: the
#: theme toggle replaces the window `app.main` holds, so the one closing last may not be the one it knows.
AFTER_CLOSE: dict = {"next": None, "session": None}


for _section, _names in PAGE_ATTRS.items():
    for _name in _names:
        setattr(WorkspaceWindow, _name, _PerPage(_section, _name))
del _section, _names, _name


def take_after_close():
    """What the closed workspace asked for (and forget it): ``"another"``, ``"menu"``, ``"session"`` (the session's
    path in `take_session_path`) or None."""
    value, AFTER_CLOSE["next"] = AFTER_CLOSE["next"], None
    return value


def take_session_path():
    """The session another project's window was asked to open (and forget it), or None."""
    value = AFTER_CLOSE.get("session")
    AFTER_CLOSE["session"] = None
    return value


def open_workspace(project, *, color_scheme: str | None = None, parent=None,
                   show: bool = True, colours=None):
    """Build and show the workspace for an ALREADY LOADED project.

    Reads `project.conditions` / `project.roots` for the sidebar header and
    the Conditions tool; `project` itself is kept on the window so Run can
    fit against it -- see the module docstring's "RUN IS WIRED". Nothing
    HERE re-reads a file; that already happened before this was called.
    """
    app = QApplication.instance() or QApplication([])
    # THE ICON, which this door was opening without. Every other QApplication site in
    # this application applies it, and `test_appicon` rediscovers the sites from the
    # tree precisely so a new door cannot quietly skip it -- which is how it was found.
    # Imported here rather than at module scope: it touches Qt, and this module is
    # imported by the analysis route before any application exists.
    from ...appicon import apply as _apply_icon
    _apply_icon()
    t = theme.current(color_scheme)

    roots = list(getattr(project, "roots", {}).values())
    # THE FOLDER THAT WAS OPENED: the conditions' common parent. Not the first condition's parent, which in a folder
    # of protein folders is one protein's ("protein I" for the whole of a user's analysis folder).
    if len(roots) > 1:
        title = Path(os.path.commonpath([str(r) for r in roots])).name
    else:
        title = roots[0].parent.name if roots else "project"
    conditions = list(getattr(project, "conditions", []))
    subtitle = workspace_subtitle(project, conditions)

    window = WorkspaceWindow(
        t, project=project, project_title=title, project_subtitle=subtitle,
        conditions=[(c, c) for c in conditions], parent=parent, colours=colours)
    # THE SCREEN DECIDES THE FIRST SIZE (30 Oct 2026, `ui.widgets.screenfit`): 1280 x 800 where it fits with its
    # frame, maximised where it does not (a 1536 x 816 laptop), and the minimum held to the screen from now on. A
    # window built hidden is placed by its caller, who installs the watch (the theme switch does).
    if show:
        show_fitted(window)
    return app, window
