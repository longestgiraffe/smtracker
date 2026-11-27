"""The catalogue of analyses the GUI offers, as DATA rather than as Qt code.

Why a registry and not a pile of tabs. The requirement is that someone who used v2 can
find every tool they had, and that is a claim about COVERAGE, so it is stated once, here,
and TESTED (``tests/test_analyses.py`` asserts every v2 tab in ``docs/v2_to_v3_map.md``
has at least one entry). A hand-built Qt tab bar makes the same claim by hand and rots the
first time a tab is renamed.

The Qt panel reads this module and builds its controls from it. Nothing here imports Qt,
and nothing here draws, so the whole catalogue is exercised headless, the same rule
``smtracker.widgets`` follows.

Three things every entry must carry, because each has already been got wrong once:

1. **``legacy_source``**, the ``.m`` file this reproduces, or ``""`` for something v2
   never had. Without it "the legacy tool" and "our better version under the old name"
   are indistinguishable in the UI, which is the failure legacy mode exists to prevent.
2. **``status``**, same / improved / new / legacy only, matching the map's vocabulary. An
   improved entry must say how to get the v2 behaviour back, because the standing
   instruction is that people who want the old behaviour may have it.
3. **Parameters with ``default=None`` mean NO DEFAULT**, and the UI must make the user
   choose. This is not a UI nicety: ``dZ`` moves a corrected fraction by tens of points,
   ``localization_sigma_um`` moves the HMM slow fraction by tens of points, and
   the confinement radius has no natural value at all. A widget that pre-fills those
   invents a measurement. Every such parameter carries a ``note`` saying what it costs.

Running is deferred. ``Analysis.run`` is a callable resolved on demand, so importing this
module does not import saspt, scipy solvers or anything heavy; the catalogue can be listed
in milliseconds and only the chosen analysis pays.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

__all__ = [
    "AnalysisError",
    "INPUT_KINDS",
    "build_input",
    "Parameter",
    "Analysis",
    "ANALYSES",
    "TABS",
    "by_tab",
    "by_key",
    "parameters_without_defaults",
    "analyses_without_a_figure",
    "result_rows",
]


class AnalysisError(RuntimeError):
    """Raised when the catalogue is asked for something it does not have."""


#: The eight v2 tabs, plus the bucket for things v2 never had. These strings must match
#: the headings in ``docs/v2_to_v3_map.md``; a test asserts it, so renaming a tab there
#: without renaming it here fails rather than quietly dropping the coverage guarantee.
TABS = (
    "Import",
    "SLA",
    "MSD",
    "Clustering",
    "GMM",
    "SQD",
    "APPD",
    "SDA",
    "Beyond v2",
)

STATUSES = ("same", "improved", "new", "legacy only")

#: What an analysis consumes as its first argument. NOT every analysis takes the canonical
#: table: the SQD fit takes squared displacements, the gamma fit takes per-track apparent
#: D, the HMM takes step sequences. Naming the input as DATA keeps the derivation in one
#: pure function (``build_input``) instead of in the Qt panel, which is the same rule the
#: rest of the interactive layer follows. A panel that derived them inline would be
#: untested code deciding what the fit sees.
INPUT_KINDS = ("canonical", "assigned", "squared_displacements", "signed_displacements",
               "appd", "step_sequences", "msd_ensemble",
               "cell_frame", "sq_disp_table", "step_vectors",
               "transition_steps",
               # THE PROJECT ITSELF, for analyses that read something the canonical
               # table does not carry: the raw acquisition beside the tracks, the
               # detections before linking, the cell outlines as images. Every other
               # kind here is DERIVED from the canonical table by `build_input`; this
               # one cannot be, which is exactly why it was the gap that kept
               # `spot_intensity` and `detection` out of the catalogue.
               "project_scope")

#: The first PARAMETER NAME each kind must land on. Checked against the real signatures by
#: ``tests/test_analyses.py``, because "the target resolves" is not "the target runs": the
#: first draft of this catalogue declared six analyses as taking the canonical table when
#: their functions take an ensemble curve, signed displacements, per-unit fits, a labelled
#: map or the pre-canonical assigned table. Every one would have been a button that raised.
INPUT_FIRST_ARG = {
    "canonical": ("canonical", "table"),
    "assigned": ("assigned",),
    "squared_displacements": ("squared_displacements",),
    "signed_displacements": ("displacement_um",),
    "appd": ("appd",),
    "step_sequences": ("steps", "sequences"),
    "msd_ensemble": ("ensemble",),
    "cell_frame": ("canonical", "table", "projected"),
    "sq_disp_table": ("observable",),
    "step_vectors": ("steps",),
    "transition_steps": ("steps",),
    "project_scope": ("scope",),
}


@dataclass(frozen=True)
class ProjectScope:
    """One condition of a loaded project, for an analysis that must read from disk.

    WHY A SCOPE RATHER THAN THE PROJECT. Every catalogue analysis runs PER CONDITION --
    `catalogue_view` loops and calls once each -- so handing over the whole project
    would make each analysis re-derive which condition it was being run for, and the
    first one to get it wrong would silently report another condition's numbers. The
    condition comes with it.

    ``root`` is that condition's folder, which is where the raw acquisitions and the
    tracker output actually live. It is the piece the canonical table cannot carry: by
    the time a table exists, the movies it came from are a path nobody kept.
    """

    project: Any
    condition: str
    root: Any = None

    @property
    def canonical(self):
        """That condition's canonical table, for an analysis that wants both."""
        return self.project.condition(self.condition)


@dataclass(frozen=True)
class Parameter:
    """One control in the panel.

    ``default is None`` means the parameter HAS no default and the user must choose. The
    UI renders that as an empty, required field, never as a pre-filled guess.
    """

    name: str
    label: str
    kind: str                       # "float" | "int" | "bool" | "choice" | "*_tuple"
    default: Any = None
    choices: tuple = ()
    note: str = ""

    def __post_init__(self):
        if self.kind not in ("float", "int", "bool", "choice", "int_tuple",
                             "str_tuple"):
            raise AnalysisError(f"{self.name}: unknown kind {self.kind!r}")
        if self.kind in ("choice", "int_tuple", "str_tuple") and not self.choices:
            raise AnalysisError(f"{self.name}: a choice parameter needs choices")
        if self.default is None and not self.note:
            raise AnalysisError(
                f"{self.name} has no default and no note. A parameter the user must "
                f"choose has to say what choosing it costs, or the empty box reads as "
                f"an oversight rather than as a decision")

    @property
    def required(self) -> bool:
        return self.default is None

    def coerce(self, value):
        """Turn the widget's value into what the function expects.

        ``int_tuple`` exists because ``k_values`` is presented as a pick-list of
        comma-separated counts but ``select_k`` wants a tuple of ints. The catalogue
        handed the STRING straight through, and the failure was
        ``'<' not supported between instances of 'str' and 'int'`` deep inside the
        clustering code. A widget value and a function argument are not the same thing,
        and the conversion belongs here rather than in the Qt layer.
        """
        if value is None:
            return None
        if self.kind == "int_tuple":
            if isinstance(value, (tuple, list)):
                return tuple(int(v) for v in value)
            return tuple(int(part) for part in str(value).split(",") if part.strip())
        if self.kind == "str_tuple":
            # Same shape as int_tuple and for the same reason: the widget holds a
            # comma-separated string and the function wants a tuple. `include=` on the
            # consistency panel is a SET OF METHODS, so handing the string through would
            # make `"sqd" in include` true for every substring of it - "sqd,appd,msd"
            # contains "sqd" AND "appd" AND "msd" by accident, which happens to look
            # right and would silently stop working the moment a method name became a
            # substring of another.
            if isinstance(value, (tuple, list)):
                return tuple(str(v) for v in value)
            return tuple(part.strip() for part in str(value).split(",") if part.strip())
        if self.kind == "int":
            return int(value)
        if self.kind == "float":
            return float(value)
        if self.kind == "bool":
            return bool(value)
        return value


@dataclass(frozen=True)
class Analysis:
    """One entry in the catalogue."""

    key: str
    label: str
    tab: str
    status: str
    summary: str
    target: str                     # "module:function", imported on demand
    input_kind: str = "canonical"
    #: ``"module:function"`` drawing the result, or ``""`` when this analysis has no
    #: figure of its own. EMPTY IS A REAL ANSWER AND NOT A GAP TO PAPER OVER: most of
    #: these return a table, and a table shown as a table is honest where an invented
    #: chart reads as a finding. The panel shows the table and says which it is doing.
    #:
    #: A drawing named here must take ``(ax, result)``. That is checked against the real
    #: signature, because the first attempt at this field claimed two functions that do
    #: not fit: ``draw_uncertainty_summary`` takes per-GROUP values rather than a unit
    #: summary, and ``draw_qc_dashboard`` takes a whole figure plus four keyword
    #: datasets. Both would have been buttons that raised, which is the third time that
    #: same class of mistake has been caught here by checking a signature.
    draw: str = ""
    #: Column of the result to hand the drawing, when the drawing wants values rather
    #: than the table. ``draw_appd_histogram`` takes an ARRAY of D, not the frame.
    draw_column: str = ""
    legacy_source: str = ""
    legacy_route: str = ""          # how to get the v2 behaviour, for status "improved"
    parameters: tuple[Parameter, ...] = ()
    caveat: str = ""

    def __post_init__(self):
        if self.input_kind not in INPUT_KINDS:
            raise AnalysisError(
                f"{self.key}: unknown input_kind {self.input_kind!r}, expected "
                f"{INPUT_KINDS}")
        if self.tab not in TABS:
            raise AnalysisError(f"{self.key}: unknown tab {self.tab!r}, expected {TABS}")
        if self.status not in STATUSES:
            raise AnalysisError(f"{self.key}: unknown status {self.status!r}")
        if self.status in ("same", "improved", "legacy only") and not self.legacy_source:
            raise AnalysisError(
                f"{self.key} claims status {self.status!r} but names no legacy source. "
                f"An entry that reproduces v2 must say which file it reproduces, or the "
                f"UI cannot tell a legacy tool from a new method wearing its name")
        if self.status == "new" and self.legacy_source:
            raise AnalysisError(
                f"{self.key} is marked new but names {self.legacy_source}; pick one")
        if self.status == "improved" and not self.legacy_route:
            raise AnalysisError(
                f"{self.key} is improved but does not say how to run the v2 behaviour. "
                f"People who want the old behaviour may have it, so the route is part "
                f"of the entry, not a thing to look up")

    def resolve_draw(self) -> Callable | None:
        """The drawing callable, or ``None`` when this analysis has no figure."""
        if not self.draw:
            return None
        module_name, _, function_name = self.draw.partition(":")
        if not function_name:
            raise AnalysisError(f"{self.key}: draw must be 'module:function'")
        module = importlib.import_module(module_name)
        try:
            return getattr(module, function_name)
        except AttributeError:
            raise AnalysisError(
                f"{self.key}: {module_name} has no {function_name}. The catalogue names "
                f"a drawing that does not exist")

    def draw_wants_data(self) -> bool:
        """Does this drawing take the analysis input alongside the result?

        A fit needs its data to be judged: a model curve with no measured points beside it
        cannot show whether it fits. Asked of the SIGNATURE rather than declared as a
        flag, so the catalogue and the drawing cannot disagree about it -- which is the
        failure mode that has already produced a button that raises three times here.
        """
        import inspect
        drawing = self.resolve_draw()
        if drawing is None:
            return False
        return "data" in inspect.signature(drawing).parameters

    def draw_payload(self, result):
        """What to hand the drawing: the whole result, or one column of it."""
        if not self.draw_column:
            return result
        try:
            return result[self.draw_column].to_numpy()
        except Exception:
            raise AnalysisError(
                f"{self.key}: the result has no column {self.draw_column!r}, so the "
                f"catalogue and the result have drifted apart")

    def resolve(self) -> Callable:
        """Import and return the callable. Deferred so listing stays cheap."""
        module_name, _, function_name = self.target.partition(":")
        if not function_name:
            raise AnalysisError(f"{self.key}: target must be 'module:function'")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise AnalysisError(
                f"{self.key}: cannot import {module_name} ({exc}). The catalogue lists "
                f"it, so this is a missing dependency rather than a missing feature")
        try:
            return getattr(module, function_name)
        except AttributeError:
            raise AnalysisError(
                f"{self.key}: {module_name} has no {function_name}. The catalogue and "
                f"the code have drifted apart")

    @property
    def required_parameters(self) -> tuple[Parameter, ...]:
        return tuple(p for p in self.parameters if p.required)


# ===================================================================================
# The catalogue
# ===================================================================================

_RADIUS = Parameter(
    "radius_um", "ROI radius [um]", "float", None,
    note="NO DEFAULT. The radius is a choice, not a measurement, and it decides the "
         "answer: the confined-segment fraction rises monotonically with it (0.173 at "
         "50 nm to 0.854 at 300 nm) and means nothing on its own. Run the null.")
_MIN_STEPS = Parameter(
    "min_steps", "minimum steps in a dwell", "int", None,
    note="NO DEFAULT, for the same reason as the radius. v2's own printout used 5.")
_SIGMA = Parameter(
    "localization_sigma_um", "localisation sigma [um]", "float", 0.0,
    note="Default 0 gives APPARENT D, comparable with SQD. Correcting is a SENSITIVITY "
         "CHECK, not an upgrade: at 76 nm the slow fraction rises by tens of points "
         "and D1 goes to zero, i.e. the slow state becomes entirely localisation noise.")
_TAU_S = Parameter(
    "tau_s", "time lag [s]", "float", None,
    note="NO DEFAULT: read from movieData.mat (20 ms here). Typing it is how a silent "
         "rescaling of every result used to happen.")
_TAU = Parameter(
    "tau_lag_s", "time lag [s]", "float", None,
    note="NO DEFAULT: read from movieData.mat (20 ms here). Typing it is how a silent "
         "rescaling of every result used to happen.")
_NPOP = Parameter(
    "n_pop", "populations", "int", 2,
    note="BIC recommends and the user decides; a close call is reported as inconclusive "
         "rather than resolved. v2 supports 1, 2 and 3 only.")
_STARTS = Parameter(
    "starts", "seeding", "choice", "multistart", choices=("multistart", "legacy"),
    note="'legacy' runs v2's single hand-placed seed and bounds verbatim. Measured on "
         "this data: no difference (rss penalty about 4e-12), so the frozen reference "
         "numbers are unaffected by that defect.")
_NSTARTS = Parameter(
    "n_starts", "restarts", "int", 50,
    note="THE SAFE COUNT DEPENDS ON THE POPULATION COUNT, and an earlier note here said "
         "otherwise because it generalised from ONE group. Swept over all 18: at 2 "
         "populations every count down to a SINGLE start reproduces the 50-start answer "
         "(0 of 18 disagree), so a live view may use 1 and say so. At 3 populations 8 "
         "starts is still wrong in 4 of 18: RSS 7.9 to 16.3 per cent worse and fractions "
         "off by up to 62 pp, and all four are conditions with a small slow fraction, which "
         "makes a 3-population fit split a population that barely exists. 16 converges "
         "them all and 200 finds nothing better, so 50 carries about a threefold margin. Leaving "
         "this blank uses `sqd.default_starts(n_pop)`, which is 8 at 2 populations and 50 "
         "above; 50 here is the conservative setting for any reported number.")
_WEIGHTING = Parameter(
    "weighting", "weighting", "choice", "step", choices=("step", "track", "cell"),
    note="'step' is the default because it reproduces the frozen reference. Long tracks "
         "are preferentially slow, so 'track' and 'cell' are less biased: the slow "
         "fraction falls by a couple of points. Report which was used.")

ANALYSES: tuple[Analysis, ...] = (
    # ------------------------------------------------------------------ 1. Import
    Analysis(
        key="qc_dashboard", label="Dataset QC", tab="Import", status="improved",
        legacy_source="SMTracker.m statistics panels",
        legacy_route="the raw per-track table is qc.track_summary; the v2 panels showed "
                     "spots per frame and track lengths, both of which are drawn here",
        summary="Four panels, each printing its own N and threshold: the length filter's "
                "loss in BOTH tracks and localisations, localisation density against an "
                "unambiguous-linking criterion, and outlier cells and movies by "
                "median +/- k*MAD with the MAD-zero fallback named.",
        target="smtracker.qc:unit_summary",
        parameters=(
            Parameter("unit", "unit", "choice", "cell", choices=("cell", "movie", "day"),
                      note="which units the outlier rule is applied over."),
            Parameter("k", "outlier k (MAD)", "float", 3.5,
                      note="flagged, never dropped. THE MAD-ZERO TRAP: with few units, "
                           "if the others agree exactly the MAD is 0 and median +/- k*0 "
                           "flags nothing, blind on exactly the one-bad-replicate case the rule "
                           "exists for. Falls back to the mean absolute deviation about "
                           "the median and reports which estimator was used."),
        ),
    ),
    Analysis(
        key="track_census", label="Track and cell census", tab="Import", status="new",
        input_kind="assigned",
        summary="Where tracks and cells are lost, named per mechanism. The cell filter "
                "drops WHOLE tracks and its loss is D-selective (discarded tracks are "
                "less low-D at matched radius in 10 of 10 groups), which moves the "
                "spatial result in the direction that flatters it.",
        target="smtracker.cellmesh:assignment_funnel",
    ),
    # --------------------------------------------------------------------- 2. SLA
    Analysis(
        key="confinement", label="Confined-track detection", tab="SLA", status="same",
        legacy_source="dwellCount.m / tauCounter3.m",
        draw="smtracker.plots:draw_confinement_classes",
        summary="The moving-ROI detector, greedy longest-first with the legacy's shared "
                "one-gap allowance and endpoint-preserving masking. Returns free / mixed "
                "/ confined per track.",
        target="smtracker.confinement:confinement_events",
        parameters=(_RADIUS, _MIN_STEPS),
        caveat="A raw confined fraction means nothing without its chance rate. Run the "
               "null: at 150 nm and D = 0.2, 79 per cent of simulated FREE localisations "
               "are flagged confined.",
    ),
    Analysis(
        key="confinement_null", label="Confinement chance rate", tab="SLA", status="new",
        draw="smtracker.plots:draw_confinement_null",
        summary="The identical detector on matched free Brownian tracks, reporting "
                "observed / chance / EXCESS. The excess peaks at 100-150 nm (+0.343 at "
                "100 nm) and collapses by 300 nm where the circle exceeds a free step. "
                "The 100 nm excess independently matches the HMM, SQD and saSPT slow "
                "fractions from a purely geometric detector.",
        target="smtracker.confinement:confinement_null",
        parameters=(
            _RADIUS, _MIN_STEPS,
            Parameter("D_um2_s", "null D [um2/s]", "float", None,
                      note="NO DEFAULT. The null D must come from the analysis (the fast "
                           "state), not from the function's imagination."),
        ),
    ),
    Analysis(
        key="sla_classification", label="Motion classification", tab="SLA",
        status="legacy only", legacy_source="confinedMSDMotion.m",
        # CANONICAL, not `pooled_msd`. The pooled matrix is the output of two earlier
        # steps and nothing ran them, so this entry resolved and then raised. The target
        # now runs the chain - confined segments, their per-track curves, the pooled
        # curve - and `radius_um` surfaces as the choice it always was.
        input_kind="canonical",
        summary="The Brownian-vs-anomalous F-test on the pooled MSD of the CONFINED "
                "SEGMENTS (not of every track, which is a different question with an "
                "identical-looking answer).",
        target="smtracker.legacy.sla:classify_from_canonical",
        parameters=(
            Parameter("radius_um", "confinement radius (um)", "float", None,
                      note="NO DEFAULT: it decides what counts as confined, and a random "
                           "walk revisits its own neighbourhood. At 150 nm and "
                           "D = 0.2 um^2/s, 79 per cent of purely FREE localisations are "
                           "flagged confined, so this number sets the answer."),
            Parameter("min_steps", "minimum steps in a segment", "int", 5,
                      note="v2's GUI minimum (SMTracker.m:2954-2963). Report it with any "
                           "count."),
            Parameter("max_lag", "lags per curve", "int", 5,
                      note="v2 fits lags 1..minimum steps (confinedMSDMotion.m:13-15)."),
        ),
        caveat="Measured: it calls 12 of 12 groups sub-diffusive, which mixing and "
               "survivorship already predict with no anomalous motion present. At "
               "realistic scatter it calls a true exponent of 0.6 Brownian in 30 of 40 "
               "draws (under v2's own weighting, reproduced 14 Oct 2026).",
    ),
    Analysis(
        key="transitions", label="Transition counts", tab="SLA", status="improved",
        legacy_source="nTrans in the SLA tab",
        legacy_route="strict=False reproduces the legacy's inconsistent minimum; both "
                     "counts are always reported and neither is called the difference",
        draw="smtracker.plots:draw_transitions",
        summary="Observed exceeds chance in 12 of 12 groups. The legacy guards an "
                "interior gap with three steps and the head and tail with nothing, and "
                "that unguarded branch supplies about HALF of all counted transitions.",
        target="smtracker.confinement:transitions",
        parameters=(_RADIUS, _MIN_STEPS),
    ),
    # --------------------------------------------------------------------- 3. MSD
    Analysis(
        key="msd", label="MSD curve and fit", tab="MSD", status="improved",
        legacy_source="calcMSDTracks_.m",
        legacy_route="ask for the full curve explicitly; v2 fitted it silently and that "
                     "is worth 51 per cent in D on this data",
        input_kind="msd_ensemble",
        draw="smtracker.plots:draw_msd_fit",
        summary="Ensemble and per-track MSD, fitted over the reliable EARLY lags only. "
                "Asking for more lags than exist RAISES rather than falling back to the "
                "whole curve. The intercept's sigma (about 76 nm) is an UPPER bound: the cell "
                "wall bends the curve, and the tracker reports about 17 nm. The workspace's "
                "MSD section draws it per condition, along or across the cell too.",
        target="smtracker.msd:fit_msd",
        parameters=(
            Parameter("n_lags", "lags to fit", "int", 4,
                      note="the tail flattens because it is a shrinking, slow-biased "
                           "subset. Asking for more lags than exist RAISES; v2 fell back "
                           "to the whole curve, worth 51 per cent in D on this data."),
            Parameter("weighted", "weight by the lag's N", "bool", True,
                      note="later lags rest on fewer tracks."),
        ),
        caveat="MSD is a CROSS-CHECK, never the fraction readout: an ensemble MSD of a "
               "mixture is dominated by the fast state. alpha about 0.69 here is expected "
               "from POPULATION MIXING plus survivorship, not evidence of anomalous "
               "diffusion.",
    ),
    # -------------------------------------------------------------- 4. Clustering
    Analysis(
        key="clustering", label="k-means on MSD curves", tab="Clustering",
        status="improved", legacy_source="clusteringMSD.m",
        legacy_route="feature='raw' is the default and is exactly v2's scale",
        draw="smtracker.plots:draw_k_selection",
        summary="k-means on per-track MSD vectors, with the null v2 never had. Measured "
                "at k = 2: raw observed 2568 against chance 2703 +/- 83, so NO EXCESS; "
                "on the log scale observed 3390 against 2020 +/- 88, so +1370. This null "
                "has no free parameter, since CH is invariant to the null D.",
        target="smtracker.clustering:select_k",
        parameters=(
            Parameter("feature", "feature scale", "choice", "raw",
                      choices=("raw", "log"),
                      note="'raw' for v2 fidelity; on this data the log scale carries "
                           "the signal. Report which."),
            Parameter("k_values", "k to try", "int_tuple", "2,3,4",
                      choices=("2,3,4", "2,3,4,5,6"),
                      note="k-means ALWAYS returns clusters and Calinski-Harabasz never "
                           "says none."),
            Parameter("D_um2_s", "null D [um2/s]", "float", None,
                      note="NO DEFAULT, and this is the whole improvement over v2, which "
                           "had no null at all. The null has no free parameter: a "
                           "Brownian MSD scales exactly with D, so changing it rescales "
                           "(raw) or shifts (log) the cloud and CH is invariant to both, "
                           "verified to machine precision. Only sigma > 0 breaks that."),
        ),
        caveat="A cluster is a CURVE SHAPE, not a diffusive state, and never the fraction "
               "readout. Per-cluster alpha is doubly confounded by mixing and "
               "survivorship, and the clustering selects on exactly those.",
    ),
    # --------------------------------------------------------------------- 5. GMM
    Analysis(
        key="jd", label="Displacement mixture", tab="GMM", status="improved",
        legacy_source="getDispl_gui.m",
        legacy_route="the legacy seeds and bounds are runnable; the five fixed defects "
                     "are listed in legacy_out/legacy_deviations.csv",
        input_kind="signed_displacements",
        draw="smtracker.plots:draw_jd_fit",
        summary="A zero-mean normal mixture on SIGNED 1-D displacements with "
                "D = sigma^2/(2*tau), which is what v2 always produced. Recovery on a "
                "known answer: seeded at D = [0.05, 0.80] with 30 per cent slow, returns "
                "[0.0508, 0.8099] and [0.300, 0.700].",
        target="smtracker.jd:fit_jd",
        parameters=(_NPOP, _TAU_S),
        caveat="Six v2 defects fixed rather than ported: the n_pop = 1 crash, a "
               "one-sided outlier rule computed on an uncleaned scale, an invented "
               "interval for the derived fraction, a fraction dropped by sort order, "
               "nlparci on pooled ECDF points, and a third fraction that could go negative "
               "(fractions on a simplex since 14 Oct 2026).",
    ),
    # --------------------------------------------------------------------- 6. SQD
    Analysis(
        key="sqd", label="Squared-displacement CDF fit", tab="SQD", status="improved",
        legacy_source="sqdFit_.m / sqdFit_Simultaneous.m",
        legacy_route="starts='legacy' runs v2's single hand-placed seed and bounds "
                     "verbatim; compare_start_strategies runs both and reports the gap",
        input_kind="squared_displacements",
        draw="smtracker.plots:draw_sqd_fit",
        summary="THE FRACTION READOUT, and validated: reproduces the frozen MATLAB "
                "reference to 3 s.f. in both the fractions and the D values. Switching-robust, confirmed by "
                "matching the HMM occupancy.",
        target="smtracker.sqd:fit_sqd",
        parameters=(_NPOP, _TAU, _STARTS, _NSTARTS, _WEIGHTING),
        caveat="D1 sits at or below the localisation floor under either sigma estimate "
               "(30 or 76 nm), so it is a BOUND and never a physical coefficient. "
               "Fractions are unaffected.",
    ),
    Analysis(
        key="sqd_n_pop", label="State-count selection", tab="SQD", status="improved",
        legacy_source="the 'choose best fit automatically' checkbox",
        legacy_route="the recommendation is always visible beside any override",
        input_kind="squared_displacements",
        draw="smtracker.plots:draw_n_pop_selection",
        summary="BIC RECOMMENDS, the user decides. .choose(n, reason=...) requires a "
                "written reason and keeps the recommendation alongside; a gap below 2 is "
                "reported as inconclusive rather than silently resolved.",
        target="smtracker.sqd:select_n_pop",
        parameters=(_TAU,),
        caveat="On this dataset BIC prefers 3, not marginally (delta about 180). But the "
               "third state's D sits near the localisation floor, so overriding to 2 is a "
               "biological judgement and must be recorded as one.",
    ),
    Analysis(
        key="sqd_bootstrap", label="Bootstrap interval", tab="SQD", status="improved",
        legacy_source="nlparci error bars",
        legacy_route="the legacy intervals are still runnable in legacy mode, and are "
                     "90 to 131 times too narrow",
        input_kind="sq_disp_table",
        summary="Resamples the INDEPENDENT UNIT (cells or replicates), never "
                "correlated CDF points. About +/- 4 pp here. Below 5 units the percentile "
                "interval is quantised and that is flagged on .attrs.",
        target="smtracker.sqd:bootstrap_sqd",
        parameters=(
            Parameter("unit", "resampling unit", "choice", "cell",
                      choices=("cell", "day", "movie"),
                      note="report the unit; never compare intervals across units."),
            Parameter("n_pop", "populations", "int", 2,
                      note="the count must match the fit being bootstrapped: fractions "
                           "from a 2-state and a 3-state fit are different quantities "
                           "and cannot be compared or averaged."),
            Parameter("n_boot", "resamples", "int", 200,
                      note="this is the slow tier by design. The cheap tier fits each "
                           "unit once and still gives an honest permutation p."),
        ),
    ),
    Analysis(
        key="sqd_map", label="Jump-density map", tab="SQD", status="new",
        input_kind="cell_frame",
        summary="WHERE the jumps of each fitted population happen, in each cell's own "
                "coordinates. No new estimator: the same two-population fit the fractions "
                "come from, read per STEP instead of per condition. Each jump contributes "
                "to every population's map in proportion to its posterior, so the maps sum "
                "to the all-jumps map exactly.",
        target="smtracker.spatialsqd:spatial_sqd_map",
        draw="smtracker.plots:draw_jump_density_map",
        parameters=(
            Parameter("n_pop", "populations", "int", 2,
                      note="two populations are named 'short jumps' and 'long jumps', "
                           "which is literally what is mapped. Above two they are "
                           "NUMBERED by D: 'medium' would assert a real intermediate "
                           "state where the middle component may be measurement "
                           "heterogeneity, and it stops scaling at four."),
            Parameter("assignment", "assignment", "choice", "soft",
                      choices=("soft", "hard"),
                      note="SOFT weights each jump by its posterior and has no cliff. "
                           "HARD gives each jump to its most likely population, which at "
                           "two populations is exactly a threshold on jump length - and "
                           "a population whose best-case posterior is at or below 0.5 is "
                           "then never selected at all, so its map does not exist - it "
                           "is blank, and nothing warns you. Across all 23 deposited "
                           "groups every one clears 0.5, but narrowly where occupancy is "
                           "low, and by a wide margin where it is high. Check max_posterior_slow before drawing "
                           "hard, not after."),
            Parameter("step_position", "jump drawn at", "choice", "A_start",
                      choices=("A_start", "B_midpoint", "C_end"),
                      note="a jump spans two localisations, so where it is drawn is a "
                           "CONVENTION. Measured over the 16 deposited groups: start and "
                           "end agree to a median 0.0008 in mean |d_norm|, while the "
                           "MIDPOINT pulls long jumps toward the axis in 16 of 16 "
                           "(median -0.0288) and leaves short jumps alone (max 0.0037). "
                           "That is Jensen's inequality on |d|, so the pull grows with "
                           "step length and would bias a long-versus-short comparison. "
                           "v2's SDA tab uses the midpoint."),
            Parameter("mirrored", "fold the map", "bool", False,
                      note="folding SUMS each bin's four reflections, removing the "
                           "arbitrary choice of which pole is l=0 and which side is d>0. "
                           "A folded panel is four-fold symmetric BY CONSTRUCTION, so "
                           "symmetry there is not a finding; unfolded, left-right and "
                           "top-bottom asymmetry is not interpretable either."),
            Parameter("bin_long_nm", "bin, long axis (nm)", "float", None,
                      note="NO DEFAULT: the bin is a resolution claim. Compare it against "
                           "the camera pixel (106 nm here) and the localisation "
                           "precision, unmeasured at 30-76 nm. A bin below that band "
                           "displays structure the measurement did not resolve, which is "
                           "the legacy location map's defect."),
            Parameter("bin_short_nm", "bin, short axis (nm)", "float", None,
                      note="NO DEFAULT, and the short axis is the tighter one: at 18 "
                           "bins across a 1 um cell the bin is 56 nm, i.e. 0.52 camera "
                           "pixels and inside the unmeasured 30-76 nm "
                           "localisation-precision band, so structure finer than that is "
                           "display and not measurement."),
            Parameter("smooth_bins", "smoothing (bins)", "float", 1.1,
                      note="Gaussian bandwidth in BINS, so it can be read against the bin "
                           "size rather than in isolation."),
            Parameter("cell_length_um", "median cell length (um)", "float", None,
                      note="NO DEFAULT: this is a measurement of YOUR cells. v2 "
                           "hard-codes 3 um with the line that read the real value "
                           "commented out above it."),
            Parameter("cell_width_um", "median cell width (um)", "float", None,
                      note="NO DEFAULT, and v2 never scaled the width at all - it draws "
                           "every cell 1.0 um wide against tracks in real um."),
        ),
        caveat="THIS IS NOT PER-MOLECULE STATE INFERENCE. The posterior depends on jump "
               "length alone, so the map shows where short and long jumps happen, not "
               "which molecule was bound. Only the HMM uses the step SEQUENCE and can "
               "make that claim. At two populations several fitted cuts also fall BELOW "
               "the localisation floor - one deposited group's cut lies below an immobile "
               "molecule's median apparent step of 50 to 127 nm - so those maps select "
               "jumps shorter than an immobile molecule usually makes.",
    ),
    # -------------------------------------------------------------------- 7. APPD
    Analysis(
        key="appd", label="Apparent D per track", tab="APPD", status="improved",
        legacy_source="getAppD_gui.m",
        legacy_route="the legacy appDFilter is RETIRED rather than ported and the reason "
                     "is recorded; keeping all tracks is the change",
        input_kind="canonical",
        draw="smtracker.plots:draw_appd_histogram",
        draw_column="appd_um2_per_s",
        summary="Per-track apparent D with no offset removed. v2's F-test-plus-R2 filter "
                "selected tracks by how well they fit the model about to be fitted, "
                "guaranteeing a Brownian-looking answer and discarding exactly the "
                "switching molecules that carry the biology (526 to 378 tracks).",
        target="smtracker.appd:apparent_D",
        parameters=(
            Parameter("mode", "sub-track handling", "choice", "truncate",
                      choices=("truncate", "pad"),
                      note="one row per sub-track by design."),
        ),
        caveat="appD COMPRESSES a condition difference by averaging over within-track "
               "switching: +0.138 where SQD gives +0.275 and the HMM +0.293. Established "
               "on ground truth, not asserted, and predicted quantitatively at the "
               "measured rates (46 +/- 2 per cent of the SQD difference). A documented "
               "estimator property, not a defect.",
    ),
    Analysis(
        key="appd_gamma", label="Gamma mixture on appD", tab="APPD", status="improved",
        legacy_source="gammafit_one.m / gammafit_two.m / gammafit_three.m",
        legacy_route="starts='legacy' and legacy_bic reproduce v2's seeding and its "
                     "BIC verbatim",
        input_kind="appd",
        draw="smtracker.plots:draw_gamma_fit",
        summary="Shape = n_steps, mean = D, fitted by EM with a closed-form M-step.",
        target="smtracker.appd_gamma:fit_appd_gamma",
        parameters=(
            _NPOP, _STARTS,
            Parameter("n_steps", "steps per track (gamma shape)", "int", None,
                      note="NO DEFAULT. Passing the median track length instead of the "
                           "exact value widens every component without moving any fitted "
                           "D, which is invisible in the fitted numbers."),
        ),
        caveat="v2's BIC multiplies densities with prod(), which UNDERFLOWS to exactly "
               "0.0 in all 24 fits on this data. The sum of log likelihoods is used here.",
    ),
    # --------------------------------------------------------------------- 8. SDA
    Analysis(
        key="sda_maps", label="Spatial distribution maps", tab="SDA",
        status="legacy only", legacy_source="plotHeatmap.m / getDynamics.m",
        # `cell_frame` means the canonical table WITH cell coordinates, and its builder
        # checks for them. The bug was upstream: nothing ever handed it a projected
        # table, so this entry passed every signature check and then raised for every
        # condition. Caught by running the catalogue against real data rather than by
        # asking whether the target resolves.
        input_kind="cell_frame",
        summary="Location and dynamics maps in the cell frame, on the legacy's MIDPOINT "
                "step convention.",
        target="smtracker.legacy.sda:sda_points",
        parameters=(
            Parameter("step_position", "step position", "choice", "midpoint",
                      choices=("midpoint", "start"),
                      note="v2 assigns a jump to its MIDPOINT, v3's HMM decoding to its "
                           "START. Measured on all twelve groups, midpoint exceeds start "
                           "in 12 of 12 (median contrast 2.92 against 1.63), and NEITHER "
                           "IS THE TRUE VALUE because both are conventions."),
        ),
        caveat="plotHeatmap.m:17 interpolates a 20-node grid to 609 nodes, 1024 displayed "
               "cells per data bin. The data bin is already sub-pixel on the SHORT axis "
               "in 12 of 12 groups (0.54 to 0.74 camera pixels) and clears a pixel on the "
               "long axis in 12 of 12, so the verdict is PER AXIS. Cubic interpolation of "
               "counts also overshoots negative and the colour floor is pinned at 0, so "
               "the overshoot reads as an empty bin.",
    ),
    Analysis(
        key="state_map", label="Per-step state map", tab="SDA", status="new",
        # CELL_FRAME, not `labelled`. A labelled table is the output of fitting an HMM and
        # decoding it, and nothing ran those steps, so this entry resolved and then
        # raised. The target now runs the chain.
        input_kind="cell_frame",
        summary="THE FLAGSHIP MAP. Per-step HMM decoding localises individual bound "
                "segments, so it maps binding EVENTS rather than where slow molecules "
                "are on average. Radial enrichment 0.54 to 1.47 for the slow state "
                "against the per-track map's 0.88 to 1.15: the same signal about twice "
                "as sharply.",
        target="smtracker.maps:state_map_from_projected",
        parameters=(
            Parameter("n_states", "states", "int", 2,
                      note="the count must be chosen once for a comparison: fractions "
                           "from a 2-state and a 3-state fit are different quantities."),
            Parameter("localization_sigma_um", "localisation sigma (um)", "float", 0.0,
                      note="ZERO by default and that is deliberate. A sigma-corrected HMM "
                           "is NOT robust to the value - slow fraction rises by tens of points "
                           "between 0 or 30 nm and 76 nm, where the slow state "
                           "is entirely explained by localisation noise - and sigma is "
                           "unmeasured here. Raise it as a sensitivity check, never as a "
                           "correction."),
        ),
        caveat="This says the slow state is MORE PERIPHERAL THAN THE FAST STATE, a "
               "relative and projection-robust claim. Calling it membrane-bound "
               "additionally requires deconvolving the 2-D projection of a 3-D cylinder. "
               "Always fold first: which pole is l = 0 is arbitrary.",
    ),
    Analysis(
        key="directionality", label="Step orientation", tab="SDA", status="improved",
        legacy_source="bias/orientation_radians.m",
        legacy_route="axial_angle reproduces the legacy fold exactly; orientation is the "
                     "signed quantity v2 could not measure",
        input_kind="step_vectors",
        draw="smtracker.plots:draw_axial_order",
        summary="v2 computes atan(|dy| / |dx|), the absolute value on BOTH components, so "
                "every step lands in the first quadrant. What survives is AXIAL "
                "ANISOTROPY, not directionality, and the two are named apart here.",
        target="smtracker.directionality:axial_null",
        parameters=(
            Parameter("tau_s", "frame interval [s]", "float", None,
                      note="NO DEFAULT: read from movieData.mat (20 ms here)."),
            Parameter("cell_length_um", "cell length [um]", "float", None,
                      note="NO DEFAULT. The chance rate depends on the box, and these "
                           "cells measure 1.63 to 4.62 um (median 2.75)."),
            Parameter("cell_width_um", "cell width [um]", "float", None,
                      note="NO DEFAULT. Measured 0.72 to 1.18 um (median 0.90). A square "
                           "box gives essentially no axial order, a rod gives plenty."),
            Parameter("D_um2_per_s", "null D [um2/s]", "float", None,
                      note="NO DEFAULT. A rod-shaped cell produces axial anisotropy with "
                           "no biology in it, and THE CHANCE RATE GROWS WITH STEP SIZE "
                           "(+0.009 at D = 0.055 against +0.055 at D = 0.85 in a "
                           "3.0 x 0.9 um cell). Comparing a low-D against a high-D "
                           "population is confounded before any biology enters."),
        ),
    ),
    # -------------------------------------------------------------- beyond v2
    Analysis(
        key="hmm", label="Diffusive-state HMM", tab="Beyond v2", status="new",
        input_kind="step_sequences",
        draw="smtracker.plots:draw_hmm_states",
        summary="The only estimator here that MEASURES state switching rather than "
                "averaging over it. Two states, with the transitions it observes counted, and every "
                "start converging. Its occupancy matches SQD track-weighted, "
                "independently confirming SQD is switching-robust.",
        target="smtracker.hmm:fit_hmm",
        parameters=(
            Parameter("tau_s", "frame interval [s]", "float", None,
                      note="NO DEFAULT: read from movieData.mat (20 ms here). Typing it "
                           "is how a silent rescaling of every result used to happen."),
            Parameter("n_states", "states", "int", 2,
                      note="BIC prefers 3 and the third does not collapse, but its "
                           "slowest D sits at or below the localisation floor."),
            _SIGMA,
        ),
        caveat="LEAD WITH RATES, NOT DWELL TIMES. A fitted dwell time longer than the "
               "median observation window of 100 ms is a model extrapolation, "
               "and most such dwells are censored. Occupancy needs no transitions and is "
               "quotable; a rate needs observed transitions, and a rarely switching "
               "condition's are NOT identifiable.",
    ),
    Analysis(
        key="consistency", label="Cross-method consistency", tab="Beyond v2",
        status="new", input_kind="canonical",
        summary="Every estimator of the slow fraction, run on ONE dataset and put side "
                "by side. AGREEMENT ACROSS METHODS WITH DIFFERENT ASSUMPTIONS IS THE "
                "EVIDENCE - on the reference data SQD (track-weighted), the HMM and "
                "saSPT independently give about 35 per cent, which no single fit can "
                "claim. Disagreement is a finding and not noise to average away: appD "
                "reading low means switching inside its averaging window, and MSD "
                "landing between the two states means it is a mixture average rather "
                "than a state at all. The panel never averages the methods together, "
                "never votes, and never picks a winner.",
        target="smtracker.panel:consistency_panel",
        draw="smtracker.plots:draw_consistency_panel",
        legacy_source="",
        caveat="COSTS ABOUT A MINUTE AND A HALF PER CONDITION with saSPT included "
               "(measured: 83 s on the 2005-track reference set, of which saSPT is the "
               "bulk), and every catalogue analysis runs on every loaded condition. Drop "
               "saSPT from the method list for a fast look.",
        parameters=(
            Parameter("n_pop", "populations", "int", 2,
                      note="SHARED across the methods deliberately. Fractions from fits "
                           "at different state counts are different quantities, so a "
                           "panel that let each method pick its own count would be "
                           "comparing things that cannot be compared."),
            Parameter("include", "methods", "str_tuple", "sqd,appd,msd,hmm",
                      choices=("sqd,appd,msd,hmm",
                               "sqd,appd,msd,hmm,saspt",
                               "sqd,hmm",
                               "sqd,appd,msd,hmm,saspt,spoton"),
                      note="the default omits saSPT because it dominates the runtime. "
                           "spoton is offered last because its defocalisation "
                           "correction needs dZ, which is UNMEASURED here and swings the "
                           "corrected fraction by tens of points - it will refuse rather than "
                           "guess, and that refusal is the correct behaviour."),
            Parameter("weighting", "weighting", "choice", "track",
                      choices=("track", "step", "cell"),
                      note="TRACK by default here, not step. The panel is for "
                           "interpretation rather than for reproducing the frozen "
                           "reference, and step weighting lets a long track speak more "
                           "loudly - which biases toward slow molecules, since fast ones "
                           "leave focus sooner."),
            Parameter("n_starts", "fit restarts", "int", 20,
                      note="20 is comfortable at two populations, where the measured "
                           "requirement for ecdf_lsq is 1. At THREE it is under the "
                           "measured requirement of 24, and an under-converged fit is "
                           "penalised through its likelihood, so raise it with n_pop "
                           "rather than leaving it here."),
        ),
    ),
    Analysis(
        key="saspt", label="saSPT state array", tab="Beyond v2", status="new",
        summary="The one method here that does not assume a state count. It resolves TWO "
                "peaks plus a broad shoulder where every method that must pick a K picks "
                "3. Its slow peak sits at the lowest grid point, i.e. pinned at its "
                "resolution limit: the same D1-is-a-bound conclusion from a third "
                "direction.",
        target="smtracker.saspt_adapter:run_saspt",
        caveat="saSPT assumes ONE D per trajectory, so it is switching-blind. With "
               "switching at 6-8 per second some of that shoulder is what switching does "
               "to a single-D-per-track model. It votes on resolvable modes, not on "
               "switching.",
    ),
    Analysis(
        key="spoton", label="Spot-On jump lengths", tab="Beyond v2", status="new",
        draw="smtracker.plots:draw_spoton_fit",
        summary="Multi-lag jump-length fit with the defocalisation correction. Native, "
                "because the fastspt package is broken on Python 3.",
        target="smtracker.spoton:fit_spoton",
        parameters=(
            Parameter("dZ_um", "axial detection depth [um]", "float", None,
                      note="NO DEFAULT, and the function REFUSES to invent one. Measured "
                           "sensitivity: the corrected slow fraction changes with the correction "
                           "by a swing of tens of points "
                           "from one unmeasured optical constant. Run dz_sensitivity "
                           "before quoting any corrected fraction."),
        ),
        caveat="With the correction OFF, Spot-On gives a fraction consistent with SQD "
               "and the HMM. Only the correction is unpinned.",
    ),
    Analysis(
        key="bubble_series", label="Bubble plot (fits overlaid)", tab="SQD",
        status="improved",
        legacy_source="SMT_GUI bubble plot",
        # SQ_DISP_TABLE, because the simultaneous arm needs the DAY and the TRACK
        # columns, not merely the displacement array. Handed the bare array it could
        # form no units and would silently fall back to one pooled fit wearing a
        # per-day label -- the shape `INPUT_FIRST_ARG` exists to catch.
        input_kind="sq_disp_table",
        summary="v2 drew ONE fit. This draws several of the same data at once, each as "
                "its own series: independent (every condition keeps its own D) against "
                "simultaneous (one D shared, fractions free), and any set of state "
                "counts. Measured on a lab project, the arrangement is worth up "
                "to several percentage points, so drawing one alone hides a choice the "
                "reader cannot see. Series are never merged: a fraction always carries "
                "the state count and the fit mode that produced it, because 38 per cent "
                "of one of two states and 38 per cent of one of three are different "
                "quantities.",
        legacy_route="Tick `independent` alone - v2 fitted each condition on its "
                     "own and drew a single set of bubbles, one state count at a time.",
        target="smtracker.compare:bubble_series",
        parameters=(
            Parameter("fit_modes", "fits to draw", "str_tuple", "independent,simultaneous",
                      choices=("independent", "simultaneous",
                               "independent,simultaneous"),
                      note="INDEPENDENT is honest about D and leaves the fractions as "
                           "percentages of different things; SIMULTANEOUS shares one D "
                           "so the fractions mean one quantity, at the cost of forcing a "
                           "D none of the conditions has. Neither is right alone, which "
                           "is why both can be drawn."),
            Parameter("n_pops", "state counts", "int_tuple", "2",
                      choices=("2", "3", "2,3", "2,3,4"),
                      note="each count is its own series and they are never averaged. "
                           "Drawing them together is a COMPARISON OF MODELS; putting "
                           "them in one column is what `compare_fractions` refuses. BIC "
                           "recommends a count, it does not decide one."),
            Parameter("unit", "independent unit", "choice", "day",
                      choices=("cell", "movie", "day"),
                      note="the unit the per-series spread is formed over. Report it and "
                           "never compare spreads across units: measured here, "
                           "replicate-to-replicate 3.2-4.9 pp against cell-to-cell 12.4-14.2 pp. "
                           "`replicate` is refused on a single-session project rather than "
                           "faked."),
            Parameter("weighting", "weighting", "choice", "track",
                      choices=("step", "track", "cell"),
                      note="STEP reproduces the frozen reference and over-represents "
                           "slow molecules, since a long track contributes more steps "
                           "and long tracks are preferentially slow (by a couple of points "
                           "on the reference data). Report which was used."),
        ),
        caveat="The SD is BIOLOGICAL SPREAD over units, not a confidence interval "
               "- the uncertainty of the mean is smaller by sqrt(n), and at three "
               "replicates the SD is itself poorly determined. Bubble AREA "
               "carries the fraction, so a disc twice as wide is four times the "
               "value: read the printed percentage. The simultaneous arm REFUSES a "
               "track identifier appearing under more than one condition - "
               "separate projects often reuse movie names, and pooling them "
               "computes displacements BETWEEN molecules in different conditions, "
               "measured on real data as 29 steps of up to 9.2 um in a 1 um cell.",
    ),
    Analysis(
        key="uncertainty", label="Per-unit fits (the independent unit)", tab="Beyond v2",
        status="new",
        # SQ_DISP_TABLE, and the target CHANGED, and the reason is architectural rather
        # than technical. This entry used to point at `compare_conditions`, which is a
        # TWO-condition statistic, inside a panel that runs every analysis on ONE
        # condition at a time. It could never have worked here, and Amendment 4 says it
        # should not: comparison belongs in the manuscript, and
        # `tests/test_catalogue_view.py` asserts no between-condition statistic appears.
        # What DOES belong per condition is the half that carries the finding - fitting
        # each independent unit once, which is where the honest spread comes from.
        # `compare_conditions` still exists and is still the right function for a
        # comparison; it is simply not a panel analysis.
        input_kind="sq_disp_table",
        summary="Fits each independent unit ONCE - the CELL or the REPLICATE DAY, never "
                "the pooled step. Measured cost of getting the unit wrong: with real "
                "cell-to-cell variation and NO true difference, a test on pooled steps "
                "had a 65 per cent false-positive rate where 5 is nominal. The per-unit "
                "fits are what a SuperPlot draws and what an honest interval and "
                "permutation p are computed from.",
        target="smtracker.aggregate:fit_by_unit",
        parameters=(
            Parameter("n_pop", "populations", "int", 2,
                      note="shared across all units by construction: per-unit state "
                           "counts would make the fractions incomparable."),
            Parameter("unit", "independent unit", "choice", "cell",
                      choices=("cell", "movie", "day"),
                      note="report the unit and never compare intervals across units. "
                           "Measured here: replicate-to-replicate spread 3.2-4.9 pp against "
                           "cell-to-cell 12.4-14.2 pp, a between-replicate share of 6-12 per "
                           "cent."),
            Parameter("shared_D", "share D across units", "bool", True,
                      note="TRUE fits D once on the pooled data and only the fractions "
                           "per unit - the legacy simultaneous fit, and the only way "
                           "per-unit fractions mean the same thing. FALSE is diagnostic "
                           "and its fractions are NOT directly comparable."),
        ),
        caveat="A cross-condition comparison is deliberately NOT offered here. Two "
               "conditions are two independent samples of replicates (glucose replicate 1 "
               "is not paired with malate replicate 1), and at three replicates a side a two-sided "
               "permutation test floors at p = 0.100 whatever the effect size.",
    ),
    Analysis(
        key="intensity", label="Spot amplitude", tab="Beyond v2", status="new",
        summary="u-track's fitted Gaussian AMPLITUDE, read since Phase 1 and never used. "
                "Finite in every localisation, within-track CV median 0.384. "
                "No bleaching signature: the share of tracks decreasing contains the 0.5 "
                "expected under no trend in both conditions of the reference project.",
        target="smtracker.intensity:amp_per_track",
        caveat="IT IS A PEAK HEIGHT, NOT AN INTEGRATED INTENSITY: no background "
               "subtracted, no illumination correction. Never present it as v2's "
               "intensity readout, whose pipeline is blocked because this project ships "
               "phase/ and sum/ rather than the raw 1001-frame movie. The bleaching null "
               "is one half, not zero.",
    ),
    Analysis(
        # IMPORT, NOT "Beyond v2". That tab means methods v2 never had, and the guard
        # requires status "new" there — correctly, since a legacy tool listed as beyond
        # v2 would be the "new method wearing an old label" this catalogue exists to
        # prevent. This reproduces a legacy quantity, so it belongs beside the other
        # reading-and-checking entries.
        key="spot_intensity", label="Integrated spot intensity", tab="Import",
        # IMPROVED, NOT NEW: it reproduces the legacy quantity and fixes a measured
        # defect in it, so the entry has to name what it reproduces. The guard caught
        # "new" beside a legacy source and was right to -- the UI cannot otherwise tell
        # a legacy tool from a new method wearing its name.
        status="improved", input_kind="project_scope",
        summary="The legacy intensity quantity, integrated out of the REAL frames rather "
                "than read off u-track's fitted amplitude. Validated against that "
                "amplitude on the same spots: Pearson r = 0.952 over 2626 spots, 0 % "
                "negative. ONE MOVIE AT A TIME by design -- the integration reads every "
                "frame, so a condition of thirty movies is minutes. NOT COMPARABLE ACROSS "
                "CONDITIONS imaged in different sessions: the background floor differs by "
                "a large fraction between this project's own conditions, and the conditions "
                "are confounded with session by up to 21 months. Fractions are untouched "
                "by any of it, since they come from displacements and not photon counts.",
        target="smtracker.spot_intensity:intensity_for_scope",
        parameters=(
            Parameter("laser_power_mw", "laser power (mW)", "float", None,
                      note="NO DEFAULT and never will have one: it appears in no sample "
                           "file, every tier scales with it, and a guess produces a "
                           "number that looks like a measurement."),
            Parameter("tier", "calibration tier", "int", 1,
                      note="1 = raw integration, 2 = + offset and illumination, "
                           "3 = + autofluorescence. Tier 3 cannot be applied to a "
                           "labelled sample here; see the module."),
        ),
        legacy_source="SMT_GUI: spotIntensity.m, bkgCell.m, unevenIllumination.m",
        legacy_route=(
            "the v2 behaviour is Tier 1 with the legacy illumination correction, which "
            "divides by the RAW MAXIMUM of the averaged field: an extreme-value "
            "statistic over ~16 000 pixels, so its scale moves with how many frames were "
            "averaged: 705.24 at 50 frames against 688.65 at 800. Pass "
            "`smooth_peak=False` to `pseudo_flat_field` to reproduce it; the default "
            "takes the peak from a SMOOTHED COPY, which drops 50-vs-800 reproducibility "
            "from 2.195 % to 0.663 % and leaves the field itself untouched."),
        caveat="Tier 1 unless you say otherwise, so no flat field and no offset -- both "
               "scale the groups together and neither closes a cross-condition gap. "
               "IT RUNS ON THE FIRST MOVIE of the condition and says which: there is no "
               "movie picker here because the panel has no free-text control, and a "
               "list of thirty would have to be built before the project is loaded. "
               "Call `spot_intensity.intensity_for_scope(movie=...)` for another.",
    ),
    Analysis(
        key="detection_fdr", label="Detection threshold vs false positives",
        tab="Beyond v2", status="new", input_kind="project_scope",
        summary="Detections OUTSIDE the cell outlines are background, so they measure the "
                "false-positive rate directly: FDR = (N_outside/area_outside x area_"
                "inside) / N_inside. Sweeps the score and picks the LOWEST threshold "
                "meeting the target, because raising it further only discards real, "
                "preferentially dim, molecules. MEASURED on one field: 7492 detections, "
                "92.8 % of them inside cells covering 4.5 % of the field, FDR ~ 0.4 %.",
        target="smtracker.detection:sweep_for_scope",
        parameters=(
            Parameter("target_fdr", "target false-positive rate", "float", 0.05,
                      note="the LOWEST threshold meeting it is chosen; a stricter one "
                           "buys purity by discarding dim real molecules."),
        ),
        caveat="IT MEASURES PURITY AND CANNOT MEASURE COMPLETENESS: it only sees "
               "detections the upstream detector already kept, so it says nothing about "
               "how many real molecules were discarded before it looked. It also assumes "
               "false positives are equally likely inside and outside cells, which is "
               "false where cells autofluoresce -- measured here, in-cell 928 counts "
               "against a 597 background, so the true in-cell rate is WORSE than "
               "reported. Use it to check a threshold, never to choose one. It runs on "
               "the FIRST movie of the condition, for the reason the intensity entry "
               "gives.",
    ),
    Analysis(
        key="switching_map", label="Where switching happens", tab="Beyond v2",
        status="new", input_kind="transition_steps",
        draw="smtracker.plots:draw_switching_profiles",
        summary="Capture and release as LOCATED EVENTS, from the HMM's per-step "
                "decoding. The state map shows where the slow state IS; this shows where "
                "molecules enter and leave it. MEASURED ON ALL TWELVE GROUPS: release per "
                "slow step falls toward the membrane in 12 of 12 (median +0.036) while "
                "capture per fast step does not (negative in 11 of 12). So the peripheral "
                "enrichment is built by molecules staying stopped longer rather than by "
                "more of them stopping there.",
        target="smtracker.switching:radial_profile",
        parameters=(
            Parameter("event", "direction", "choice", "release",
                      choices=("capture", "release"),
                      note="different questions -- where binding begins and where it "
                           "ends -- and pooling them answers neither."),
            Parameter("slow_state", "slow state label", "int", None,
                      note="NO DEFAULT. decode_states labels states 1..n while fit.D is "
                           "0-indexed, and getting it wrong puts EVERY transition in one "
                           "direction, which reads as a finding rather than an "
                           "off-by-one. Use switching.slow_state_label."),
            Parameter("n_shells", "shells", "int", 5,
                      note="pooled along the cell, so a flat profile is not a flat map."),
        ),
        caveat="A GRADIENT HERE IS NOT NULL-TESTED BY ITSELF. The uniform-switching null "
               "was run on ONE group (where the null's spread covered zero and the observed "
               "gradient lay well outside it) and eleven have none; consistency across groups does not address "
               "that, because a systematic artefact is systematic in every group. The "
               "cell filter accounts for about 16 per cent. LB and Glu+Mal are not "
               "covered. A transition spans two localisations and is placed at the step's "
               "START, a convention that moves the magnitude.",
    ),
)


def build_input(kind: str, canonical, *, tau_lag_s=None, max_lag=4, mode="truncate",
                scope=None):
    """Derive an analysis's first argument from the canonical table.

    ONE PLACE, so the panel, a script and a test all feed a fit the same thing. Every
    derivation here already exists as a module function; this only chooses which.

    ``scope`` is required for -- and only for -- `project_scope`, the one kind that is
    NOT derived from the table. Refused rather than defaulted: an analysis that reads
    the raw acquisition cannot be handed a table and left to guess where it came from.
    """
    if kind not in INPUT_KINDS:
        raise AnalysisError(f"unknown input_kind {kind!r}, expected {INPUT_KINDS}")
    if kind == "project_scope":
        if scope is None:
            raise AnalysisError(
                "this analysis reads from the project itself (the raw acquisition, or "
                "the detections before linking), so it needs the loaded project and "
                "not only the canonical table. `catalogue_view` passes one; a caller "
                "building an input by hand must too")
        return scope
    if kind in ("canonical", "cell_frame", "assigned"):
        if kind == "cell_frame" and "l_norm" not in getattr(canonical, "columns", ()):
            raise AnalysisError(
                "this analysis needs cell-frame coordinates, which cellcoords."
                "project_tracks supplies. Running it on raw positions would put every "
                "map in the camera's frame and still draw")
        return canonical
    if kind == "squared_displacements":
        from .observables import squared_displacements
        return squared_displacements(canonical)["sq_disp_um2"].to_numpy()
    if kind == "sq_disp_table":
        # The TABLE, not the array: the bootstrap resamples cells or days, so it needs
        # the unit columns. Handing it the bare array would silently resample steps,
        # which is the pseudoreplication this project exists to refuse.
        from .observables import squared_displacements
        return squared_displacements(canonical)
    if kind == "step_vectors":
        # `step_vectors` needs the cell-frame POSITIONS in um (`L_um`, `D_um`), not the
        # normalised ones: an axial angle is a direction in the cell and l_norm/d_norm
        # are scaled differently per axis, so an angle computed from them would be
        # squashed by the cell's aspect ratio. Checked here rather than left to fail with
        # `missing column L_um` from three frames down.
        from .directionality import step_vectors
        needed = [c for c in ("L_um", "D_um") if c not in getattr(canonical, "columns", ())]
        if needed:
            raise AnalysisError(
                f"this analysis needs the cell-frame positions in um ({needed} missing), "
                f"which cellcoords.project_tracks supplies. The normalised l_norm and "
                f"d_norm are scaled differently per axis, so an angle computed from them "
                f"would be squashed by the cell's aspect ratio")
        return step_vectors(canonical)
    if kind == "appd":
        # THE VALUES, NOT THE TABLE. `fit_appd_gamma` calls np.asarray on what it is
        # handed and the table carries a uid column, so it raised on 't0'. The parameter
        # name `appd` matched perfectly, which is why the signature check passed.
        from .appd import apparent_D
        return apparent_D(canonical, mode=mode)["appd_um2_per_s"].to_numpy()
    if kind == "step_sequences":
        from .hmm import step_sequences
        return step_sequences(canonical)
    if kind == "signed_displacements":
        # THE COLUMN, NOT THE FRAME. `signed_displacements` returns a table (uid, axis,
        # displacement_um, cell, day, condition) and `fit_jd` calls np.asarray on what it
        # is handed, so the frame raises on the uid column. The parameter NAME matched
        # (`displacement_um`), which is why the signature check passed and the button
        # still raised: a name says nothing about a type.
        #
        # This takes BOTH axes, which is v2's "xy" direction. Per-axis selection is a v2
        # control that is not wired here yet, and saying so is better than a silent xy.
        from .jd import signed_displacements
        return signed_displacements(canonical)["displacement_um"].to_numpy()
    if kind == "msd_ensemble":
        from .msd import ensemble_msd, msd_per_track
        return ensemble_msd(msd_per_track(canonical, max_lag=max_lag))
    if kind == "transition_steps":
        raise AnalysisError(
            "'transition_steps' cannot be derived from the canonical table alone: it is "
            "the OUTPUT of an earlier step, an HMM fit plus its per-step decoding. An "
            "entry declaring it "
            "needs a target that RUNS that step, the way `sla_classification`, "
            "`state_map` and `sqd_map` now do; otherwise it is a button that raises")
    raise AnalysisError(f"no builder for {kind!r}")   # pragma: no cover


# ===================================================================================
# Lookup
# ===================================================================================
def by_tab(tab: str) -> tuple[Analysis, ...]:
    """Every analysis on one tab, in catalogue order."""
    if tab not in TABS:
        raise AnalysisError(f"unknown tab {tab!r}, expected one of {TABS}")
    return tuple(a for a in ANALYSES if a.tab == tab)


def by_key(key: str) -> Analysis:
    for analysis in ANALYSES:
        if analysis.key == key:
            return analysis
    raise AnalysisError(
        f"no analysis {key!r}. Available: {', '.join(a.key for a in ANALYSES)}")


def result_rows(result):
    """Any analysis result as ``[(name, value), ...]``, for display.

    Fifteen of the twenty-two analyses return a fit object or a dict, not a table --
    ``SQDFit``, ``HMMFit``, ``MSDFit``, ``GammaFit``, ``SpotOnFit``, ``KSelection``,
    ``NPopSelection``, ``ComparisonResult``, ``SASPTResult`` and six dicts. A panel that
    looks for ``.columns`` finds none and renders an empty box, which is what the first
    version did: clicking Run on the SQD fit, the fraction readout this whole project
    rests on, produced a title and a caveat and NO NUMBERS.

    So the rendering is defined over what the results actually are. A DataFrame keeps its
    own table (the caller uses it directly); everything else is flattened to name/value
    pairs here, in the pure module, so the Qt panel still computes nothing.

    Sequences are shown whole rather than summarised: ``D`` and ``fractions`` are the
    answer, and a mean of them would be a different quantity.
    """
    import dataclasses

    if hasattr(result, "columns"):
        raise AnalysisError(
            "a DataFrame carries its own table; show it directly rather than flattening "
            "it to name/value pairs, which would hide every row after the first")

    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        items = [(f.name, getattr(result, f.name)) for f in dataclasses.fields(result)]
    elif isinstance(result, dict):
        items = list(result.items())
    else:
        items = [(name, getattr(result, name)) for name in dir(result)
                 if not name.startswith("_")
                 and not callable(getattr(result, name, None))]
        if not items:
            items = [("value", result)]

    rows = []
    for name, value in items:
        if isinstance(value, (str, bytes)):
            text = str(value)
        elif hasattr(value, "tolist"):
            text = str(value.tolist())
        elif isinstance(value, (list, tuple)):
            text = str(list(value))
        elif isinstance(value, dict):
            text = "{" + ", ".join(f"{k}: {v}" for k, v in list(value.items())[:8]) + "}"
        elif value is None or isinstance(value, (int, float, bool)):
            text = str(value)
        else:
            text = type(value).__name__
        rows.append((str(name), text))
    return rows


def analyses_without_a_figure() -> tuple[str, ...]:
    """Keys of the analyses whose result is shown as a TABLE rather than a plot.

    Reported, never hidden. ``smtracker.plots`` holds eleven drawing functions and almost
    none of them takes an analysis result directly, so most entries here have no figure
    yet. Showing the table is honest; inventing a chart for it would be decoration that
    reads as a finding. This list is the remaining work, in one place, and it shrinks as
    real drawings are written.
    """
    return tuple(a.key for a in ANALYSES if not a.draw)


def parameters_without_defaults() -> tuple[tuple[str, Parameter], ...]:
    """``(analysis key, parameter)`` for every choice the user must make.

    The UI walks this to render required fields, and a test walks it to check each one
    says what choosing it costs.
    """
    return tuple((a.key, p) for a in ANALYSES for p in a.required_parameters)
