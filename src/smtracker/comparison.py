"""Comparison-first assembly, the tool measures the shift between conditions.

The single-condition views were the accident of a tool that grew one folder at a
time. The science is a *difference*: how the slow-population fraction moves from
glucose to malate, and whether that move clears its own uncertainty. So this
module assembles every method across N conditions at once; one condition is just
the N = 1 case.

The comparison basis: a mobility threshold, not a state index
------------------------------------------------------------
Two conditions may be best-fit with different state counts (BIC picks per
condition). "Fraction of state 1" is then not the same quantity in each, a
2-state and a 3-state fit slice the population differently. So the cross-condition
fraction is defined as **the fraction of the population with D < X** (default
X = 0.15 µm²/s, user-editable): integrate the fitted mixture below the threshold.
This is comparable across conditions regardless of how many states each was fit
with, and it is what every Δ / forest number here uses.

NEUTRAL by construction
-----------------------
Nothing here names a state "bound" / "free" / "slow" / "fast". A state is an
index ordered slowest→fastest by D and a numeric D value; interpretation is the
user's, in their caption. The assembled data structures carry ``D`` and
``fraction`` and a state *index*, never a semantic word, so a plot built from
them cannot print one.

Everything is pure (numpy/pandas/scipy) and returns plain dataclasses / dicts,
the Qt shell and the plot layer consume these, so a wrong number fails a test
here, never subtly on screen. Caching is the caller's job (pass results through
``ResultCache``); this module recomputes when asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "ComparisonError",
    "DEFAULT_THRESHOLD_UM2_S",
    "MODEL_CHOICES",
    "ComparisonSpec",
    "slow_fraction_below",
    "resolve_n_pop",
    "fit_condition",
    "sqd_bubbles",
    "appd_points",
    "hmm_kinetics",
    "fraction_superplot",
    "forest",
    "fraction_matrix",
    "saspt_spectra",
    "spatial_maps",
    "warm_conditions",
    "active_shortcuts",
    "HEATMAP_METHODS",
]


class ComparisonError(RuntimeError):
    """Raised when a comparison cannot be assembled honestly."""


class CancelledError(RuntimeError):
    """Raised (and caught by the worker) when a compute is cancelled mid-run.

    Cooperative: the assembly checks ``spec.should_cancel`` between conditions, so
    a cancel stops BEFORE the next condition's fit: it actually stops work rather
    than only hiding a bar. A cancelled assembly writes no whole-view result.
    """


def _check_cancel(spec):
    fn = getattr(spec, "should_cancel", None)
    if fn is not None and fn():
        raise CancelledError("compute cancelled")


#: The mobility threshold defining the comparable slow fraction: population with
#: D < X µm²/s. Editable by the user; this is only the default.
DEFAULT_THRESHOLD_UM2_S = 0.15

#: The largest count "Best" considers unless asked for more -- ruled by the
#: experimenter. THREE, because the legacy SQD fit supported 1-3 populations only (a
#: 4-population fit would be a new method under an old label), and because the
#: 4-population step is most of the cost: measured on the reference project,
#: 67.5 of 90.4 s, since it needs 200 starts at 337 ms each. The cost of the cap is
#: recorded, not hidden: the reference project prefers 4 once converged, so a condition that
#: lands ON the cap is named in the figure's caption.
DEFAULT_BEST_MAX_POP = 3

#: D-sharing scope, ONE setting, three mutually exclusive levels (Item 2, 6a):
#:   "replicate": every replicate fitted independently (own D + fractions).
#:   "condition": D shared across a condition's replicate days, fractions free
#:                 per replicate; conditions independent of each other.
#:   "global"   : D shared across ALL conditions AND replicates at once, fractions
#:                 free per condition-replicate. DEFAULT (6a): the PER-STATE primary
#:                 endpoint needs state correspondence, "state 1" must be the same D
#:                 in every condition for its fractions to be comparable. Only a
#:                 shared-D fit guarantees that.
#:   "group"    : D shared across the conditions of each GROUP -- one value of a folder design's factor
#:                 (`ComparisonSpec.d_sharing_groups`; asked 25 Oct 2026: "shared D should also have within same
#:                 group. so a group gets fittet together") -- with fractions free per condition-replicate and each
#:                 group its own state count: protein A glucose and protein A malate share one D, protein B's conditions another.
#:                 A group with one fittable condition is that condition's own fit.
#: A single enum makes a contradictory combination (e.g. "share across conditions
#: but not replicates") structurally unrepresentable.
D_SHARING_SCOPES = ("replicate", "condition", "group", "global")
DEFAULT_D_SHARING_SCOPE = "condition"
#: THE SCOPES A CONDITION CAN HAVE ON ITS OWN (`ComparisonSpec.d_sharing_by_condition`; asked 1 Nov 2026, a Shared D
#: per row of the Compute dialog's Conditions page): under each, a condition is fitted from its own data alone, so one
#: condition fitted Across replicates beside another fitted Not shared is two ordinary fits drawn side by side. "group"
#: and "global" share D ACROSS conditions -- properties of the whole set, never of one condition. The same two values
#: are spelled in `bubble_view.APART_SCOPES` and `workspace_view.ROW_SHARED_D`, which import nothing of the fit;
#: `tests/test_shared_d_per_condition.py` holds the three equal.
CONDITION_SCOPES = ("condition", "replicate")

#: Bump when the SHAPE or LOGIC of a cached comparison result changes but its
#: inputs/settings do not. The warm cache keys on inputs+settings, NOT on the code,
#: so without this a structural change (the bubble aggregation, the inspector
#: payload, the fraction weighting) would keep serving a pre-change cached result,
#: exactly the "different D per replicate" stale-payload the user hit. Bumping it
#: invalidates every comparison payload on the next run.
#: 3: the pooled fit now follows the weighting (it was always track) and the
#: replicate payload carries the pooled estimate. The warm cache is keyed on CONTENT
#: and deliberately ignores the commit, so without this bump an entry fit before the
#: change -- a track-weighted D -- would be served as "Step", and an old payload would
#: arrive without the pooled fit the Uncertainty setting "None" draws.
#: 4 (14 Oct 2026): "Best" now SELECTS the state count at the fit's own weighting. It was
#: always step-weighted, so a Track fit could carry a count chosen on step weights. The
#: keys already carry the weighting, but a Track entry cached before the change has the
#: same key as one made after it and would be served with the step-chosen count.
#: 5 (14 Oct 2026): Shared D = All Conditions follows the Weighting too (the joint fit and its
#: count were step-weighted whatever the setting).
#: 6 (14 Oct 2026): the replicate fit carries each cell's fractions (the nested mixed model's data).
#: 7 (14 Oct 2026): and each cell's share below the D threshold.
#: 8 (4 Nov 2026): Chunks are withheld on a condition with two or more replicates (`chunks_withheld`): its
#: payload under a chunk key holds its real replicates and the reason, where it held chunks dealt across cultures.
#: 9 (4 Nov 2026, raised from main's 8, Chunks withheld): exact-zero jumps are an immobile population, counted (`sqd`'s
#: module head): a synthetic truth condition's fits, its replicates' and its Best count all change, and the payload
#: names the rule (`zero_jumps`).
#: 10 (4 Nov 2026): the joint fit over conditions (Whole data set, Conditions) follows the Run's ECDF setting -- each
#: condition's ECDF thinned to the budget, as its own fit is; it fitted every point. Its D move (the middle
#: of three populations shifted by a few per cent), so every fraction fitted at them does too.
#: 11 (4 Nov 2026): and it uses the exact derivatives where the Run's Optimiser is Fast (the dialog's default), as
#: every other fit does; it used finite differences. The same optimum within the optimiser's tolerance, not the same
#: last digits, so a Fast joint D made before is not served.
#: 12 (4 Nov 2026, merged on top of 11): the exact-zero rule's floor (a 1 % floor, withdrawn the same day): zeros under
#: it stay in the fit as any jump, said; the joint fit decides under its weighting.
#: 13 (4 Nov 2026): the exact-zero rule has no floor: a handful of zeros is the counted immobile population too
#: (`sqd.zero_jumps`), so a joint fit with a few zeros has its populations changed.
#: 14 (4 Nov 2026): the Whole data set payload carries the joint fit's exact-zero decision (``zero_jumps``), as the
#: other scopes carry theirs; a payload kept before has none.
#: (23 Nov 2026: the chunk rule changed, chunks of at least three movies, and its key carries that,
#: `replicate_settings`' ``pseudo_min``, rather than this version: a Run without Chunks keeps every entry it had.)
COMPARISON_CACHE_VERSION = 14

#: A cell's fraction from fewer steps than this is mostly noise; such cells are left out of the
#: mixed model's data and counted (``cells_left_out``).
MIN_CELL_STEPS = 20

#: Multistart seeds for the INTERACTIVE SQD fits. The library default is 50, but
#: on real data the moment-based seed already finds the optimum, measured
#: byte-identical fractions/D at 6 vs 50 starts, while 50 is ~8x slower. So the
#: comparison views use a small keep-best count (still robust: one moment start +
#: a handful of random ones). The frozen reference validation keeps the library
#: default of 50 (it calls fit_sqd directly, not through here), so golden numbers
#: are untouched.
SQD_INTERACTIVE_STARTS = 12


def interactive_starts(n_pop: int, method: str = "ecdf_lsq") -> int:
    """Interactive multistart count: per population count, because 12 is not enough.

    THE FLAT 12 was a correctness bug, not merely a speed setting. It was justified
    against a measurement at TWO populations ("byte-identical at 6 vs 50"), and then
    applied to whatever ``resolve_n_pop`` returned, THREE included. Swept over all 18
    groups, a 3-population fit is still in a local minimum at 8 starts in 4 of them --
    RSS 7.9 to 16.3 per cent worse, fractions off by up to 62 pp -- and converges only
    at 16. All four are conditions with a small slow fraction, which makes a 3-population
    fit split a population that barely exists. So the interactive view could draw a
    failed fit, silently, on exactly the condition that carries the low-fraction claim.

    ``sqd.default_starts`` is the one source of truth for the measured requirement: 8 at
    one or two populations (requirement 1, so 8x margin), 50 above (requirement 16, so
    ~3x). Fifty is affordable here only because the ECDF budget below cuts the per-fit
    cost about fourteenfold -- the two changes are what make the SAFE count interactive,
    which is why they land together.

    THE ESTIMATOR'S OWN COUNT (10 Nov 2026). ``method`` is the fit's: a Per spot Run is maximum likelihood, and it
    got the least-squares count, 8 at two populations, against the 50 maximum likelihood needs
    (`sqd.MLE_STARTS_BY_N_POP`). On four of a lab project's conditions its
    two-population fit stopped in a local optimum, one population holding every step at D 0.83-0.88 and the other
    empty at D 3.5-8.2, drawn as 100 % slow (BIC 578-1895 worse than the optimum, which reads 6.6-9.6 %). The groups
    the note at `MLE_STARTS_BY_N_POP` names. Read off the spec by `ComparisonSpec.sqd_method`.
    """
    from .sqd import default_starts

    return default_starts(n_pop, method=method)


#: ECDF-quantile budget for the INTERACTIVE fits, and the reason it is 2000 rather than
#: smaller. The fit is a least-squares against the empirical CDF, so its per-iteration
#: cost scales with the number of ECDF POINTS -- 42 587 of them on the reference project. Thinning
#: keeps every measurement (each point still reports a quantile of the FULL sample) and
#: only evaluates the residual at fewer places along the curve, so unlike dropping tracks
#: it adds no sampling variance at all: it is deterministic and the data is untouched.
#:
#: It is nearly free for the same reason the legacy `nlparci` on ECDF points is WRONG:
#: adjacent points differ by 1/N and are almost perfectly correlated, so they are not
#: independent constraints on three parameters.
#:
#: MEASURED (reference project, n_pop 2, 8 starts) -- and 2000 is chosen at the KNEE, not at the
#: maximum speedup:
#:     points   42587    5000    2000    1000     500     200     100
#:     seconds  2.810   0.360   0.204   0.176   0.141   0.118   0.120
#:     speedup   1.0x    7.8x   13.8x   16.0x   19.9x   23.9x   23.3x
#:     slow pp  0.0000  0.0003  0.0008  0.0001  0.0022  0.0094  0.0046
#: Below about 1000 the TIME STOPS FALLING -- optimiser overhead dominates, not residual
#: evaluation -- while the error keeps growing, so 2000 to 100 buys 84 ms and costs an
#: order of magnitude of accuracy. Paying accuracy for time nobody can feel is the
#: definition of the wrong side of a diminishing return.
#:
#: 0.0008 pp is about 1/5000th of the +/-4 pp cell bootstrap half-width. The BOOTSTRAP
#: STAYS EXACT and the frozen reference validation calls `fit_sqd` directly, so no
#: reported number moves. `_rigor_notes` prints the budget on every figure regardless.
SQD_INTERACTIVE_ECDF_POINTS = 2000


def _ecdf_budget(spec):
    """The ECDF budget for an interactive fit: the spec's, or the measured default.

    An explicit ``ecdf_subsample`` on the spec always wins, including ``0``/``None``
    meaning "exact" for anyone who wants the untouched curve.
    """
    if getattr(spec, "ecdf_subsample", None) is not None:
        return spec.ecdf_subsample or None
    return SQD_INTERACTIVE_ECDF_POINTS

#: The library's full multistart count (fit_sqd default), what BIC model
#: selection always uses, since the selected COUNT is seed-sensitive.
SQD_FULL_STARTS = 50

#: A condition needs at least this many tracks before a mixture fit means
#: anything; below it the view states "n = … too few tracks to fit" rather than
#: plotting a fit of noise. FOR REVIEW: the exact value is a judgement call.
MIN_TRACKS_TO_FIT = 20


def _condition_track_count(project, condition):
    try:
        return int(project.condition(condition)["uid"].nunique())
    except Exception:                                  # pragma: no cover
        return 0


def active_shortcuts(spec: "ComparisonSpec") -> list:
    """Human-readable names of every speed shortcut active for ``spec``.

    Every shortcut that trades rigor for speed must be visible (in the provenance
    stamp AND on the exported figure) so a fast/locked result can never be
    mistaken for a full-rigor one. The list: a locked model (BIC selection
    bypassed), dev fast mode, the reduced interactive multistart, and (when on) an
    ECDF subsample. An empty list means full rigor.
    """
    out = []
    if spec.lock_model:
        out.append(f"model locked = {spec.lock_model} (BIC selection bypassed)")
    if spec.is_fast:
        # neutral wording ("fast" is on the figure-neutrality banned list)
        out.append("DEV MODE: reduced rigor")
    if spec.is_fast:
        out.append(f"SQD multistart = {spec.fast.sqd_n_starts} (DEV MODE)")
    else:
        # THE COUNT DEPENDS ON n_pop AND THIS FUNCTION DOES NOT KNOW IT, so the rule is
        # printed rather than a number that would be wrong for half the views. Saying
        # "12" here while the fit used 50 is a caveat that misinforms.
        method = spec.sqd_method()
        need = "24 and 50" if method == "mle" else "1 and 16"
        out.append(
            f"SQD multistart = {interactive_starts(2, method)} at 1-2 populations, "
            f"{interactive_starts(3, method)} at 3+ (the measured requirement is {need}; "
            f"BIC selection still uses full {SQD_FULL_STARTS})")
    budget = _ecdf_budget(spec)
    if budget:
        out.append(f"ECDF subsample budget = {budget} "
                   f"(point fit only, applied when N > budget)")
    return out

#: The per-condition model choices a user may pick. "best" = lowest-BIC count;
#: "saspt" = the count saSPT resolves nonparametrically, then fit parametrically.
MODEL_CHOICES = ("best", "1", "2", "3", "4", "saspt")


#: The time lags an SQD Run may use, in frames (`ComparisonSpec.sqd_lag`): v2's 1 to 4, and v2's "1-4 together"
#: (`JOINT_LAG`, 17 Oct 2026): the four lags in one fit, D and fractions shared across them.
SQD_LAGS = (1, 2, 3, 4)
JOINT_LAG = "1-4"                     # == observables.JOINT
SQD_LAG_CHOICES = SQD_LAGS + (JOINT_LAG,)

#: RANDOM CHUNKS AS PSEUDO-REPLICATES (asked 24 Oct 2026: "methods when people are not using replicates and just
#: pooling data. so that still we can get good error bars etc, like randomize the data set into x chunks"; built:
#: "Ur proposal sounds good build it"). The Compute dialog's Uncertainty "Chunks" deals each condition's movies at
#: random into `PSEUDO_CHUNKS` chunks -- its cells when it has fewer than `PSEUDO_MIN_PER_CHUNK` movies a chunk --
#: and every per-replicate step then runs on the chunks: a fraction per chunk with the condition's D held, the mean
#: +- SD drawn, Welch or the equivalence test between conditions.
#: MEASURED BEFORE IT WAS BUILT (`scripts/pooled_false_alarm.py`, docs/claude/sqd_and_statistics.md): two replicates
#: of the SAME condition, 76 pairs on the MSD set, compared as if each were a condition imaged once -- 5 chunks of
#: movies called them different 4 % of the time, 5 chunks of cells 5 %, against 58 % for a test on pooled steps. So
#: chunks of whole movies or cells hold their level WITHIN one culture. They cannot see a second culture: where
#: cultures differ (as in one lab project, where two conditions differed by a large margin), any test without replicates calls the culture a
#: condition, which is why every chunk figure says "pseudo-replicates within one culture".
PSEUDO_CHUNKS = 5
#: AT LEAST THREE UNITS A CHUNK (23 Nov 2026, asked: "Chunks maybe not a fixed number but depends on sample size?"): the
#: count set is the MOST chunks; a condition with fewer movies gets fewer chunks of whole movies (9 movies: 3 chunks
#: of 3) rather than chunks of cells, whose cells share a movie with cells in other chunks and so look more alike than
#: chunks are. Cells only where there are not movies enough for two chunks. The count never changes the standard error
#: of a condition's mean (that is the movies'); it keeps each chunk's value steady (`pseudo_chunk_count`).
PSEUDO_MIN_PER_CHUNK = 3
#: The deal's seed, recorded in every caption and key; `PSEUDO_SPLITS` deals in all, the first drawn, every one
#: tested, so a reader sees how far the p moves with the deal.
PSEUDO_SEED = 0
PSEUDO_SPLITS = 20

#: CHUNKS ARE FOR DATA IMAGED WITHOUT REPLICATES (ruled 4 Nov 2026, the overnight tester's A4: on a project with three
#: replicates a condition Chunks dealt 15 movies of three cultures into 5 chunks, every caption said "pseudo-replicates
#: within one culture", false there, and a bracket read "*** 5 v 5 chunks"). As Movies and Cells are withheld where
#: the replicate is the unit, so are Chunks: such a condition is drawn as its pooled fit with no spread, and the
#: caption, the card and the bracket say why. One clause for SQD, the apparent-D bubbles and the HMM.
CHUNKS_WITHHELD = ("chunks withheld: two or more replicates make the replicate the unit, and chunks dealt across "
                   "them would mix cultures (choose Replicates)")


def chunks_withheld(n_replicates: int) -> str | None:
    """Why Chunks are withheld from a condition holding ``n_replicates`` real replicates, as one sentence -- or None
    where they may run (one replicate: data imaged without replicates). The rule of `CHUNKS_WITHHELD`."""
    n = int(n_replicates)
    if n < 2:
        return None
    return (f"Chunks withheld: this condition holds {n} replicates, so the replicate is the unit, and chunks dealt "
            f"across them would mix cultures (choose Replicates)")


def pseudo_chunk_count(n_units: int, k: int) -> int:
    """How many chunks ``n_units`` movies (or cells) fill: ``k`` at most, at least `PSEUDO_MIN_PER_CHUNK` units each;
    0 where not even two chunks fill."""
    n = int(n_units) // PSEUDO_MIN_PER_CHUNK
    return min(int(k), n) if n >= 2 else 0


def pseudo_replicate_deal(table, k: int, *, seed: int = PSEUDO_SEED) -> tuple:
    """``(mapping, by)``: each movie's chunk (1..n) -- or each cell's, keyed ``"movie\x00cell"``, where the movies do
    not fill two chunks -- dealt at random in near-equal numbers, and the unit dealt. ``n`` is `pseudo_chunk_count`:
    ``k`` at most, at least `PSEUDO_MIN_PER_CHUNK` units a chunk (the chunks made: ``max(mapping.values())``).
    Deterministic: the units are sorted by name before the deal, so one seed is one deal. ComparisonError when there
    are not enough cells either."""
    movies = sorted(table["movie"].astype(str).unique())
    n = pseudo_chunk_count(len(movies), k)
    if n:
        by, units = "movie", movies
    else:
        cells = table[table["cell"].notna()]
        by = "cell"
        units = sorted((cells["movie"].astype(str) + "\x00" + cells["cell"].astype(str)).unique())
        n = pseudo_chunk_count(len(units), k)
        if not n:
            raise ComparisonError(
                f"{len(movies)} movies and {len(units)} cells cannot fill two chunks of at least "
                f"{PSEUDO_MIN_PER_CHUNK}: too little data for random chunks")
    order = np.random.default_rng(seed).permutation(len(units))
    return {units[j]: (pos % n) + 1 for pos, j in enumerate(order)}, by


def pseudo_replicate_labels(table, mapping: dict, by: str):
    """Each row's chunk under ``mapping`` (`pseudo_replicate_deal`), as an int array; 0 for a row whose unit the
    deal did not see (a track outside every cell under a deal of cells)."""
    keys = table["movie"].astype(str)
    if by == "cell":
        keys = keys + "\x00" + table["cell"].astype(str)
    return keys.map(mapping).fillna(0).astype(int).to_numpy()


def _sqd_observable(canonical, spec):
    """The squared displacements every SQD fit of ``spec`` uses: ``spec.sqd_lag`` frames apart (1 = consecutive
    frames), carrying their own lag time, which the fit reads as its tau. ONE place, so no fit of a Run can use
    another lag than its neighbours. Under `JOINT_LAG` the four lags together (`observables.joint_lag_displacements`,
    v2's "1-4 together")."""
    from .observables import joint_lag_displacements, squared_displacements

    if getattr(spec, "sqd_lag", 1) == JOINT_LAG:
        return joint_lag_displacements(canonical)
    lag = int(getattr(spec, "sqd_lag", 1))
    # AT LAG 1 THE CALL IT REPLACED, EXACTLY: a stand-in for `squared_displacements` that takes only the table
    # (`tests/test_best_follows_weighting.py`) broke on the keyword when it was always passed.
    return squared_displacements(canonical) if lag == 1 else squared_displacements(canonical, lag_frames=lag)


def _count_observable(canonical, spec, obs=None):
    """What a Run's STATE COUNT is chosen on: the fit's own observable, except under `JOINT_LAG`, where it is lag 1 --
    the joint set holds each jump at four lags, overlapping, so its BIC would count every jump up to four times and
    prefer more populations than the data hold. ``obs`` is the fit's observable, when the caller has it."""
    if getattr(spec, "sqd_lag", 1) != JOINT_LAG:
        return obs if obs is not None else _sqd_observable(canonical, spec)
    from .observables import squared_displacements

    return squared_displacements(canonical)


#: RESAMPLES FOR THE SQD SECTION'S SINGLE-SESSION INTERVAL (Uncertainty Cells or Movies: the condition's cells or
#: movies resampled, the model refitted each time, the 2.5 and 97.5 % points). 50 until 10 Nov 2026: a percentile
#: interval over 50 resamples holds its 95 % less well (the V1 draft's coverage, 91 % of the session's
#: own cells; the review's row P29, the experimenter: "do all"). 1000 for an afternoon, about twentyfold the time of 50 (the experimenter:
#: "1000 resamples too slow then"); 200 since, `bootstrap_sqd`'s own default and the apparent D's count, about
#: fourfold. `ComparisonSpec.interval_resamples`; DEV mode keeps its own.
SINGLE_SESSION_RESAMPLES = 200


@dataclass
class ComparisonSpec:
    """What to compare and how: the shared state every method view reads.

    ``models`` maps a condition to its model choice (see ``MODEL_CHOICES``);
    ``lock_model`` forces one choice on all conditions. ``threshold_um2_s`` is the
    mobility threshold for the comparable slow fraction. ``reference`` names the
    condition Δ's are measured against (defaults to the first selected).
    """

    conditions: list[str]
    models: dict[str, str] = field(default_factory=dict)
    lock_model: str | None = None
    threshold_um2_s: float = DEFAULT_THRESHOLD_UM2_S
    reference: str | None = None
    n_boot: int = 50
    #: A FastProfile (from smtracker.fastmode) for reduced-rigor dev iteration, or
    #: None for full rigor. Presence is loud: stamped into provenance + bannered.
    fast: "object | None" = None
    #: The largest state count "Best" (and saSPT's clamp) may return. 3 by default --
    #: see `DEFAULT_BEST_MAX_POP`; 4 is the opt-in. It never limits an explicitly
    #: chosen count: picking "4" on a condition still fits four.
    best_max_pop: int = DEFAULT_BEST_MAX_POP
    #: Worker count for per-condition parallelism. None/-1 = all-but-one core; 1 =
    #: serial. Records into provenance DISPLAY-ONLY (not the key), because parallel
    #: fits are byte-identical to serial, 8 and 64 cores give the same numbers.
    n_jobs: "int | None" = None
    #: Cooperative cancel hook: a zero-arg callable returning True to request a
    #: cancel. Checked between conditions (``_check_cancel``). Excluded from the
    #: cache key / provenance: it is control flow, not an input.
    should_cancel: "object | None" = None
    #: ECDF-subsample budget for the POINT SQD fit (Item 6). None = exact (default).
    #: When set, a fit with more than this many displacements is fit to that many
    #: evenly-spaced ECDF quantile points, validated IDENTICAL to 3 s.f. at
    #: budget ≥ 2000 on real data, ~3.4x faster. The bootstrap stays EXACT.
    ecdf_subsample: "int | None" = None
    #: D-sharing scope, ONE setting, three mutually exclusive levels (Item 2).
    #: "condition" (default) shares D across a condition's replicate days with free
    #: per-replicate fractions; "global" shares D across all conditions too;
    #: "replicate" fits every replicate independently. Stamped in provenance and
    #: named in the caption whenever it is non-default. There is no separate
    #: replicate toggle: the enum is the only control, so a contradictory
    #: combination cannot be expressed. A condition may be fitted at the other
    #: scope of `CONDITION_SCOPES` (`d_sharing_by_condition`, `scope_for`); this
    #: field is then the Run's own, the one every condition not named has.
    d_sharing_scope: str = DEFAULT_D_SHARING_SCOPE

    #: Which ESTIMAND the fraction fit targets (the simultaneous fit's fractions,
    #: D̂ held):
    #:   "track" (DEFAULT) → the MOLECULE fraction: each track counts once, so it is
    #:     the proportion of the POPULATION in a state, the pre-registered endpoint
    #:     (proportion of molecules below 0.15 µm²/s).
    #:   "step"  → the DISPLACEMENT (time) fraction: each displacement counts once,
    #:     i.e. the fraction of observed TIME in a state, which over-weights longer
    #:     (slow-biased) tracks. It reproduces the legacy step-weighted simultaneous
    #:     fit, a consistency check, never the reason for the endpoint.
    #: Stamped in provenance; labelled by estimand in the inspector/bubble.
    fraction_weighting: str = "track"

    #: D-axis Berglund correction (item C), default OFF. When True, the SQD bubble
    #: view displays D_true = (3/2)·(D_app − σ²/τ), static localization error then
    #: motion blur removed, applied to ALL states or none (never a subset), σ from
    #: the acquisition params. States whose corrected D falls at/below the floor
    #: render as BOUNDS, not points. It is DISPLAY-only (the fit/fractions are
    #: untouched: a fraction is set by the split, not the magnitude), but it IS in
    #: the provenance key and the caption: a corrected figure can never be mistaken
    #: for an apparent-D one, the same register as the fast-mode fence.
    d_correction: bool = False
    #: Which fits the OVERLAY view draws together. Both by default: the point of that
    #: view is that neither arrangement is right alone, and drawing one hides a choice.
    bubble_fit_modes: tuple = ("independent", "simultaneous")
    #: Which state counts the OVERLAY view draws together. Each is its own series and
    #: they are never averaged -- side by side is a COMPARISON OF MODELS, and putting
    #: them in one column is what `compare_fractions` refuses.
    bubble_n_pops: tuple = (2,)
    #: The unit the overlay view forms its per-series spread over.
    bubble_unit: str = "day"
    #: The SQD optimiser's gradient (`sqd.GRADIENTS`): "numeric" or "analytic", the exact
    #: derivatives -- the same fits, faster (measured 15 Oct 2026: 43 conditions of 7 proteins,
    #: fractions within 0.0001 pp). In every fit key when not numeric. The APP's default is
    #: analytic, set by the Compute dialog (`DEFAULT_COMPUTE`, `options` "sqd_optimiser",
    #: ruled 16 Oct 2026); a spec built in a script keeps the numeric path unless it asks.
    sqd_gradient: str = "numeric"
    #: THE TIME LAG every SQD fit of the Run uses, in frames (`SQD_LAGS`; asked 16 Oct 2026, v2's "Time lag 2/3/4"):
    #: displacements that many frames apart, fitted at tau = lag x the frame interval, so D is the APPARENT D at that
    #: lag. A diffusing state keeps its D across lags; a population of localisation error alone keeps its MSD
    #: (4 sigma^2), so its apparent D falls as 1/lag -- the test the open two-versus-three question wants. 1 (the
    #: default) adds nothing to the fit keys, so every cache made before the option still holds. Displacements at a
    #: lag above 1 overlap within a track: intervals stay over days or cells, as always. `JOINT_LAG` ("1-4") is v2's
    #: "1-4 together": every fit made to lags 1 to 4 at once, the count chosen at lag 1 (`_count_observable`).
    sqd_lag: "int | str" = 1
    #: RANDOM CHUNKS IN PLACE OF THE REPLICATES (`PSEUDO_CHUNKS`): 0, the default, is the real replicates; k >= 2
    #: deals each condition's movies (or cells) into k chunks and treats them as its replicates. In the replicate
    #: fits' keys when not 0, never in the pooled fit's (which does not read the replicates).
    pseudo_replicates: int = 0
    #: SHARED D WITHIN A GROUP (scope "group"): each condition's group, ``{condition: value}`` of the factor
    #: ``d_sharing_factor`` names (`workspace_view.spec_from_values` reads both off the folder design). Required,
    #: and covering every condition, under that scope; read by nothing else.
    d_sharing_groups: "dict | None" = None
    d_sharing_factor: str = ""
    #: FITS SIDE BY SIDE (asked 28 Oct 2026: "add multiple fits next to each other ... group with 1 pop and 2 pop ...
    #: same colour with different shades"; ticked on the Conditions page since 30 Oct): the MODELS (`MODEL_CHOICES`:
    #: "best", "1".."4", "saspt") every condition is fitted at, each the Run's own fit with the model locked to it
    #: (`bubble_view.compute_bubble_view`), drawn as a group of columns per condition, in `MODEL_CHOICES` order.
    #: Empty (the default) draws each condition at its own model. Read by the bubble view only: every fit inside is a
    #: fit at a locked model, so no fit key carries it. Counts given as integers (a session from 28-29 Oct) are read
    #: as those models.
    side_by_side: tuple = ()
    #: WHICH OF THEM EACH CONDITION SHOWS (asked 31 Oct 2026: "sqd bubble plot real data 1 and 2 pop fit, simulation
    #: only 1 pop fit ... choose for every protein / condition how many and which fits are shown and not for whole data
    #: set"): ``{condition: models}``, each a non-empty subset of ``side_by_side`` (kept in its order); a condition not
    #: named shows every one. Only with ``side_by_side``, and read by the bubble view only, as it is: a fit at a locked
    #: model is the same fit whichever columns are drawn.
    side_by_side_shown: "dict | None" = None
    #: A SHARED D PER CONDITION (asked 1 Nov 2026: the Compute dialog's Conditions page gives each row its own Shared
    #: D): ``{condition: "condition" | "replicate"}`` (`CONDITION_SCOPES`) for the conditions fitted otherwise than
    #: ``d_sharing_scope`` says; `scope_for` reads it. Only beside a ``d_sharing_scope`` of `CONDITION_SCOPES`: "global"
    #: and "group" share D across conditions, so one condition cannot leave them, and the pair is refused. NORMALISED
    #: in `__post_init__`: entries for conditions not in the Run and entries equal to ``d_sharing_scope`` are dropped,
    #: and when every condition of the Run is named with one scope that scope becomes ``d_sharing_scope`` and the dict
    #: None -- so a Run whose conditions all share one scope is, field for field and key for key, the uniform Run it
    #: draws. Each condition is fitted by the one per-scope path, `condition_replicate_fit`, at its own scope; no fit
    #: is written twice.
    d_sharing_by_condition: "dict | None" = None
    #: PER-SPOT PRECISION (6 Nov 2026; asked: the option in the SQD section, "yes sure"): ``(c_nm, p, source)`` of a
    #: GIVEN calibration (`precision.Calibration`: measured on immobile spots, never learned -- ruled 6 Nov 2026), or
    #: None, the model as before. Every SQD fit of the Run is then maximum likelihood with each step's own noise
    #: (`sqd.fit_sqd`'s ``precision``; `sqd_fit_kwargs`), and its numbers enter every fit key (`fast_settings`).
    #: A D shared ACROSS conditions (Conditions, Whole data set) is the joint maximum-likelihood fit with each step's
    #: noise (`aggregate.fit_simultaneous`'s ``precision``; since 6 Nov 2026, asked: "the spot precision stuff for
    #: simultanous fits, everywhere its possible"). Not beside lags fitted together or the Berglund D correction (the
    #: D's are corrected already).
    sqd_precision: "tuple | None" = None
    #: The conditions of a Per spot Run whose tracks carry NO per-spot precision (`workspace_view.precision_off`: a
    #: truth condition, a track table): fitted with no localisation-error model, by the same maximum likelihood
    #: (`sqd_fit_kwargs(condition)`), and named in the caption. Not inside a D shared across conditions with
    #: conditions that carry it (refused by name).
    sqd_precision_off: tuple = ()

    def __post_init__(self):
        if self.d_sharing_scope not in D_SHARING_SCOPES:
            raise ComparisonError(
                f"d_sharing_scope must be one of {D_SHARING_SCOPES}, "
                f"got {self.d_sharing_scope!r}")
        if not (isinstance(self.best_max_pop, int) and 1 <= self.best_max_pop <= 4):
            raise ComparisonError(
                f"best_max_pop must be an int from 1 to 4, got {self.best_max_pop!r}")
        if self.fraction_weighting not in ("step", "track"):
            raise ComparisonError(
                "fraction_weighting must be 'step' or 'track', "
                f"got {self.fraction_weighting!r}")
        if self.sqd_lag not in SQD_LAG_CHOICES:
            raise ComparisonError(f"sqd_lag must be one of {SQD_LAG_CHOICES}, got {self.sqd_lag!r}")
        if self.sqd_precision is not None:
            c_nm, p, source = self.sqd_precision
            if not (float(c_nm) > 0 and float(p) > 0 and str(source)):
                raise ComparisonError(f"sqd_precision is (c_nm > 0, p > 0, source), got {self.sqd_precision!r}")
            self.sqd_precision = (float(c_nm), float(p), str(source))
            if self.sqd_lag == JOINT_LAG:
                raise ComparisonError("per-spot precision is not built for lags fitted together (they reuse each jump in up to four displacements, so a likelihood over them counts it up to four times); "
                                      "fit one lag")
            if self.d_correction:
                raise ComparisonError("per-spot precision corrects the D's already: the Berglund correction would "
                                      "take the localisation error off twice")
        self.sqd_precision_off = tuple(str(c) for c in (self.sqd_precision_off or ()))
        if self.sqd_precision_off:
            if self.sqd_precision is None:
                raise ComparisonError("sqd_precision_off names conditions fitted without per-spot precision in a Run "
                                      "that has none (sqd_precision is None)")
            if self.d_sharing_scope in ("global", "group"):
                groups = ({"": list(self.conditions)} if self.d_sharing_scope == "global" else {})
                for c in self.conditions:
                    if self.d_sharing_scope == "group":
                        groups.setdefault(str((self.d_sharing_groups or {}).get(c, "")), []).append(c)
                for members in groups.values():
                    off = [c for c in members if c in self.sqd_precision_off]
                    if off and len(off) < len(members):
                        raise ComparisonError(
                            f"one D forced across conditions with and without per-spot precision ({', '.join(off)}: "
                            f"their tracks carry none) cannot model the error in some of them only: force D across "
                            f"Replicates, or leave {', '.join(off)} out")
        if isinstance(self.pseudo_replicates, bool) or not isinstance(self.pseudo_replicates, int) \
                or self.pseudo_replicates == 1 or not 0 <= self.pseudo_replicates <= 20:
            raise ComparisonError(f"pseudo_replicates is 0 (the real replicates) or 2 to 20 chunks, got "
                                  f"{self.pseudo_replicates!r}")
        models = tuple(self.side_by_side or ())
        if any(isinstance(m, bool) or str(m) not in MODEL_CHOICES for m in models) \
                or len({str(m) for m in models}) != len(models) or len(models) == 1:
            raise ComparisonError(f"side_by_side is empty or two or more different models of {MODEL_CHOICES}, got "
                                  f"{self.side_by_side!r}")
        self.side_by_side = tuple(m for m in MODEL_CHOICES if m in {str(x) for x in models})
        if self.side_by_side_shown:
            if not self.side_by_side:
                raise ComparisonError("side_by_side_shown says which of the fits side by side a condition shows; "
                                      "there are none (side_by_side is empty)")
            shown = {}
            for condition, wanted in dict(self.side_by_side_shown).items():
                wanted = {str(m) for m in wanted}
                if not wanted or wanted - set(self.side_by_side):
                    raise ComparisonError(f"{condition}: shows one or more of the models side by side "
                                          f"{self.side_by_side}, got {sorted(wanted)}")
                shown[str(condition)] = tuple(m for m in self.side_by_side if m in wanted)
            self.side_by_side_shown = shown
        else:
            self.side_by_side_shown = None
        if self.d_sharing_scope == "group":
            groups = self.d_sharing_groups or {}
            missing = [str(c) for c in self.conditions if c not in groups]
            if missing:
                from .workspace_view import setting_words

                raise ComparisonError(f"{setting_words('group')} needs every condition's group; none for "
                                      + ", ".join(missing))
        self._normalise_scopes()

    def _normalise_scopes(self) -> None:
        """`d_sharing_by_condition` checked and put in its one form (see the field): each value one of
        `CONDITION_SCOPES`, only beside a ``d_sharing_scope`` of them, kept to the Run's conditions and to those
        fitted otherwise than ``d_sharing_scope``; one scope for every condition becomes ``d_sharing_scope``."""
        own = dict(self.d_sharing_by_condition or {})
        bad = {str(c): s for c, s in own.items() if s not in CONDITION_SCOPES}
        if bad:
            from .workspace_view import SHARED_D_NAME, shared_d_pill

            raise ComparisonError(f"a condition's own {SHARED_D_NAME} is one of {CONDITION_SCOPES} ("
                                  f"{' or '.join(shared_d_pill(s) for s in CONDITION_SCOPES)}), got {bad}")
        names = {str(c) for c in self.conditions}
        own = {str(c): s for c, s in own.items() if str(c) in names}
        if own and self.d_sharing_scope not in CONDITION_SCOPES:
            from .workspace_view import setting_words

            raise ComparisonError(
                f"{setting_words(self.d_sharing_scope, factor=self.d_sharing_factor or '')} "
                f"({self.d_sharing_scope!r}) shares one D across conditions, so no condition can have its own "
                f"(asked for {', '.join(sorted(own))}); give every condition the set's choice")
        scopes = {own.get(str(c), self.d_sharing_scope) for c in self.conditions}
        if len(scopes) == 1:
            (only,) = scopes
            self.d_sharing_scope = only
            own = {}
        own = {c: s for c, s in own.items() if s != self.d_sharing_scope}
        self.d_sharing_by_condition = own or None

    def scope_for(self, condition) -> str:
        """The D-sharing scope ``condition`` is fitted at: its own (`d_sharing_by_condition`), else the Run's. Read
        with ``getattr`` because a spec pickled in a session before the field existed has no such attribute."""
        own = getattr(self, "d_sharing_by_condition", None) or {}
        return own.get(str(condition), self.d_sharing_scope)

    def scope_settings(self) -> dict:
        """The per-condition scopes for a WHOLE VIEW's cache key (`comparison_controller.provenance_for`): every
        condition's scope when they differ, nothing when the Run has one -- so every key made before stays, and a view
        assembled with one condition Not shared is never served for the Run with all of them Across replicates."""
        if not getattr(self, "d_sharing_by_condition", None):
            return {}
        return {"d_sharing_by_condition": {str(c): self.scope_for(c) for c in self.conditions}}

    @property
    def conditions_simultaneous(self) -> bool:
        """Back-compat read: True iff D is shared ACROSS conditions (global scope).
        The canonical control is ``d_sharing_scope``; this is a derived view of it,
        so a contradictory pair of flags cannot exist."""
        return self.d_sharing_scope == "global"

    @property
    def is_fast(self) -> bool:
        return self.fast is not None

    def effective_n_boot(self) -> int:
        return self.fast.n_boot if self.is_fast else self.n_boot

    def interval_resamples(self) -> int:
        """The resamples behind the SQD section's single-session interval (`SINGLE_SESSION_RESAMPLES`, 10 Nov 2026);
        DEV mode's own count where it is on. In the interval's cache key, so an interval kept at 50 is made again."""
        return self.fast.n_boot if self.is_fast else SINGLE_SESSION_RESAMPLES

    def fast_settings(self) -> dict:
        """The provenance stamp for the fit rigor: the honesty guard's key part -- and the optimiser's
        gradient when it is not the default, so a fit made one way is never served for the other (the
        default adds nothing, so every key made before the option existed still holds)."""
        out = self.fast.as_settings() if self.is_fast else {"fast": False}
        if getattr(self, "sqd_gradient", "numeric") != "numeric":
            out = {**out, "sqd_gradient": self.sqd_gradient}
        if getattr(self, "sqd_lag", 1) != 1:
            lag = self.sqd_lag
            out = {**out, "sqd_lag": lag if lag == JOINT_LAG else int(lag)}
        if getattr(self, "sqd_precision", None) is not None:
            # the calibration's numbers (its words are not the fit), only when per-spot: every key made before holds
            out = {**out, "sqd_precision": [round(self.sqd_precision[0], 9), round(self.sqd_precision[1], 9)],
                   # the maximum-likelihood start count (10 Nov 2026): per-spot fits kept before had least squares'
                   "starts": "mle"}
            if getattr(self, "sqd_precision_off", ()):
                out = {**out, "sqd_precision_off": sorted(self.sqd_precision_off)}
        return out

    def precision_calibration(self, condition=None):
        """The given calibration as a `precision.Calibration`, or None (the model as before) -- None too for a
        ``condition`` whose tracks carry no per-spot precision (`sqd_precision_off`)."""
        if getattr(self, "sqd_precision", None) is None:
            return None
        if condition is not None and str(condition) in (getattr(self, "sqd_precision_off", ()) or ()):
            return None
        from .precision import Calibration

        c_nm, p, source = self.sqd_precision
        return Calibration(c_nm=c_nm, p=p, source=source)

    def sqd_fit_kwargs(self, condition=None) -> dict:
        """What every SQD fit of the Run adds: per-spot precision and the maximum likelihood it needs, or nothing. A
        ``condition`` without per-spot precision in a Per spot Run (`sqd_precision_off`) keeps the maximum likelihood,
        so every condition of the Run is fitted by one estimator, without the noise model."""
        if getattr(self, "sqd_precision", None) is None:
            return {}
        calibration = self.precision_calibration(condition)
        return {"method": "mle"} if calibration is None else {"precision": calibration, "method": "mle"}

    def sqd_method(self, condition=None) -> str:
        """The estimator every SQD fit of the Run uses for ``condition``: "mle" under per-spot precision (with or
        without the noise model, `sqd_fit_kwargs`), else the default least squares. What sets the start count
        (`interactive_starts`, 10 Nov 2026)."""
        return self.sqd_fit_kwargs(condition).get("method", "ecdf_lsq")

    def replicate_settings(self) -> dict:
        """What the replicate fits read beyond `fast_settings`: the chunks, when the replicates are random chunks
        (nothing otherwise, so every key made before chunks existed still holds). ``pseudo_min`` since 23 Nov 2026,
        chunks of at least `PSEUDO_MIN_PER_CHUNK` units (`pseudo_chunk_count`): a chunk entry made under the rule of
        two is not served."""
        k = int(getattr(self, "pseudo_replicates", 0) or 0)
        return {"pseudo_replicates": k, "pseudo_seed": PSEUDO_SEED, "pseudo_splits": PSEUDO_SPLITS,
                "pseudo_min": PSEUDO_MIN_PER_CHUNK} if k else {}

    def group_of(self, condition) -> str:
        """The group ``condition`` shares D with under scope "group" (its value of ``d_sharing_factor``)."""
        return str((self.d_sharing_groups or {})[condition])

    def group_members(self, condition) -> list:
        """The conditions of this Run in ``condition``'s group, in the Run's order."""
        mine = self.group_of(condition)
        return [c for c in self.conditions if self.group_of(c) == mine]

    def sharing_settings(self, condition=None) -> dict:
        """What a fit that shares D ACROSS conditions reads beyond its own condition: the conditions it shares with
        (for ``condition``'s payload, its own group; for the whole view, every group). Nothing under condition or
        replicate scope, so every key made before stays. Under Across all conditions it was missing from the replicate
        payload's key, so a payload fitted beside one set of conditions could be served beside another.

        AND WHAT THE SHARED COUNT IS CHOSEN FROM (1 Nov 2026, the night), which the condition's own ``model`` in a key
        does not say: under Across all conditions the lock alone (`_global_joint_D`: a number, else Best on the pooled
        data -- every row at 2 with nothing locked was Best's count, keyed "2" like the lock "2" fitted at 2), under a
        group every member's model (`_group_count`). Missing until then, so one key could hold either fit; the keys
        of these two scopes moved, and nothing else's."""
        scope = self.d_sharing_scope
        if scope == "group":
            if condition is not None:
                members = self.group_members(condition)
                return {"d_sharing_with": [str(c) for c in members],
                        "d_sharing_models": {str(c): self.model_for(c) for c in members}}
            return {"d_sharing_groups": {str(c): self.group_of(c) for c in self.conditions}}
        if scope == "global":
            if condition is not None:
                return {"d_sharing_with": [str(c) for c in self.conditions], "d_sharing_lock": self.lock_model}
            return {"d_sharing_lock": self.lock_model}
        return {}

    def model_for(self, condition: str) -> str:
        if self.lock_model:
            return self.lock_model
        return self.models.get(condition, "best")

    def reference_condition(self) -> str:
        if not self.conditions:
            raise ComparisonError("no conditions selected")
        if self.reference and self.reference in self.conditions:
            return self.reference
        return self.conditions[0]


# --------------------------------------------------------------------------- #
# The comparison basis: fraction below a mobility threshold
# --------------------------------------------------------------------------- #


def slow_fraction_below(D, fractions, threshold_um2_s: float) -> float:
    """Fraction of the fitted mixture with D < ``threshold``, the comparable basis.

    A mixture fit gives (D_k, fraction_k). The slow fraction is the summed
    fraction of every population whose D sits below the threshold. Because it is
    a property of the *fitted distribution*, not of a state index, it is directly
    comparable between a 2-state and a 3-state fit of two conditions.
    """
    D = np.asarray(D, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    if D.shape != fractions.shape:
        raise ComparisonError(
            f"D and fractions differ in length: {D.shape} vs {fractions.shape}")
    return float(fractions[D < threshold_um2_s].sum())


def below_populations(D, threshold_um2_s: float) -> str:
    """WHICH POPULATIONS a share below the D threshold sums, for every caption that prints one: "populations 1-2",
    "population 1" or "none" (7 Nov 2026, the per-spot audit's CORRECTION 1: per spot the middle population's D fell
    under 0.15 in every SynR condition, so the caption's "below-threshold" read 55 % on low against 12 % plain, a
    threshold crossing and not a change in the fit; the share is comparable between two fits only where it sums the
    same populations). Populations are numbered slowest first, so those below are always the first ones."""
    below = [k + 1 for k, d in enumerate(np.sort(np.asarray(D, dtype=float))) if d < threshold_um2_s]
    if not below:
        return "none"
    return f"population {below[0]}" if len(below) == 1 else f"populations {below[0]}-{below[-1]}"


# --------------------------------------------------------------------------- #
# Model resolution
# --------------------------------------------------------------------------- #



def resolve_n_pop(observable, choice: str, *, weighting: str, part=None,
                  max_pop: int = DEFAULT_BEST_MAX_POP, n_jobs: int | None = 1, precision=None) -> int:
    """Turn a model choice into a concrete state count for one condition.

    ``"best"`` → lowest-BIC count (``select_n_pop``). ``"1".."4"`` → that count.
    ``"saspt"`` → the number of modes saSPT resolves nonparametrically, then that
    many are fit parametrically (clamped to ``[1, max_pop]``).

    ``weighting`` is the weighting of THE FIT THIS COUNT IS FOR, and "best" selects at
    it: the weighted log-likelihood with the Kish effective n in BIC's penalty, which
    ``sqd`` already computes for every weighted fit (no second BIC). It is REQUIRED,
    not defaulted: the selection used to receive none, so it ran at ``select_n_pop``'s
    own default ("step") while the fit beside it followed Weighting, and a Track fit
    carried a count chosen on step weights. A default would let the next caller repeat
    that without a sign; a missing argument is a TypeError.
    """
    if weighting not in ("step", "track", "cell"):
        raise ComparisonError(
            f"weighting must be 'step', 'track' or 'cell', got {weighting!r}")
    if choice in ("1", "2", "3", "4"):
        return int(choice)
    if choice == "best":
        from .sqd import select_n_pop

        # Full multistart here: BIC model SELECTION (which count wins) is sensitive
        # to the seed count, measured one condition flipping 2->3 pop at reduced starts.
        # Keep it deterministic; the reduction is applied only to FIXED-count fits.
        # `n_jobs` spreads the selection's starts over cores; it never changes the
        # answer (`sqd._run_plans`), so it is not part of any cache key.
        # per-spot precision: the count chosen with the same noise as the fit it is for
        selection = select_n_pop(observable, max_pop=max_pop, method="mle",
                                 n_jobs=n_jobs, weighting=weighting, precision=precision)
        return int(selection.recommended.n_pop)
    if choice == "saspt":
        if part is None:
            raise ComparisonError("'saspt' model needs the canonical part")
        from .saspt_adapter import run_saspt

        n_modes = len(run_saspt(part).modes())
        return int(np.clip(n_modes, 1, max_pop))
    raise ComparisonError(
        f"unknown model choice {choice!r}; use one of {MODEL_CHOICES}")


def fit_condition(project, condition: str, spec: ComparisonSpec):
    """Fit one condition per the spec; return ``(SQDFit, n_pop)``.

    WEIGHTED AS THE SPEC SAYS, for D as well as for the fractions. This used to be
    track-weighted unconditionally ("the weighting the fraction claims rest on"),
    written when the GUI defaulted to track. The experimenter then made STEP the
    default because it is the legacy fit -- and with D fixed at track, "step" became a
    hybrid the legacy never computed: a track-weighted shared D with step-weighted
    fractions on top. The weighting control now governs both halves, so step
    reproduces the legacy simultaneous design end to end and track is the
    switching-robust variant end to end.

    The state count comes from ``resolve_n_pop`` so the per-condition dropdown and the
    lock toggle both land here.
    """
    from .observables import squared_displacements
    from .sqd import fit_sqd

    part = _subsample(project.condition(condition), spec)
    obs = _sqd_observable(part, spec)
    # THE COUNT IS CHOSEN AT THE WEIGHTING IT IS FITTED AT (the fit_sqd call below); under lags 1-4 together, at lag 1.
    n_pop = resolve_n_pop(_count_observable(part, spec, obs), spec.model_for(condition), part=part,
                          weighting=spec.fraction_weighting,
                          max_pop=spec.best_max_pop, n_jobs=spec.n_jobs,
                          precision=spec.precision_calibration(condition))
    n_starts = (spec.fast.sqd_n_starts if spec.is_fast
                else interactive_starts(n_pop, spec.sqd_method(condition)))
    fit = fit_sqd(obs, n_pop=n_pop, weighting=spec.fraction_weighting,
                  n_starts=n_starts, ecdf_max_points=_ecdf_budget(spec), **spec.sqd_fit_kwargs(condition))
    return fit, n_pop


def _subsample(part, spec):
    """Optionally keep a fixed-seed fraction of tracks (fast mode only)."""
    if not spec.is_fast or not spec.fast.track_subsample:
        return part
    frac = spec.fast.track_subsample
    uids = part["uid"].drop_duplicates()
    keep = uids.sample(frac=frac, random_state=0)
    return part[part["uid"].isin(keep)].copy()


# --------------------------------------------------------------------------- #
# Per-method assembly: each takes N conditions
# --------------------------------------------------------------------------- #


@dataclass
class BubbleColumn:
    """One condition's column in the SQD bubble plot: its fitted populations.

    The bubble aggregation is EXPLICIT (Item 3). Per population:
    - ``D`` is the shared D (one value) under condition/global scope; under
      replicate scope it is the MEAN of the per-replicate D and ``D_sd`` carries
      their spread (there is no ± on D when it is shared by construction).
    - ``fractions`` == ``fraction_mean`` is the MEAN of the per-replicate
      fractions; ``fraction_sd`` is their SD across replicates, the spread of
      fractions CONDITIONAL on the shared D̂ (it excludes the common-mode
      uncertainty in D̂), over ``n_replicates`` replicates.
    All of it is built from the same ``condition_replicate_fit`` payload the
    inspector reads: one source of truth (Item 4).
    """

    condition: str
    n_pop: int
    D: list[float]                     # slowest→fastest (shared, or mean under replicate)
    fractions: list[float]             # == fraction_mean; matches D
    slow_fraction: float               # fraction below the threshold, from the means
    D_ci: list[tuple] | None = None    # per-population (lo, hi) on D, bootstrap-opt-in
    fraction_ci: list[tuple] | None = None
    #: explicit aggregation (Item 3)
    fraction_mean: list[float] | None = None
    fraction_sd: list[float] | None = None       # SD across replicates | shared-D
    n_replicates: int = 1
    per_replicate_fractions: list[list[float]] | None = None   # [replicate][pop]
    D_sd: list[float] | None = None              # only under replicate scope
    scope: str = "condition"
    #: every replicate day pooled into one set (Uncertainty "None"): its own D and
    #: fractions, since under a condition's own D the pooled fit's D is the shared one
    pooled_D: list[float] | None = None
    pooled_fractions: list[float] | None = None
    #: each cell's fractions with the shared D held (the mixed model's data), or None
    per_cell: list[dict] | None = None
    #: cells with fewer than `MIN_CELL_STEPS` steps, left out of ``per_cell``
    cells_left_out: int = 0


def _joint_observable(project, spec: ComparisonSpec):
    """The squared displacements of every condition of ``spec`` as ONE table: what the joint fit is made on and its
    exact-zero decision read (`_global_joint_D`, `_global_zero_jumps`)."""
    import pandas as pd

    parts = [_sqd_observable(project.condition(c), spec) for c in spec.conditions]
    obs = pd.concat(parts, ignore_index=True)
    obs.attrs.update(parts[0].attrs)
    return obs


def _global_zero_jumps(project, spec: ComparisonSpec) -> "dict | None":
    """The exact-zero decision of the joint fit over ``spec.conditions`` (`aggregate.joint_zero_jumps`, the rule
    `fit_simultaneous` applies), as the payload's ``zero_jumps`` dict, or None without an exact zero. Under Whole data
    set a condition has no fit of its own to carry it (4 Nov 2026: the bubbles said nothing, the other scopes did)."""
    from .aggregate import joint_zero_jumps

    zeros = joint_zero_jumps(_joint_observable(project, spec), spec.fraction_weighting)
    return None if zeros is None else zeros.to_dict()


def _global_joint_D(project, spec: ComparisonSpec):
    """The D shared across ALL conditions (global scope): the joint LSQ fit's D.

    Returns ``(shared_D, n_pop)``. D is shared across conditions by construction,
    so it is read from any row of the joint table."""
    import pandas as pd

    from .aggregate import fit_simultaneous
    from .observables import squared_displacements

    obs = _joint_observable(project, spec)
    lock = spec.lock_model
    # THE FIT'S OWN WEIGHTING, for the count and for the joint fit (14 Oct 2026: the
    # joint fit used to ignore it, so both were step-weighted). A uid carries its
    # condition (`<condition>_<movie>:<track>`), so per-track weights on the pooled
    # frame never join two conditions; the joint fit forms them per condition anyway.
    weighting = spec.fraction_weighting
    # PER SPOT, unless no condition joined here carries precision (a group of truth conditions: fitted as before; the
    # spec refuses a group that mixes them with tracked ones)
    off = set(getattr(spec, "sqd_precision_off", ()) or ())
    calibration = None if all(c in off for c in spec.conditions) else spec.precision_calibration()
    # named only where set: a Run without per-spot precision calls both fits exactly as before
    per_spot = {} if calibration is None else {"precision": calibration}
    if lock and str(lock).isdigit():
        n_pop = int(lock)
    else:
        counted = obs
        if getattr(spec, "sqd_lag", 1) == JOINT_LAG:       # the count at lag 1 (`_count_observable`)
            counted = pd.concat([_count_observable(project.condition(c), spec) for c in spec.conditions],
                                ignore_index=True)
            counted.attrs.update(_count_observable(project.condition(spec.conditions[0]), spec).attrs)
        n_pop = resolve_n_pop(counted, "best", weighting=weighting,
                              max_pop=spec.best_max_pop, n_jobs=spec.n_jobs, **per_spot)
    # The joint D uses the INTERACTIVE start count (the moment-based seed already
    # finds the optimum) so the global-scope DEFAULT stays responsive; the count is
    # already chosen (lock or the full-multistart BIC in resolve_n_pop).
    # per spot the joint fit is maximum likelihood and takes its count (10 Nov 2026)
    n_starts = (spec.fast.sqd_n_starts if spec.is_fast
                else interactive_starts(n_pop, "mle" if per_spot else "ecdf_lsq"))
    # ITS STARTS SIDE BY SIDE (`spec.n_jobs`, as the count's selection above): 377 s one after another on
    # a lab project's four conditions at 3 populations; the same fit whatever the cores (`fit_simultaneous`).
    # AT THE RUN'S ECDF SETTING (4 Nov 2026, ruled by the experimenter: "follows the Run's ECDF setting, as every
    # other fit does, so the Metadata receipt is true"): each condition's ECDF thinned to the budget, as its own fit
    # is. It fitted every point: on a four-condition lab project at 3 populations the D values moved by up to a few per cent
    # at 2000 points, a fraction up to 0.67 percentage points apart (`COMPARISON_CACHE_VERSION` 10).
    # AND THE RUN'S OPTIMISER (4 Nov 2026, ruled): Fast's exact derivatives
    # (`aggregate._simultaneous_jacobian`), Standard's finite differences -- named from the spec, as the key is
    # (`fast_settings`), so the two never part.
    # PER-SPOT PRECISION (6 Nov 2026): the joint MAXIMUM-LIKELIHOOD fit, each step its own noise, each condition the
    # same say in D (`aggregate.fit_simultaneous`'s ``precision``)
    table = fit_simultaneous(obs, n_pop=n_pop, by="condition", n_starts=n_starts,
                             weighting=weighting, n_jobs=spec.n_jobs, ecdf_max_points=_ecdf_budget(spec),
                             gradient=getattr(spec, "sqd_gradient", None), **per_spot)
    shared_D = [float(table[f"D_{k}"].iloc[0]) for k in range(1, n_pop + 1)]
    return shared_D, n_pop


def shared_model_bic(project, spec: ComparisonSpec, *, max_pop=4) -> dict:
    """The SHARED-model BIC diagnostic (6a): under global scope the state count is
    chosen ONCE for the whole comparison, on the POOLED displacements, not per
    condition. Returns the BIC curve over counts (with ΔBIC to the best), the
    recommended shared count, and (as a stability CHECK) what each condition would
    have selected ALONE. Days disagreeing with the shared choice is a QC signal, not
    licence to fit different models per condition.
    """
    import pandas as pd

    from .observables import squared_displacements
    from .sqd import select_n_pop

    parts = [_count_observable(project.condition(c), spec) for c in spec.conditions]
    pooled = pd.concat(parts, ignore_index=True)
    pooled.attrs.update(parts[0].attrs)
    sel = select_n_pop(pooled, max_pop=max_pop, method="mle", precision=spec.precision_calibration())
    curve = [{"n_pop": f.n_pop, "bic": float(f.bic),
              "delta_bic": float(sel.delta_bic(f.n_pop))}
             for f in sorted(sel.fits, key=lambda x: x.n_pop)]
    per_condition = {c: int(select_n_pop(obs, max_pop=max_pop, method="mle", precision=spec.precision_calibration(c))
                            .recommended.n_pop)
                     for c, obs in zip(spec.conditions, parts)}
    return {"recommended": int(sel.recommended.n_pop), "curve": curve,
            "per_condition_alone": per_condition, "max_pop": int(max_pop)}


def _cached_global_D(project, spec: ComparisonSpec, cache):
    """The global joint D (``_global_joint_D``), memoized on a PROJECT-level key,
    it is one joint fit over ALL selected conditions, so a warm hit makes the
    global-scope DEFAULT (6a) as fast as the per-condition path once computed."""
    if cache is None:
        return _global_joint_D(project, spec)
    import pandas as pd

    from .provenance import capture

    pooled = pd.concat([project.condition(c) for c in spec.conditions],
                       ignore_index=True)
    provenance = capture(
        settings={"view": "global_joint_D", "conditions": list(spec.conditions),
                  "lock": spec.lock_model, "cache_version": COMPARISON_CACHE_VERSION,
                  "best_max_pop": spec.best_max_pop,
                  # THE WEIGHTING THE JOINT FIT IS MADE AT (`_global_joint_D` follows it since 14 Oct 2026). Missing
                  # from this key until 25 Oct 2026, so a step-weighted joint D could be served to a track Run.
                  "weighting": spec.fraction_weighting,
                  # AND ITS ECDF BUDGET (the joint fit follows the Run's ECDF setting since 4 Nov 2026), resolved as
                  # `_cached_fit` keys it: a Full Run is never served a Fast joint D
                  "ecdf": _ecdf_budget(spec),
                  **spec.fast_settings()},
        seeds={"sqd": 0}, canonical=pooled, unit="cell")
    key = "cmp_global_D_" + "_".join(spec.conditions)
    return cache.get_or_compute(key, provenance,
                                lambda: _global_joint_D(project, spec))


def _group_count(spec: ComparisonSpec, members) -> "str | None":
    """The count a group's shared D is fitted at: the lock; else the number every member's own row names, when
    they all name the same one; else None -- "Best" on the group's pooled data (`_global_joint_D`)."""
    if spec.lock_model:
        return spec.lock_model
    models = {spec.model_for(c) for c in members}
    if len(models) == 1:
        (only,) = models
        if str(only).isdigit():
            return str(only)
    return None


def _group_joint_spec(spec: ComparisonSpec, members) -> ComparisonSpec:
    """The joint fit over one group's conditions: Across all conditions, restricted to ``members`` and its count."""
    from dataclasses import replace

    return replace(spec, conditions=list(members), d_sharing_scope="global", d_sharing_groups=None,
                   lock_model=_group_count(spec, members))


def _fittable(project, conditions) -> list:
    return [c for c in conditions if _condition_track_count(project, c) >= MIN_TRACKS_TO_FIT]


def _held_fractions(g, shared_D, tau, weighting, spec, condition=None):
    """The fractions of ``g``'s steps with ``shared_D`` held, at ``weighting`` -- and, under the spec's per-spot
    precision, with each step's own noise (`aggregate._fit_fractions`'s ``step_noise``), the steps lacking a spot's
    precision left out (`precision.usable_steps`). ONE place, so no fraction of a Run is made without it. ``g`` is a
    part of the condition (a replicate, a cell, a chunk) or its whole at a joint D; the share of steps without a
    precision is refused once, by the fit that made the D (`sqd.fit_sqd`, `aggregate.fit_simultaneous`), never again
    for a part: a cell's few such steps are left out, not the Run (7 Nov 2026, found on SynR tracked by TrackMate)."""
    from .aggregate import _fit_fractions
    from .observables import displacement_weights

    calibration = spec.precision_calibration(condition)
    noise = None
    if calibration is not None:
        from .precision import step_noise, usable_steps

        ok = usable_steps(g, max_missing=None)
        if not ok.all():
            g = g.loc[ok]
        noise = step_noise(g, calibration)
    sw = None if weighting == "step" else displacement_weights(g, weighting)
    return _fit_fractions(g["sq_disp_um2"].to_numpy(dtype=float), np.asarray(shared_D, dtype=float), tau,
                          sample_weights=sw, step_noise=noise)


def _linking_record(project, condition, spec, tau_lag_s) -> "dict | None":
    """``{"radius_px", "pixel_size_nm", "tau_s"}`` of ``condition``'s linking, for `bubble_view.radius_notes`; None at
    a lag above 1 (the radius cuts one-frame steps, not the displacement fitted) or where no radius was recorded."""
    if getattr(spec, "sqd_lag", 1) != 1:
        return None
    params = getattr(project, "params", None)
    try:
        radius = params.search_radius_px_for(condition)
        pixel = float(params.pixel_size_nm)
    except Exception:                                  # noqa: BLE001 - no radius, no note
        return None
    if not radius or not pixel:
        return None
    return {"radius_px": float(radius), "pixel_size_nm": pixel, "tau_s": float(tau_lag_s)}


def condition_replicate_fit(project, condition: str, spec: ComparisonSpec, *,
                            cache=None, global_D_override=None, shared_with=None,
                            global_zero_jumps="auto") -> dict:
    """One condition's per-replicate SIMULTANEOUS SQD fit, the bubble-inspector
    payload. Honours the D-sharing SCOPE (Item 2), which is stamped into the
    returned payload so a reader can never mistake which fit produced it:

    - ``"condition"`` (default), **shared D** = the condition's pooled
      track-weighted fit; each replicate's fractions are the EM with that D held
      fixed (`_fit_fractions`), so they are conditionally independent given D̂.
    - ``"global"``: the shared D is the JOINT fit's D (shared across every
      condition); per-replicate fractions again with that D held.
    - ``"group"``: the joint fit over the fittable conditions of ``condition``'s group (``shared_with``, when the
      bubble builder has found them, with their D as ``global_D_override``); alone in its group, its own D. The
      payload carries ``group`` and, when shared, ``own_D``: this condition fitted alone at the group's count,
      which `bubble_view.group_warnings` holds against the shared D.
    - ``"replicate"``: each replicate is fit INDEPENDENTLY at the condition's
      state count, so it carries its OWN ``D`` and fractions. This is the only
      scope in which a per-replicate D exists; the payload then has NO single
      ``shared_D`` (it is ``None``) and each replicate dict carries ``D``.

    With one imaging day there is one replicate (the pooled fit). NEUTRAL:
    populations by index + numeric D.

    THE SCOPE IS THE CONDITION'S OWN (`ComparisonSpec.scope_for`, 1 Nov 2026): a Run may fit one condition Within
    Condition and another Not shared, and each is made here, by the branch of its scope -- the payload says which.
    """

    from .observables import squared_displacements
    from .sqd import fit_sqd, zero_rule_of

    scope = spec.scope_for(condition)
    weighting = spec.fraction_weighting              # "track" (default) | "step"
    part = project.condition(condition)
    obs = _sqd_observable(part, spec)
    tau = float(obs.attrs["lag_time_s"])
    threshold = spec.threshold_um2_s
    # RANDOM CHUNKS IN PLACE OF THE REPLICATES (`PSEUDO_CHUNKS`): the same deal relabels the table and its
    # displacements, so every step below -- per-replicate fractions, per-cell rows -- runs on the chunks.
    k_pseudo = int(getattr(spec, "pseudo_replicates", 0) or 0)
    pseudo = None
    # WITHHELD WHERE THE CONDITION HAS REPLICATES (`chunks_withheld`): no deal, the real replicates kept, the reason
    # carried for the bubble (drawn pooled), its caption and its bracket
    chunks_held = chunks_withheld(part["day"].nunique()) if k_pseudo else None
    if chunks_held:
        k_pseudo = 0
    if k_pseudo:
        mapping, by = pseudo_replicate_deal(part, k_pseudo)
        # the chunks this condition fills (`pseudo_chunk_count`): the count set at most
        k_pseudo = int(max(mapping.values()))
        pseudo = {"k": k_pseudo, "by": by, "seed": PSEUDO_SEED, "real_replicates": int(part["day"].nunique()),
                  "splits": []}
        part = part.copy()
        part["day"] = pseudo_replicate_labels(part, mapping, by)
        attrs = dict(obs.attrs)
        obs = obs.copy()
        obs.attrs = attrs
        obs["day"] = pseudo_replicate_labels(obs, mapping, by)
        part = part[part["day"] > 0]
        obs = obs[obs["day"] > 0]

    # The condition's state count anchors ALL replicates (so populations align even
    # under replicate scope: the count is chosen ONCE per condition, never per
    # replicate). The pooled fit also supplies the shared D for condition scope.
    # NOT MADE UNDER WHOLE DATA SET (4 Nov 2026): the joint fit gives the count and the D and the pooled estimate is
    # fitted at them, so this condition's own fit was never read -- yet made, Best's BIC selection and all, about
    # 150 s a condition on a lab project's (four of them, one after another, in the Run's one job).
    pooled_fit, n_pop = (None, None) if scope == "global" else _cached_fit(project, condition, spec, cache)
    #: D is a JOINT fit's (shared with other conditions), so the pooled estimate below is fitted with it held
    joint = scope == "global"
    group = None
    own_D = None
    joint_zeros = None
    if scope == "global":
        # the joint D is shared across ALL conditions; the bubble builder computes
        # it ONCE and injects it here so we don't refit the joint per condition.
        if global_D_override is not None:
            shared_D, n_pop = list(global_D_override), len(global_D_override)
        else:
            shared_D, n_pop = _cached_global_D(project, spec, cache)
        # the joint fit's exact-zero decision (the builder decides it once; "auto": made here)
        joint_zeros = _global_zero_jumps(project, spec) if global_zero_jumps == "auto" else global_zero_jumps
    elif scope == "group":
        members = list(shared_with) if shared_with is not None else _fittable(project, spec.group_members(condition))
        if global_D_override is not None:
            shared_D, n_pop = list(global_D_override), len(global_D_override)
        elif len(members) >= 2:
            shared_D, n_pop = _cached_global_D(project, _group_joint_spec(spec, members), cache)
        else:
            shared_D = None
        joint = shared_D is not None
        if joint:
            # THIS CONDITION ALONE, AT THE GROUP'S COUNT: what sharing would have to be moving into the fractions
            from dataclasses import replace

            own = (pooled_fit if len(pooled_fit.D) == n_pop
                   else _cached_fit(project, condition, replace(spec, lock_model=str(n_pop)), cache)[0])
            own_D = [float(d) for d in own.D]
        else:
            shared_D = [float(d) for d in pooled_fit.D]
        group = {"name": spec.group_of(condition), "factor": spec.d_sharing_factor,
                 "with": [str(c) for c in members] if joint else [str(condition)]}
    elif scope == "condition":
        shared_D = [float(d) for d in pooled_fit.D]
    else:                                            # replicate: no shared D
        shared_D = None

    n_days = int(part["day"].nunique())
    keys = sorted(part["day"].unique()) if n_days >= 2 else ["all"]
    n_starts = (spec.fast.sqd_n_starts if spec.is_fast
                else interactive_starts(n_pop, spec.sqd_method(condition)))
    replicates = []
    for key in keys:
        g = obs if key == "all" else obs[obs["day"] == key]
        cell_frame = part if key == "all" else part[part["day"] == key]
        rep = {"replicate": ("all" if key == "all" else int(key)),
               "n_cells": int(cell_frame["cell"].nunique()),
               "n_movies": int(cell_frame["movie"].nunique()) if "movie" in cell_frame.columns else None,
               "n_tracks": int(cell_frame["uid"].nunique())}
        if scope == "replicate":
            # fit this replicate on its own: its OWN D and fractions, at the
            # condition's fixed n_pop so populations stay index-aligned. CACHED like the
            # pooled fit (`_cached_replicate_fit`): with Per Day the Compute dialog's default
            # (17 Oct 2026), every Run made these again, on every condition and day.
            # with the pooled fit's rule for exact-zero jumps (`sqd.zero_rule_of`): a replicate holds the same
            # populations, the immobile one included where the condition has it
            rfit = _cached_replicate_fit(project, condition, key, cell_frame, g, spec, cache,
                                         n_pop=n_pop, weighting=weighting, n_starts=n_starts,
                                         zero_rule=zero_rule_of(pooled_fit))
            rep["D"] = [float(d) for d in rfit.D]
            rep["fractions"] = [float(f) for f in rfit.fractions]
            rep["slow_fraction"] = slow_fraction_below(rep["D"], rep["fractions"],
                                                        threshold)
        else:
            # Per-replicate fractions with the shared D held, at the chosen estimand:
            # "track" (default) = the MOLECULE fraction (each track once, the
            # pre-registered endpoint); "step" = the displacement/time fraction (each
            # displacement once, the legacy consistency check).
            fractions = [float(f) for f in _held_fractions(g, shared_D, tau, weighting, spec, condition)]
            rep["fractions"] = fractions
            rep["slow_fraction"] = float(
                sum(f for d, f in zip(shared_D, fractions) if d < threshold))
        replicates.append(rep)

    # EACH CELL'S FRACTIONS, with the same shared D and the same EM (14 Oct 2026): what the
    # nested mixed model tests (`mixedmodel`, cells within movies within days). A cell
    # with fewer than `MIN_CELL_STEPS` steps is left out and counted: its fraction would be
    # mostly noise. None under the replicate scope, which has no D shared by the cells; a
    # project without outlines has no cells (NA), so the list is empty.
    cells, cells_left_out = None, 0
    if shared_D is not None and n_days >= 2:
        cells = []
        for (day, movie, cell), g in obs.groupby(["day", "movie", "cell"], sort=True,
                                                 observed=True):
            if len(g) < MIN_CELL_STEPS:
                cells_left_out += 1
                continue
            fr = _held_fractions(g, shared_D, tau, weighting, spec, condition)
            cells.append({"cell": str(cell), "movie": str(movie), "day": int(day),
                          "n_steps": int(len(g)),
                          "fractions": [float(f) for f in fr],
                          # the share in states below the D threshold: comparable between
                          # fits of different state counts (`slow_fraction_below`)
                          "below": slow_fraction_below(shared_D, fr, threshold)})

    # EVERY OTHER DEAL OF THE CHUNKS, so a reader sees how far a p moves with the deal: each deal's per-chunk
    # fractions with the same shared D (cheap: the D is held). The first deal is the one drawn.
    if pseudo is not None and shared_D is not None:
        whole = project.condition(condition)
        for seed in range(PSEUDO_SEED + 1, PSEUDO_SEED + PSEUDO_SPLITS):
            mapping, by = pseudo_replicate_deal(whole, k_pseudo, seed=seed)
            labels = pseudo_replicate_labels(obs, mapping, by)
            split = []
            for chunk in range(1, k_pseudo + 1):
                g = obs[labels == chunk]
                split.append([float(f) for f in _held_fractions(g, shared_D, tau, weighting, spec, condition)])
            pseudo["splits"].append(split)

    # THE POOLED ESTIMATE, every replicate day as one set -- what the Uncertainty
    # setting "None" draws. It was computed all along (the shared D comes from it) and
    # then thrown away. Under a condition's own D it is the plain pooled fit itself,
    # the same fit the frozen MATLAB reference is checked against; under the joint D
    # (global scope) there is no free-D pooled fit that would share it, so the
    # fractions are fit on all of the condition's data with the joint D held.
    if joint:
        pooled_D = list(shared_D)
        pooled_fractions = [float(f) for f in _held_fractions(obs, pooled_D, tau, weighting, spec, condition)]
    else:
        pooled_D = [float(d) for d in pooled_fit.D]
        pooled_fractions = [float(f) for f in pooled_fit.fractions]
    # the SAMPLE the bubbles are drawn from, read by the bubble chart's "n" line (`bubble_view.sample_text`): the
    # cells, movies and tracks fitted (30 Oct 2026 the movies, for "8 movies" under Uncertainty Movies)
    pooled = {"D": pooled_D, "fractions": pooled_fractions,
              "n_cells": int(part["cell"].nunique()),
              "n_movies": int(part["movie"].nunique()) if "movie" in part.columns else None,
              "n_tracks": int(part["uid"].nunique())}

    # The inspector renders a per-replicate BUBBLE plot (x = replicate, y = D,
    # area ∝ fraction), so the payload carries only what that needs, the fit and
    # its fractions. No ECDF/CDF fields: the CDF fit-quality view was retired.
    return {"condition": condition, "n_pop": n_pop, "shared_D": shared_D,
            "scope": scope, "threshold_um2_s": threshold, "weighting": weighting,
            "unit": ("chunk" if pseudo is not None else "day") if n_days >= 2 else "cell",
            "replicates": replicates, "pooled": pooled,
            # what the fit's checks need to say how far the linking radius cuts the fast steps (10 Nov 2026,
            # `bubble_view.radius_notes`); None where the radius or the lag does not allow it
            "linking": _linking_record(project, condition, spec, tau),
            "cells": cells, "cells_left_out": cells_left_out,
            **({"pseudo": pseudo} if pseudo is not None else {}),
            **({"chunks_withheld": chunks_held} if chunks_held else {}),
            **({"group": group} if group is not None else {}),
            **({"own_D": own_D} if own_D is not None else {}),
            # exact-zero jumps met by the condition's fit, and what was done (`sqd.zero_jumps`; its note is shown)
            **({"zero_jumps": pooled_fit.zero_jumps.to_dict()}
               if getattr(pooled_fit, "zero_jumps", None) is not None else {}),
            # Whole data set: the one decision of the joint fit, the same in every condition it joins
            **({"zero_jumps": joint_zeros} if joint_zeros is not None else {})}


def _d_correction_stamp(project, spec: ComparisonSpec) -> dict:
    """The D-axis Berglund-correction directive carried into the bubble view (item C).

    Default OFF. When on it records everything a reader or exporter needs (σ, its
    source, τ, R, the floor) so the correction is reproducible and the caption /
    provenance can fence it. The correction MATH lives once in
    ``msd.berglund_correct_apparent_D``; this only assembles its inputs from the
    acquisition params, so a σ change (or a manual σ) flows straight through.
    """
    from .msd import BERGLUND_R_CONTINUOUS

    params = project.params
    return {
        "on": bool(spec.d_correction),
        "sigma_nm": float(params.localization_sigma_nm),
        "sigma_source": params.localization_sigma_source,
        "tau_s": float(params.tau_s),
        "R": float(BERGLUND_R_CONTINUOUS),
        "floor_D": float(params.localization_floor_D),
    }


def sqd_bubbles(project, spec: ComparisonSpec, *, cache=None,
                with_ci: bool = False, ci_unit: str = "cell") -> dict:
    """Assemble the SQD bubble comparison: one column of populations per condition.

    Each population is a bubble at (condition, D) with area ∝ its fraction and
    colour = state index. Conditions fit with different counts simply show
    different bubble counts: that difference is informative, not an error.

    The D-sharing SCOPE (``spec.d_sharing_scope``, Item 6a) selects the fit:
    ``"global"`` (DEFAULT) shares D across conditions (the joint fit) so states
    correspond; ``"condition"`` and ``"replicate"`` fit each condition independently,
    each at its OWN scope where the spec names one (`ComparisonSpec.scope_for`): the
    column's ``scope`` is the one its payload was fitted at.

    ``ci_unit`` is the unit ``with_ci`` resamples on a single session: ``"cell"``, or
    ``"movie"`` (19 Oct 2026; cells in one movie share its focus and field).
    """
    if spec.d_sharing_scope == "global" and len(spec.conditions) >= 2:
        return _sqd_bubbles_joint(project, spec, cache=cache)
    if spec.d_sharing_scope == "group":
        return _sqd_bubbles_grouped(project, spec, cache=cache)
    columns = []
    for condition in spec.conditions:
        _check_cancel(spec)
        n_tracks = _condition_track_count(project, condition)
        if n_tracks < MIN_TRACKS_TO_FIT:
            column = BubbleColumn(condition=condition, n_pop=0, D=[], fractions=[],
                                  slow_fraction=float("nan"))
            column.error = f"n = {n_tracks}: too few tracks to fit"  # type: ignore
            columns.append(column)
            continue
        try:
            # ONE source of truth: the bubble is the per-replicate simultaneous fit
            # aggregated (Item 3/4), the exact payload the inspector renders.
            payload = condition_replicate_fit(project, condition, spec, cache=cache)
        except Exception as exc:                          # noqa: BLE001 - reported
            column = BubbleColumn(condition=condition, n_pop=0, D=[], fractions=[],
                                  slow_fraction=float("nan"))
            column.error = f"n = {n_tracks}: fit failed ({exc})"  # type: ignore
            columns.append(column)
            continue
        column = _column_from_payload(payload, spec.threshold_um2_s)
        # HARD GATE (Report A, decided): the cell bootstrap CI resamples CELLS, so it
        # is the wrong unit on a day-level figure (cells share a day → anti-
        # conservative). It is STRUCTURALLY unavailable when the condition has ≥2
        # replicate days: the whisker cannot be attached, so a cropped/pasted figure
        # can never carry it. It is computed ONLY for single-session data, where the
        # cell IS the fallback unit.
        if with_ci and column.n_replicates < 2:
            # The FRACTION interval is now kept too: under the Uncertainty setting
            # "Cells" it is drawn around the POOLED point, which is the fit the
            # bootstrap resamples -- never around a per-day mean. Same gate as
            # before: a single session only, where the cell IS the fallback unit.
            column.D_ci, column.fraction_ci = _bubble_ci(
                project, condition, spec, column.n_pop, cache, unit=ci_unit)
        columns.append(column)
    # flag when the CI was asked for but withheld because the figure is day-level,
    # so the view can say why no whisker appears (never silently drop it).
    ci_gated = bool(with_ci) and any(getattr(c, "n_replicates", 1) >= 2
                                     for c in columns)
    return {"columns": columns, "threshold_um2_s": spec.threshold_um2_s,
            "d_sharing_scope": spec.d_sharing_scope,
            "fraction_weighting": spec.fraction_weighting,
            "ci_gated_multiday": ci_gated,
            "d_correction": _d_correction_stamp(project, spec),
            "conditions_simultaneous": spec.conditions_simultaneous}


def _column_from_payload(payload: dict, threshold: float) -> "BubbleColumn":
    """Aggregate a ``condition_replicate_fit`` payload into one bubble column,
    the EXPLICIT bubble aggregation (Item 3), so the bubble, the inspector and the
    data table all read the same numbers.

    D = the shared D (condition/global) or the mean per-replicate D with its SD
    (replicate). fraction = the MEAN of per-replicate fractions; fraction_sd =
    their SD across replicates (the spread conditional on the shared D̂)."""
    n_pop = int(payload["n_pop"])
    scope = payload.get("scope", "condition")
    reps = payload["replicates"]
    per_rep_fr = np.asarray([r["fractions"] for r in reps], dtype=float)  # [rep,pop]
    n_rep = len(reps)
    fraction_mean = per_rep_fr.mean(axis=0)
    fraction_sd = (per_rep_fr.std(axis=0, ddof=1) if n_rep >= 2
                   else np.zeros(n_pop))
    if scope == "replicate":
        per_rep_D = np.asarray([r["D"] for r in reps], dtype=float)
        D = per_rep_D.mean(axis=0)
        D_sd = (per_rep_D.std(axis=0, ddof=1) if n_rep >= 2
                else np.zeros(n_pop)).tolist()
    else:
        D = np.asarray(payload["shared_D"], dtype=float)
        D_sd = None                                       # shared → no ± by construction
    fm = fraction_mean.tolist()
    Dl = D.tolist()
    return BubbleColumn(
        condition=payload["condition"], n_pop=n_pop, D=Dl, fractions=fm,
        slow_fraction=slow_fraction_below(Dl, fm, threshold),
        fraction_mean=fm, fraction_sd=fraction_sd.tolist(), n_replicates=n_rep,
        per_replicate_fractions=per_rep_fr.tolist(), D_sd=D_sd, scope=scope,
        pooled_D=(payload.get("pooled") or {}).get("D"),
        pooled_fractions=(payload.get("pooled") or {}).get("fractions"),
        per_cell=payload.get("cells"), cells_left_out=int(payload.get("cells_left_out") or 0))


def _sqd_bubbles_joint(project, spec, *, cache=None) -> dict:
    """SQD bubbles with D shared across conditions (global scope, the DEFAULT). One
    shared count is required (the lock, else BIC on the pooled data), since a shared
    D needs one model."""
    from dataclasses import replace

    # Conditions with too few tracks state the count, they are never fit to noise,
    # the same guard the condition-scope path applies. The joint fit shares D across
    # only the FITTABLE conditions; a thin one gets an error column, in place.
    def _thin_column(condition, n_tracks):
        col = BubbleColumn(condition=condition, n_pop=0, D=[], fractions=[],
                           slow_fraction=float("nan"))
        col.error = f"n = {n_tracks}: too few tracks to fit"   # type: ignore
        return col

    fittable = [c for c in spec.conditions
                if _condition_track_count(project, c) >= MIN_TRACKS_TO_FIT]
    # Fewer than two fittable conditions → nothing to share a D with; fall back to
    # the per-condition path (each fittable condition fit on its own).
    if len(fittable) < 2:
        return sqd_bubbles(project, replace(spec, d_sharing_scope="condition"),
                           cache=cache)

    # The joint D is shared across the FITTABLE conditions; compute it ONCE (cached).
    joint_spec = replace(spec, conditions=fittable)
    shared_D, _ = _cached_global_D(project, joint_spec, cache)
    joint_zeros = _global_zero_jumps(project, joint_spec)       # decided once for every column
    columns = []
    for condition in spec.conditions:
        _check_cancel(spec)
        n_tracks = _condition_track_count(project, condition)
        if n_tracks < MIN_TRACKS_TO_FIT:
            columns.append(_thin_column(condition, n_tracks))
            continue
        try:
            payload = condition_replicate_fit(project, condition, spec, cache=cache,
                                              global_D_override=shared_D, global_zero_jumps=joint_zeros)
        except Exception as exc:                          # noqa: BLE001 - reported
            col = BubbleColumn(condition=condition, n_pop=0, D=[], fractions=[],
                               slow_fraction=float("nan"))
            col.error = f"no joint fit ({exc})"                 # type: ignore
            columns.append(col)
            continue
        columns.append(_column_from_payload(payload, spec.threshold_um2_s))
    return {"columns": columns, "threshold_um2_s": spec.threshold_um2_s,
            "d_sharing_scope": "global",
            "fraction_weighting": spec.fraction_weighting,
            "ci_gated_multiday": False,
            "d_correction": _d_correction_stamp(project, spec),
            "conditions_simultaneous": True}


def _sqd_bubbles_grouped(project, spec, *, cache=None) -> dict:
    """SQD bubbles with D shared WITHIN each group (scope "group"): per group, the joint fit over its fittable
    conditions (`_cached_global_D` on that group alone, at its own count), then each condition's fractions with
    that D held. A group with one fittable condition is that condition's own fit. Columns keep the Run's order; a
    group whose joint fit fails names the failure on each of its columns, the other groups are drawn."""
    def _error_column(condition, why):
        col = BubbleColumn(condition=condition, n_pop=0, D=[], fractions=[], slow_fraction=float("nan"))
        col.error = why                                         # type: ignore
        return col

    shared: dict = {}
    for condition in spec.conditions:
        name = spec.group_of(condition)
        if name in shared:
            continue
        _check_cancel(spec)
        members = _fittable(project, spec.group_members(condition))
        if len(members) < 2:
            shared[name] = (None, members, None)
            continue
        try:
            found = _cached_global_D(project, _group_joint_spec(spec, members), cache)
            shared[name] = (found, members, None)
        except Exception as exc:                          # noqa: BLE001 - reported on the group's columns
            shared[name] = (None, members, f"no joint fit within {name} ({exc})")
    columns = []
    for condition in spec.conditions:
        _check_cancel(spec)
        n_tracks = _condition_track_count(project, condition)
        if n_tracks < MIN_TRACKS_TO_FIT:
            columns.append(_error_column(condition, f"n = {n_tracks}: too few tracks to fit"))
            continue
        found, members, failed = shared[spec.group_of(condition)]
        if failed:
            columns.append(_error_column(condition, failed))
            continue
        try:
            payload = condition_replicate_fit(
                project, condition, spec, cache=cache, shared_with=members,
                global_D_override=None if found is None else found[0])
        except Exception as exc:                          # noqa: BLE001 - reported
            columns.append(_error_column(condition, f"n = {n_tracks}: fit failed ({exc})"))
            continue
        columns.append(_column_from_payload(payload, spec.threshold_um2_s))
    return {"columns": columns, "threshold_um2_s": spec.threshold_um2_s,
            "d_sharing_scope": "group",
            "d_sharing_factor": spec.d_sharing_factor,
            "d_sharing_groups": {str(c): spec.group_of(c) for c in spec.conditions},
            "fraction_weighting": spec.fraction_weighting,
            "ci_gated_multiday": False,
            "d_correction": _d_correction_stamp(project, spec),
            "conditions_simultaneous": False}


def _bubble_ci(project, condition, spec, n_pop, cache, *, unit: str = "cell"):
    """Per-population ``(D_ci, fraction_ci)`` percentile intervals from the CELL (or,
    ``unit="movie"``, the MOVIE) bootstrap: opt-in.

    The FRACTION interval is returned as well now, for exactly one use: the
    Uncertainty setting "Cells", which draws it around the POOLED point -- the fit
    the bootstrap resamples. It is still never attached to a per-day mean (Report A):
    that bubble's uncertainty is the day-level ± SD, and a cell-level interval there
    would double up at the wrong unit. The caller only asks on single-session data.
    The CI does not depend on the mobility threshold, so it is cached on a
    threshold-independent key: a threshold change reuses it.

    NOTE (unit): the resampled unit is the CELL. On a ≥2-day figure that is
    anti-conservative for a day-level claim; the whisker is labelled cell-level and
    is really the single-session fallback interval. See docs/uncertainty_tier_audit.md.
    """
    boot = _cached_bubble_boot(project, condition, spec, n_pop, cache, unit=unit)
    D_ci, fraction_ci = [], []
    for k in range(1, n_pop + 1):
        dcol, fcol = f"D_{k}", f"fraction_{k}"
        D_ci.append(tuple(boot[dcol].quantile([0.025, 0.975]).tolist())
                    if dcol in boot else None)
        fraction_ci.append(tuple(boot[fcol].quantile([0.025, 0.975]).tolist())
                           if fcol in boot else None)
    return D_ci, fraction_ci


def appd_points(project, spec: ComparisonSpec) -> dict:
    """Per-track and per-cell apparent-D for a ridgeline/violin + SuperPlot swarm.

    One entry per condition: ``track`` values (log10 appD, one per track) for the
    violin, and ``cells`` (per-cell mean log10 appD, coloured by replicate day)
    for the SuperPlot overlay. Neutral: values are numeric log10 D, never a label.
    """
    from .qc import track_summary

    per_condition = {}
    for condition in spec.conditions:
        _check_cancel(spec)
        summary = track_summary(project.condition(condition))
        keep = summary[["uid", "cell", "day", "appd_um2_per_s"]].dropna()
        keep = keep[keep["appd_um2_per_s"] > 0]
        track_log = np.log10(keep["appd_um2_per_s"].to_numpy())
        cells = (
            keep.groupby("cell", observed=True)
            .agg(day=("day", "first"),
                 log_appd=("appd_um2_per_s", lambda s: float(np.log10(s.mean()))))
            .reset_index()
        )
        per_condition[condition] = {
            "track_log_appd": track_log,
            "cells": cells,
            "n_tracks": int(len(keep)),
            "n_cells": int(cells.shape[0]),
        }
    return {"per_condition": per_condition}


@dataclass
class KineticsRow:
    """One condition's HMM kinetics: occupancy + on/off rates with bounds."""

    condition: str
    occupancy_slow: float
    on_rate: float                     # fast→slow
    off_rate: float                    # slow→fast
    #: percentile CI of the occupancy over the SAME unit resamples as the rates.
    #: The bootstrap already fits occupancy per resample, so quoting the rates with
    #: intervals and the occupancy without one was a reporting gap, not a cost:
    #: occupancy is the quantity that SURVIVES when the rates do not (it needs no
    #: observed transition), so it is the one that most needs its interval shown.
    occupancy_ci: tuple | None = None
    #: True when an occupancy arm SITS ON the [0, 1] range limit. A clipped arm and a
    #: short arm mean different things, the first says the interval ran into the
    #: parameter's edge and the estimate is not resolved on that side, the second says
    #: the estimate is precise. They must never render identically. (Named "limit",
    #: not "bound": the neutral-labels gate forbids that word in figure text, and a
    #: field name is what ends up pasted into a caption.)
    occupancy_ci_at_limit: bool = False
    on_ci: tuple | None = None
    off_ci: tuple | None = None
    on_identifiable: bool = True
    off_identifiable: bool = True
    error: str | None = None           # stated reason when a condition can't be fit


def hmm_kinetics(project, spec: ComparisonSpec, *, cache=None) -> dict:
    """Assemble HMM occupancy + switching rates per condition, with CIs.

    A rate that the data cannot pin (a condition that rarely switches: a release
    rate is the standing example) is marked ``*_identifiable = False`` and
    its CI reaches down to ~0: the plot must render it as an open bound, never a
    point estimate. Identifiability is read from the bootstrap: if a large share
    of resamples are degenerate (no observed transition), or the CI floor sits at
    zero, the rate is not identifiable.
    """
    from .hmm import HMMError, bootstrap_hmm, fit_hmm, step_sequences

    tau = project.params.tau_s
    rows = []
    unit = None                       # the bootstrap's resampling unit, for the caption
    for condition in spec.conditions:
        _check_cancel(spec)
        part = project.condition(condition)
        n_tracks = _condition_track_count(project, condition)
        if n_tracks < MIN_TRACKS_TO_FIT:
            rows.append(KineticsRow(
                condition=condition, occupancy_slow=float("nan"),
                on_rate=float("nan"), off_rate=float("nan"),
                on_identifiable=False, off_identifiable=False,
                error=f"n = {n_tracks}: too few tracks to fit"))
            continue
        try:
            fit = fit_hmm(step_sequences(part), n_states=2, tau_s=tau,
                          **_hmm_kwargs(spec, default_starts=8))
        except HMMError as exc:
            # A condition the HMM cannot fit (too few switching steps) is marked
            # with its reason rather than sinking the whole comparison.
            rows.append(KineticsRow(
                condition=condition, occupancy_slow=float("nan"),
                on_rate=float("nan"), off_rate=float("nan"),
                on_identifiable=False, off_identifiable=False,
                error=f"n = {n_tracks}: HMM did not converge ({exc})"))
            continue
        rates = fit.rates_per_s
        on, off = float(rates[1, 0]), float(rates[0, 1])
        row = KineticsRow(
            condition=condition, occupancy_slow=float(fit.occupancy[0]),
            on_rate=on, off_rate=off,
        )
        try:
            boot = _cached_hmm_boot(project, condition, spec, cache)
            unit = boot.attrs.get("unit", unit)
            occ_lo, occ_hi = boot["occupancy_1"].quantile([0.025, 0.975]).tolist()
            # Occupancy is bounded on [0, 1]. A PERCENTILE interval cannot leave the
            # range (every resample's occupancy is already inside it), so clipping is
            # a no-op here, but an arm that LANDS ON a bound is flagged rather than
            # drawn as an ordinary short arm, and any future non-percentile interval
            # is clipped here rather than silently escaping the parameter space.
            row.occupancy_ci = (min(max(occ_lo, 0.0), 1.0), min(max(occ_hi, 0.0), 1.0))
            row.occupancy_ci_at_limit = bool(
                occ_lo <= 0.0 or occ_hi >= 1.0)
            row.on_ci = tuple(boot["rate_2_to_1"].quantile([0.025, 0.975]).tolist())
            row.off_ci = tuple(boot["rate_1_to_2"].quantile([0.025, 0.975]).tolist())
            degenerate_share = boot.attrs["n_degenerate"] / max(boot.attrs["n_ok"], 1)
            # A rate is unidentifiable if the resamples frequently see no
            # transition, or the lower CI is pinned at zero.
            row.on_identifiable = not (degenerate_share > 0.1 or row.on_ci[0] <= 1e-6)
            row.off_identifiable = not (degenerate_share > 0.1 or row.off_ci[0] <= 1e-6)
        except HMMError:
            row.on_identifiable = row.off_identifiable = False
        rows.append(row)
    return {"rows": rows, "unit": unit or "unit"}


def fraction_superplot(project, spec: ComparisonSpec, *, cache=None,
                       uncertainty="analytic") -> dict:
    """Per-cell slow fraction (threshold basis) per condition, + CI + permutation p.

    Each condition contributes per-cell slow fractions (the fraction of each
    cell's tracks whose per-track appD is below the threshold, a per-cell version
    of the comparable basis), coloured by replicate day. The CI tier
    (``uncertainty``) is per the tier policy: ``"analytic"`` (default) is the
    fast per-cell mean ± SD/√n; ``"bootstrap"`` is the cell resample; ``"none"``
    draws no interval. Each condition also carries ``escalation`` reasons (near
    the 0/1 boundary, too few cells) so the UI can flag where the analytic
    interval misleads. The between-condition test is always a CELL-LEVEL
    permutation p, never a pooled-step test.
    """
    from .qc import track_summary
    from .uncertainty import analytic_summary, permutation_p

    threshold = spec.threshold_um2_s
    per_condition = {}
    unit_values = {}                         # condition -> the per-UNIT values used
    for condition in spec.conditions:
        _check_cancel(spec)
        summary = track_summary(project.condition(condition))
        keep = summary[["cell", "day", "appd_um2_per_s"]].dropna()
        per_cell = (
            keep.assign(is_slow=(keep["appd_um2_per_s"] < threshold).astype(float))
            .groupby("cell", observed=True)
            .agg(day=("day", "first"), slow_fraction=("is_slow", "mean"),
                 n=("is_slow", "size"))
            .reset_index()
        )
        # THE UNIT: the replicate DAY is the honest independent unit and is the
        # DEFAULT the moment there are ≥2 real days, cells within a day share the
        # culture, so a cell-level interval is anti-conservative for a biological
        # claim (on a lab project the cell spread hid a larger between-day SD). With one
        # day there is no day structure, so it falls back to the CELL.
        per_day = per_cell.groupby("day", observed=True)["slow_fraction"].mean()
        n_days = int(per_day.shape[0])
        unit = "day" if n_days >= 2 else "cell"
        u_values = (per_day.to_numpy() if unit == "day"
                    else per_cell["slow_fraction"].to_numpy())
        unit_values[condition] = u_values

        summary_stats = analytic_summary(u_values, is_fraction=True)
        if uncertainty == "bootstrap":
            # the cell bootstrap stays available (opt-in) in BOTH cases, a deeper
            # look at cell-to-cell spread, whatever the default unit is
            ci = _cell_fraction_bootstrap(per_cell["slow_fraction"].to_numpy(),
                                          n_boot=spec.effective_n_boot())
        elif uncertainty == "none":
            ci = None
        else:                                # analytic (default): mean ± SD/√n_unit
            ci = summary_stats.ci
        per_condition[condition] = {
            "per_cell": per_cell,
            "mean": float(summary_stats.mean),
            "sd": float(summary_stats.sd),
            "ci": ci,
            "n_cells": int(len(per_cell)),
            "n_days": n_days,
            "unit": unit,
            "day_means": {int(d): float(v) for d, v in per_day.items()},
            "escalation": list(summary_stats.escalation),
            "tier": uncertainty,
        }

    # the between-condition permutation test is over the UNIT values (days when
    # there are ≥2), never pooled steps and never the anti-conservative cells when
    # a day structure exists
    pvalues = {}
    conds = spec.conditions
    for i in range(len(conds)):
        for j in range(i + 1, len(conds)):
            pvalues[(conds[i], conds[j])] = permutation_p(
                unit_values[conds[i]], unit_values[conds[j]])

    overall_unit = ("day" if all(per_condition[c]["unit"] == "day" for c in conds)
                    else "cell")
    return {"per_condition": per_condition, "pvalues": pvalues,
            "threshold_um2_s": threshold, "tier": uncertainty, "unit": overall_unit}


#: Methods that yield a comparable "fraction below the threshold" per condition,
#: for the heatmap. Each maps a condition to that fraction in [0, 1]. Confinement
#: is excluded: its excess-over-null is a different quantity, not a fraction.
HEATMAP_METHODS = ("SQD", "appD", "HMM", "saSPT")


def fraction_matrix(project, spec: ComparisonSpec, *, methods=None, cache=None) -> dict:
    """A conditions × methods matrix of the fraction below the mobility threshold.

    The overview a user needs across many conditions: every method's estimate of
    the below-threshold fraction, one cell each, so a condition's shift shows as a
    row that lights up across methods. Values are in [0, 1]; a method that fails
    for a condition is ``nan`` (drawn blank), never a fake 0. NEUTRAL: the colour
    encodes the fraction and the axis says ``D < X``, no interpretive word.
    """
    methods = list(methods) if methods is not None else list(HEATMAP_METHODS)
    conditions = list(spec.conditions)
    fns = {"SQD": _hm_sqd, "appD": _hm_appd, "HMM": _hm_hmm, "saSPT": _hm_saspt}
    for m in methods:
        if m not in fns:
            raise ComparisonError(
                f"method {m!r} has no below-threshold fraction; "
                f"choose from {HEATMAP_METHODS}")

    values = np.full((len(conditions), len(methods)), np.nan)
    for i, condition in enumerate(conditions):
        _check_cancel(spec)
        for j, method in enumerate(methods):
            try:
                values[i, j] = fns[method](project, condition, spec, cache)
            except Exception:                             # noqa: BLE001 - blank cell
                values[i, j] = np.nan
    return {"conditions": conditions, "methods": methods, "values": values,
            "threshold_um2_s": spec.threshold_um2_s}


def _hm_sqd(project, condition, spec, cache):
    fit, _ = _cached_fit(project, condition, spec, cache)
    return slow_fraction_below(fit.D, fit.fractions, spec.threshold_um2_s)


def _hm_appd(project, condition, spec, cache):
    from .qc import track_summary

    summary = track_summary(project.condition(condition))
    keep = summary["appd_um2_per_s"].dropna()
    return float((keep < spec.threshold_um2_s).mean())


def _hm_hmm(project, condition, spec, cache):
    from .hmm import fit_hmm, step_sequences

    fit = fit_hmm(step_sequences(project.condition(condition)), n_states=2,
                  tau_s=project.params.tau_s, **_hmm_kwargs(spec, default_starts=8))
    occ = np.asarray(fit.occupancy, dtype=float)
    D = np.asarray(fit.D, dtype=float)
    return float(occ[D < spec.threshold_um2_s].sum())


def _hm_saspt(project, condition, spec, cache):
    return float(_cached_saspt(project, condition, spec, cache).occupancy_below(
        spec.threshold_um2_s))


def _cached_saspt(project, condition, spec, cache):
    """The saSPT posterior for one condition, memoized THRESHOLD-INDEPENDENTLY so
    the heatmap, the forest and the spectrum view all share a single run."""
    from .saspt_adapter import run_saspt

    cache = _for_condition(cache, condition)
    part = project.condition(condition)
    compute = lambda: run_saspt(part)
    if cache is None:
        return compute()
    from .provenance import capture

    provenance = capture(
        settings={"view": "saspt", "condition": condition, **spec.fast_settings()},
        seeds={"saspt": 0}, canonical=part, unit="cell",
    )
    return cache.get_or_compute(f"cmp_saspt_{condition}", provenance, compute)


#: A panel needs at least this many localizations to be mappable; below it the
#: panel is greyed out with an honest "too sparse" note rather than rendering
#: confetti (the spatial analogue of the HMM open-arrow / report-as-a-bound rule).
SPATIAL_N_PANEL_MIN = 150
#: …or if more than this fraction of its bins are masked (no-data speckle).
SPATIAL_MAX_MASKED_FRAC = 0.80


def spatial_maps(project, spec: ComparisonSpec, *, source="appd", mode="enrichment",
                 n_bins=2, sigma_um=0.06, n_min=None, bins=(24, 12),
                 n_panel_min=SPATIAL_N_PANEL_MIN, max_masked_frac=SPATIAL_MAX_MASKED_FRAC,
                 cache=None) -> dict:
    """WHERE the populations sit in the cell: the legacy speed map, ALL conditions.

    A grid (rows = conditions, columns = populations slowest→fastest by D). Fixes
    the sparse-noise blowout, per the render spec + the rules in the FIX:

    - ``mode="enrichment"`` (DEFAULT, for claims): each population's density is
      divided by the matched all-localization null (log scale, symmetric about 1),
      which divides out the projection geometry and most of the spike problem.
      ``mode="density"`` is the raw-density ORIENTATION toggle.
    - Colour scale is shared **within a population across conditions** (so slow-A
      vs slow-B are comparable) but NEVER across populations of different
      abundance: one colour scale per population. Limits are percentile-robust
      (2–98 %), so a single low-count spike cannot set the top.
    - A panel with too few localizations (< ``n_panel_min``) or too much masked
      area (> ``max_masked_frac``) is flagged ``sparse``, the plot greys it out
      with an honest ``n = …`` note instead of bright noise.
    - Coarser bins + a wider σ-kernel by default, so the no-data mask is a
      contiguous region, not speckle.

    Returns plain arrays so it caches/serialises without the render objects.
    NEUTRAL: populations by INDEX + numeric D, the d-axis is a LOCATION.
    """
    from .render import robust_color_limits

    symmetric = 1.0 if mode == "enrichment" else None
    per_condition, grids_by_pop, skipped_by_condition = {}, {}, {}
    for condition in spec.conditions:
        _check_cancel(spec)
        try:
            entries, n_skipped = _condition_grids(
                project, spec, condition, source, mode, n_bins, sigma_um, n_min, bins)
            per_condition[condition] = entries
            skipped_by_condition[condition] = n_skipped
            for pop, (_state, _D, grid) in enumerate(entries, start=1):
                grids_by_pop.setdefault(pop, []).append(grid)
        except Exception as exc:                          # noqa: BLE001 - reported
            per_condition[condition] = {"error": str(exc)}

    n_pops = max((len(v) for v in per_condition.values()
                  if isinstance(v, list)), default=0)
    if n_pops == 0:
        return {"conditions": list(spec.conditions), "source": source, "mode": mode,
                "n_pops": 0, "panels": {c: [] for c in spec.conditions},
                "limits_by_pop": {}, "n_localizations": 0}

    # One robust, per-population colour scale (shared across conditions).
    limits_by_pop = {}
    for pop, grids in grids_by_pop.items():
        try:
            limits_by_pop[pop] = robust_color_limits(grids, symmetric_about=symmetric)
        except Exception:                                 # pragma: no cover
            limits_by_pop[pop] = (0.0, 1.0)

    panels, total = {}, 0
    for condition, entries in per_condition.items():
        if not isinstance(entries, list):
            panels[condition] = [{"error": entries["error"]}]
            continue
        rows = []
        for pop, (state, D, grid) in enumerate(entries, start=1):
            values = np.asarray(grid.values, dtype=float)
            n_loc = int(grid.n_localizations)
            masked_frac = float(np.mean(~np.isfinite(values)))
            sparse = n_loc < n_panel_min or masked_frac > max_masked_frac
            label = (f"pop {state}: D = {D:.3g} µm²/s" if D is not None
                     else f"pop {state}")
            rows.append({"state": state, "pop": pop, "D": D, "label": label,
                         "values": values,
                         "extent": tuple(float(x) for x in grid.extent),
                         "n": n_loc, "masked_frac": masked_frac,
                         "sparse": bool(sparse)})
            total += n_loc
        panels[condition] = rows
    return {"conditions": list(spec.conditions), "source": source, "mode": mode,
            "n_pops": n_pops, "panels": panels,
            "limits_by_pop": {p: (float(lo), float(hi))
                              for p, (lo, hi) in limits_by_pop.items()},
            "n_localizations": total,
            # cells skipped because they carry no Oufti mesh rib (no centerline),
            # per condition: reported so the skip is honest, never silent
            "skipped_no_rib": {c: int(n) for c, n in skipped_by_condition.items()
                               if n}}


def _condition_grids(project, spec, condition, source, mode, n_bins, sigma_um,
                     n_min, bins):
    """The (state, D, MapGrid) list for one condition, ordered slowest→fastest.

    ``mode="enrichment"`` divides each state by the all-localization null;
    ``mode="density"`` is the raw density.
    """
    from .cellcoords import project_tracks
    from .maps import fold_map, labels_from_appd, labels_from_hmm
    from .render import enrichment_grid, map_grid

    # A cell with no Oufti mesh RIBS has no centerline, so its localizations cannot
    # be projected, SKIP those cells rather than fail the whole condition's map
    # (require_centerline=False leaves their coords NaN, so they drop out of every
    # grid). The count of skipped cells is reported, never silently dropped.
    cond_df = project.condition(condition)
    n_cells = int(cond_df["cell"].dropna().nunique())
    projected = project_tracks(cond_df, project.meshes, require_centerline=False)
    n_skipped = max(0, n_cells - int(projected.attrs.get("n_cells_projected", n_cells)))
    if source == "hmm":
        from .hmm import fit_hmm, step_sequences

        fit = fit_hmm(step_sequences(projected), n_states=n_bins,
                      tau_s=project.params.tau_s, **_hmm_kwargs(spec, default_starts=6))
        labelled = fold_map(labels_from_hmm(projected, fit, min_posterior=0.5))
        state_D = {i + 1: float(d) for i, d in enumerate(fit.D)}
    else:
        labelled = fold_map(labels_from_appd(projected, n_bins=n_bins))
        state_D = _appd_bin_D(labelled)

    states = sorted(int(s) for s in labelled["state"].unique())
    out = []
    for s in states:
        if mode == "enrichment":
            grid = enrichment_grid(labelled, state=s, bins=bins, sigma_um=sigma_um,
                                   n_min=n_min)
        else:
            grid = map_grid(labelled, state=s, bins=bins, sigma_um=sigma_um,
                            n_min=n_min)
        out.append((s, state_D.get(s), grid))
    return out, n_skipped


def _appd_bin_D(labelled):
    """Median apparent D per speed bin, for a neutral panel label."""
    from .appd import apparent_D

    try:
        table = apparent_D(labelled, n_steps=4, mode="split")
        per_track = table.groupby("uid")["appd_um2_per_s"].mean()
    except Exception:                                  # pragma: no cover - label only
        return {}
    out = {}
    for state in sorted(labelled["state"].unique()):
        uids = labelled.loc[labelled["state"] == state, "uid"].unique()
        vals = per_track.reindex(uids).dropna()
        if len(vals):
            out[int(state)] = float(vals.median())
    return out


def saspt_spectra(project, spec: ComparisonSpec, *, cache=None) -> dict:
    """saSPT posterior occupancy over the D grid, one curve per condition.

    The nonparametric cross-check as a COMPARISON: overlay each condition's
    posterior so a shift in where the mass sits is visible without asserting a
    state count. Neutral: the x-axis is D, the y-axis posterior occupancy, curves
    are labelled by condition. Grids/occupations are plain arrays for the plot.
    """
    per_condition = {}
    for condition in spec.conditions:
        _check_cancel(spec)
        n_tracks = _condition_track_count(project, condition)
        if n_tracks < MIN_TRACKS_TO_FIT:
            # honest edge state: too little data to infer a posterior, say so,
            # never draw a spectrum from noise (matches sqd_bubbles / hmm_kinetics)
            per_condition[condition] = {
                "grid": np.empty(0), "occupations": np.empty(0),
                "occupancy_below": float("nan"),
                "error": f"n = {n_tracks}: too few tracks to infer",
            }
            continue
        try:
            result = _cached_saspt(project, condition, spec, cache)
        except Exception as exc:                        # fail loud, per condition
            per_condition[condition] = {
                "grid": np.empty(0), "occupations": np.empty(0),
                "occupancy_below": float("nan"),
                "error": f"n = {n_tracks}: saSPT failed ({exc})",
            }
            continue
        per_condition[condition] = {
            "grid": np.asarray(result.diffusion_grid, dtype=float),
            "occupations": np.asarray(result.occupations, dtype=float),
            "occupancy_below": float(result.occupancy_below(spec.threshold_um2_s)),
        }
    return {"per_condition": per_condition, "threshold_um2_s": spec.threshold_um2_s}


@dataclass
class ForestRow:
    """One method × target's Δ slow-fraction vs the reference condition, with a CI."""

    method: str
    delta: float
    ci: tuple
    shares_machinery: bool = True      # False for saSPT / confinement (mark them)
    target: str = ""                   # the compared condition (vs the reference)


def forest(project, spec: ComparisonSpec, *, methods=None, cache=None) -> dict:
    """Cross-method forest: Δ slow-fraction (target − reference) per method.

    Generalised to N conditions: for the chosen reference, every OTHER selected
    condition is a target, and each (method, target) is one row, Δ in the
    threshold-basis slow fraction, point + bootstrap CI, with a line at 0.
    ``methods`` is the ticked set (default all); the caption counts THIS set.

    The two-condition case (one target) is just the single-column forest; with
    more, the plot groups rows by target so several conditions read against one
    reference at once.
    """
    if len(spec.conditions) < 2:
        raise ComparisonError(
            "the forest needs a reference and at least one other condition; "
            f"got {len(spec.conditions)}")
    reference = spec.reference_condition()
    targets = [c for c in spec.conditions if c != reference]

    available = {
        "SQD": (_forest_sqd, True),
        "HMM": (_forest_hmm, True),
        "appD": (_forest_appd, True),
        "saSPT": (_forest_saspt, False),
        "confinement": (_forest_confinement, False),
    }
    methods = list(methods) if methods is not None else list(available)
    for method in methods:
        if method not in available:
            raise ComparisonError(f"unknown forest method {method!r}")

    rows = []
    for target in targets:
        for method in methods:
            fn, shares = available[method]
            try:
                delta, ci = fn(project, reference, target, spec, cache)
                rows.append(ForestRow(method=method, delta=delta, ci=ci,
                                      shares_machinery=shares, target=target))
            except Exception as exc:                      # noqa: BLE001 - reported
                row = ForestRow(method=method, delta=float("nan"),
                                ci=(float("nan"), float("nan")),
                                shares_machinery=shares, target=target)
                row.error = str(exc)                       # type: ignore[attr-defined]
                rows.append(row)
    return {"reference": reference, "target": targets[0], "targets": targets,
            "methods": methods, "rows": rows, "n_methods": len(methods),
            "threshold_um2_s": spec.threshold_um2_s}


#: The pre-specified σ sweep (nm) for the sensitivity panel (consequence 3C):
#: 0, the derived pixelation floor 30.6, and up to the retired MSD-intercept 76.
SIGMA_SWEEP_NM = (0.0, 20.0, 30.6, 45.0, 60.0, 76.0)


def sigma_sensitivity(project, spec: ComparisonSpec, *, cache=None,
                      sigmas=SIGMA_SWEEP_NM, n_pop=None) -> dict:
    """The pre-specified σ-sensitivity sweep, as a GUI-reachable panel (item D).

    Refits each condition at every σ in ``sigmas`` (the SQD static-error offset),
    holding the state COUNT fixed across σ so the states stay comparable, and
    reports the lowest-D state's occupancy per condition and, for two conditions,
    the between-condition difference Δ against σ. The question it answers: is the
    endpoint FLAT across the plausible σ range (then σ is not a threat) or not
    (then that is the finding). This is the panel form of
    ``scripts/sigma_sensitivity.py``: same computation, one source of truth here.

    Expensive by nature (one refit per σ per condition); it is a deliberate panel
    action, not part of the warm view set. NEUTRAL: states by index + numeric D.
    """
    from .observables import squared_displacements
    from .sqd import fit_sqd

    conds = list(spec.conditions)
    if len(conds) < 1:
        raise ComparisonError("σ-sensitivity needs at least one condition")
    obs = {c: _sqd_observable(project.condition(c), spec) for c in conds}
    # the state count is chosen ONCE (at the BIC default, or the spec's lock) and
    # held across σ: a per-σ count would make the occupancies incomparable.
    if n_pop is None:
        # chosen at "track", the weighting every fit of this sweep is made at (below)
        n_pop = resolve_n_pop(_count_observable(project.condition(conds[0]), spec, obs[conds[0]]),
                              spec.model_for(conds[0]),
                              weighting="track",
                              max_pop=spec.best_max_pop, n_jobs=spec.n_jobs)
    n_starts = (spec.fast.sqd_n_starts if spec.is_fast
                else interactive_starts(n_pop))

    slow_occ = {c: [] for c in conds}
    slow_D = {c: [] for c in conds}
    for s in sigmas:
        _check_cancel(spec)
        for c in conds:
            fit = fit_sqd(obs[c], n_pop=n_pop, weighting="track", n_starts=n_starts,
                          seed=0, localization_sigma_um=s / 1000.0)
            i = int(np.argmin(fit.D))                    # the lowest-D state
            slow_occ[c].append(float(fit.fractions[i]))
            slow_D[c].append(float(fit.D[i]))
    delta = None
    if len(conds) == 2:
        a, b = conds
        delta = [slow_occ[b][i] - slow_occ[a][i] for i in range(len(sigmas))]
    return {"sigmas_nm": list(sigmas), "n_pop": int(n_pop), "conditions": conds,
            "slow_occ": slow_occ, "slow_D": slow_D, "delta": delta,
            "delta_range": (min(delta), max(delta)) if delta else None,
            "pixelation_nm": float(project.params.localization_sigma_nm),
            "sigma_source": project.params.localization_sigma_source}


# --------------------------------------------------------------------------- #
# Forest per-method deltas
# --------------------------------------------------------------------------- #


def _forest_sqd(project, reference, target, spec, cache):
    from .observables import squared_displacements
    from .sqd import bootstrap_sqd

    def slow_from_boot(condition):
        fit, n_pop = _cached_fit(project, condition, spec, cache)
        point = slow_fraction_below(fit.D, fit.fractions, spec.threshold_um2_s)
        obs = _sqd_observable(project.condition(condition), spec)
        boot = bootstrap_sqd(obs, n_pop=n_pop, unit="cell", n_boot=spec.effective_n_boot(),
                             weighting="track")
        samples = _boot_slow_below(boot, n_pop, spec.threshold_um2_s)
        return point, samples

    ref_pt, ref_s = slow_from_boot(reference)
    tar_pt, tar_s = slow_from_boot(target)
    return _delta_ci(ref_pt, ref_s, tar_pt, tar_s)


def _forest_appd(project, reference, target, spec, cache):
    from .qc import track_summary

    def per_cell_slow(condition):
        summary = track_summary(project.condition(condition))
        keep = summary[["cell", "appd_um2_per_s"]].dropna()
        per_cell = (
            keep.assign(is_slow=(keep["appd_um2_per_s"] < spec.threshold_um2_s))
            .groupby("cell", observed=True)["is_slow"].mean().to_numpy())
        return per_cell

    ref = per_cell_slow(reference)
    tar = per_cell_slow(target)
    return _delta_ci(float(ref.mean()), _resample_mean(ref, spec.effective_n_boot()),
                     float(tar.mean()), _resample_mean(tar, spec.effective_n_boot()))


def _forest_hmm(project, reference, target, spec, cache):
    tau = project.params.tau_s

    def occ(condition):
        boot = _cached_hmm_boot(project, condition, spec, cache)
        col = "occupancy_1"
        samples = boot[col].to_numpy() if col in boot else np.array([np.nan])
        return float(np.nanmean(samples)), samples

    ref_pt, ref_s = occ(reference)
    tar_pt, tar_s = occ(target)
    return _delta_ci(ref_pt, ref_s, tar_pt, tar_s)


def _forest_saspt(project, reference, target, spec, cache):
    def occ_below(condition):
        return _cached_saspt(project, condition, spec, cache).occupancy_below(
            spec.threshold_um2_s)

    ref = occ_below(reference)
    tar = occ_below(target)
    # saSPT gives no per-unit resample here; report the point Δ with a degenerate
    # CI (the plot marks saSPT as sharing no machinery).
    return tar - ref, (tar - ref, tar - ref)


def _forest_confinement(project, reference, target, spec, cache):
    from .confinement import confinement_null
    from .observables import squared_displacements
    from .sqd import fit_sqd

    def excess(condition):
        part = project.condition(condition)
        fast_D = float(max(fit_sqd(squared_displacements(part), n_pop=2).D))
        null = confinement_null(part, radius_um=0.1, D_um2_s=fast_D, min_steps=3,
                                n_replicates=3)
        return float(null.excess["confined_segment_fraction"])

    ref = excess(reference)
    tar = excess(target)
    return tar - ref, (tar - ref, tar - ref)


# --------------------------------------------------------------------------- #
# Bootstrap helpers
# --------------------------------------------------------------------------- #


def _boot_slow_below(boot: pd.DataFrame, n_pop: int, threshold: float) -> np.ndarray:
    """Per-replicate slow-fraction-below-threshold from a bootstrap frame."""
    out = []
    for _, row in boot.iterrows():
        D = [row.get(f"D_{k}") for k in range(1, n_pop + 1)]
        fr = [row.get(f"fraction_{k}") for k in range(1, n_pop + 1)]
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in D + fr):
            continue
        out.append(slow_fraction_below(D, fr, threshold))
    return np.asarray(out, dtype=float)


def _cell_fraction_bootstrap(values: np.ndarray, *, n_boot: int) -> tuple:
    """Percentile CI of the mean of per-cell values, resampling cells."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(0)
    means = [rng.choice(values, size=values.size, replace=True).mean()
             for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def _resample_mean(values: np.ndarray, n_boot: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.array([np.nan])
    rng = np.random.default_rng(0)
    return np.array([rng.choice(values, size=values.size, replace=True).mean()
                     for _ in range(n_boot)])


def _delta_ci(ref_pt, ref_samples, tar_pt, tar_samples) -> tuple:
    """Δ = target − reference, with a percentile CI on the paired resamples."""
    delta = float(tar_pt - ref_pt)
    ref_samples = np.asarray(ref_samples, dtype=float)
    tar_samples = np.asarray(tar_samples, dtype=float)
    n = min(ref_samples.size, tar_samples.size)
    if n < 2:
        return delta, (delta, delta)
    diffs = tar_samples[:n] - ref_samples[:n]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return delta, (float(lo), float(hi))


def _permutation_p(a: np.ndarray, b: np.ndarray, *, n_perm: int = 2000) -> float:
    """Two-sided permutation test on the difference of means. Cell-level input.

    p is never zero: the observed labelling is counted, so the floor is
    ``1 / (n_perm + 1)``. This is the only kind of significance test allowed,
    the input is one value per independent unit (a cell), never pooled steps.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    observed = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(0)
    na = a.size
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


# --------------------------------------------------------------------------- #
# Cache glue, memoize the expensive fits by provenance
# --------------------------------------------------------------------------- #


def _for_condition(cache, condition):
    """Route a per-condition result into that condition's own folder cache when the
    cache supports it, so the result travels with the condition folder."""
    if cache is not None and hasattr(cache, "for_condition"):
        return cache.for_condition(condition)
    return cache


# --------------------------------------------------------------------------- #
# Per-condition PARALLELISM, warm the expensive per-condition caches concurrently
# --------------------------------------------------------------------------- #

#: The expensive per-condition results, warmed in parallel. Each writes to the
#: CONDITION's own folder cache (distinct folders → no write race); all are
#: deterministic (seed 0), so parallel and serial are byte-identical.
_WARM_JOBS = ("sqd_fit", "hmm_boot", "saspt")


def _warm_one_condition(condition, shared):            # module-level: picklable
    """Compute (and cache) one condition's expensive results. Side-effecting on the
    condition's folder cache; returns the condition for ordering/reporting."""
    project = shared["project"]
    spec = shared["spec"]
    cache = shared["cache"]
    jobs = shared.get("jobs", _WARM_JOBS)
    for job in jobs:
        try:
            if job == "sqd_fit":
                _cached_fit(project, condition, spec, cache)
            elif job == "hmm_boot":
                _cached_hmm_boot(project, condition, spec, cache)
            elif job == "saspt":
                _cached_saspt(project, condition, spec, cache)
        except Exception:                              # noqa: BLE001 - warm best-effort
            continue
    return condition


def warm_conditions(project, spec: ComparisonSpec, *, cache=None, jobs=None):
    """Warm every selected condition's expensive per-condition caches CONCURRENTLY.

    Runs the SQD fit, the HMM rate bootstrap and the saSPT posterior for each
    condition on its own worker (``spec.n_jobs`` cores), each writing into that
    condition's folder cache. Because the fits are deterministic and the folders
    are distinct, the warmed caches (and therefore every view assembled from them
   ) are BYTE-IDENTICAL to the serial path regardless of worker count. Degrades to
    serial for a single condition, ``n_jobs=1``, or a pool that will not start.
    Returns the list of conditions warmed.
    """
    if cache is None:
        return []
    from dataclasses import replace

    from .parallel import condition_map

    # should_cancel is a callable (not picklable) and is control flow, not an
    # input, strip it before the spec ships to workers.
    spec = replace(spec, should_cancel=None)
    shared = {"project": project, "spec": spec, "cache": cache,
              "jobs": tuple(jobs) if jobs else _WARM_JOBS}
    return condition_map(_warm_one_condition, list(spec.conditions),
                         shared=shared, n_jobs=spec.n_jobs)


def _cached_fit(project, condition, spec, cache):
    """SQDFit + n_pop, memoized by provenance when a cache is supplied.

    Stored in the CONDITION's own folder cache (see ``WarmCache.for_condition``)
    so a per-condition fit travels with the condition folder.
    """
    cache = _for_condition(cache, condition)
    if cache is None:
        return fit_condition(project, condition, spec)
    from .provenance import capture

    model = spec.model_for(condition)
    provenance = capture(
        settings={"view": "sqd_fit", "condition": condition, "model": model,
                  # the count "Best" may reach is part of which fit this is
                  "best_max_pop": spec.best_max_pop,
                  "cache_version": COMPARISON_CACHE_VERSION,
                  # THE RESOLVED BUDGET, never the raw spec field. The field is None
                  # by default and the budget is not, so keying on the field would give
                  # an exact run and a thinned run the SAME key -- the `require_all_inside`
                  # defect exactly: a setting that changes the table, absent from the key.
                  # THE WEIGHTING THE FIT WAS MADE AT. It was the literal "track"
                  # while the fit was always track; once the fit follows the spec, a
                  # literal here would hand a step request the track fit.
                  "weighting": spec.fraction_weighting, "ecdf": _ecdf_budget(spec),
                  **spec.fast_settings()},
        seeds={"sqd": 0},
        canonical=project.condition(condition),
        unit="cell",
    )
    return cache.get_or_compute(
        f"cmp_fit_{condition}_{model}", provenance,
        lambda: fit_condition(project, condition, spec))


def _best_bic(project, condition, spec, lo: int, hi: int) -> list:
    """``[[n, BIC], ...]`` for the counts ``lo..hi`` as Best's own selection scores them for ``condition``
    (`resolve_n_pop`'s: `select_n_pop`, maximum likelihood, the full multistart, at the fit's weighting, on the
    observable the count is chosen on). Every count's fit is its own plan and seed, so a count scored here scores what
    it does in Best's whole table."""
    from .sqd import select_n_pop

    part = _subsample(project.condition(condition), spec)
    selection = select_n_pop(_count_observable(part, spec), min_pop=int(lo), max_pop=int(hi), method="mle",
                             n_jobs=spec.n_jobs, weighting=spec.fraction_weighting,
                             precision=spec.precision_calibration(condition))
    return [[int(f.n_pop), float(f.bic)] for f in sorted(selection.fits, key=lambda f: f.n_pop)]


def forced_count_bic(project, condition, spec, cache=None) -> list | None:
    """A COUNT SET BY HAND, CHECKED AGAINST BEST (ruled 4 Nov 2026, `bubble_view.forced_count_warning`): Best's scores
    beside a condition fitted at a number of two or more, as far as the answer needs them
    (`bubble_view.forced_count_scores`: the counts below and the count set, by Best's own maximum likelihood -- the
    bubble's own fit is not what Best scores -- and Best's counts above only where one below wins) -- memoized like
    the fits (`_cached_fit`), so a Run pays for it once -- or None where nothing is checked: Best itself, one
    population (none fewer), and a set-wide Force same D across (Conditions, Whole data set), whose count is the joint
    fit's. The fit is never made at Best's count instead: the user decides."""
    from .bubble_view import forced_count_scores

    model = str(spec.model_for(condition))
    if not model.isdigit() or int(model) < 2 or spec.scope_for(condition) in ("global", "group"):
        return None
    max_pop = max(int(spec.best_max_pop), int(model))

    def scores():
        return forced_count_scores(int(model), range(1, max_pop + 1),
                                   lambda lo, hi: _best_bic(project, condition, spec, lo, hi))

    cache = _for_condition(cache, condition)
    if cache is None:
        return scores()
    from .provenance import capture

    # the count set is in the key: which counts are scored depends on it (the rows held for a set 2 lack 3)
    provenance = capture(
        settings={"view": "sqd_best_bic", "condition": condition, "max_pop": max_pop, "n_set": int(model),
                  "cache_version": COMPARISON_CACHE_VERSION, "weighting": spec.fraction_weighting,
                  **spec.fast_settings()},
        seeds={"sqd": 0}, canonical=project.condition(condition), unit="cell")
    return cache.get_or_compute(f"cmp_bestbic_{condition}", provenance, scores)


def _cached_replicate_fit(project, condition, key, frame, observable, spec, cache, *, n_pop, weighting, n_starts,
                          zero_rule: str = "auto"):
    """One replicate day's own SQD fit (the Per Day scope), memoized by provenance like `_cached_fit`: the key
    hashes the day's own table (``frame``), so a different day, condition or exclusion never shares an entry, and
    holds every setting the fit reads (count, weighting, starts, the spec's fit settings -- optimiser and lag
    included). Without a cache it is the plain fit, exactly as before. ``zero_rule`` is the condition's rule for
    exact-zero jumps (`sqd.zero_rule_of` on its pooled fit), in the key when it is not "auto"."""
    from .sqd import fit_sqd

    def fit():
        return fit_sqd(observable, n_pop=n_pop, weighting=weighting, n_starts=n_starts, zero_jumps=zero_rule,
                       **spec.sqd_fit_kwargs(condition))

    cache = _for_condition(cache, condition)
    if cache is None:
        return fit()
    from .provenance import capture

    provenance = capture(
        settings={"view": "sqd_replicate_fit", "condition": condition, "replicate": str(key),
                  "n_pop": int(n_pop), "weighting": weighting, "n_starts": int(n_starts),
                  "cache_version": COMPARISON_CACHE_VERSION, **spec.fast_settings(),
                  **spec.replicate_settings(),
                  **({"zero_jumps": zero_rule} if zero_rule != "auto" else {})},
        seeds={"sqd": 0}, canonical=frame, unit="cell")
    return cache.get_or_compute(f"cmp_rep_{condition}_{key}_{n_pop}", provenance, fit)


def _hmm_kwargs(spec, *, default_starts):
    """fit_hmm knobs honoring fast mode (fewer starts, capped EM iterations)."""
    if spec.is_fast:
        return {"n_starts": spec.fast.hmm_n_starts, "max_iter": spec.fast.hmm_max_iter}
    return {"n_starts": default_starts}


def _cached_bubble_boot(project, condition, spec, n_pop, cache, *, unit: str = "cell"):
    """The per-population SQD CI bootstrap, memoized on a THRESHOLD-INDEPENDENT
    key (condition + n_pop + n_boot), so a threshold change reuses it. ``unit`` is
    what is resampled, ``"cell"`` or ``"movie"``; a cell interval keeps the key and
    provenance it always had, so caches made before the movie unit still hit."""
    from .observables import squared_displacements
    from .sqd import bootstrap_sqd

    cache = _for_condition(cache, condition)
    obs = _sqd_observable(project.condition(condition), spec)
    # THE SAME WEIGHTING AS THE POINT IT BRACKETS. This was the literal "track", so under
    # a step-weighted fit the interval resampled a different quantity from the one drawn.
    weighting = spec.fraction_weighting
    compute = lambda: bootstrap_sqd(obs, n_pop=n_pop, unit=unit,
                                    n_boot=spec.interval_resamples(), weighting=weighting,
                                    **spec.sqd_fit_kwargs(condition))
    if cache is None:
        return compute()
    from .provenance import capture

    by_unit = {} if unit == "cell" else {"unit": unit}
    provenance = capture(
        settings={"view": "bubble_ci", "condition": condition, "n_pop": n_pop,
                  "cache_version": COMPARISON_CACHE_VERSION, "weighting": weighting,
                  "n_boot": spec.interval_resamples(), **spec.fast_settings(), **by_unit},
        seeds={"sqd": 0, "bootstrap": 0},
        canonical=project.condition(condition), unit=unit,
    )
    suffix = "" if unit == "cell" else f"_{unit}"
    return cache.get_or_compute(f"cmp_bubbleci_{condition}_{n_pop}_{weighting}{suffix}",
                                provenance, compute)


def _cached_hmm_boot(project, condition, spec, cache):
    """HMM rate bootstrap frame, memoized by provenance when a cache is supplied."""
    from .hmm import bootstrap_hmm

    cache = _for_condition(cache, condition)
    part = _subsample(project.condition(condition), spec)
    tau = project.params.tau_s
    boot_kwargs = {"n_starts": spec.fast.hmm_n_starts} if spec.is_fast else {}
    compute = lambda: bootstrap_hmm(part, n_states=2, tau_s=tau, unit="cell",
                                    n_boot=spec.effective_n_boot(), **boot_kwargs)
    if cache is None:
        return compute()
    from .provenance import capture

    provenance = capture(
        settings={"view": "hmm_boot", "condition": condition,
                  "n_boot": spec.effective_n_boot(), **spec.fast_settings()},
        seeds={"hmm": 0, "bootstrap": 0},
        canonical=part, unit="cell",
    )
    return cache.get_or_compute(f"cmp_hmmboot_{condition}", provenance, compute)
