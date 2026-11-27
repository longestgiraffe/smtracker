"""Significance brackets on exported figures: which values, which test, which words. Headless.

RULED BY THE EXPERIMENTER, 13 Oct 2026 (deferred until then "to be discussed"):

* **The test is Welch's t on PER-DAY values** (`uncertainty.welch_test`). Each condition's
  replicate days are ONE independent sample. Never paired by day number (RULE 0: ``day``
  is an ordinal within its condition, not a date, so glucose day 1 and malate day 1 share
  nothing but an integer), and never per cell or per track (pseudoreplication: a test on
  pooled steps had a 65 % false-positive rate here). There is no path from pooled tracks
  to a p in this module: the only values it reads are the per-day fractions behind a
  drawn day mean.
* **The bracket prints the EXACT p**, Holm-adjusted over the pairs picked ("p = 0.012";
  below 0.001, "p < 0.001"). The legend names the test, the days per condition and the
  correction.
* **The pairs are picked by the user** in the Export dialog, and Holm runs over exactly
  those (`uncertainty.holm`; a pair with no test is not part of the family).
* **Holm is the standard correction; Benjamini-Hochberg is a choice beside it** (10 Nov 2026,
  `CORRECTIONS`, `uncertainty.adjust`): the lab's own scripts and standing legends use
  Benjamini-Hochberg, so a legend written by habit named a correction the figure had not used.
  Chosen on the chart (`ViewChoice.correction`), exported as shown
  (`FigureSpec.significance_correction`), and NAMED wherever a p is adjusted: the legend, the
  sheets' columns ("p (Benjamini-Hochberg)"), the block's lines. It holds the false discovery
  rate, not the chance of any false positive, and the legend says so. Wherever this text says
  "Holm" below, read "the correction chosen"; with Holm chosen nothing reads differently from
  before. `Comparison.p_holm` keeps its name and holds the adjusted p under either.
* **Opt-in per figure**: off by default, never carried to the next chart.

WHERE THE VALUES COME FROM. The SQD bubble chart under Uncertainty = Days draws each
population's MEAN over the condition's replicate days; `bubble_view.compute_bubble_view`
keeps the per-day fractions behind each mean (`BubblePlotData.unit_values`), and those are
what is tested. Under Uncertainty = None or Cells the bubble IS the pooled fit, so there is
no per-day sample behind it and no p is offered (one from pooled tracks would be the
pseudoreplication this project measured). An overlay of several fit arrangements draws
several means per column, so none is offered there either. The apparent-D bubble chart
follows the same rules since 14 Oct 2026: under Uncertainty = Days it draws the mean of
per-day gamma-mixture fractions (D shared by a condition's days) and keeps them in the same
``unit_values``; before, it was one pooled fit per condition and was refused outright.

A PAIR THAT CANNOT BE TESTED IS NAMED, NEVER DRAWN WITH A NUMBER: a condition imaged in one
session, a condition with no fit, differing state counts ("x % of 2 states" and "x % of 3
states" are different quantities, the refusal `compare.compare_fractions` makes), or a
condition no longer in the figure. Its row in the source data says why. Since 1 Nov 2026 also
two conditions FITTED WITH THEIR D FORCED DIFFERENTLY (a row of the Compute dialog's Fits page
set Replicates, another Don't force anywhere; `bubble_view.mixed_scopes`): one bubble is the
replicates' fractions at the condition's shared D, the other the mean of each replicate's own
fit, so their difference would include the method. Pairs fitted alike are tested as before.

THE STRICTER SECOND TEST (asked for 14 Oct 2026: "can we have more of strict mixed model
statistics as well that looks at the distribution of the data, but not like on v2 there
everything was significant"; ruled: cells within days, a parametric-bootstrap p, beside Welch
in the Export dialog). ``test="mixed"`` fits `mixedmodel` to EACH CELL's fraction (D held at
the condition's shared D, `comparison.condition_replicate_fit`): cells within movies within
replicate days, the condition the fixed effect, and a p from simulating the fit without a
condition effect. It uses the cells' spread and still judges the effect against the scatter
of days and movies, so it cannot become v2's test on pooled steps. It is offered on the SQD
bubbles and, since 4 Nov 2026, on the apparent-D bubbles (`appd_fits` keeps each cell's fractions at
the D its condition's replicates share, cells in movies in replicates, as SQD does), and needs the
same two days per condition. A p costs seconds per pair, so it is CACHED (`_MIXED`): the Export dialog's preview
worker computes it (its build runs `compare`), and the dialog's lines read only the cache
(``compute=False``), so until then the pair is ``pending``, never drawn.

ON THE CHART FIRST, THE EXPORT AS SHOWN (asked for 14 Oct 2026: "just add in the viewing
options next to the plots a way to show significance in the plot in smtracker. then during
export, it exports how u see it depending on the choices on the plot"). The choices live with
the CHART (`ViewChoice`, set in the Significance tool's popover); `chart_view` turns
them into the brackets the workspace paints and the lines the popover shows; the Export dialog
starts from them (`ViewChoice.spec_changes`) and shows them, read-only. On screen a mixed-model
p is computed in the workspace's process pool (`pending_mixed` / `compute_mixed` /
`store_mixed`), never on the GUI thread, and the pair stays undrawn until it is in. Which
bracket stands over which (`bracket_tiers`) is one rule for the screen and the export.

STARS, ALL POPULATIONS, AND WHERE A BRACKET STANDS (asked for 14 Oct 2026: "the stars are
missing for significance with option to change what each stars mean. standard should be the
normal values"; "when i choose slow the significance bar and stars and values should be at the
bubble i choose ... for slow, make it to the bottom"; "1,2,3,4,all"):
* A bracket says STARS by default (`DEFAULT_STARS`: * p < 0.05, ** < 0.01, *** < 0.001, else
  "ns", on the Holm-adjusted p), or the exact p, or both (`LABELS`); the thresholds are the
  user's to change (`parse_stars`). The exact p stays in the legend and the source data.
* ``population="all"`` compares every population. With TWO populations the second fraction is
  one minus the first, so its test IS the first's: it is drawn at both bubbles but counted ONCE
  in Holm, and the mixed model's result is mirrored rather than simulated again (bootstrap
  noise would otherwise give the same test two p values). With three or more, each population
  is its own quantity and its own test.
* A bracket stands AT ITS POPULATION's bubbles (`bracket_side`): the slowest below them (the
  chart's bottom), every other above its own bubbles, so the fastest is at the top.
* ``population="below"`` compares the SHARE BELOW THE RUN's D THRESHOLD (`comparison.
  slow_fraction_below`): the summed fractions of every state slower than the threshold, which
  means the same for a 2-state and a 3-state fit, so it is the comparison offered when two
  conditions were fitted with different state counts (asked 14 Oct 2026: "yes"). Its bracket
  stands under the bubbles below the threshold.

RANDOM CHUNKS (24 Oct 2026, Uncertainty "Chunks", `comparison.PSEUDO_CHUNKS`): where a condition's
replicates are random chunks of its movies or cells, Welch and the equivalence test run on the
chunks as they would on replicates -- measured to hold their level within one culture
(`scripts/pooled_false_alarm.py`) -- and every line says they are pseudo-replicates; each pair is
tested on every other deal of the chunks too, and the range of p is reported beside the one drawn
(`Comparison.split_p`). The cell tests are refused there: they read the real replicates, and a
condition imaged once has one.

A BRACKET ON A FLAGGED POPULATION IS DRAWN AS ANY OTHER, AND A WARNING NAMES IT (ruled 4 Nov 2026, D8; the overnight
tester's A8: a bracket compared population 1 of a fit whose two populations had the same D). "Check the fit"
(`bubble_view.fit_checks`: a population under 5 %, or two populations within twofold in D) is not a refusal, so the
bracket keeps its p and carries NO mark; `bracket_warnings` says which bracket stands on which flagged population and
why, under the chart (`chart_view`'s ``warnings``, the SQD section's flag line, the apparent-D section's) and in the
exported legend (`figure_export._significance_legend`).

BRACKETS ON THE HMM COMPARISON (asked 5 Nov 2026: "yes but not as default"). The HMM section's "Across conditions"
strip (`hmm_view.comparison_payload`, kind `HMM_KIND`) draws each condition's per-replicate HMM fits as open circles on
four panels -- the state's occupancy, the rates into and out of it, its apparent D. A bracket there compares ONE
PANEL's per-replicate values between two conditions (``population`` is the panel's key, or "all" for every panel), by
Welch on replicates or, where replicates were declared as pairs, the paired test (`HMM_TESTS`; the equivalence margin is
in percentage points of a fraction and the cell tests read cells the HMM does not keep, so neither is offered), Holm
over every bracket shown. Off by default, as on the bubbles. Refused by name: a pooled fit or a movie or cell interval
(no per-replicate sample), random chunks (one deal, no range of p), two conditions fitted with different state counts
(their "slow state" is not one quantity), and the D panel where the D was held across replicates (no spread).

Nothing here draws; `figure_export` draws the brackets and writes the sheets from what this
returns, and the Export dialog shows `status_lines`.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field, replace

__all__ = [
    "OFFERED",
    "HMM_KIND",
    "HMM_TESTS",
    "MIXED_OFFERED",
    "SECTION_KINDS",
    "ViewChoice",
    "DEFAULT_STARS",
    "CORRECTIONS",
    "CORRECTION_NAMES",
    "CORRECTION_TIPS",
    "DEFAULT_CORRECTION",
    "correction_name",
    "LABELS",
    "LABEL_WORDS",
    "bracket_side",
    "bracket_text",
    "parse_stars",
    "star_text",
    "stars_sentence",
    "bracket_tiers",
    "chart_view",
    "compute_mixed",
    "pending_mixed",
    "store_mixed",
    "MIXED_TEST_NAME",
    "TESTS",
    "TEST_LABELS",
    "TEST_NAME",
    "CORRECTION",
    "RULE0_SENTENCE",
    "Comparison",
    "add_pair",
    "FLOOR_TESTS",
    "floor_words",
    "bracket_warnings",
    "compare",
    "comparison_table",
    "conditions",
    "legend_sentence",
    "offered",
    "p_text",
    "population_choices",
    "whose_words",
    "sheet_notes",
    "status_lines",
    "unavailable",
    "values_sheet",
    "values_table",
    "without_pair",
]

#: The HMM section's comparison across conditions (5 Nov 2026), and the tests offered on it: those on one value per
#: replicate that read no fraction margin.
HMM_KIND = "hmm_comparison"
HMM_TESTS = ("welch", "paired")
#: The figure kinds whose Export dialog shows the Significance row.
OFFERED = ("sqd_bubbles", "appd_bubbles", HMM_KIND)

TEST_NAME = "Welch's t-test (two-sided, unequal variances) on per-replicate values"
#: The minimum-effect test's name on the Comparisons sheet (its rows are Welch's columns, its p the test's).
MET_TEST_NAME = ("minimum-effect test: two one-sided Welch tests against plus and minus the margin, on per-replicate "
                 "values")
#: The tests a figure can ask for, and the words the Export dialog's choice shows. Asked 24 Oct 2026 beside Welch and
#: the mixed model ("lets try all of them u listed"; "1, good idea. 2. true. 3. we can include if people do more
#: replicates"; "Ur proposal sounds good build it"): the equivalence test for "similar", the Bayesian model, and the
#: meta-analysis and hierarchical bootstrap, the last two offered from three replicates a side (`nested`).
#: THE PAIRED TEST (5 Nov 2026, asked: "optional pair ... in day picker?", "a tick box for activate pairing in day
#: picker"): offered only where replicates were DECLARED as pairs on the group board (`declared_pairs`), never by
#: replicate number; Welch stays the standard.
#: THE MINIMUM-EFFECT TEST (23 Nov 2026, asked: "My prof said don't make to many movies or everything will get
#: significant", then "yes" to it as the standard under Chunks): do the two differ by MORE than the margin
#: (`uncertainty.minimum_effect_test`), the equivalence test's mirror, on the same margin. Offered wherever the
#: equivalence test is; the test a chunk chart starts with (`resolved`) until one is picked by hand.
TESTS = ("welch", "paired", "tost", "met", "mixed", "bayes", "meta", "hboot")
TEST_LABELS = {"welch": "Welch on replicates", "paired": "Paired t (declared pairs)", "tost": "Equivalence (TOST)",
               "met": "Difference beyond margin",
               "mixed": "Mixed model", "bayes": "Bayesian model", "meta": "Meta-analysis",
               "hboot": "Hierarchical bootstrap"}
#: What each test's choice says on hover.
TEST_TIPS = {
    "welch": "Welch's t on each condition's per-replicate values: the replicate is the unit (the standard).",
    "paired": "A paired t-test on the replicates declared as pairs on the Groups board (Pair replicates): each pair "
              "one culture or session in both conditions. A replicate without a partner is left out of it.",
    "tost": "Are the two conditions the same within the margin? Two one-sided Welch tests on the per-replicate "
            "values; a bracket shows ≈ when they are equivalent. A non-significant Welch p does not show this.",
    "met": "Do the two conditions differ by more than the margin? Two one-sided Welch tests against ±margin on the "
           "per-replicate values: a smaller difference is never significant, however much data there is. The "
           "standard under Chunks, where two cultures always differ a little.",
    "mixed": "Each cell's fraction, cells within movies within replicates, a parametric-bootstrap p; a few seconds "
             "a pair.",
    "bayes": "Each cell's fraction in a Bayesian hierarchical model (half-normal priors on the replicate, movie and "
             "cell SDs, written in the legend); works with one replicate on a side. About a second a pair.",
    "meta": "Random-effects meta-analysis over the replicates, each with its own variance; from three replicates "
            "a side.",
    "hboot": "Replicates, then movies, then cells resampled; from three replicates a side.",
}
#: Tests on one value per replicate (instant), and tests on each cell's fraction (computed off the GUI, cached).
REPLICATE_TESTS = ("welch", "paired", "tost", "met")
#: The tests that read the margin (the popover's margin row; percentage points of the fraction).
MARGIN_TESTS = ("tost", "met")
CELL_TESTS = ("mixed", "bayes", "meta", "hboot")
#: THE TESTS WHOSE p IS PRINTED WITH THE DESIGN'S FLOOR (ruled 4 Nov 2026, D13; the overnight tester, round 2: on 3 v 3
#: replicates they printed "p < 0.001" where Welch gave 0.004 and an exact test cannot go below 0.1). Their p comes
#: from a model or a resample of cells and movies, so it can fall far under what the replicates alone allow; beside it
#: stands `Comparison.floor`, the smallest p an exact permutation test on the same replicate counts can give
#: (`aggregate.achievable_p`), on screen (`status_lines`) and in the export (`legend_sentence`, the Comparisons sheet).
FLOOR_TESTS = ("bayes", "meta", "hboot")
#: The figure kinds that keep each cell's fractions (`bubble_view`'s ``unit_values``): the cell tests' charts. The
#: apparent-D bubbles since 4 Nov 2026 (`appd_fits.unit_values_of`: each cell's fractions at the shared D, cells in
#: movies in replicates; found by the overnight tester, the manual promised them and the chart refused them).
MIXED_OFFERED = ("sqd_bubbles", "appd_bubbles")
#: The equivalence margin a new chart starts from, in percentage points of the fraction (chosen before looking).
DEFAULT_MARGIN_PP = 5.0
#: What an equivalence bracket says when the two are shown equivalent, and when they are not.
EQUIVALENT_MARK = "≈"
NOT_EQUIVALENT_MARK = "not ≈"


def tests_for(cells: bool, paired: bool = False, kind: str = "") -> tuple:
    """The tests a chart offers: every one where each cell's fraction is kept, else those on per-replicate values (on
    the HMM comparison `HMM_TESTS`); the paired test only where replicates were declared as pairs (``paired``,
    `declared_pairs`)."""
    pool = HMM_TESTS if kind == HMM_KIND else (TESTS if cells else REPLICATE_TESTS)
    return tuple(t for t in pool if paired or t != "paired")


def declared_pairs(payload) -> dict:
    """``{condition: {replicate: tag}}``: the replicates declared as pairs on the group board, as the workspace puts
    them in a chart's payload (``"replicate_pairs"``, from `project.Project.replicate_pairs`); empty where none."""
    found = payload.get("replicate_pairs") if isinstance(payload, dict) else None
    return {str(c): {int(d): str(t) for d, t in tags.items()} for c, tags in (found or {}).items() if tags} \
        if isinstance(found, dict) else {}
#: The workspace sections whose bubble chart can carry brackets, and the figure kind it draws.
SECTION_KINDS = {"sqd": "sqd_bubbles", "appd": "appd_bubbles", "hmm": HMM_KIND}
MIXED_TEST_NAME = ("nested linear mixed model on each cell's fraction (cells within movies "
                   "within replicates, condition as the fixed effect, REML), two-sided "
                   "parametric-bootstrap p")
CORRECTION = "Holm"
#: THE CORRECTION FOR SEVERAL BRACKETS ON ONE FIGURE. Holm is the standard (ruled 13 Oct 2026) and what every figure
#: made before the choice used; Benjamini-Hochberg is a CHOICE beside it (10 Nov 2026: the lab's own scripts and
#: standing legends use it, so a legend written by habit named a correction the figure did not use). The keys are
#: `uncertainty.CORRECTIONS`'s; the names are what a reader sees, in the legend, the sheets and the block.
CORRECTIONS = ("holm", "bh")
DEFAULT_CORRECTION = "holm"
CORRECTION_NAMES = {"holm": "Holm", "bh": "Benjamini-Hochberg"}
CORRECTION_TIPS = {
    "holm": ("Holm: holds the chance of any false positive among the brackets shown at the threshold, whatever the "
             "dependence between them. The standard, and the stricter of the two."),
    "bh": ("Benjamini-Hochberg: holds the expected share of false positives among the brackets called different "
           "(the false discovery rate), under independent or positively dependent tests. Its p is never larger than "
           "Holm's; with one bracket both leave the p unchanged."),
}


def correction_name(correction: str = DEFAULT_CORRECTION) -> str:
    """The correction's name as a reader sees it; an unknown key is refused, never read as Holm."""
    if correction not in CORRECTION_NAMES:
        raise ValueError(f"the correction is one of {', '.join(CORRECTIONS)}; got {correction!r}")
    return CORRECTION_NAMES[correction]


def _correction_of(comparisons) -> str:
    """The correction a set of `Comparison` rows was adjusted with (rows made before the choice: Holm)."""
    return getattr(comparisons[0], "correction", DEFAULT_CORRECTION) if comparisons else DEFAULT_CORRECTION
#: The star thresholds everyone uses, on the adjusted p: * p < 0.05, ** < 0.01, *** < 0.001.
DEFAULT_STARS = (0.05, 0.01, 0.001)
#: What a bracket says: stars (the default), the exact p, or both.
LABELS = ("stars", "p", "both")
LABEL_WORDS = {"stars": "Stars", "p": "Exact p", "both": "Both"}
#: RULE 0, beside the numbers (CLAUDE.md): what dropping the pairing does, and why.
RULE0_SENTENCE = (
    "Unpaired: each condition's replicates are one independent sample, and replicate n of "
    "one condition is not matched to replicate n of another. Unpaired intervals are "
    "narrower, through the test's degrees of freedom, which flatters the effect.")
PAIRED_TEST_NAME = "Paired t-test (two-sided) on the differences between replicates declared as pairs"
#: What the paired test says beside its numbers: who paired the replicates, and what that rests on.
PAIRED_SENTENCE = (
    "Paired: replicates declared as pairs on the Groups board (Pair replicates), each pair one "
    "culture or imaging session shared by both conditions. The test is on the within-pair "
    "differences, n - 1 degrees of freedom for n pairs; unpartnered replicates are left out. "
    "Valid only if the pairing is real.")


@dataclass
class Comparison:
    """One pair a bracket compares. ``test`` is the `uncertainty.WelchTest` (None when the
    pair could not be tested, ``reason`` then says why); ``p_holm`` the ADJUSTED p the
    bracket prints, under ``correction`` (the field keeps its first name: Holm was the only correction until
    10 Nov 2026, and sessions and caches hold rows made then); ``family`` how many tests the correction adjusted
    over."""

    a: str
    b: str
    population: int
    days_a: list
    days_b: list
    values_a: list
    values_b: list
    d_a: float | None = None
    d_b: float | None = None
    test: object = None
    p_holm: float = float("nan")
    family: int = 0
    floor: float = float("nan")
    reason: str = ""
    #: "welch" or "mixed"; for "mixed", ``mixed`` is the `mixedmodel.MixedTest` and ``cells_a``
    #: / ``cells_b`` the per-cell rows it was fitted to
    kind: str = "welch"
    mixed: object = None
    cells_a: list = field(default_factory=list)
    cells_b: list = field(default_factory=list)
    #: a mixed-model p not computed yet (the dialog computes it in the background)
    pending: bool = False
    #: how many populations the two fits have (where the bracket stands: `bracket_side`)
    n_pop: int = 0
    #: the mixed-model cache key this row reads (two populations: the first's)
    key: object = None
    #: the D threshold (µm²/s) when the share below it is compared (``population="below"``)
    threshold: float | None = None
    #: for "bayes" / "meta" / "hboot", the `nested.NestedTest` (it is ``test`` too)
    nested: object = None
    #: for "tost", the equivalence margin in fraction units
    margin: float | None = None
    #: the replicates are random chunks (Uncertainty "Chunks"); ``split_p`` is ``(lowest, median, highest, deals)``:
    #: the raw p over every deal of the chunks, None where there are no other deals
    chunks: bool = False
    split_p: tuple | None = None
    #: for "paired": the declared pairs tested, ``[(tag, replicate of A, replicate of B)]``, and how many replicates
    #: of A and of B had no partner (left out of the test)
    pairs: list = field(default_factory=list)
    unpaired: tuple = (0, 0)
    #: on the HMM comparison: the panel compared, as its title and unit ("Slow-state occupancy (%)"); "" on bubbles
    what: str = ""
    #: the correction ``p_holm`` was adjusted with (`CORRECTIONS`); a row made before the choice reads Holm
    correction: str = DEFAULT_CORRECTION

    @property
    def tested(self) -> bool:
        return self.test is not None and math.isfinite(self.test.p)


class _MixedAsTest:
    """A `mixedmodel.MixedTest` read through the fields the tables and lines read on a
    `uncertainty.WelchTest`. No interval: a Wald interval from three days is too narrow, and
    the bootstrap gives a p, not an interval."""

    def __init__(self, mixed):
        self.mixed = mixed
        self.p, self.t = mixed.p, mixed.t
        self.mean_a, self.mean_b, self.difference = mixed.mean_a, mixed.mean_b, mixed.difference
        self.ci_low = self.ci_high = self.df = float("nan")
        self.n_a, self.n_b = mixed.n_days
        self.reason = ""


@dataclass(frozen=True)
class ViewChoice:
    """What a chart's Significance block holds: brackets on or off, the test, whose fraction
    (a population, or "all"), the pairs (real condition names, in the order picked), what a
    bracket says and the star thresholds. Off by default, per chart."""

    on: bool = False
    test: str = "welch"
    population: object = 1
    pairs: tuple = ()
    label: str = "stars"
    stars: tuple = DEFAULT_STARS
    #: the equivalence test's margin, percentage points of the fraction (``test="tost"``)
    margin: float = DEFAULT_MARGIN_PP
    #: the correction over the brackets shown (`CORRECTIONS`); a choice saved before it existed reads Holm
    correction: str = DEFAULT_CORRECTION
    #: the test was picked by hand (the popover's list): `resolved` then never swaps it for a chart's standard
    test_chosen: bool = False

    def spec_changes(self) -> dict:
        """The `figure_export.FigureSpec` fields that export the chart as it is shown."""
        return {"significance": bool(self.on and self.pairs),
                "significance_pairs": tuple(tuple(p) for p in self.pairs),
                "significance_population": self.population,
                "significance_test": self.test,
                "significance_label": self.label,
                "significance_stars": tuple(float(t) for t in self.stars),
                "significance_margin": float(self.margin),
                "significance_correction": self.correction}


def chunked(data) -> bool:
    """Whether a chart's replicates are random chunks of one culture (Uncertainty "Chunks") for some condition."""
    return bool((getattr(data, "pseudo", None) or {}).get("by"))


def resolved(choice: ViewChoice, data) -> ViewChoice:
    """The choice a chart computes with: on random chunks the minimum-effect test ("met") until a test is picked by
    hand, Welch on real replicates (a test not picked by hand follows the chart). With one culture per condition the
    question is never "do they differ at all" (two cultures always do, and enough movies find it) but "do they differ
    by more than the margin"."""
    if choice.test_chosen or choice.test not in ("welch", "met"):
        return choice
    want = "met" if chunked(data) else "welch"
    return choice if choice.test == want else replace(choice, test=want)


def star_text(p: float, thresholds=DEFAULT_STARS, test: str = "welch") -> str:
    """"*" for each threshold ``p`` is below ("***" under the third), "ns" under none. For the equivalence test
    one mark: "\u2248" below the first threshold (shown equivalent within the margin), "not \u2248" otherwise."""
    if p is None or not math.isfinite(p):
        raise ValueError("no p to mark: a bracket is drawn only for a pair with a test")
    if test == "tost":
        return EQUIVALENT_MARK if thresholds and p < float(max(thresholds)) else NOT_EQUIVALENT_MARK
    k = sum(1 for t in thresholds if p < float(t))
    return "*" * k if k else "ns"


def bracket_text(p: float, label: str = "stars", thresholds=DEFAULT_STARS, test: str = "welch") -> str:
    """What a bracket prints for the adjusted ``p``: stars (or the equivalence mark), the exact p, or both."""
    if label == "p":
        return p_text(p)
    stars = star_text(p, thresholds, test)
    return stars if label == "stars" else f"{stars} ({p_text(p)})"


def stars_sentence(thresholds=DEFAULT_STARS, test: str = "welch", margin_pp: float = DEFAULT_MARGIN_PP) -> str:
    """The legend's key to the stars: "* p < 0.05, ** p < 0.01, *** p < 0.001; ns, p >= 0.05" (for the
    equivalence test, what \u2248 and "not \u2248" mean)."""
    ts = sorted((float(t) for t in thresholds), reverse=True)
    if test == "tost":
        return (f"{EQUIVALENT_MARK}, equivalent within \u00b1{margin_pp:g} percentage points (TOST p < {ts[0]:g}); "
                f"{NOT_EQUIVALENT_MARK}, equivalence not shown" if ts else "")
    keys = ", ".join(f"{'*' * (i + 1)} p < {t:g}" for i, t in enumerate(ts))
    return f"{keys}; ns, p \u2265 {ts[0]:g}" if ts else ""


def parse_stars(text) -> tuple:
    """The star thresholds typed as "0.05, 0.01, 0.001": strictly decreasing, each between 0
    and 1, at most five (one star each). ValueError naming what is wrong."""
    parts = [x for x in str(text).replace(";", ",").replace(" ", ",").split(",") if x]
    try:
        values = tuple(float(x) for x in parts)
    except ValueError:
        raise ValueError("star thresholds are numbers, e.g. 0.05, 0.01, 0.001") from None
    if not values:
        raise ValueError("give at least one star threshold, e.g. 0.05")
    if len(values) > 5:
        raise ValueError("at most five star thresholds")
    if any(not 0 < v < 1 for v in values):
        raise ValueError("each star threshold is a p between 0 and 1")
    if any(b >= a for a, b in zip(values, values[1:])):
        raise ValueError("star thresholds go from the largest p down: 0.05, 0.01, 0.001")
    return values


def bracket_side(population, n_pop: int) -> str:
    """Where a bracket for ``population`` (1 = slowest) of an ``n_pop`` fit stands: "below"
    the slowest population's bubbles (the bottom of the chart), "above" every other's (the
    fastest at the top). The share below the D threshold stands below too."""
    if population == "below":
        return "below"
    return "below" if int(population) == 1 and int(n_pop) > 1 else "above"


#: The mixed model's results by their data (key: population and the per-cell rows), so the
#: dialog's every redraw and the export's two passes cost one fit. Bounded.
_MIXED: dict = {}
_MIXED_LOCK = threading.Lock()
_MIXED_KEEP = 64


def offered(kind: str) -> bool:
    """Whether ``kind``'s Export dialog shows a Significance row at all."""
    return kind in OFFERED


def _data(payload):
    return payload.get("data") if isinstance(payload, dict) else None


def _hmm_data(payload):
    """The HMM comparison's payload (`hmm_view.comparison_payload`): the chart's ``"data"``, or the export's payload
    itself; None where there is none."""
    if not isinstance(payload, dict):
        return None
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    return data if isinstance(data, dict) and data.get("panels") else None


def _hmm_rows(data) -> list:
    return list(data["panels"][0].get("rows") or []) if data and data.get("panels") else []


def _column_names(payload) -> list:
    """The columns a chart draws, in its order: the bubbles' conditions, or the HMM comparison's."""
    hmm = _hmm_data(payload)
    if hmm is not None:
        return [str(r["key"]) for r in _hmm_rows(hmm)]
    return [str(c) for c in getattr(_data(payload), "conditions", [])]


def conditions(kind: str, payload) -> list:
    """The conditions a pair can be picked from: the figure's own, in its order."""
    return _column_names(payload) if offered(kind) else []


def unavailable(kind: str, payload, test: str = "welch") -> str:
    """Why this figure can carry no significance bracket at all, or "" when it can."""
    if not offered(kind):
        return "significance brackets are offered on the bubble charts and the HMM comparison only"
    if kind == HMM_KIND:
        return _hmm_unavailable(payload, test)
    if test not in TESTS:
        return f"the test must be one of {', '.join(TESTS)}"
    if test == "paired" and not declared_pairs(payload):
        return ("no replicates are declared as pairs: tick Pair replicates on the Groups board when the project "
                "opens and give each pair one tag (Welch on replicates needs no pairs)")
    if test in CELL_TESTS and kind not in MIXED_OFFERED:
        return (f"the {TEST_LABELS[test].lower()} is offered on the bubble charts that keep each cell's fractions "
                f"(SQD and Apparent Diffusion); use Welch on replicates here")
    data = _data(payload)
    if data is None:
        return "there is no fit to compare yet"
    if getattr(data, "series_labels", None):
        return ("several fit arrangements are overlaid, so each column carries several "
                "means; choose one arrangement to compare conditions")
    mode = getattr(data, "uncertainty", "")
    if mode != "day":
        shown = {"none": "the pooled fit (Uncertainty: None)",
                 "cell": "the pooled fit with a cell interval (Uncertainty: Cells)"}.get(
                     mode, "a fit with no per-replicate values behind it")
        which = "Apparent Diffusion" if kind == "appd_bubbles" else "SQD"
        return (f"the bubbles show {shown}, so there is no per-replicate sample to test; run the "
                f"{which} with Uncertainty = Replicates for a p (none is computed from pooled "
                f"tracks or cells)")
    values = getattr(data, "unit_values", None) or {}
    if not any(_days(entry) >= 2 for entry in values.values()):
        return ("no condition has two or more replicates: a p needs a sample of replicates "
                "on each side (one from cells or tracks would be pseudoreplication)")
    if test == "paired" and getattr(data, "pseudo", None):
        return ("the replicates here are random chunks of one culture (Uncertainty: Chunks): chunks are never pairs; "
                "run with Uncertainty = Replicates for the paired test")
    if test in CELL_TESTS and getattr(data, "pseudo", None):
        return (f"the replicates here are random chunks of one culture (Uncertainty: Chunks); the "
                f"{TEST_LABELS[test].lower()} reads real replicates. Welch or the equivalence test compare the "
                f"chunks, or run the {'Apparent Diffusion' if kind == 'appd_bubbles' else 'SQD'} with "
                f"Uncertainty = Replicates")
    if test in CELL_TESTS and not any(entry.get("cells") for entry in values.values()):
        from .workspace_view import setting_words

        return (f"no condition has per-cell fractions: the {TEST_LABELS[test].lower()} needs cell outlines and "
                f"a D shared by a condition's replicates ({setting_words('condition', 'global')})")
    return ""


def _days(entry: dict) -> int:
    return len(entry.get("fractions") or []) if entry.get("unit") in ("day", "chunk") else 0


def population_choices(kind: str, payload) -> tuple:
    """``((population, words), ...)``: whose fraction a bracket compares. 1 is the slowest;
    "all" compares every population (offered when there are two or more). On the HMM comparison: which PANEL, by its
    key ("occupancy", "rate_in", "rate_out", "D", and "dwell" where drawn), or "all"."""
    if kind == HMM_KIND:
        data = _hmm_data(payload)
        panels = [(p["key"], HMM_PANEL_WORDS.get(p["key"], str(p["title"]))) for p in (data or {}).get("panels") or ()]
        return tuple(panels) + ((("all", "All"),) if len(panels) > 1 else ())
    data = _data(payload)
    values = getattr(data, "unit_values", None) or {}
    counts = [len(entry["fractions"][0]) for entry in values.values()
              if entry.get("fractions")]
    most = max(counts) if counts else 0
    choices = []
    for k in range(1, most + 1):
        words = str(k)
        if k == 1:
            words = "1 (slowest)"
        elif k == most and len(set(counts)) == 1:
            words = f"{k} (fastest)"
        choices.append((k, words))
    if most >= 2:
        choices.append(("all", "all"))
    threshold = getattr(data, "threshold_um2_s", None)
    if threshold is not None and any(entry.get("below") for entry in values.values()):
        choices.append(("below", f"D < {threshold:g}"))
    return tuple(choices)


def p_text(p: float) -> str:
    """The p a bracket prints: exact, to two significant figures, "p < 0.001" below that."""
    if p is None or not math.isfinite(p):
        raise ValueError("no p to print: a bracket is drawn only for a pair with a test")
    if p < 0.001:
        return "p < 0.001"
    if p >= 0.995:
        return "p = 1.0" if p < 1.0 else "p = 1"
    digits = max(2, 1 - math.floor(math.log10(p)))
    return f"p = {round(p, digits):.{digits}f}"


def add_pair(pairs, a: str, b: str) -> tuple:
    """``pairs`` with ``(a, b)`` added; ValueError naming why it cannot be (the same
    condition twice, or a pair already listed either way round)."""
    a, b = str(a), str(b)
    if a == b:
        raise ValueError("pick two different conditions")
    for x, y in pairs:
        if {x, y} == {a, b}:
            raise ValueError(f"{a} and {b} are already compared")
    return tuple(tuple(p) for p in pairs) + ((a, b),)


def without_pair(pairs, index: int) -> tuple:
    """``pairs`` without the one at ``index``."""
    return tuple(tuple(p) for i, p in enumerate(pairs) if i != index)


def _counted(n: int, word: str) -> str:
    """"1 state", "2 states": a count with its noun in the right number."""
    return f"{n} {word}" + ("" if int(n) == 1 else "s")


def _entry_reason(condition: str, values: dict, data, test: str = "welch") -> str:
    """Why ``condition`` has no per-day sample (and, for the mixed model, no cells), or ""."""
    names = [str(c) for c in getattr(data, "conditions", [])]
    if condition not in names:
        # FITS SIDE BY SIDE draw a condition as one column per model ("glucose (1-pop)", "glucose (2-pop)"): a pair
        # picked on the plain name before is still there, under other names (3 Nov 2026, reported: it said "is not
        # in this figure" beside its own columns)
        fits = getattr(data, "fits", None) or []
        columns = [name for name, fit in zip(names, fits) if str(fit.get("condition")) == condition]
        if columns:
            listed = ", ".join(columns[:-1]) + " and " + columns[-1] if len(columns) > 1 else columns[0]
            return f"{condition} is drawn as {listed} now: pick the pair again from these columns"
        return f"{condition} is not in this figure"
    errors = getattr(data, "errors", {}) or {}
    column = names.index(condition)
    if column in errors:
        return f"{condition} has no fit ({errors[column]})"
    from .nested import MIN_REPLICATES_RICH

    entry = values.get(condition)
    if entry is None:
        # CHUNKS WITHHELD on a condition with replicates (`comparison.chunks_withheld`): its bubble is the pooled fit
        from .bubble_view import column_condition

        held = ((getattr(data, "pseudo", None) or {}).get("withheld") or {}).get(column_condition(data, column))
        if held:
            return f"{condition}: {held[0].lower()}{held[1:]}"
        return f"{condition} has no per-replicate values"
    n = len(entry.get("fractions") or [])
    # THE BAYESIAN MODEL TAKES ONE REPLICATE ON A SIDE: its replicate SD is learnt from the other side and the
    # prior (`nested.bayes_test`, which also wants three in all and says so). Every other test needs a sample.
    if test == "bayes":
        if n < 1:
            return f"{condition} has no per-replicate values"
    elif entry.get("unit") not in ("day", "chunk") or n < 2:
        return (f"{condition} was imaged in one session: one value, not a sample of replicates "
                f"(a p from its cells or tracks would be pseudoreplication)")
    if test in ("meta", "hboot") and n < MIN_REPLICATES_RICH:
        return (f"{condition} has {n} replicates; the {TEST_LABELS[test].lower()} is offered from "
                f"{MIN_REPLICATES_RICH} a side (with fewer it has at most one degree of freedom or a handful of "
                f"distinct resamples)")
    if test in CELL_TESTS and not entry.get("cells"):
        return f"{condition} has no per-cell fractions (no cell outlines, or no shared D)"
    return ""


def _shared_d_of(name: str, data) -> str:
    """The Shared D column ``name`` was fitted with, where the figure's conditions differ in it
    (`bubble_view.mixed_scopes`), else "" -- one Shared D for every condition refuses nothing."""
    from .bubble_view import column_condition, mixed_scopes

    names = [str(c) for c in getattr(data, "conditions", [])]
    if name not in names:
        return ""
    return str(mixed_scopes(data).get(column_condition(data, names.index(name)), ""))


def compare(kind: str, payload, pairs, population=1, test: str = "welch", *,
            compute: bool = True, margin_pp: float = DEFAULT_MARGIN_PP,
            correction: str = DEFAULT_CORRECTION) -> list:
    """One `Comparison` per pair and population, in the order picked, adjusted over the
    distinct tests (``population="all"``: every population of each pair) by ``correction``
    (`CORRECTIONS`: Holm, the standard, or Benjamini-Hochberg; an unknown one is refused).

    Raises ValueError when the figure can carry no bracket at all (`unavailable`) or the
    population is not a whole number from 1 or "all"; a single pair that cannot be tested
    comes back with ``test`` None and its ``reason``. Under a cell test (`CELL_TESTS`) with
    ``compute`` False, a pair whose p is not cached yet comes back ``pending`` (`status_lines`
    then holds Holm back: the family is every test shown). A two-population fit's second
    fraction is one minus the first: its test is the first's, counted once. ``margin_pp`` is
    the equivalence test's margin, in percentage points."""
    from .uncertainty import adjust

    correction_name(correction)                                   # refuses an unknown one before anything is run
    if kind == HMM_KIND:
        return _compare_hmm(payload, pairs, population, test, correction)
    why = unavailable(kind, payload, test)
    if why:
        raise ValueError(why)
    if test in MARGIN_TESTS and not (isinstance(margin_pp, (int, float)) and math.isfinite(margin_pp)
                                     and 0 < margin_pp < 100):
        raise ValueError("the margin is a number of percentage points above 0 and below 100")
    every = population == "all"
    below = population == "below"
    if below:
        threshold = getattr(_data(payload), "threshold_um2_s", None)
        if threshold is None:
            raise ValueError("this chart has no D threshold to compare the share below")
    elif not every:
        if isinstance(population, bool) or not isinstance(population, (int, float)) \
                or int(population) != population or population < 1:
            raise ValueError("the population compared is a whole number from 1, all, or below")
        population = int(population)
    data = _data(payload)
    values = getattr(data, "unit_values", None) or {}
    declared = declared_pairs(payload) if test == "paired" else {}

    def tags_of(name: str) -> dict:
        """The declared tags of the condition column ``name`` draws (`bubble_view.column_condition`)."""
        from .bubble_view import column_condition

        names = [str(c) for c in getattr(data, "conditions", [])]
        return declared.get(column_condition(data, names.index(name)), {}) if name in names else {}

    # FITS SIDE BY SIDE (`bubble_view.side_by_side`): a column is one condition's fit at one count
    fit_of = dict(zip((str(c) for c in getattr(data, "conditions", [])), getattr(data, "fits", None) or []))
    # BEST ONCE (`bubble_view.fold_best`, 1 Nov 2026): a pair naming a count's column that Best's column now stands
    # for (the same fit) is that column's pair, and a pair it then repeats is tested once
    from .bubble_view import folded_names

    folded = folded_names(data)
    found, seen = [], set()
    for a, b in pairs:
        a, b = folded.get(str(a), str(a)), folded.get(str(b), str(b))
        if a == b and str(a) in folded.values():
            row = Comparison(a=a, b=b, population=1 if every else population, days_a=[], days_b=[], values_a=[],
                             values_b=[], kind=test)
            row.reason = f"{a} is one column: Best chose the count also picked, drawn once"
            found.append(row)
            continue
        if frozenset((a, b)) in seen:
            continue
        seen.add(frozenset((a, b)))
        reasons = [r for r in (_entry_reason(a, values, data, test),
                               _entry_reason(b, values, data, test)) if r]
        if a in fit_of and b in fit_of and fit_of[a]["condition"] == fit_of[b]["condition"]:
            # ONE SET OF REPLICATES FITTED TWICE is not two samples: no test of any kind applies (Rule 0's mirror)
            reasons = [f"{a} and {b} are two fits of the same replicates, not two samples: compare conditions at "
                       f"one count"]
        scope_a, scope_b = _shared_d_of(a, data), _shared_d_of(b, data)
        if scope_a and scope_b and scope_a != scope_b:
            # A SHARED D PER CONDITION (1 Nov 2026): one fraction at its condition's shared D, the other the mean of
            # each replicate's own fit -- two methods, so a difference between them is partly the method's
            from .bubble_view import scope_words

            reasons = [f"{a} and {b} were fitted with their D forced differently ({a}: {scope_words(scope_a)}; "
                       f"{b}: {scope_words(scope_b)}): their difference would include the method. Compare "
                       f"conditions fitted alike"] + reasons
        na = nb = 0
        if not reasons and below:
            ea, eb = values[a], values[b]
            reasons = [f"{name} has no share below the D threshold" for name, e
                       in ((a, ea), (b, eb)) if not e.get("below")]
            if not reasons:
                found.append(_below_row(a, b, ea, eb, threshold, test, compute, margin_pp,
                                        tags=(tags_of(a), tags_of(b))))
                continue
        if not reasons:
            ea, eb = values[a], values[b]
            na, nb = len(ea["fractions"][0]), len(eb["fractions"][0])
            if na != nb:
                reasons.append(
                    f"{a} is fitted with {_counted(na, 'population')} and {b} with {nb}: 'x % of "
                    f"{_counted(na, 'state')}' and 'x % of {_counted(nb, 'state')}' are different quantities "
                    f"(compare the share below the D threshold instead)")
            elif not every and population > na:
                reasons.append(f"the fits have {_counted(na, 'population')}, not {population}")
        if reasons:
            row = Comparison(a=a, b=b, population=1 if every else population, days_a=[],
                             days_b=[], values_a=[], values_b=[], kind=test)
            row.reason = "; ".join(reasons)
            found.append(row)
            continue
        for k in (range(1, na + 1) if every else (population,)):
            row = Comparison(a=a, b=b, population=k, days_a=[], days_b=[], values_a=[],
                             values_b=[], kind=test)
            row.n_pop = na
            row.days_a, row.days_b = list(ea["replicates"]), list(eb["replicates"])
            row.values_a = [float(day[k - 1]) for day in ea["fractions"]]
            row.values_b = [float(day[k - 1]) for day in eb["fractions"]]
            row.d_a = _d(ea, k)
            row.d_b = _d(eb, k)
            # Two populations: the second fraction is one minus the first, so its test IS
            # the first's -- mirrored, never simulated again.
            _run_row(row, ea, eb, test, compute, margin_pp, own=1 if na == 2 else k,
                     tags=(tags_of(a), tags_of(b)))
            _split_ps(row, ea, eb, test, margin_pp, lambda split, k=k: [float(c[k - 1]) for c in split])
            found.append(row)
    # THE CORRECTION OVER THE DISTINCT TESTS: a two-population fit's second fraction is the first test
    # again, drawn at both bubbles but one member of the family.
    keys = [(r.a, r.b, 1 if getattr(r, "n_pop", 0) == 2 else r.population) for r in found]
    first = {}
    for key, r in zip(keys, found):
        if r.tested and key not in first:
            first[key] = r.test.p
    distinct = list(first)
    adjusted = dict(zip(distinct, adjust([first[k] for k in distinct], correction)))
    for key, row in zip(keys, found):
        row.correction = correction
        row.p_holm = adjusted.get(key, float("nan")) if row.tested else float("nan")
        row.family = len(distinct) if row.tested else 0
    return found


#: The HMM comparison's panels as the Significance block's pills say them.
HMM_PANEL_WORDS = {"occupancy": "Occupancy", "rate_in": "Rate in", "rate_out": "Rate out", "dwell": "Dwell", "D": "D"}


def whose_words(kind: str, population) -> str:
    """What a bracket compares, in the Export dialog's words: "population 1", "the share below the D threshold", or on
    the HMM comparison "the occupancy panel" / "every panel"."""
    if kind == HMM_KIND:
        return "every panel" if population == "all" else \
            f"the {HMM_PANEL_WORDS.get(population, str(population)).lower()} panel"
    if population == "below":
        return "the share below the D threshold"
    return "every population" if population == "all" else f"population {population}"


def _hmm_unavailable(payload, test: str) -> str:
    """Why the HMM comparison can carry no bracket at all under ``test``, or ""."""
    if test not in HMM_TESTS:
        return (f"the {TEST_LABELS.get(test, test).lower()} is offered on the SQD and Apparent Diffusion bubbles; the "
                f"HMM comparison offers Welch on replicates and, with declared pairs, the paired test")
    if test == "paired" and not declared_pairs(payload):
        return ("no replicates are declared as pairs: tick Pair replicates on the Groups board when the project "
                "opens and give each pair one tag (Welch on replicates needs no pairs)")
    data = _hmm_data(payload)
    if data is None:
        return "there is no HMM fit to compare yet"
    rows = [r for p in data["panels"] for r in p.get("rows") or ()]
    if any(r.get("days") and (r.get("unit") or ("replicate",))[0] == "chunk" for r in rows):
        return ("the open circles here are random chunks of one culture (Uncertainty: Chunks), one deal of them: no p "
                "between conditions is drawn on the HMM from chunks; run the HMM with Uncertainty = Replicates")
    if not any(len(r.get("days") or ()) >= 2 for r in rows):
        return ("no condition has per-replicate fits here (a pooled fit, or a movie or cell interval), so there is no "
                "sample of replicates to test; run the HMM with Uncertainty = Replicates for a p")
    return ""


def _compare_hmm(payload, pairs, population, test: str, correction: str = DEFAULT_CORRECTION) -> list:
    """`compare` on the HMM comparison: one `Comparison` per pair and panel, on each condition's per-replicate values
    as drawn (occupancy in %, rates per second, D in µm²/s), Welch or the paired test (`_run_row`, the bubbles' own),
    the correction (Holm unless chosen otherwise) over every bracket tested."""
    from .uncertainty import adjust

    why = _hmm_unavailable(payload, test)
    if why:
        raise ValueError(why)
    data = _hmm_data(payload)
    panels = {p["key"]: p for p in data["panels"]}
    if population == "all":
        keys = list(panels)
    elif population in panels:
        keys = [population]
    else:
        raise ValueError(f"the panel compared is one of {', '.join(panels)}, or all")
    declared = declared_pairs(payload) if test == "paired" else {}
    found, seen = [], set()
    for a, b in pairs:
        a, b = str(a), str(b)
        if frozenset((a, b)) in seen:
            continue
        seen.add(frozenset((a, b)))
        for key in keys:
            panel = panels[key]
            rows = {str(r["key"]): r for r in panel.get("rows") or ()}
            row = Comparison(a=a, b=b, population=key, days_a=[], days_b=[], values_a=[], values_b=[], kind=test)
            row.what = f"{panel['title']} ({panel['unit']})"
            reasons = []
            for name in (a, b):
                r = rows.get(name)
                if r is None:
                    reasons.append(f"{name} is not in the comparison")
                elif r.get("missing"):
                    reasons.append(f"{name} has {r['missing']}")
                elif len(r.get("days") or ()) < 2:
                    reasons.append(f"{name} has fewer than two replicates with a value here")
            ra, rb = rows.get(a), rows.get(b)
            if not reasons and int(ra["n_states"]) != int(rb["n_states"]):
                reasons.append(f"{a} is fitted with {_counted(int(ra['n_states']), 'state')} and {b} with "
                               f"{int(rb['n_states'])}: a state of the same name is not the same quantity in both")
            if not reasons and key == "D" and (ra.get("held_D") or rb.get("held_D")):
                from .workspace_view import SHARED_D_NAME      # the setting's name, spelled once

                reasons.append(f"the D was held the same across replicates ({SHARED_D_NAME}), so it has no spread "
                               "between replicates to test")
            if reasons:
                row.reason = "; ".join(reasons)
                found.append(row)
                continue
            for side, r in (("a", ra), ("b", rb)):
                labels = [n if n is not None else lab
                          for n, lab in zip(r.get("day_numbers") or [None] * len(r["days"]), r.get("day_labels") or ())]
                setattr(row, f"days_{side}", labels)
                setattr(row, f"values_{side}", [float(v) for v in r["days"]])
            _run_row(row, {}, {}, test, False, DEFAULT_MARGIN_PP, tags=(declared.get(a, {}), declared.get(b, {})))
            found.append(row)
    keys_tested = [(r.a, r.b, r.population) for r in found if r.tested]
    adjusted = dict(zip(keys_tested, adjust([r.test.p for r in found if r.tested], correction)))
    for row in found:
        row.correction = correction
        row.p_holm = adjusted.get((row.a, row.b, row.population), float("nan")) if row.tested else float("nan")
        row.family = len(keys_tested) if row.tested else 0
    return found


def _split_ps(row: Comparison, ea: dict, eb: dict, test: str, margin_pp: float, values_of) -> None:
    """Under random chunks: the same test on every other deal of the chunks (deal s of A against deal s of B), and
    the range of raw p beside the one drawn. ``values_of(split)``: one deal's values."""
    from .uncertainty import equivalence_test, minimum_effect_test, welch_test

    row.chunks = ea.get("unit") == "chunk" or eb.get("unit") == "chunk"
    if not row.chunks or test not in REPLICATE_TESTS or not row.tested:
        return
    ps = [row.test.p]
    for split_a, split_b in zip(ea.get("splits") or [], eb.get("splits") or []):
        a, b = values_of(split_a), values_of(split_b)
        if test == "welch":
            t = welch_test(a, b)
        elif test == "met":
            t = minimum_effect_test(a, b, margin_pp / 100.0)
        else:
            t = equivalence_test(a, b, margin_pp / 100.0)
        if math.isfinite(t.p):
            ps.append(t.p)
    if len(ps) > 1:
        ordered = sorted(ps)
        row.split_p = (ordered[0], ordered[len(ordered) // 2], ordered[-1], len(ordered))


def floor_words(c: Comparison) -> str:
    """"an exact test on 3 v 3 replicates cannot go below 0.1" beside a `FLOOR_TESTS` p; "" for every other test, an
    untested pair and a design with one replicate on a side (no exact test there)."""
    if c.kind not in FLOOR_TESTS or not c.tested or not math.isfinite(c.floor):
        return ""
    return (f"an exact test on {len(c.values_a)} v {len(c.values_b)} replicates cannot go below "
            f"{c.floor:.2g}")


def _floor_sentence(tested, shown_name) -> str:
    """The legend's sentence on the floor of every `FLOOR_TESTS` pair, one per design: "" where there is none."""
    designs: dict = {}
    for c in tested:
        words = floor_words(c)
        if words:
            designs.setdefault(words, []).append(f"{shown_name(c.a)} vs {shown_name(c.b)}")
    if not designs:
        return ""
    bits = "; ".join(f"{words} ({', '.join(dict.fromkeys(pairs))})" for words, pairs in designs.items())
    return (f" Floor: {bits}. A p below that floor comes from the model and the cells and movies it resamples, not "
            f"from the number of replicates.")


def _floor(n_a: int, n_b: int) -> float:
    """The permutation floor of a design, or NaN where a side has one replicate (the Bayesian model's)."""
    from .aggregate import achievable_p

    return achievable_p([n_a, n_b])["smallest_p"] if min(n_a, n_b) >= 2 else float("nan")


def _replicate_key(label):
    """A replicate label as the declared pairs key it (the ``day`` number), or None for a label that is not one."""
    try:
        return int(label)
    except (TypeError, ValueError):
        return None


def _run_paired(row: Comparison, tags) -> None:
    """``row``'s paired test: the replicates of A and B tagged alike on the group board (`declared_pairs`), each pair
    one difference; a replicate without a partner is counted and left out. Never matched by replicate number."""
    from .uncertainty import paired_test

    tags_a, tags_b = tags
    of_b = {}
    for label, value in zip(row.days_b, row.values_b):
        tag = tags_b.get(_replicate_key(label))
        if tag is not None:
            of_b[tag] = (label, value)
    xs, ys, pairs = [], [], []
    for label, value in zip(row.days_a, row.values_a):
        tag = tags_a.get(_replicate_key(label))
        if tag is not None and tag in of_b:
            other, other_value = of_b[tag]
            pairs.append((tag, label, other))
            xs.append(float(value))
            ys.append(float(other_value))
    row.pairs = sorted(pairs, key=lambda p: str(p[0]))
    row.unpaired = (len(row.values_a) - len(pairs), len(row.values_b) - len(pairs))
    order = sorted(range(len(pairs)), key=lambda i: str(pairs[i][0]))
    row.test = paired_test([xs[i] for i in order], [ys[i] for i in order])
    row.reason = row.test.reason
    if len(pairs) < 2:
        row.reason = (f"{row.a} and {row.b} share {len(pairs)} declared pair{'' if len(pairs) == 1 else 's'} "
                      f"(tags on the Groups board); a paired test needs two or more")


def _run_row(row: Comparison, ea: dict, eb: dict, test: str, compute: bool, margin_pp: float,
             own=None, tags=({}, {})) -> None:
    """Fill ``row``'s test: Welch, the paired test (on ``tags``, the two conditions' declared pairs) or the
    equivalence test on its per-replicate values (instant), or a cell test read from the cache (computed here when
    ``compute``). ``own``: the population whose cell test this row reads (a two-population fit's second is the
    first's, mirrored)."""
    from .uncertainty import equivalence_test, minimum_effect_test, welch_test

    row.floor = _floor(len(row.values_a), len(row.values_b))
    if test == "paired":
        _run_paired(row, tags)
        return
    if test == "welch":
        row.test = welch_test(row.values_a, row.values_b)
        row.reason = row.test.reason
        return
    if test == "tost":
        row.margin = margin_pp / 100.0
        row.test = equivalence_test(row.values_a, row.values_b, row.margin)
        row.reason = row.test.reason
        return
    if test == "met":
        row.margin = margin_pp / 100.0
        row.test = minimum_effect_test(row.values_a, row.values_b, row.margin)
        row.reason = row.test.reason
        return
    row.cells_a, row.cells_b = list(ea.get("cells") or []), list(eb.get("cells") or [])
    if row.population == "below" and any("below" not in c for c in row.cells_a + row.cells_b):
        row.reason = "the cells carry no share below the D threshold: run the SQD fit again"
        return
    k = row.population if own is None else own
    row.key = _cell_key(row, k, test)
    result = _cell_result(row.key, compute)
    if result is None:
        row.pending = True
        row.reason = f"the {TEST_LABELS[test].lower()} is being computed"
        return
    if isinstance(result, str):
        row.reason = result
        return
    if k != row.population:
        result = _mirrored(result)
    if test == "mixed":
        row.mixed = result
        row.test = _MixedAsTest(result)
    else:
        row.nested = result
        row.test = result


def _mirrored(result):
    """A two-population test read for the second population: the means are one minus the first's, the difference
    and its interval change sign."""
    if hasattr(result, "ci_low"):
        return replace(result, mean_a=1.0 - result.mean_a, mean_b=1.0 - result.mean_b,
                       difference=-result.difference, ci_low=-result.ci_high, ci_high=-result.ci_low,
                       t=-result.t)
    return replace(result, mean_a=1.0 - result.mean_a, mean_b=1.0 - result.mean_b,
                   difference=-result.difference, t=-result.t)


def _cell_value(cell: dict, k) -> float:
    """One cell's value for population ``k``: its fraction, or its share below the D
    threshold (``k == "below"``)."""
    return float(cell["below"] if k == "below" else cell["fractions"][k - 1])


def _cell_key(row: Comparison, population, test: str = "mixed") -> tuple:
    """A cell test's data as a cache key: ``(test, population, cells)``, each cell ``(condition, replicate,
    movie, value)`` -- with its step count for the tests that weigh by it (meta-analysis, hierarchical bootstrap;
    the Bayesian model carries it unused) -- A first, so the difference is B minus A."""
    cells = []
    for name, rows in ((row.a, row.cells_a), (row.b, row.cells_b)):
        if test == "mixed":
            cells += [(name, int(c["day"]), str(c["movie"]), _cell_value(c, population)) for c in rows]
        else:
            cells += [(name, int(c["day"]), str(c["movie"]), _cell_value(c, population),
                       int(c.get("n_steps") or 1)) for c in rows]
    return (test, population, tuple(cells))


def _below_row(a, b, ea, eb, threshold, test, compute, margin_pp=DEFAULT_MARGIN_PP, tags=({}, {})) -> Comparison:
    """One pair compared on the share below the D threshold: per replicate for Welch and the equivalence test, per
    cell for the cell tests; the state counts may differ."""
    row = Comparison(a=a, b=b, population="below", days_a=list(ea["replicates"]),
                     days_b=list(eb["replicates"]), values_a=[float(v) for v in ea["below"]],
                     values_b=[float(v) for v in eb["below"]], kind=test)
    row.threshold = float(threshold)
    row.n_pop = max(len(ea["fractions"][0]), len(eb["fractions"][0]))
    _run_row(row, ea, eb, test, compute, margin_pp, tags=tags)
    row.chunks = ea.get("unit") == "chunk" or eb.get("unit") == "chunk"
    return row


def _cell_result(key: tuple, compute: bool):
    """The cached result of a cell test (a `mixedmodel.MixedTest`, a `nested.NestedTest`, or the refusal's words)
    for ``key``; computed here when ``compute``, else None while it is not cached."""
    with _MIXED_LOCK:
        if key in _MIXED:
            return _MIXED[key]
    if not compute:
        return None
    result = _run_cell_test(key)
    store_mixed({key: result})
    return result


def _run_cell_test(key: tuple):
    """One cell test, by the name its key carries; a refusal comes back as its words."""
    from . import mixedmodel, nested

    test, _population, cells = key
    run = {"mixed": mixedmodel.test, "bayes": nested.bayes_test, "meta": nested.meta_test,
           "hboot": nested.hierarchical_bootstrap}[test]
    try:
        return run(list(cells))
    except (mixedmodel.MixedModelError, nested.NestedError) as exc:
        return str(exc)


def pending_mixed(kind: str, payload, pairs, population=1, test: str = "mixed") -> list:
    """The cell-test fits ``pairs`` still need under ``test`` (cache keys for `compute_mixed`): empty when every
    one is cached, the test is not a cell test, or nothing can be tested."""
    if test not in CELL_TESTS:
        return []
    try:
        found = compare(kind, payload, pairs, population, test, compute=False)
    except ValueError:
        return []
    return list(dict.fromkeys(c.key for c in found if c.pending))


def compute_mixed(keys) -> dict:
    """``{key: result}`` for each of ``keys`` (`pending_mixed`): the test's result or the refusal's words. Pure, so
    it can run in a worker PROCESS; the caller hands the result to `store_mixed` in its own. Named for the first
    cell test; it runs every one (`CELL_TESTS`)."""
    return {key: _run_cell_test(key) for key in keys}


def store_mixed(results: dict) -> None:
    """Put `compute_mixed`'s results in this process's cache."""
    with _MIXED_LOCK:
        for key, value in results.items():
            while len(_MIXED) >= _MIXED_KEEP:
                _MIXED.pop(next(iter(_MIXED)))
            _MIXED[key] = value


def chart_view(kind: str, payload, choice: ViewChoice) -> dict:
    """What the workspace shows for ``choice`` on the chart drawing ``payload``:
    ``{"marks": [{"a", "b", "population", "side", "text"}], "lines": [...], "pending": [keys],
    "why": str}``.

    ``marks`` are the brackets to paint (column indices, low first; the population whose
    bubbles they stand at and on which side, `bracket_side`; the text, `bracket_text`), only
    for tests with a p; ``lines`` what the Significance block says under the pairs
    (`status_lines`); ``pending`` the cell-test fits still to compute; ``why`` the reason
    no bracket can be drawn at all ("" when they can). Never computes a cell test."""
    out = {"marks": [], "lines": [], "pending": [], "why": "", "warnings": [],
           "correction_name": correction_name(choice.correction)}
    why = unavailable(kind, payload, choice.test)
    if why:
        out["why"] = why
        return out
    if not choice.pairs:
        return out
    out["lines"] = status_lines(kind, payload, choice.pairs, choice.population, choice.test,
                                stars=choice.stars, margin_pp=choice.margin, correction=choice.correction)
    out["pending"] = pending_mixed(kind, payload, choice.pairs, choice.population, choice.test)
    if not choice.on:
        return out
    try:
        found = compare(kind, payload, choice.pairs, choice.population, choice.test, compute=False,
                        margin_pp=choice.margin, correction=choice.correction)
    except ValueError:
        return out                       # the lines already say why (a margin out of range)
    if any(c.pending for c in found):
        return out                       # the correction needs the whole family: draw when all are in
    out["marks"] = bracket_marks(found, payload, choice.label, choice.stars)
    out["warnings"] = bracket_warnings(found, _data(payload))
    return out


def bracket_warnings(found, data, shown_name=str) -> list:
    """A sentence per drawn bracket standing on a population the "Check the fit" rule flags (`bubble_view
    .fit_checks`) in either of its conditions: the bracket, its population and the flag's reason (ruled 4 Nov 2026:
    no mark on the bracket, a warning beside the chart and in the exported legend). Only tested brackets (the ones
    drawn) and a numbered population: the share below the D threshold sums every state slower than it, whatever the
    count, so a population split in two or a small one does not change what it compares. [] when nothing is
    flagged."""
    from .bubble_view import fit_checks

    if data is None:
        return []
    names = [str(c) for c in getattr(data, "conditions", [])]
    checks = [c for c in fit_checks(getattr(data, "points", None) or [], names) if c["style"] == 0]
    out = []
    for c in found:
        if not c.tested or not isinstance(c.population, int):
            continue
        for side in (c.a, c.b):
            if side not in names:
                continue
            column = names.index(side)
            for check in checks:
                if check["column"] != column or c.population not in check["populations"]:
                    continue
                line = (f"{shown_name(c.a)} vs {shown_name(c.b)}, population {c.population}: in {shown_name(side)} "
                        f"{check['reason']}, so the bracket's p compares a population the fit may not hold (fit one "
                        f"population fewer)")
                if line not in out:
                    out.append(line)
    return out


def bracket_marks(found, payload, label: str = "stars", thresholds=DEFAULT_STARS) -> list:
    """``[{"a", "b", "population", "side", "text"}]`` for the tested rows of ``found``: what
    the chart and the exported figure both draw."""
    names = _column_names(payload)
    marks = []
    for c in found:
        if c.tested:
            lo, hi = sorted((names.index(c.a), names.index(c.b)))
            mark = {"a": lo, "b": hi, "population": c.population,
                    "side": "above" if c.what else bracket_side(c.population, c.n_pop),
                    "text": bracket_text(c.p_holm, label, thresholds, c.kind)}
            if c.population == "below":
                mark["below_d"] = c.threshold
            marks.append(mark)
    return marks


def bracket_tiers(marks, label_half) -> list:
    """Each bracket's tier, 0 nearest the plot; the ONE rule for the screen and the export.

    ``marks``: dicts with ``a``, ``b`` (its columns, low first) and ``x0``, ``x1`` (where its
    legs stand, in column units); ``label_half(mark)``: half its p's width plus a margin, in
    column units. A bracket's footprint is its legs' span and its p's width; it sits one tier
    above every bracket already placed whose footprint it meets, shortest spans first, so a
    long bracket always stands over the short ones it covers and brackets that do not meet
    share a tier."""
    def footprint(mark):
        half = label_half(mark)
        mid = (mark["a"] + mark["b"]) / 2.0
        return min(mark["x0"], mid - half), max(mark["x1"], mid + half)

    feet = [footprint(m) for m in marks]
    order = sorted(range(len(marks)), key=lambda i: (abs(marks[i]["a"] - marks[i]["b"]),
                                                     min(marks[i]["a"], marks[i]["b"])))
    levels, placed = [0] * len(marks), []
    for i in order:
        x0, x1 = feet[i]
        under = [levels[j] for j in placed if x0 < feet[j][1] and feet[j][0] < x1]
        levels[i] = max(under) + 1 if under else 0
        placed.append(i)
    return levels


def _d(entry: dict, population: int):
    ds = entry.get("D") or []
    return float(ds[population - 1]) if len(ds) >= population else None


def _short(c: Comparison) -> str:
    """How a bracket's quantity is named after its pair: its panel on the HMM comparison, else its population."""
    return c.what or f"population {c.population}"


def _population_words(population, threshold=None) -> str:
    if population == "all":
        return "each population"
    if population == "below":
        return (f"the states below the D threshold ({threshold:g} µm²/s, summed: the same "
                f"quantity whatever the number of states fitted)")
    return "population 1 (the slowest)" if population == 1 else f"population {population}"


def _units_text(c: Comparison, name: str) -> str:
    """"3 replicates, 24 movies, 140 cells" for one side of a cell-test comparison."""
    m = c.mixed if c.mixed is not None else c.nested
    side = 0 if name == c.a else 1
    return (f"{m.n_days[side]} replicates, {m.n_movies[side]} movies, {m.n_cells[side]} cells")


def _cell_counts(tested, shown_name) -> str:
    units = {}
    for c in tested:
        units.setdefault(c.a, _units_text(c, c.a))
        units.setdefault(c.b, _units_text(c, c.b))
    return "; ".join(f"{shown_name(name)} {text}" for name, text in units.items())


def _replicate_counts(tested, shown_name) -> str:
    days = {}
    for c in tested:
        days.setdefault(c.a, len(c.values_a))
        days.setdefault(c.b, len(c.values_b))
    return ", ".join(f"{shown_name(name)} {n}" for name, n in days.items())


def _intervals(tested, shown_name, several) -> str:
    """"A vs B [-0.04, 0.12]; ...": each tested pair's interval of the difference, in fraction units."""
    return "; ".join(f"{shown_name(c.a)} vs {shown_name(c.b)}"
                     + (f", {_short(c)}" if several else "")
                     + f" [{c.test.ci_low:.3f}, {c.test.ci_high:.3f}]" for c in tested)


def _caveats(tested, shown_name) -> str:
    seen = []
    for c in tested:
        caveat = getattr(c.nested, "caveat", "") if c.nested is not None else ""
        if caveat and caveat not in seen:
            seen.append(caveat)
    return (" Caveat: " + "; ".join(seen) + ".") if seen else ""


def legend_sentence(comparisons, shown_name=str, label: str = "p",
                    thresholds=DEFAULT_STARS) -> str:
    """The drafted legend's sentence for the brackets: the test, the units per condition,
    the correction (and, for the cell tests, the spread they fitted), the key to the stars and
    every exact p when the brackets show stars -- and every pair with no p, named with why."""
    if not comparisons:
        return ""
    several = len({c.population for c in comparisons}) > 1
    population = "all" if several else comparisons[0].population
    threshold = comparisons[0].threshold
    kind = comparisons[0].kind
    tested = [c for c in comparisons if c.tested]
    margin_pp = 100.0 * (comparisons[0].margin or DEFAULT_MARGIN_PP / 100.0)
    correction = _correction_of(comparisons)
    named = correction_name(correction)
    parts = []
    if tested:
        family = tested[0].family
        holm_words = (f"{named}-adjusted over the {family} comparisons shown"
                      + (" (this holds the false discovery rate, the expected share of false positives among the "
                         "comparisons called different, not the chance of any false positive)."
                         if correction == "bh" else ".")
                      if family > 1 else f"from a single comparison ({named} leaves one p unchanged).")
        what = (f"Brackets compare the fraction in {_population_words(population, threshold)} between two "
                f"conditions: ")
        noun = "fractions"
        if comparisons[0].what:
            # THE HMM COMPARISON: a panel's quantity, in its unit as drawn
            noun = "values"
            what = ("Brackets compare one panel's quantity between two conditions, each bracket in the panel it "
                    "tests: " if several else
                    f"Brackets compare the {comparisons[0].what[0].lower()}{comparisons[0].what[1:]} between two "
                    f"conditions: ")
        if kind == "mixed":
            spread = "; ".join(
                f"{shown_name(c.a)} vs {shown_name(c.b)} {c.mixed.s_day:.3f}, "
                f"{c.mixed.s_movie:.3f}, {c.mixed.s_cell:.3f}" for c in tested)
            parts.append(
                what + f"a linear mixed model on each cell's fraction (cells within "
                f"movies within replicates, condition as the fixed effect, REML), with a "
                f"two-sided parametric-bootstrap p ({tested[0].mixed.n_boot} simulations); "
                f"each condition's replicates are its own (unpaired; "
                f"{_cell_counts(tested, shown_name)}). Spread fitted (SD between replicates, "
                f"movies and cells): {spread}. p values are " + holm_words)
        elif kind == "tost":
            parts.append(
                f"Brackets test whether the fraction in {_population_words(population, threshold)} is the same "
                f"in two conditions: two one-sided Welch tests (TOST) on the per-replicate fractions against an "
                f"equivalence margin of \u00b1{margin_pp:g} percentage points, set before the data were seen; "
                f"each condition's replicates one independent sample (unpaired; replicates per condition: "
                f"{_replicate_counts(tested, shown_name)}). 90 % intervals of the difference: "
                f"{_intervals(tested, shown_name, several)}. p (the larger one-sided p) values are " + holm_words
                + (" The replicates are random chunks of one culture (pseudo-replicates): this tests equivalence of "
                   "these cells, not of another culture." + _split_sentence(tested, shown_name)
                   if any(c.chunks for c in tested) else ""))
        elif kind == "met":
            parts.append(
                f"Brackets test whether the fraction in {_population_words(population, threshold)} differs between "
                f"two conditions by more than \u00b1{margin_pp:g} percentage points (a minimum-effect test: two "
                f"one-sided Welch tests against the margin, set before the data were seen, so a smaller difference "
                f"is never significant however much data there is) on the per-replicate fractions; each condition's "
                f"replicates one independent sample (unpaired; replicates per condition: "
                f"{_replicate_counts(tested, shown_name)}). p values are " + holm_words
                + (" The replicates are random chunks of one culture (pseudo-replicates): a difference beyond the "
                   "margin between these cells, not shown to hold for another culture." + _split_sentence(tested,
                                                                                                       shown_name)
                   if any(c.chunks for c in tested) else ""))
        elif kind == "bayes":
            d = tested[0].nested.details
            prior = ", ".join(f"{v:g}" for v in d["prior"])
            spread = "; ".join(
                f"{shown_name(c.a)} vs {shown_name(c.b)} {c.nested.details['s_day']:.3f}, "
                f"{c.nested.details['s_movie']:.3f}, {c.nested.details['s_cell']:.3f}" for c in tested)
            parts.append(
                what + f"a Bayesian hierarchical model on each cell's fraction (cells within movies within "
                f"replicates, a mean per condition; half-normal priors on the SDs, scales {prior}), "
                f"sampled in two Gibbs chains ({d['draws']} draws kept, R-hat "
                f"\u2264 {max(c.nested.details['rhat'] for c in tested):.3f}); each condition's replicates are its "
                f"own (unpaired; {_cell_counts(tested, shown_name)}). 95 % credible intervals of the difference: "
                f"{_intervals(tested, shown_name, several)}. SDs fitted (posterior medians, between replicates, "
                f"movies and cells): {spread}. p is the two-sided posterior tail probability, "
                f"2 min(P(difference < 0), P(difference > 0)); p values are " + holm_words
                + _caveats(tested, shown_name) + _floor_sentence(tested, shown_name))
        elif kind == "meta":
            taus = "; ".join(f"{shown_name(c.a)} vs {shown_name(c.b)} {c.nested.details['tau']:.3f}"
                             for c in tested)
            parts.append(
                what + "a random-effects meta-regression over replicates: each replicate's step-weighted "
                "fraction with its own variance (movies as clusters), the between-replicate variance by "
                "DerSimonian-Laird, condition as the moderator, and Knapp-Hartung's t on k - 2 degrees of freedom; "
                "each condition's replicates are its own (unpaired; "
                f"{_cell_counts(tested, shown_name)}). 95 % intervals of the difference: "
                f"{_intervals(tested, shown_name, several)}. Between-replicate SD (tau): {taus}. p values are "
                + holm_words + _floor_sentence(tested, shown_name))
        elif kind == "hboot":
            parts.append(
                what + f"a hierarchical bootstrap ({tested[0].nested.details['n_boot']} resamples: replicates "
                f"within each condition, then movies, then cells), the "
                f"step-weighted fraction of each resample; each condition's replicates are its own (unpaired; "
                f"{_cell_counts(tested, shown_name)}). 95 % percentile intervals of the difference: "
                f"{_intervals(tested, shown_name, several)}. Two-sided p from the share of resampled differences "
                f"on either side of zero; p values are " + holm_words + _caveats(tested, shown_name)
                + _floor_sentence(tested, shown_name))
        elif kind == "paired":
            parts.append(
                what + f"a paired t-test (two-sided) on the per-replicate {noun} of replicates declared as pairs "
                "on the Groups board, each pair one culture or imaging session shared by the two "
                "conditions (never matched by replicate number); the test is on the within-pair differences, "
                f"n - 1 degrees of freedom for n pairs ({_pairs_counts(tested, shown_name)}); p values are "
                + holm_words)
        elif any(c.chunks for c in tested):
            parts.append(
                what + f"Welch's t-test (two-sided, unequal variances) on the fractions of random chunks of each "
                f"condition's movies or cells (pseudo-replicates within one culture: the test asks whether these "
                f"cells differ, not whether another culture would; chunks per condition: "
                f"{_replicate_counts(tested, shown_name)}); p values are " + holm_words + _split_sentence(tested,
                                                                                         shown_name))
        else:
            parts.append(
                what + f"Welch's t-test (two-sided, unequal variances) on the "
                f"per-replicate {noun}, each condition's replicates one independent sample "
                f"(unpaired; replicates per condition: {_replicate_counts(tested, shown_name)}); p values are "
                + holm_words)
    if tested and label in ("stars", "both"):
        key = "Marks" if kind == "tost" else "Stars"
        parts.append(f"{key} on the {named}-adjusted p: {stars_sentence(thresholds, kind, margin_pp)}.")
    if tested and label == "stars":
        exact = "; ".join(
            f"{shown_name(c.a)} vs {shown_name(c.b)}"
            + (f", {_short(c)}" if several else "") + f", {p_text(c.p_holm)}"
            for c in tested)
        parts.append(f"Exact p ({named}-adjusted): {exact}.")
    for c in comparisons:
        if not c.tested:
            parts.append(f"No p for {shown_name(c.a)} against {shown_name(c.b)}: {c.reason}.")
    return " ".join(parts)


def _pairs_counts(tested, shown_name) -> str:
    """"Glucose vs Malate: pairs A, B, C; 1 replicate of Malate without a partner, left out" for each pair tested."""
    bits = []
    for c in tested:
        left = [f"{n} replicate{'' if n == 1 else 's'} of {shown_name(name)}"
                for name, n in ((c.a, c.unpaired[0]), (c.b, c.unpaired[1])) if n]
        bit = (f"{shown_name(c.a)} vs {shown_name(c.b)}: {len(c.pairs)} pairs, "
               + ", ".join(str(tag) for tag, _ra, _rb in c.pairs))
        if left:
            bit += "; " + " and ".join(left) + " without a partner, left out"
        if bit not in bits:
            bits.append(bit)
    return "; ".join(bits)


def _p_bare(p: float) -> str:
    """`p_text` without its "p = " ("0.012"; "< 0.001" below that)."""
    text = p_text(p)
    return text[4:] if text.startswith("p = ") else text[2:]


def _split_sentence(tested, shown_name) -> str:
    """" Over 20 deals of the chunks the raw p ran from 0.01 to 0.09 (median 0.03) for A vs B." -- or ""."""
    bits = [f"{_p_bare(c.split_p[0])} to {_p_bare(c.split_p[2])} (median {_p_bare(c.split_p[1])}) for "
            f"{shown_name(c.a)} vs {shown_name(c.b)}" for c in tested if c.split_p]
    if not bits:
        return ""
    n = next(c.split_p[3] for c in tested if c.split_p)
    return f" Over {n} random deals of the chunks the raw p ran from " + "; ".join(bits) + "."


def _fdr_words(correction: str) -> str:
    """What Benjamini-Hochberg holds, said once in the sheet's note; nothing under Holm (the sheets made before)."""
    return (" Benjamini-Hochberg holds the false discovery rate (the expected share of false positives among the "
            "pairs called different), not the chance of any false positive." if correction == "bh" else "")


def _correction_note(correction: str = DEFAULT_CORRECTION) -> str:
    named = correction_name(correction)
    return (f"Correction: {named}, over the pairs picked for this figure; 'p ({named})' is what the bracket "
            "prints. A pair with no test is not counted in the family and is named with why."
            + _fdr_words(correction))


def sheet_notes(test: str = "welch", kind: str = "", correction: str = DEFAULT_CORRECTION) -> tuple:
    """The header lines of the source data's Comparisons sheet."""
    from .comparison import MIN_CELL_STEPS

    note = _correction_note(correction)
    named = correction_name(correction)

    if kind == HMM_KIND:
        what = ("One row per bracket picked: one panel's quantity compared between two conditions, in the panel's "
                "unit as drawn (occupancy in per cent, rates per second, D in µm²/s); each value is one replicate's own "
                "HMM fit.")
        if test == "paired":
            return (what + " Difference = mean over pairs of (B - A), with its 95 % paired interval.",
                    f"Test: {PAIRED_TEST_NAME}, on n - 1 degrees of freedom for n pairs.", PAIRED_SENTENCE, note,
                    "The per-replicate values themselves, with each replicate's pair tag, are on the 'Per-replicate "
                    "values' sheet, as numbers.")
        return (what + " Difference = mean B - mean A, with its 95 % Welch interval.",
                f"Test: {TEST_NAME}. The unit is the replicate: one HMM fit per replicate, never a cell, track or "
                f"step.", RULE0_SENTENCE, note,
                "The per-replicate values themselves are on the 'Per-replicate values' sheet, as numbers.")

    cells_line = (f"Each cell's fraction is fitted with the condition's shared D held; a cell with fewer than "
                  f"{MIN_CELL_STEPS} steps is left out.")
    if test == "mixed":
        return (
            "One row per bracket picked: one population's fraction (0 to 1) compared between "
            "two conditions; Difference = model mean B - model mean A.",
            f"Test: {MIXED_TEST_NAME}. {cells_line} The "
            "effect is judged against the scatter of replicates and movies, not against the number "
            "of cells.",
            "The p is the share of simulated |t| at least the observed, (k + 1) / (B + 1), "
            "with data simulated B times from the fit without a condition effect; the smallest "
            "reportable p is 1 / (B + 1). 'Model SE' is REML's and is optimistic with few "
            "replicates, so no interval is given from it.",
            "SD replicate / movie / cell: the spread the model fitted at each level, in fraction "
            "units (the same for both conditions).",
            RULE0_SENTENCE,
            note,
            "The per-cell values themselves are on the 'Per-cell values' sheet, as numbers.")
    if test in ("bayes", "meta", "hboot"):
        from .nested import METHOD_NAMES, PRIOR_SCALES, RHAT_LIMIT

        how = {
            "bayes": (f"Half-normal priors on the SD between replicates, movies and cells (scales "
                      f"{', '.join(f'{v:g}' for v in PRIOR_SCALES)}); flat on each condition's mean. Two Gibbs "
                      f"chains; R-hat above {RHAT_LIMIT:g} is refused. 'Interval' is the 95 % credible "
                      f"interval; p is 2 min(P(difference < 0), P(difference > 0)). One replicate on a side "
                      f"is allowed."),
            "meta": ("Each replicate's value is the step-weighted mean of its cells' fractions, with its "
                     "variance over movies. Between-replicate variance: DerSimonian-Laird; interval and p: "
                     "Knapp-Hartung's t on (replicates - 2) degrees of freedom. From three replicates a side."),
            "hboot": ("Each resample draws replicates within each condition, then movies, then cells, and "
                      "takes the step-weighted fraction. 'Interval' is the 95 % percentile interval; "
                      "p = (2 min(k below 0, k above 0) + 1) / (B + 1). From three replicates a side "
                      "(three allow only ten distinct draws)."),
        }[test]
        return (
            "One row per bracket picked: one population's fraction (0 to 1) compared between two conditions; "
            "Difference = B - A.",
            f"Test: {METHOD_NAMES[test]}. {cells_line}",
            how,
            "Permutation floor: the smallest two-sided p an exact permutation test on the same replicate counts "
            "could give (3 against 3 replicates: 0.100). A p below it comes from the model and the cells and movies "
            "it resamples, not from the number of replicates; empty with one replicate on a side.",
            RULE0_SENTENCE,
            note,
            "The per-cell values themselves are on the 'Per-cell values' sheet, as numbers.")
    if test == "paired":
        return (
            "One row per bracket picked: one population's fraction (0 to 1) compared between two conditions on the "
            "replicates declared as pairs; Difference = mean over pairs of (B - A), with its 95 % paired interval.",
            f"Test: {PAIRED_TEST_NAME}, on n - 1 degrees of freedom for n pairs. The unit is the declared pair of "
            "replicates, never a cell or a track.",
            PAIRED_SENTENCE,
            note,
            "The per-replicate values themselves, with each replicate's pair tag, are on the 'Per-replicate values' "
            "sheet, as numbers.")
    if test == "met":
        return (
            "One row per bracket picked: whether one population's fraction (0 to 1) differs between two conditions "
            "by more than the margin; Difference = mean B - mean A, with its 95 % Welch interval.",
            "Test: a minimum-effect test, two one-sided Welch tests against plus and minus the margin on "
            "per-replicate values; 'p' is twice the smaller one-sided p (at a margin of 0, Welch's two-sided p). The "
            "margin is the user's, set before looking; a difference inside it is never significant.",
            RULE0_SENTENCE,
            note,
            "The per-replicate values themselves are on the 'Per-replicate values' sheet, as numbers.")
    if test == "tost":
        return (
            "One row per bracket picked: whether one population's fraction (0 to 1) is the same in two "
            "conditions within the margin; Difference = mean B - mean A, with its 90 % Welch interval (the "
            "interval two one-sided tests at 0.05 are equivalent to).",
            "Test: two one-sided Welch tests (TOST) on per-replicate values; 'p' is the larger of the two "
            "one-sided p values. Equivalence is shown where p is below the threshold, i.e. where the 90 % "
            "interval lies inside the margin. The margin is the user's, set before looking.",
            RULE0_SENTENCE,
            note,
            "The per-replicate values themselves are on the 'Per-replicate values' sheet, as numbers.")
    return (
        "One row per bracket picked: one population's fraction (0 to 1) compared between "
        "two conditions; Difference = mean B - mean A, with its 95 % Welch interval.",
        f"Test: {TEST_NAME}. The unit is the replicate: one value per replicate, never per "
        "cell or per track (pseudoreplication).",
        RULE0_SENTENCE,
        f"Correction: {named}, over the pairs picked for this figure; 'p ({named})' is "
        "what the bracket prints (exact, two significant figures; below 0.001 as "
        "p < 0.001). A pair with no test is not counted in the family and is named with why."
        + _fdr_words(correction),
        "Permutation floor: the smallest two-sided p an exact permutation test on the same "
        "replicate counts could give (3 against 3 replicates: 0.100). A Welch p below it rests on the "
        "t distribution, i.e. on per-replicate values that are roughly normal.",
        "The per-replicate values themselves are on the 'Per-replicate values' sheet, as numbers.")


def values_sheet(test: str = "welch", kind: str = "") -> tuple:
    """``(sheet name, title, notes)`` of the sheet holding the values a test used."""
    if kind == HMM_KIND:
        return ("Per-replicate values", "Per-replicate values behind the brackets",
                ("Each condition's value on each of its replicates, in the panel's unit as drawn: the values the "
                 + ("paired test used, with the pair tag each replicate was given on the Groups board."
                    if test == "paired" else "Welch test used, one per replicate."),
                 PAIRED_SENTENCE if test == "paired" else RULE0_SENTENCE))
    if test in CELL_TESTS:
        return ("Per-cell values", "Per-cell values behind the brackets",
                (f"Each cell's fraction (0 to 1), fitted with its condition's shared D held: "
                 f"the values the {TEST_LABELS[test].lower()} used, one per cell, with the replicate and movie "
                 f"that nest it.", RULE0_SENTENCE))
    if test == "paired":
        return ("Per-replicate values", "Per-replicate values behind the brackets",
                ("Each condition's fraction (0 to 1) on each of its replicates, with the pair tag it was given on "
                 "the Groups board; a replicate with no tag, or none shared with the other condition, was left out "
                 "of the paired test.", PAIRED_SENTENCE))
    return ("Per-replicate values", "Per-replicate values behind the brackets",
            (f"Each condition's fraction (0 to 1) on each of its replicates: the values "
             f"the {_values_test_words(test)} used, one per replicate.", RULE0_SENTENCE))


def _values_test_words(test: str) -> str:
    return {"tost": "equivalence test", "met": "minimum-effect test"}.get(test, "Welch test")


def _joined(values) -> str:
    return "; ".join(f"{v:.4g}" for v in values)


def _num(value):
    return float(value) if value is not None and math.isfinite(value) else None


def comparison_table(comparisons, shown_name=str, label: str = "p",
                     thresholds=DEFAULT_STARS) -> tuple:
    """``(columns, rows)`` of the Comparisons sheet: numbers as numbers, an untested pair a
    row with its reason and no statistic; the last column is what its bracket prints."""
    columns, rows = _tables(comparisons, shown_name)
    columns = columns + ["On the bracket"]
    for c, row in zip(comparisons, rows):
        row.append(bracket_text(c.p_holm, label, thresholds, c.kind) if c.tested else "")
    return columns, rows


def _tables(comparisons, shown_name) -> tuple:
    kind = comparisons[0].kind if comparisons else "welch"
    if kind == "mixed":
        return _mixed_table(comparisons, shown_name)
    if kind in ("bayes", "meta", "hboot"):
        return _nested_table(comparisons, shown_name)
    if kind == "tost":
        return _tost_table(comparisons, shown_name)
    if kind == "paired":
        return _paired_table(comparisons, shown_name)
    named = correction_name(_correction_of(comparisons))
    columns = ["Pair", _what_column(comparisons), "Condition A", "Shown as A", "Replicates A",
               "Per-replicate values A", "Mean A", "D A (µm²/s)", "Condition B", "Shown as B",
               "Replicates B", "Per-replicate values B", "Mean B", "D B (µm²/s)", "Difference (B - A)",
               "95 % CI low", "95 % CI high", "t", "df (Welch)", "p", f"p ({named})",
               f"{named} family", "n replicates A", "n replicates B", "Permutation floor", "Test", "Note"]
    rows = []
    for index, c in enumerate(comparisons, 1):
        test = c.test
        rows.append([
            index, c.what or c.population, c.a, shown_name(c.a), ", ".join(str(d) for d in c.days_a),
            _joined(c.values_a), _num(test.mean_a) if test else None, _num(c.d_a),
            c.b, shown_name(c.b), ", ".join(str(d) for d in c.days_b),
            _joined(c.values_b), _num(test.mean_b) if test else None, _num(c.d_b),
            _num(test.difference) if test else None,
            _num(test.ci_low) if test else None, _num(test.ci_high) if test else None,
            _num(test.t) if test else None, _num(test.df) if test else None,
            _num(test.p) if test else None, _num(c.p_holm),
            c.family if c.tested else None,
            test.n_a if test else None, test.n_b if test else None,
            _num(c.floor), _row_test_name(c) if c.tested else "", c.reason])
    return columns, rows


def _row_test_name(c) -> str:
    """The Test column of a Welch-shaped row: Welch's, or the minimum-effect test's with its margin."""
    if c.kind == "met":
        return f"{MET_TEST_NAME} (\u00b1{100.0 * (c.margin or DEFAULT_MARGIN_PP / 100.0):g} percentage points)"
    return TEST_NAME


def _what_column(comparisons) -> str:
    """The Comparisons sheet's second column: the population a bubble bracket compares, the panel an HMM one does."""
    return "Panel" if comparisons and comparisons[0].what else "Population"


def _paired_table(comparisons, shown_name) -> tuple:
    named = correction_name(_correction_of(comparisons))
    columns = ["Pair", _what_column(comparisons), "Condition A", "Shown as A", "Condition B", "Shown as B",
               "Declared pairs (tag: replicate A / replicate B)", "Values A (in pair order)",
               "Values B (in pair order)", "Mean A", "Mean B", "Mean difference (B - A)", "95 % CI low",
               "95 % CI high", "t", "df (pairs - 1)", "p", f"p ({named})", f"{named} family", "n pairs",
               "Replicates of A without a partner", "Replicates of B without a partner", "Test", "Note"]
    rows = []
    for index, c in enumerate(comparisons, 1):
        test = c.test
        value_a = dict(zip(c.days_a, c.values_a))
        value_b = dict(zip(c.days_b, c.values_b))
        rows.append([
            index, c.what or c.population, c.a, shown_name(c.a), c.b, shown_name(c.b),
            "; ".join(f"{tag}: {ra} / {rb}" for tag, ra, rb in c.pairs),
            _joined(value_a[ra] for _tag, ra, _rb in c.pairs), _joined(value_b[rb] for _tag, _ra, rb in c.pairs),
            _num(test.mean_a) if test else None, _num(test.mean_b) if test else None,
            _num(test.difference) if test else None,
            _num(test.ci_low) if test else None, _num(test.ci_high) if test else None,
            _num(test.t) if test else None, _num(test.df) if test else None,
            _num(test.p) if test else None, _num(c.p_holm), c.family if c.tested else None,
            len(c.pairs), c.unpaired[0], c.unpaired[1], PAIRED_TEST_NAME if c.tested else "", c.reason])
    return columns, rows


def _tost_table(comparisons, shown_name) -> tuple:
    named = correction_name(_correction_of(comparisons))
    columns = ["Pair", "Population", "Condition A", "Shown as A", "Replicates A", "Per-replicate values A",
               "Mean A", "D A (µm²/s)", "Condition B", "Shown as B", "Replicates B", "Per-replicate values B",
               "Mean B", "D B (µm²/s)", "Difference (B - A)", "Margin (+-)", "90 % CI low", "90 % CI high",
               "df (Welch)", "p lower", "p upper", "p (TOST)", f"p ({named})", f"{named} family", "n replicates A",
               "n replicates B", "Test", "Note"]
    rows = []
    for index, c in enumerate(comparisons, 1):
        test = c.test
        rows.append([
            index, c.population, c.a, shown_name(c.a), ", ".join(str(d) for d in c.days_a),
            _joined(c.values_a), _num(test.mean_a) if test else None, _num(c.d_a),
            c.b, shown_name(c.b), ", ".join(str(d) for d in c.days_b),
            _joined(c.values_b), _num(test.mean_b) if test else None, _num(c.d_b),
            _num(test.difference) if test else None, _num(c.margin),
            _num(test.ci_low) if test else None, _num(test.ci_high) if test else None,
            _num(test.df) if test else None, _num(test.p_lower) if test else None,
            _num(test.p_upper) if test else None, _num(test.p) if test else None, _num(c.p_holm),
            c.family if c.tested else None, test.n_a if test else None, test.n_b if test else None,
            "two one-sided Welch tests (TOST) on per-replicate values" if c.tested else "", c.reason])
    return columns, rows


def _nested_table(comparisons, shown_name) -> tuple:
    from .nested import METHOD_NAMES

    named = correction_name(_correction_of(comparisons))
    columns = ["Pair", "Population", "Condition A", "Shown as A", "Replicates A", "Movies A", "Cells A",
               "Mean A", "D A (µm²/s)", "Condition B", "Shown as B", "Replicates B", "Movies B", "Cells B",
               "Mean B", "D B (µm²/s)", "Difference (B - A)", "Interval low", "Interval high", "p",
               f"p ({named})", f"{named} family", "SD replicate", "SD movie", "SD cell", "Between-replicate SD (tau)",
               "df", "Draws or resamples", "R-hat", "Permutation floor", "Caveat", "Test", "Note"]
    rows = []
    for index, c in enumerate(comparisons, 1):
        m = c.nested
        d = m.details if m is not None else {}
        rows.append([
            index, c.population, c.a, shown_name(c.a), m.n_days[0] if m else None,
            m.n_movies[0] if m else None, m.n_cells[0] if m else None, _num(m.mean_a) if m else None,
            _num(c.d_a), c.b, shown_name(c.b), m.n_days[1] if m else None, m.n_movies[1] if m else None,
            m.n_cells[1] if m else None, _num(m.mean_b) if m else None, _num(c.d_b),
            _num(m.difference) if m else None, _num(m.ci_low) if m else None, _num(m.ci_high) if m else None,
            _num(m.p) if m else None, _num(c.p_holm), c.family if c.tested else None,
            _num(d.get("s_day")), _num(d.get("s_movie")), _num(d.get("s_cell")), _num(d.get("tau")),
            _num(m.df) if m else None, d.get("draws") or d.get("n_boot"), _num(d.get("rhat")), _num(c.floor),
            m.caveat if m else "", METHOD_NAMES[c.kind] if c.tested else "", c.reason])
    return columns, rows


def _mixed_table(comparisons, shown_name) -> tuple:
    named = correction_name(_correction_of(comparisons))
    columns = ["Pair", "Population", "Condition A", "Shown as A", "Replicates A", "Movies A",
               "Cells A", "Model mean A", "D A (µm²/s)", "Condition B", "Shown as B", "Replicates B",
               "Movies B", "Cells B", "Model mean B", "D B (µm²/s)", "Difference (B - A)",
               "Model SE", "t", "p (bootstrap)", "Simulations", f"p ({named})", f"{named} family",
               "SD replicate", "SD movie", "SD cell", "Test", "Note"]
    rows = []
    for index, c in enumerate(comparisons, 1):
        m = c.mixed
        rows.append([
            index, c.population, c.a, shown_name(c.a), m.n_days[0] if m else None,
            m.n_movies[0] if m else None, m.n_cells[0] if m else None,
            _num(m.mean_a) if m else None, _num(c.d_a), c.b, shown_name(c.b),
            m.n_days[1] if m else None, m.n_movies[1] if m else None,
            m.n_cells[1] if m else None, _num(m.mean_b) if m else None, _num(c.d_b),
            _num(m.difference) if m else None, _num(m.se) if m else None,
            _num(m.t) if m else None, _num(m.p) if m else None, m.n_boot if m else None,
            _num(c.p_holm), c.family if c.tested else None,
            _num(m.s_day) if m else None, _num(m.s_movie) if m else None,
            _num(m.s_cell) if m else None, MIXED_TEST_NAME if c.tested else "", c.reason])
    return columns, rows


def values_table(comparisons, shown_name=str) -> tuple:
    """``(columns, rows)``: every value a bracket was computed from, as a number, each
    condition and population once (per replicate for Welch and TOST, per cell for the cell tests)."""
    per_cell = bool(comparisons) and comparisons[0].kind in CELL_TESTS
    paired = bool(comparisons) and comparisons[0].kind == "paired"
    if per_cell:
        columns = ["Condition", "Shown as", "Population", "Replicate", "Movie", "Cell",
                   "Steps", "Fraction"]
    elif comparisons and comparisons[0].what:
        columns = ["Condition", "Shown as", "Panel", "Replicate", "Value"] + (["Pair tag"] if paired else [])
    else:
        columns = ["Condition", "Shown as", "Population", "Replicate", "Fraction"] + (["Pair tag"] if paired else [])
    rows, seen = [], set()
    for c in comparisons:
        if per_cell:
            sides = ((c.a, c.cells_a), (c.b, c.cells_b))
            for name, cells in sides:
                if (name, c.population) in seen or not cells or not c.tested:
                    continue
                seen.add((name, c.population))
                rows += [[name, shown_name(name), c.population, int(cell["day"]),
                          str(cell["movie"]), str(cell["cell"]), int(cell["n_steps"]),
                          _cell_value(cell, c.population)] for cell in cells]
            continue
        tag_of = {(c.a, ra): tag for tag, ra, _rb in c.pairs}
        tag_of.update({(c.b, rb): tag for tag, _ra, rb in c.pairs})
        for name, days, values in ((c.a, c.days_a, c.values_a), (c.b, c.days_b, c.values_b)):
            if (name, c.population) in seen or not values:
                continue
            seen.add((name, c.population))
            rows += [[name, shown_name(name), c.what or c.population, day, float(v)]
                     + ([tag_of.get((name, day), "")] if paired else [])
                     for day, v in zip(days, values)]
    return columns, rows


def status_lines(kind: str, payload, pairs, population=1, test: str = "welch", *,
                 stars=DEFAULT_STARS, margin_pp: float = DEFAULT_MARGIN_PP,
                 correction: str = DEFAULT_CORRECTION) -> list:
    """What the Significance block and the Export dialog say under the pairs: each test's
    stars and p (adjusted by ``correction``, and raw) and its units, or why it has none. Never computes a cell
    test (a pending pair says so)."""
    if not pairs:
        return []
    try:
        found = compare(kind, payload, pairs, population, test, compute=False, margin_pp=margin_pp,
                        correction=correction)
    except ValueError as exc:
        return [str(exc)]
    named = correction_name(correction)
    waiting = any(c.pending for c in found)
    several = population == "all"
    lines = []
    for c in found:
        name = f"{c.a} vs {c.b}" + ((f", {HMM_PANEL_WORDS.get(c.population, c.population)}" if c.what
                                     else f", pop {c.population}") if several and (c.n_pop or c.what) else "")
        if c.pending:
            lines.append(f"{name}: computing the {TEST_LABELS[c.kind].lower()} ...")
        elif c.tested and c.kind in CELL_TESTS:
            m = c.mixed if c.mixed is not None else c.nested
            units = (f"{m.n_days[0]}/{m.n_cells[0]} v {m.n_days[1]}/"
                     f"{m.n_cells[1]} replicates/cells")
            floor = floor_words(c)
            lines.append((f"{name}: raw {p_text(c.test.p)}, {units}" if waiting else
                          f"{name}: {star_text(c.p_holm, stars, c.kind)} {p_text(c.p_holm)} "
                          f"{named}-adjusted (raw {p_text(c.test.p)}), {units}") + (f"; {floor}" if floor else ""))
        elif c.tested:
            within = (f" within \u00b1{margin_pp:g} pp" if c.kind == "tost" else
                      f" beyond \u00b1{margin_pp:g} pp" if c.kind == "met" else "")
            unit = "chunks" if c.chunks else "replicates"
            if c.kind == "paired":
                left = sum(c.unpaired)
                lines.append(f"{name}: {star_text(c.p_holm, stars, c.kind)} {p_text(c.p_holm)} {named}-adjusted "
                             f"(raw {p_text(c.test.p)}), {len(c.pairs)} declared pairs"
                             + (f"; {left} replicate{'' if left == 1 else 's'} without a partner left out"
                                if left else ""))
                continue
            spread = (f"; over {c.split_p[3]} deals raw p {_p_bare(c.split_p[0])} to {_p_bare(c.split_p[2])}"
                      if c.split_p else "")
            lines.append(f"{name}: {star_text(c.p_holm, stars, c.kind)}{within} {p_text(c.p_holm)} "
                         f"{named}-adjusted (raw {p_text(c.test.p)}), {c.test.n_a} v "
                         f"{c.test.n_b} {unit}{spread}")
        else:
            lines.append(f"{name}: no p ({c.reason})")
    return lines
