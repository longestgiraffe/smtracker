"""The workspace's right-hand tool rail: its rules, with no Qt in them (the thin-Qt rule; `ui.workspace.toolbar`
draws the rail, `ui.workspace.window` asks here).

* `rail_section` -- what the rail's popovers describe for the page on screen. A section's Visibility once followed
  the SECTION alone, so Jump distances, Signed jumps and the Apparent Diffusion pages without a bubble chart offered
  the Populations chart's Significance block: brackets for a chart not on screen (4 Nov 2026, rail review).
* `compute_job` / `compute_words` -- what the round Compute mark runs on a page, and what its hover says. It said
  "Compute" everywhere, on Composer and Images & tracks too, where a press did nothing; and while a job ran it showed
  a stop square on hover while a press opened the settings again (4 Nov 2026, rail review).
"""
from __future__ import annotations

__all__ = ["ASKS_FIRST", "NOTHING_TO_COMPUTE", "PAGE_JOBS", "REPLICATE_PAGE", "SECTION_JOBS", "SIGNIFICANCE_PAGES",
           "NO_MAP_YET", "SIGNIFICANCE_EMPTY_TIP", "SIGNIFICANCE_TIP", "SWATCHES", "compute_job", "compute_words", "hidden_count",
           "pair_badge", "pair_key", "pairs_against", "pairs_every", "pairs_toggled", "rail_section",
           "swatch_colours"]

#: The sections whose Significance block belongs to ONE page, and that page: the one drawing the bubble chart (the
#: HMM's comparison, its Slow state tab: since 23 Nov 2026 the second, the state diagrams first, where no bracket draws).
SIGNIFICANCE_PAGES: dict = {"sqd": "populations", "appd": "populations", "hmm": "overview"}
#: The rail's key for a condition's replicate page, drawn in place of a section's bubbles ("sqd.replicates"): no
#: brackets there either, so its Visibility is dimmed like any other page without a bubble chart.
REPLICATE_PAGE = "replicates"


def rail_section(view: str) -> str:
    """What the rail's popovers describe for ``view`` ("section.page"): its section, except a page of a section in
    `SIGNIFICANCE_PAGES` other than the chart's own, which gets ``view`` itself -- a key that lists nothing
    (`toolbar.FIGURES.get`), so its Visibility is dimmed rather than offering another page's brackets."""
    section, _, page = str(view).partition(".")
    own = SIGNIFICANCE_PAGES.get(section)
    if own is not None and page not in ("", own):
        return str(view)
    return section


#: The job a PAGE's Compute mark runs where the page has its own (`WorkspaceWindow._running`'s keys) ...
PAGE_JOBS: dict = {"sqd.jumps": "jumps", "sqd.signed": "jd"}
#: ... else its section's, for these sections. Composer and Images & tracks run nothing: they show what is loaded
#: or what the sections made.
SECTION_JOBS: tuple = ("sqd", "appd", "qc", "spatial", "kinetics", "msd", "hmm", "clusters")
#: The jobs whose Run opens its settings first: the hover ends in an ellipsis, as a menu item that asks does.
ASKS_FIRST: frozenset = frozenset({"sqd", "jumps", "appd", "spatial", "kinetics", "msd", "hmm"})
#: The hover of a page with nothing to compute, where the mark is dimmed.
NOTHING_TO_COMPUTE = "Nothing to compute on this page"


def compute_job(view: str):
    """The job the Compute mark runs on ``view``, or None where there is nothing to compute."""
    view = str(view)
    if view in PAGE_JOBS:
        return PAGE_JOBS[view]
    section = view.partition(".")[0]
    return section if section in SECTION_JOBS else None


def compute_words(job, title: str, running: bool) -> str:
    """The Compute mark's hover: "Run SQD\u2026" (its settings first), "Run QC", "Stop SQD \u00b7 Jump distances"
    while that page's job runs (a press stops it), or `NOTHING_TO_COMPUTE`."""
    if job is None:
        return NOTHING_TO_COMPUTE
    if running:
        return f"Stop {title}"
    return f"Run {title}" + ("\u2026" if job in ASKS_FIRST else "")


# -- the Significance tool's pairs (rail review ideas 1 and 2, 4 Nov 2026) ----------------------------------------- #
#: What the Spatial Map's Visibility says in place of its figure rows until a map is drawn.
NO_MAP_YET = "Run the map to choose which maps it shows."

#: The Significance tool's hover, with brackets on offer and without. The tool is its own button since 4 Nov 2026; the
#: brackets sat under Visibility, a name that says "show or hide".
SIGNIFICANCE_TIP = "Significance: which conditions are compared, with what test, and the brackets on the chart"
SIGNIFICANCE_EMPTY_TIP = ("Significance: brackets appear on the Populations bubbles of SQD and Apparent "
                          "Diffusion and on the HMM comparison. This page has none.")


def pair_key(a: str, b: str) -> frozenset:
    """A pair as the checklist knows it: two conditions, either way round."""
    return frozenset((str(a), str(b)))


def pairs_every(names) -> tuple:
    """Every pair of ``names``, in their order: (first, later) for each, ``n (n - 1) / 2`` of them."""
    names = [str(n) for n in names]
    return tuple((a, b) for i, a in enumerate(names) for b in names[i + 1:])


def pairs_against(names, control: str) -> tuple:
    """Each condition against ``control``: (control, other) in the order of ``names``. ValueError when ``control`` is
    not one of them (a stale choice must not draw brackets for a condition that is not there)."""
    names = [str(n) for n in names]
    control = str(control)
    if control not in names:
        raise ValueError(f"{control} is not one of the conditions on the chart")
    return tuple((control, n) for n in names if n != control)


def pairs_toggled(pairs, a: str, b: str, on: bool) -> tuple:
    """``pairs`` with the pair {a, b} in (``on``; appended as (a, b) unless already there either way round) or out
    (every orientation of it). The order of the others is kept."""
    key = pair_key(a, b)
    pairs = tuple(tuple(p) for p in pairs)
    present = any(pair_key(*p) == key for p in pairs)
    if not on:
        return tuple(p for p in pairs if pair_key(*p) != key)
    return pairs if present else pairs + ((str(a), str(b)),)


def pair_badge(pairs) -> str | None:
    """The count on the Significance tool: how many pairs are set, or None for none (no badge)."""
    n = len(tuple(pairs or ()))
    return str(n) if n else None


def hidden_count(states) -> int:
    """How many figures a Visibility popover has switched off (``states``: name -> shown)."""
    return sum(1 for shown in dict(states or {}).values() if not shown)


#: The Conditions icon carries at most this many condition colours.
SWATCHES = 3


def swatch_colours(colours) -> tuple:
    """The first `SWATCHES` of ``colours`` (condition order), for the dots under the Conditions icon."""
    return tuple(str(c) for c in list(colours or ())[:SWATCHES] if c)
