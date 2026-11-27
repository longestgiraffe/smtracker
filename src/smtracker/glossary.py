"""Explain-this-number, a registry of every reported quantity, with its caveat.

Each number the tool reports has a definition, a formula, a unit, and, crucially
an honest caveat. Those caveats are not new prose: they are the same warnings
the analysis modules already enforce (D₁ is a bound, rates need transitions, σ is
unmeasured, a fraction needs its unit). This module gathers them into one
searchable registry so a reader can hover a number and learn what it is and what
not to over-read, and so the glossary panel and any tooltip draw from one source.

``explain(key)`` returns the static term; ``explain(key, context=…)`` additionally
folds in the *live* caveat where the measurement limits supply it (the actual
localization floor, the actual dZ span), so the warning carries this dataset's
numbers, not a generic sentence.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GlossaryError", "Term", "GLOSSARY", "explain", "search", "tooltip"]


class GlossaryError(RuntimeError):
    """Raised when an unknown quantity is requested."""


@dataclass(frozen=True)
class Term:
    """One reported quantity."""

    key: str
    name: str
    definition: str
    formula: str
    unit: str
    caveat: str

    def as_text(self) -> str:
        parts = [f"{self.name} ({self.unit})", self.definition]
        if self.formula:
            parts.append(f"formula: {self.formula}")
        if self.caveat:
            parts.append(f"CAVEAT: {self.caveat}")
        return "\n".join(parts)


def _term(key, name, definition, formula, unit, caveat):
    return Term(key, name, definition, formula, unit, caveat)


#: The registry. Caveats mirror the guarantees the analysis modules enforce.
GLOSSARY: dict[str, Term] = {t.key: t for t in [
    _term(
        "slow_fraction", "slow / bound fraction",
        "The share of molecules (or steps) in the slow, near-immobile state: the "
        "primary readout of the whole analysis.",
        "f_slow = frac_1 of the SQD/HMM mixture", "fraction or %",
        "Base claims on this fraction rather than on the absolute D. Say which unit the "
        "bootstrap used (cell or replicate); intervals from different units cannot be compared.",
    ),
    _term(
        "D_slow", "slow-state apparent D",
        "The apparent diffusion coefficient of the slow state.",
        "D_1 of the fit; mean r² = 4·D·τ (+4σ²)", "µm²/s",
        "Read it as an upper bound: it sits at or below the localization floor "
        "(0.011–0.072 µm²/s, depending on σ). Report it as '≤ floor'.",
    ),
    _term(
        "D_fast", "fast-state apparent D",
        "The apparent diffusion coefficient of the fast state.",
        "D_2 of the fit", "µm²/s",
        "Underestimated by motion blur and by fast molecules leaving focus "
        "(survivorship); apparent, near the detection ceiling. Berglund-correctable.",
    ),
    _term(
        "on_rate", "binding rate (fast→slow)",
        "The per-second rate of entering the slow state: how often binding happens.",
        "k_on = A[fast,slow] / τ", "s⁻¹",
        "A rate needs observed transitions. Where molecules rarely switch, its "
        "interval can be unbounded; quote it with its bootstrap CI.",
    ),
    _term(
        "off_rate", "release rate (slow→fast)",
        "The per-second rate of leaving the slow state.",
        "k_off = A[slow,fast] / τ", "s⁻¹",
        "As with the on-rate: it cannot be determined when switching is rare, and "
        "then only its interval is meaningful.",
    ),
    _term(
        "dwell_time", "state dwell time",
        "Mean time spent in a state before switching: a model extrapolation.",
        "τ_dwell = τ / (1 − A_kk)", "ms",
        "Model-derived. It often exceeds the observation window, so most "
        "dwells are censored. Do not quote a dwell for a condition that barely "
        "switches, because its rate is undetermined.",
    ),
    _term(
        "occupancy", "HMM occupancy",
        "The time-averaged posterior fraction in each state, switching-corrected.",
        "stationary of the fitted chain", "fraction",
        "Well determined even when switching is rare, because a fraction needs no "
        "transitions; the rates do. This is the fraction that is robust to switching.",
    ),
    _term(
        "localization_floor_D", "localization floor",
        "The apparent D a perfectly immobile molecule shows from localization "
        "noise alone: the resolution limit for slow motion.",
        "σ² / (4τ)", "µm²/s",
        "σ is not measured (30–76 nm is the range used), so the floor is a range. "
        "Any D at or below it is unresolved.",
    ),
    _term(
        "detection_ceiling_D", "detection ceiling",
        "The apparent D above which molecules move further than the tracker's "
        "search radius in one frame and are lost.",
        "(r_search · pixel)² / (4τ)", "µm²/s",
        "Fast molecules above this are truncated or missed, so the fast D is a "
        "lower bound that motion blur makes worse.",
    ),
    _term(
        "bootstrap_ci", "bootstrap confidence interval",
        "A percentile interval from resampling independent units (cells or replicates), "
        "not steps.",
        "2.5–97.5th percentile over unit resamples", "same as the quantity",
        "The unit is the cell or the replicate, never the displacement: a step-level "
        "interval is far too narrow (pseudoreplication). Report which unit.",
    ),
    _term(
        "wasserstein", "Wasserstein distance (effect size)",
        "A model-free measure of how far apart two conditions' step "
        "distributions are. It is a distance and is not a test.",
        "1-Wasserstein (earth-mover) on squared displacements", "µm²",
        "A distance whose interval includes ~0 is not evidence of a difference; a "
        "p-value, if wanted, comes from the unit-level permutation test.",
    ),
    _term(
        "confinement_excess", "confinement excess over null",
        "Observed confined fraction minus what the same detector finds on matched "
        "free-Brownian tracks: the part that is not chance.",
        "excess = observed − chance", "fraction",
        "The raw confined fraction means little alone (a random walk revisits "
        "itself); only the excess over the null supports a claim. It peaks at "
        "100–150 nm radius.",
    ),
    _term(
        "calinski_harabasz", "Calinski-Harabasz index",
        "A cluster-separation score used by the MSD clustering, judged against a "
        "free-Brownian null.",
        "between/within dispersion per d.o.f.", "dimensionless",
        "k-means always returns clusters, so read the index only as excess over the null. "
        "A cluster is a curve shape and does not stand for a state.",
    ),
    _term(
        "bic", "Bayesian Information Criterion",
        "The model-selection score across population counts; lower is preferred.",
        "BIC = k·ln(n) − 2·ln(L)", "dimensionless",
        "BIC recommends a count; you decide. A statistically preferred extra "
        "state can still be measurement heterogeneity at the floor.",
    ),
    _term(
        "saspt_occupancy", "saSPT occupancy",
        "Nonparametric posterior occupancy over a D grid, with no assumed state count.",
        "saSPT state-array posterior", "fraction",
        "saSPT assumes one D per trajectory, so it cannot see switching; its slow "
        "peak sits at the grid's lowest D (a bound, as elsewhere).",
    ),
    _term(
        "radial_enrichment", "radial enrichment",
        "How much more a state is found near the membrane than expected, in "
        "normalised cell coordinates.",
        "state density / overall density, folded", "ratio (1 = none)",
        "The magnitude depends on convention (start point or midpoint); the "
        "direction (slow more peripheral than fast) does not. Projecting a "
        "3-D cell also biases the absolute profile.",
    ),
]}


def explain(key: str, *, context=None) -> Term:
    """The term for a quantity, with a live caveat folded in where available.

    ``context`` may be an ``AcquisitionParams`` (or anything exposing
    ``localization_floor_D`` / ``detection_ceiling_D``); when given, the floor and
    ceiling terms carry this dataset's actual numbers.
    """
    if key not in GLOSSARY:
        raise GlossaryError(
            f"no glossary entry for {key!r}; known: {sorted(GLOSSARY)[:6]}…")
    term = GLOSSARY[key]
    if context is None:
        return term

    live = None
    if key == "localization_floor_D":
        floor = getattr(context, "localization_floor_D", None)
        if floor is not None:
            live = f"on this dataset the floor is ≈ {floor:.4g} µm²/s"
    elif key == "detection_ceiling_D" and getattr(context, "has_mixed_search_radius", False):
        # LINKED AT DIFFERENT RADII (tester D10): each condition's own ceiling, never one of them for all
        said = []
        for condition in context.search_radius_px_by_condition:
            ceiling = context.detection_ceiling_D_for(condition)
            if ceiling is not None:
                said.append(f"{condition} ≈ {ceiling:.4g} µm²/s")
        if said:
            live = "on this dataset each condition has its own ceiling: " + ", ".join(said)
    elif key == "detection_ceiling_D":
        ceiling = getattr(context, "detection_ceiling_D", None)
        if ceiling is not None:
            live = f"on this dataset the ceiling is ≈ {ceiling:.4g} µm²/s"
    if live:
        return Term(term.key, term.name, term.definition, term.formula,
                    term.unit, f"{term.caveat} ({live})")
    return term


def search(query: str) -> list[Term]:
    """Every term whose name/definition/caveat contains ``query`` (case-fold)."""
    q = query.strip().lower()
    if not q:
        return list(GLOSSARY.values())
    return [
        t for t in GLOSSARY.values()
        if q in t.name.lower() or q in t.definition.lower()
        or q in t.caveat.lower() or q in t.key.lower()
    ]


def tooltip(key: str, *, context=None) -> str:
    """The hover text for a quantity: the term as a compact block."""
    return explain(key, context=context).as_text()
