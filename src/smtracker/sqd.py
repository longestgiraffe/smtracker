"""SQD: exponential-mixture fit of the squared-displacement CDF.

The primary readout of the whole tool: the **fractions** of a mixture of
diffusive states, and their condition-dependence. ``D`` values are apparent and
bounded by the measurement limits (see ``AcquisitionParams``); the fractions are
what claims rest on.

Model (ndim = 2)
----------------
    P(r² < X)  =  1 − Σ_k  f_k · exp( −X / (4 · D_k · τ_lag) )

with ``Σ f_k = 1``, ``f_k ≥ 0``, ``D_k > 0`` and ``τ_lag = lag_frames · τ``.
Verified against the legacy ``twoPopulation.m``, which writes the denominator as
``2 · ndim · D · τ``: the same thing for ndim = 2.

Two estimators, deliberately
----------------------------
``method="ecdf_lsq"`` (default)
    Least squares on the empirical CDF points, exactly what the legacy MATLAB
    does (``lsqnonlin`` over ``fitfun(x, τ, cdf_x) − cdf_y``). This is the
    estimator to use when **validating against the frozen reference**.
``method="mle"``
    Maximum likelihood on the squared displacements themselves. r² from a
    mixture of 2-D Brownian states is a mixture of *exponentials* with means
    ``4·D_k·τ``, so the likelihood is exact and needs no binning. Statistically
    the better estimator: ECDF points are strongly correlated, which is
    precisely why least squares on them cannot yield honest uncertainties.

Both are offered because they answer different questions (reproduce-the-oracle
versus estimate-it-properly) and neither is a silent substitute for the other.
**BIC is always computed from the true likelihood**, never from ECDF residuals
(the legacy appD path computed BIC on ECDF points, a known bug).

Choosing the number of states
-----------------------------
``select_n_pop`` returns an ``NPopSelection``: BIC **recommends**, the analyst
**decides**. The data is never asked to settle a question it cannot: a ΔBIC of
1–2 is a coin flip, and a statistically preferred extra state can still be
biologically implausible (e.g. one sitting at the localization floor, where it
may be measurement heterogeneity rather than a population). ``.choose(n,
reason=...)`` records the decision *and its justification* with the result, so
an override is visible rather than buried. This is not a licence to assert a
state count: the comparison table always travels with the choice.

Whose displacements count: ``weighting``
----------------------------------------
``"step"`` (default) pools every displacement equally: what the legacy does,
and what reproduces the frozen reference. But a track with ``N`` localizations
contributes ``N − 1`` steps, and **track length is not neutral**: fast molecules
leave focus or exceed the tracker's search radius sooner, so long tracks are
preferentially *slow*. Step-pooling therefore over-represents slow molecules and
inflates the slow fraction. This is the same survivorship effect that flattens
the MSD tail, acting here at a single lag.

``"track"`` gives every track an equal say, ``"cell"`` every cell. Measured on
the reference dataset: the slow fraction falls by a couple of points (ecdf_lsq or
mle) under per-track weighting, a real, systematic bias of
about 0.6× the current cell-bootstrap half-width. It will **not** shrink as more
cells are added, unlike the statistical uncertainty.

What per-track weighting does NOT fix: molecules too fast to form a
≥5-localization track at all are absent from the pool entirely, and no
reweighting can recover them. The residual bias runs the same way (slow fraction
still overestimated), so treat the weighted number as a *less* biased estimate,
not an unbiased one.

BIC uses the Kish effective sample size under weighting, so its penalty reflects
the information actually present rather than the raw step count.

Divergences from the legacy code, on purpose
--------------------------------------------
- **Random multistart** (keep-best) instead of one hand-placed seed, with a
  cross-start identifiability report. The legacy single seed
  ``[0.04, 0.4, 0.5]`` can land in a local minimum without any sign of it.
  The legacy seeds and bounds are kept verbatim (``V2_SQD_SEEDS``, the one
  source; ``LEGACY_STARTS`` and ``starts="legacy"`` derive from it) so the old
  behaviour can still be run, and
  ``compare_start_strategies`` measures what the single seed actually cost on a
  given dataset rather than assuming it mattered, or that it did not.
- **Fractions are a proper simplex.** Legacy parameterises ``f_last = 1 − Σ f``
  with each ``f`` bounded in ``[1e-3, 1]`` independently, so a 3-population fit
  can return a *negative* last fraction. Stick-breaking here keeps every
  fraction in ``[0, 1]`` and summing to 1 by construction. For n_pop ≤ 2 the two
  parameterisations are identical, so reference validation is unaffected.
- **n_pop = 1 works** (it crashed in the legacy tool).
- **No ``nlparci`` error bars.** Uncertainties come from
  ``bootstrap_sqd`` over independent units (cells / days / movies).
- Fail loud: a non-converged or degenerate fit raises; it never returns zeros.

Ordering: populations are always returned sorted by ascending ``D``, so
``fractions[0]`` is the slow/bound state (as the legacy tool also sorts).

Exact-zero jumps: an immobile population that is COUNTED, not fitted (ruled 4 Nov 2026)
--------------------------------------------------------------------------------------
A jump of length exactly 0 is a molecule localised twice at the same point: no motion AND no localisation error.
Measured data never holds one (every localisation carries its error); a synthetic TRUTH condition (each molecule's
instantaneous position, the 30 Oct 2026 ruling) does, one per bound step. Fitted as any jump, the zeros pull one
population onto the fitter's lowest D (1e-4), where the likelihood has no maximum (an exponential's density at 0 is
1/m, unbounded as m -> 0): 26 000 jumps with 3 000 zeros took 5 s at one population, 54 s at two, 353 s at three and
did not finish four. So, where the zeros are real (`zero_jumps`):

* the zeros are their OWN population, D = 0 exactly, its fraction their share of the jumps under the fit's
  weighting (each step once, or each track once), never fitted;
* the other ``n_pop - 1`` populations are fitted to the NON-ZERO jumps alone, by either method, and scaled by
  ``1 - share`` (`with_immobile`); the result holds ``n_pop`` populations, the immobile one first (D 0 is the
  lowest D), so ``n_pop`` is always the TOTAL count and a fit of 3 reads bound / middle / free;
* the likelihood is taken against the measure "a point at 0 plus the line" (`_log_mixture_density`): an exact zero
  scores the immobile fraction, a moving jump the continuous mixture. It is the zero / non-zero split's Bernoulli
  likelihood plus the moving jumps' own, each scaled to its own effective n, and BIC adds the two parts' BICs, so
  Best (BIC) chooses the count of MOVING populations on the non-zero jumps exactly (the split's part is the same
  for every count). One population cannot hold an immobile population beside moving ones and is refused by name;
  Best then compares two and up;
* the result says so: ``SQDFit.zero_jumps`` (`ZeroJumps`, its ``note``) and ``SQDFit.notes``.

THE QUANTIZATION GUARD. Exact zeros also come from QUANTIZED coordinates (a tracker run without sub-pixel
localisation writes integer pixels, and a slow molecule's jump then rounds to 0): those zeros are not a population
at rest. They are told apart by the moving jumps (`_lattice_spacing_um`): on a lattice every non-zero squared jump
is a whole number of one unit, the unit no larger than the shortest jumps, and few values repeat many times;
coordinates written to six decimals of a pixel (the truth tables) are a lattice too, but so fine that the shortest
moving jump is tens of millions of units, which the test does not call a lattice, and rightly: rounding at 1e-6 px cannot make a
moving molecule's jump exactly 0. On a lattice the zeros are KEPT in the fit as any jump (the behaviour before the
ruling, slow and with the slowest D on the fitter's floor) and the note says why. Fewer than
`ZERO_TEST_MIN_MOVING` moving jumps cannot be tested and are kept the same way.

NO FLOOR (ruled 4 Nov 2026, the same day as a 1 % floor, which was withdrawn): ANY exact zeros that pass the
guard above form the counted immobile population, however few. Below 1 % the zeros had stayed in the fit, and a
three-population mle fit then invented a population at the D floor (about 1e-4) holding the same 0.6-1 % anyway,
which is the problem the rule was built to remove. The note says how many zeros there are (count and share). A part
given the data set's decision ("immobile" / "kept") is not tested again.

``zero_jumps=`` on `fit_sqd` overrides the test with the decision made on the whole data set ("immobile" / "kept"),
so a bootstrap resample, a replicate or a cell fitted alone has the same populations as the fit it belongs to
(`bootstrap_sqd`, `aggregate.fit_by_unit` and the comparison's replicate fits pass it). Data without an exact zero
take none of these paths: every number is what it was.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from collections import Counter
from math import comb
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

from .observables import ObservableError, displacement_weights, ecdf, fit_ecdf, lag_labels
from .parallel import bootstrap_map, resample_indices

__all__ = [
    "GRADIENTS",
    "default_gradient",
    "gradient_mode",
    "SQDError",
    "SQDFit",
    "log_likelihood_of",
    "NPopSelection",
    "sqd_cdf",
    "ecdf_r_squared",
    "fit_sqd",
    "select_n_pop",
    "compare_start_strategies",
    "legacy_start",
    "bootstrap_sqd",
    "LEGACY_STARTS",
    "V2_SQD_SEEDS",
    "v2_sqd_start",
    "START_STRATEGIES",
    "D_BOUNDS",
    "at_bound",
    "BOOTSTRAP_UNITS",
    "EVIDENCE_THRESHOLDS",
    "ZeroJumps",
    "zero_jumps",
    "zero_rule_of",
    "with_immobile",
    "posteriors",
    "IMMOBILE_LABEL",
    "ZERO_JUMP_RULES",
]

#: Conventional strength-of-evidence bands for a BIC difference (Kass & Raftery).
#: Below the first threshold the two models are practically indistinguishable,
#: a "close call" where the choice is the analyst's, not the number's.
EVIDENCE_THRESHOLDS = ((2.0, "weak"), (6.0, "positive"), (10.0, "strong"))

#: Independent units to bootstrap over. Never bootstrap over displacements:
#: steps within a track, and tracks within a cell, are correlated.
BOOTSTRAP_UNITS = ("cell", "movie", "day")

_NDIM = 2
_MIN_FRACTION = 0.0
_DEFAULT_STARTS = 50

#: MEASURED multistart requirement, per population count. Deposited in
#: `legacy_out/tables/sqd_starts_sweep.csv` (18 groups x {1,5,8,50} starts x n_pop 2,3)
#: and `sqd_starts_convergence_3pop.csv` (the four failures x up to 200 starts).
#:
#: THE ANSWER IS PER n_pop, AND THAT IS THE WHOLE POINT. Blanket "50 is wasted" is what
#: the 2-population evidence alone would have said, and it would have broken the
#: 3-population fit on a quarter of the dataset:
#:
#:   n_pop = 2 | identical to the 50-start answer in 18 of 18 groups at EVERY count
#:             | tested, down to a single start. Requirement measured: 1.
#:   n_pop = 3 | 4 of 18 groups still wrong at 8 starts -- RSS 7.9 to 16.3 per cent
#:             | worse and fractions off by up to 62 pp, i.e. failed fits rather than
#:             | equivalent optima. 16 starts converges all four to 0.00 excess, and
#:             | 100 and 200 find nothing better. Requirement measured: 16.
#:
#: All four failures are conditions with a small slow fraction, so a
#: 3-population fit is splitting a population that barely exists and the surface carries
#: local minima the 2-population one does not. This is the case the multistart is
#: insurance FOR, and it is not hypothetical: it is a quarter of the real groups.
#:
#: The shipped numbers keep margin over the measured requirement (8x at n_pop <= 2, 3x
#: above) because the requirement is a property of THIS dataset and the tool must survive
#: the next one. Margin is cheap at 2 populations and the reason 50 stays at 3.
#:
#: SWEPT AGAIN TO 200 STARTS PER METHOD AND PER n_pop, because the paragraph above was
#: measured on `ecdf_lsq` at n_pop 2 and 3 only and then applied to everything.
#: `legacy_out/tables/sqd_starts_by_method.csv`. BIC excess over the best ever found,
#: 0.0 = converged (three groups x two methods x n_pop 2,3,4):
#:
#:   ecdf_lsq         A/2     A/3     A/4      B/2     B/3     B/4     C/2   C/3   C/4
#:        8 starts      0.0    29.9     4.0      0.0    46.6     0.0     0.0   0.0   0.0
#:       24 starts      0.0     0.0     4.0      0.0     0.0     0.0     0.0   0.0   0.0
#:       50 starts      0.0     0.0     4.0      0.0     0.0     0.0     0.0   0.0   0.0
#:      100 starts      0.0     0.0     0.0      0.0     0.0     0.0     0.0   0.0   0.0
#:   mle
#:        8 starts    499.7    18.4    18.4   1252.1    21.5    21.5     0.0   0.0  49.6
#:       24 starts      0.0     0.0    18.4      0.0    21.5     0.0     0.0   0.0   0.0
#:       50 starts      0.0     0.0    18.4      0.0     0.0     0.0     0.0   0.0   0.0
#:      100 starts      0.0     0.0     0.0      0.0     0.0     0.0     0.0   0.0   0.0
#:
#: MEASURED REQUIREMENT, and the shipped number is ~2x it (the one exception is
#: `ecdf_lsq` at 1-2 populations, already at 8x and too cheap to bother trimming):
#:
#:     method     n_pop     required     ships
#:     ecdf_lsq     1-2            1         8
#:     ecdf_lsq       3           24        50
#:     ecdf_lsq      4+          100       200
#:     mle          1-2           24        50
#:     mle            3           50       100
#:     mle           4+          100       200
#:
#: FOUR POPULATIONS NEEDS 100 UNDER BOTH METHODS, so the `n_pop >= 3 -> 50` rule written
#: from the first sweep was itself insufficient at 4 -- the third time in this file that
#: a number measured in one regime was applied to another. That a 4-population fit needs
#: an order of magnitude more starts than a 2-population one is not an inconvenience to
#: engineer around: it is the optimiser reporting that the surface is barely constrained,
#: which is the same message as saSPT resolving only two modes.
STARTS_BY_N_POP = {1: 8, 2: 8, 3: 50}
_STARTS_ABOVE = 200

#: MLE GETS THE HIGH COUNT AT EVERY POPULATION COUNT, and the reason is a mistake made
#: here rather than a caution invented here. The sweep above ran the DEFAULT estimator,
#: `ecdf_lsq`; `default_starts` was then written from it and silently applied to `mle`
#: too -- the identical shape as the flat `SQD_INTERACTIVE_STARTS = 12` it replaced, a
#: number justified in one regime and used in another.
#:
#: MEASURED under `mle`, the 2-population fit -- the one `ecdf_lsq` reaches from a SINGLE
#: start in 18 of 18 groups -- is badly unconverged at 12 starts, BIC improving on the
#: way to 50 by:
#:     group B  -1252.1      group D  -1387.6      group A  -499.7
#: For scale, the whole open state-count question rests on a ΔBIC of about 180. So the
#: cheap count is a property of the ESTIMATOR, not of the population count, and it does
#: not transfer.
#:
#: CONSEQUENCE ALREADY OBSERVED, and it is not academic: BIC's chosen state count moved
#: in 2 of 5 groups between 12 and 50 starts (one moved from 3 to 2, another from 3 to 4).
#: An unconverged fit has a worse likelihood, so BIC penalises exactly the counts that
#: are hardest to fit -- a model-selection answer that is really an optimiser artefact.
#:
#: MEASURED SINCE, and the interim floor of 50 was indeed too low -- at four populations
#: `mle` still carried 18.4 BIC of excess at 50 and converged only at 100. The numbers
#: below are the sweep's requirement with ~2x margin; see the table above STARTS_BY_N_POP.
#:
#: This costs nothing where speed matters: `ecdf_lsq` is the default and is what the
#: interactive browser fits, so the 6-14x GUI win is untouched. It DOES make
#: `select_n_pop` dearer, which is the correct place to spend it -- that is the one path
#: where the chosen state count was observed to move with the start count.
MLE_STARTS_BY_N_POP = {1: 50, 2: 50, 3: 100}


def default_starts(n_pop: int, *, method: str = "ecdf_lsq") -> int:
    """Multistart count for ``n_pop`` UNDER ``method``, from the measurements above.

    Exposed rather than inlined so a GUI can show which count it used, and so lowering
    one is a documented edit to a named constant instead of a call-site literal.

    ``method`` is not optional in spirit even though it has a default: the cheap counts
    belong to ``ecdf_lsq`` alone, and passing the wrong one is how this went wrong once.

    Both tables fall through to ``_STARTS_ABOVE`` for four populations and more, where
    the requirement is 100 REGARDLESS of method, so the fallback must never be the
    per-method cheap value, which is the shape of the two mistakes this file records.
    """
    table = MLE_STARTS_BY_N_POP if method == "mle" else STARTS_BY_N_POP
    return table.get(int(n_pop), _STARTS_ABOVE)
#: Finite stand-in for an infinite negative log-likelihood, per observation.
_PENALTY_PER_POINT = 1e6
#: Below this many independent units, a percentile bootstrap is flagged as coarse.
FEW_UNITS = 5

#: v2's SQD seeds and bounds, VERBATIM -- THE ONE PLACE THEY ARE WRITTEN DOWN. Every
#: other v2 seed in this project is derived from this table: `LEGACY_STARTS` below (the
#: modern estimator's ``starts="legacy"``) and `smtracker.legacy.sqd` (legacy mode, both
#: fits). Two sets, because v2 runs two fits with two different literal sets:
#:
#: * ``"per_condition"``: ``SMT_GUI/@SMT/initializeVars.m:145-155``, read by
#:   ``sqdFit_.m:4-6``. The lower bound on every D is the string ``"floor"``, meaning
#:   ``0.03**2 / (4*tau)`` (``initializeVars.m:146``, "Diffusion from localization error
#:   = lower bound"), filled in at call time because it depends on τ.
#: * ``"simultaneous"``: ``SMT_GUI/@SMT/sqdFit_Simultaneous.m:41-57``, its own literals.
#:   D's lower bound is 1e-3 (not the floor) and a fraction's lower bound is 0 (not
#:   1e-3). ``p0_frac`` / the fraction bounds repeat once per unit ("condition" in v2,
#:   replicate days in practice). The n_pop = 1 entry is v2's own case-1 literals; v2
#:   then crashes on them (``sqdFit_Simultaneous.m:112``), legacy mode does not.
#:
#: ``p0_frac`` holds v2's FREE fractions only (alpha; alpha and beta); the last is
#: ``1 - sum`` (twoPopulation.m:6, threePopulation.m:9). At 3 populations v2 seeds
#: alpha = beta = 0.5, so the derived third fraction STARTS AT EXACTLY 0.
V2_SQD_SEEDS = {
    "per_condition": {
        1: {"p0_D": (0.5,), "lb_D": "floor", "ub_D": (10.0,),
            "p0_frac": (), "lb_frac": 1e-3, "ub_frac": 1.0},
        2: {"p0_D": (0.04, 0.4), "lb_D": "floor", "ub_D": (3.0, 10.0),
            "p0_frac": (0.5,), "lb_frac": 1e-3, "ub_frac": 1.0},
        3: {"p0_D": (0.01, 0.04, 0.4), "lb_D": "floor", "ub_D": (10.0, 10.0, 10.0),
            "p0_frac": (0.5, 0.5), "lb_frac": 1e-3, "ub_frac": 1.0},
    },
    "simultaneous": {
        1: {"p0_D": (0.05,), "lb_D": 1e-3, "ub_D": (10.0,),
            "p0_frac": (), "lb_frac": 0.0, "ub_frac": 1.0},
        2: {"p0_D": (0.05, 0.5), "lb_D": 1e-3, "ub_D": (10.0, 10.0),
            "p0_frac": (0.5,), "lb_frac": 0.0, "ub_frac": 1.0},
        3: {"p0_D": (0.05, 0.2, 0.5), "lb_D": 1e-3, "ub_D": (10.0, 10.0, 10.0),
            "p0_frac": (0.5, 0.5), "lb_frac": 0.0, "ub_frac": 1.0},
    },
}


def v2_sqd_start(n_pop: int, tau_s: float, *, fit: str = "per_condition",
                 n_units: int = 1):
    """v2's start vector and bounds in v2's OWN parameterisation, from `V2_SQD_SEEDS`.

    Returns ``(p0, lower, upper)`` as lists over ``[D_1..D_n, free fractions of unit 1,
    ..., free fractions of unit n_units]`` -- the vector v2 hands ``lsqnonlin``. ``fit``
    is ``"per_condition"`` (``sqdFit_``) or ``"simultaneous"`` (``sqdFit_Simultaneous``).
    Nothing is clipped here: a seed outside its bound (the 3-population 0.01 against the
    floor of 0.01125 at 20 ms) is returned as v2 wrote it, and the caller says so.
    """
    if fit not in V2_SQD_SEEDS:
        raise SQDError(f"fit must be one of {sorted(V2_SQD_SEEDS)}, got {fit!r}")
    table = V2_SQD_SEEDS[fit]
    if n_pop not in table:
        raise SQDError(
            f"the legacy tool defined seeds only for n_pop in {sorted(table)}, got "
            f"{n_pop}. Use starts='multistart'.")
    if n_units < 1:
        raise SQDError(f"n_units must be >= 1, got {n_units}")
    spec = table[n_pop]
    lb_D = (LEGACY_PRECISION_UM ** 2 / (2.0 * _NDIM * float(tau_s))
            if spec["lb_D"] == "floor" else float(spec["lb_D"]))
    free = len(spec["p0_frac"]) * n_units
    p0 = list(spec["p0_D"]) + list(spec["p0_frac"]) * n_units
    lower = [lb_D] * n_pop + [float(spec["lb_frac"])] * free
    upper = list(spec["ub_D"]) + [float(spec["ub_frac"])] * free
    return p0, lower, upper


def _legacy_starts_from_v2() -> dict:
    """`LEGACY_STARTS` in this module's parameterisation, derived from `V2_SQD_SEEDS`."""
    out = {}
    for n_pop, spec in V2_SQD_SEEDS["per_condition"].items():
        free = list(spec["p0_frac"])
        out[n_pop] = {"D": list(spec["p0_D"]),
                      "fractions": free + [1.0 - sum(free)] if free else [1.0],
                      "D_upper": list(spec["ub_D"])}
    return out


#: The modern estimator's ``starts="legacy"``: v2's PER-CONDITION seed and D bounds
#: (`V2_SQD_SEEDS["per_condition"]`), fractions stored as all n of them (v2 stores the
#: free ones; the last is implied, so 3 populations read [0.5, 0.5, 0.0]). DERIVED, not
#: retyped. The D lower bound is the floor ``0.03**2 / (4*tau)``, filled in by
#: `legacy_start`.
LEGACY_STARTS = _legacy_starts_from_v2()
#: The 0.03 µm the legacy MATLAB hard-codes as its GMM classifier's localization-error
#: lower bound (initializeVars.m:99, `obj.gmm.lb{1} = 0.03; % localization error`), a
#: fit CONSTRAINT, not a measured precision, and in the GMM path, NOT the SQD path. It
#: is reused here ONLY to reproduce the legacy SQD seed strategy's D floor
#: (`0.03²/(4τ)`); it is not the project's σ (which is derived, see
#: AcquisitionParams.localization_sigma_nm). Kept for legacy fidelity only.
LEGACY_PRECISION_UM = 0.03
#: Start strategies accepted by ``fit_sqd``.
START_STRATEGIES = ("multistart", "legacy")
#: The D bounds (µm²/s) of every multistart fit whose caller gives none: the SQD section's Run, its joint fits
#: (`aggregate.fit_simultaneous`) and its replicates' fits all fit within them.
D_BOUNDS = (1e-4, 10.0)


def at_bound(D, bounds=D_BOUNDS, *, rtol: float = 1e-3) -> list:
    """Each D's place against the ``bounds`` it was fitted within: ``"lower"``, ``"upper"`` or ``""``.

    A D ON A BOUND IS NOT ESTIMATED (7 Nov 2026, the per-spot audit's finding 1: per spot, SynR's bound population sat
    on 1e-4 in every condition, a number the optimiser stopped at, not one the data chose). The fit wanted to go
    further, so it reads "at most the lower bound" or "at least the upper", never as a value; its fraction stands.
    The immobile population of exact-zero jumps (D 0, counted, not fitted) is never on a bound. ``bounds`` are
    ``(lo, hi)``, scalars or one per population."""
    D = np.asarray(D, dtype=float).ravel()
    lo, hi = (np.broadcast_to(np.asarray(b, dtype=float), D.shape) for b in bounds)
    return ["" if d == 0 else "lower" if d <= lo_k * (1.0 + rtol) else "upper" if d >= hi_k * (1.0 - rtol) else ""
            for d, lo_k, hi_k in zip(D, lo, hi)]


class SQDError(RuntimeError):
    """Raised when an SQD fit cannot be performed or did not converge."""


# --------------------------------------------------------------------------
# Exact-zero jumps (ruled 4 Nov 2026; see the module head)
# --------------------------------------------------------------------------

#: What the immobile population is called wherever it is named.
IMMOBILE_LABEL = "exact-zero jumps"
#: `fit_sqd`'s ``zero_jumps``: "auto" tests the data (`zero_jumps`); "immobile" and "kept" impose a decision made on
#: the whole data set, so a subset fitted alone keeps the populations of the fit it belongs to.
ZERO_JUMP_RULES = ("auto", "immobile", "kept")
#: Below this many MOVING jumps the lattice test cannot tell rounding from rest, and the zeros are kept.
ZERO_TEST_MIN_MOVING = 20
#: The squared length, in lattice units, the SHORTEST moving jump on a lattice can have: a sum of two squares
#: (1 one step along an axis, 2 one diagonal step, 4 two steps, ...). The unit is that jump over one of these.
_LATTICE_MULTIPLES = (1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25)
#: A squared jump is ON the lattice when, in lattice units, it is within this of a whole number. Arithmetic on
#: lattice coordinates misses a whole number by ~1e-15 of it; a continuous value hits the window 0.2 % of the time.
_LATTICE_TOLERANCE = 1e-3
#: The lattice is found when this share of the moving jumps sits on it...
LATTICE_SHARE = 0.9
#: ... or when the moving jumps hold fewer distinct values than this share of their count: a coarse lattice repeats
#: a handful of values thousands of times, continuous coordinates never repeat one (a bootstrap resample's
#: duplicated units repeat each value at most a few times).
LATTICE_DISTINCT_SHARE = 0.05
#: How many of the shortest distinct moving jumps are tried as the lattice's shortest (one odd value is not enough
#: to hide a lattice).
_LATTICE_CANDIDATES = 5


@dataclass(frozen=True)
class ZeroJumps:
    """What a fit found and did about jumps of length exactly 0 (`zero_jumps`)."""

    n_zero: int
    n_jumps: int
    #: the zeros' share under the fit's weighting: the immobile population's fraction when ``immobile``
    share: float
    #: True: their own population (D 0, counted); False: kept in the fit as any jump (a lattice, or too few moving)
    immobile: bool
    #: the coordinates' lattice spacing (µm) when the moving jumps sit on one, else None
    lattice_um: "float | None"
    weighting: str
    note: str

    def to_dict(self) -> dict:
        return {"n_zero": self.n_zero, "n_jumps": self.n_jumps, "share": self.share,
                "immobile": self.immobile, "lattice_um": self.lattice_um,
                "weighting": self.weighting, "note": self.note}


def _lattice_spacing_um(moving_r2) -> "float | None":
    """The lattice spacing (µm) the moving jumps' coordinates sit on, or None for continuous coordinates.

    ``moving_r2`` are the non-zero squared jumps at their OWN lag (µm²). On a lattice of spacing a, every one is
    a² times a whole number (a sum of two squares), and the shortest is a² times a small one (`_LATTICE_MULTIPLES`):
    so a lattice is found when `LATTICE_SHARE` of them are whole multiples of (one of the `_LATTICE_CANDIDATES`
    shortest distinct values) / (one of the multiples), or when they repeat values so much that fewer than
    `LATTICE_DISTINCT_SHARE` of them are distinct. A lattice so fine that the shortest moving jump is millions of
    units (six decimals of a pixel) is not found: rounding there cannot make a moving jump 0."""
    r2 = np.asarray(moving_r2, dtype=float).ravel()
    distinct = np.unique(r2)
    if len(distinct) <= LATTICE_DISTINCT_SHARE * len(r2):
        return float(np.sqrt(distinct[0]))
    for shortest in distinct[:_LATTICE_CANDIDATES]:
        for multiple in _LATTICE_MULTIPLES:
            units = r2 * (multiple / shortest)
            on = np.abs(units - np.round(units)) <= _LATTICE_TOLERANCE
            if on.mean() >= LATTICE_SHARE:
                return float(np.sqrt(shortest / multiple))
    return None


#: HOW THE RULE'S NOTES NAME WHAT WAS COUNTED, AND WHAT "KEPT" DOES TO IT, per reader (`zero_jumps`' ``noun``): the
#: SQD fit's jumps are kept in the fit as any jump; the apparent D's tracks (`appd_gamma.zero_tracks`, ruled 4 Nov
#: 2026: "the same counted-population rule as SQD") cannot be: a gamma mixture has density 0 at an apparent D of
#: exactly 0, so for it "kept" means LEFT OUT of the fit, what it did with them before the ruling, and the note says
#: so. ``natural`` is the weighting that needs no words ("each track once" said only where it is not the unit itself).
_ZERO_WORDS = {
    "jump": {"natural": "step", "kept": "kept in the fit as any jump",
             "data_set": "keeps its zeros in the fit as any jump",
             "rest": "a molecule found twice at the same point",
             "lattice": "every moving jump a whole number of lattice steps",
             "rounded": "a short jump rounded to nothing",
             "lattice_cost": "They are kept in the fit as any jump: the slowest population may sit at the fitter's "
                             "lowest D, and the fit is slow."},
    "track": {"natural": "track", "kept": "left out of the gamma fit, which cannot hold an apparent D of exactly 0",
              "data_set": "leaves its zeros out of the gamma fit (which cannot hold an apparent D of exactly 0)",
              "rest": "a molecule found at the same point all along its track",
              "lattice": "every moving track's jumps whole numbers of lattice steps",
              "rounded": "a slow track rounded to nothing",
              "lattice_cost": "They are left out of the gamma fit, which cannot hold an apparent D of exactly 0."},
}
_ZERO_WORDS["sub-track"] = {**_ZERO_WORDS["track"], "natural": "step",
                            "rest": "a molecule found at the same point all along its sub-track",
                            "lattice": "every moving sub-track's jumps whole numbers of lattice steps",
                            "rounded": "a slow sub-track rounded to nothing"}


def zero_jumps(values, weights=None, lags=None, *, weighting: str = "step",
               rule: str = "auto", noun: str = "jump", lattice_in_um: bool = True) -> "ZeroJumps | None":
    """The rule for jumps of length exactly 0, decided once for every fit that reaches it. None without a zero.

    ``values`` are the squared jumps a fit receives (µm²; under lags 1-4 together divided by their lag, ``lags``
    each row's lag), ``weights`` the fit's per-jump weights (None: each step once) and ``weighting`` their name.
    The zeros are an IMMOBILE population (``immobile`` True, ``share`` their weighted share, however small: there
    is no floor, ruled and withdrawn 4 Nov 2026) unless the moving jumps sit on a lattice
    (`_lattice_spacing_um`: the zeros may be rounding) or are too few to tell (`ZERO_TEST_MIN_MOVING`); then they
    are kept in the fit as any jump. ``rule`` "immobile" / "kept" imposes the
    decision (made on the whole data set) without a test, and "immobile" holds even where this subset has no zero
    (an immobile population of share 0), so a subset keeps its data set's populations.

    THE SAME RULE FOR THE APPARENT D (``noun`` "track" or "sub-track", ruled 4 Nov 2026; `appd_gamma.zero_tracks`):
    ``values`` are then per-track apparent D's and ``lags`` each track's steps times 4 tau, so ``values * lags`` is
    the track's summed squared jumps (µm²), which on a lattice is a whole number of lattice units as a jump is. Only
    the notes' words change (`_ZERO_WORDS`): "kept" there means left out of the gamma fit. Without the frame interval
    (``lattice_in_um`` False, ``lags`` the steps alone) the test is the same, since it reads whole multiples of a
    unit, but the lattice's size is unknown: ``lattice_um`` stays None and the note says "a lattice"."""
    if rule not in ZERO_JUMP_RULES:
        raise SQDError(f"zero_jumps must be one of {ZERO_JUMP_RULES}, got {rule!r}")
    if noun not in _ZERO_WORDS:
        raise SQDError(f"noun must be one of {tuple(_ZERO_WORDS)}, got {noun!r}")
    words = _ZERO_WORDS[noun]
    values = np.asarray(values, dtype=float).ravel()
    zero = values == 0
    n_zero, n = int(zero.sum()), int(len(values))
    if n_zero == 0 and rule != "immobile":
        return None
    w = None if weights is None else np.asarray(weights, dtype=float).ravel()
    share = n_zero / n if w is None else float(w[zero].sum() / w.sum())
    # display only: a share under 1 % in two figures ("0.42 %"), so a handful of zeros never reads "0.0 %"
    percent = f"{100.0 * share:.1f}" if share >= 0.01 or share == 0 else f"{100.0 * share:.2g}"
    counted = (f"{n_zero} of {n} {noun}s ({percent} %"
               + ("" if weighting == words["natural"] else f" weighting each {weighting} once") + ")")
    if rule == "immobile":
        return ZeroJumps(n_zero, n, share, True, None, weighting, _immobile_note(counted, n - n_zero, noun))
    if rule == "kept":
        return ZeroJumps(n_zero, n, share, False, None, weighting, (
            f"{counted} are exactly zero; the data set they belong to {words['data_set']}, so this part does too."))
    moving = values[~zero]
    if len(moving) < ZERO_TEST_MIN_MOVING:
        return ZeroJumps(n_zero, n, share, False, None, weighting, (
            f"{counted} are exactly zero and only {len(moving)} move: too few moving {noun}s to tell rounding from "
            f"molecules at rest, so the zeros are {words['kept']}."))
    raw = moving if lags is None else moving * np.asarray(lags, dtype=float).ravel()[~zero]
    lattice = _lattice_spacing_um(raw)
    if lattice is not None:
        size = f" of about {1000.0 * lattice:.3g} nm" if lattice_in_um else ""
        return ZeroJumps(n_zero, n, share, False, lattice if lattice_in_um else None, weighting, (
            f"{counted} are exactly zero, but the coordinates sit on a lattice{size} "
            f"({words['lattice']}, as coordinates in whole pixels give), so a zero "
            f"may be {words['rounded']} rather than a molecule at rest. {words['lattice_cost']}"))
    return ZeroJumps(n_zero, n, share, True, None, weighting, _immobile_note(counted, n - n_zero, noun))


def _immobile_note(counted: str, n_moving: int, noun: str = "jump") -> str:
    return (f"{counted} are exactly zero: an immobile population, D 0 and no localisation error "
            f"({_ZERO_WORDS[noun]['rest']}). It is counted, not fitted; the other populations are fitted to the "
            f"{n_moving} moving {noun}s and scaled by the share that moves.")


def with_immobile(D, fractions, share: float) -> tuple[np.ndarray, np.ndarray]:
    """``(D, fractions)`` of the moving populations with the immobile one put first: D 0, fraction ``share``, the
    others scaled by ``1 - share``. The one place a fit's populations gain it."""
    share = float(share)
    return (np.concatenate([[0.0], np.asarray(D, dtype=float).ravel()]),
            np.concatenate([[share], (1.0 - share) * np.asarray(fractions, dtype=float).ravel()]))


def zero_rule_of(fit) -> str:
    """The ``zero_jumps`` rule that gives a subset of ``fit``'s data the same populations: "immobile" when the fit
    holds the immobile population, "kept" when it kept zeros in the fit, "auto" when it met none."""
    zeros = getattr(fit, "zero_jumps", None)
    if zeros is None:
        return "auto"
    return "immobile" if zeros.immobile else "kept"


@dataclass
class SQDFit:
    """Result of one SQD fit. Populations are sorted by ascending ``D``."""

    n_pop: int
    D: np.ndarray                      # µm²/s, ascending
    fractions: np.ndarray              # sums to 1
    tau_lag_s: float
    n_observations: int
    method: str
    log_likelihood: float
    bic: float
    rss: float | None = None           # ECDF residual sum of squares (lsq only)
    starts_strategy: str = "multistart"
    weighting: str = "step"
    n_starts: int = 1
    n_converged_starts: int = 0
    start_spread: dict = field(default_factory=dict)
    localization_sigma_um: float = 0.0          # static-error offset used (0 = none)
    #: The lags fitted together (v2's "1-4 together", `observables.joint_lag_displacements`), or None for one lag.
    #: Then ``tau_lag_s`` is one frame interval and the BIC counts each jump at every lag (choose the count at lag 1).
    joint_lags: tuple | None = None
    #: What was found and done about jumps of length exactly 0 (`zero_jumps`), or None when there were none.
    zero_jumps: "ZeroJumps | None" = None
    #: Fitted with each step's own localisation noise (`fit_sqd`'s ``step_noise_um2``, the `precision` module): the
    #: D's are then corrected for localisation error, not for motion blur (a moving population's D is still apparent);
    #: the calibration is the caller's to record.
    per_spot_precision: bool = False
    #: The GIVEN calibration (c in nm, p) behind the noise when the fit was made with ``precision`` (never learned:
    #: ruled 6 Nov 2026), and how many steps were left out for a spot without a precision (`precision.usable_steps`).
    precision_calibration: tuple | None = None
    precision_left_out: int = 0
    #: Each population's place against the D bounds it was fitted within (`at_bound`: "lower", "upper" or ""), or ()
    #: where none sits on one. A D on a bound is not estimated (7 Nov 2026). Not in `to_dict`, so no table written from
    #: fits changes its columns.
    D_at_bound: tuple = ()

    @property
    def immobile(self) -> bool:
        """True when the first population is the immobile one of exact-zero jumps (D 0, counted, not fitted)."""
        return self.zero_jumps is not None and self.zero_jumps.immobile

    @property
    def notes(self) -> tuple:
        """What a reader of this fit must be told beside its numbers (the exact-zero jumps' rule, when met)."""
        return (self.zero_jumps.note,) if self.zero_jumps is not None else ()

    @property
    def fractions_percent(self) -> np.ndarray:
        return self.fractions * 100.0

    def summary(self) -> str:
        parts = [
            (f"pop{i + 1}: {f * 100:.1f} %, D = 0 ({IMMOBILE_LABEL})" if d == 0 and self.immobile
             else f"pop{i + 1}: {f * 100:.1f} %, D = {d:.4g} µm²/s")
            for i, (f, d) in enumerate(zip(self.fractions, self.D))
        ]
        return (
            f"SQD {self.n_pop}-population fit ({self.method}, n = {self.n_observations}, "
            f"τ_lag = {self.tau_lag_s * 1000:g} ms): " + "; ".join(parts)
            + f" | BIC = {self.bic:.1f}" + "".join(f" | {note}" for note in self.notes)
        )

    def to_dict(self) -> dict:
        out = {
            "n_pop": self.n_pop,
            "method": self.method,
            "tau_lag_s": self.tau_lag_s,
            "n_observations": self.n_observations,
            "log_likelihood": self.log_likelihood,
            "bic": self.bic,
            "rss": self.rss,
            "starts_strategy": self.starts_strategy,
            "weighting": self.weighting,
            "n_starts": self.n_starts,
            "n_converged_starts": self.n_converged_starts,
        }
        for i, (d, f) in enumerate(zip(self.D, self.fractions), start=1):
            out[f"D_{i}"] = d
            out[f"fraction_{i}"] = f
        if self.zero_jumps is not None:              # only where zeros were met: other rows are as they were
            out["n_zero_jumps"] = self.zero_jumps.n_zero
            out["zero_jumps_immobile"] = self.zero_jumps.immobile
        if self.per_spot_precision:                  # only where used: other rows are as they were
            out["per_spot_precision"] = True
            out["precision_left_out"] = int(self.precision_left_out)
        return out


def ecdf_r_squared(fit: "SQDFit", values, weights=None) -> float:
    """R² of a fit against the ECDF it was fitted to.

    The legacy tool quotes this alongside every fit, so it is reproduced for
    the benchmark comparison. Read it for what it is: agreement with a
    *cumulative* curve, whose points are strongly correlated by construction.
    It says the mixture describes the data's shape; it says nothing about the
    uncertainty on the fractions, which is what bootstrapping over cells is
    for. An R² of 0.998 on an ECDF is unremarkable, not impressive.
    """
    from .observables import fit_ecdf, lag_labels

    lags = lag_labels(values)
    values = np.asarray(
        values["sq_disp_um2"] if hasattr(values, "columns") else values, dtype=float
    )
    x, y = fit_ecdf(values, weights=weights, lags=lags)
    # read on the ECDF's own side (<=), so the immobile population of exact-zero jumps is counted at 0
    predicted = sqd_cdf(x, fit.D, fit.fractions, fit.tau_lag_s,
                        immobile=getattr(fit, "immobile", False), inclusive=True)
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 0:
        raise SQDError("ECDF is constant; R² is undefined")
    return float(1.0 - np.sum((y - predicted) ** 2) / total)


def _population_msd(D, tau_lag_s: float, localization_sigma_um: float = 0.0):
    """Mean squared displacement per population, INCLUDING static localization
    error: MSD = 2·ndim·(D·τ + σ²) = 4·D·τ + 4·σ² for ndim = 2 (two localizations,
    each with isotropic error σ per coordinate, add a constant 4σ² offset to the
    MSD). ``localization_sigma_um = 0`` (DEFAULT) recovers the pure 4·D·τ model
    exactly: the pre-registered endpoint; σ > 0 is the pre-specified sensitivity."""
    return 2.0 * _NDIM * (np.asarray(D, dtype=float) * tau_lag_s
                          + float(localization_sigma_um) ** 2)


def sqd_cdf(x, D: Sequence[float], fractions: Sequence[float], tau_lag_s: float,
            localization_sigma_um: float = 0.0, *, immobile: bool = False,
            inclusive: bool = False) -> np.ndarray:
    """``P(r² < x)`` for the exponential mixture. ``x`` in µm², ``D`` in µm²/s.

    ``localization_sigma_um`` adds the static-error offset to each population's MSD
    (default 0 = the pure-diffusion model).

    A D of 0 is refused, unless ``immobile`` says the model holds the immobile population of exact-zero jumps
    (module head; pass ``fit.immobile``): all of its jumps are 0 and it has no localisation error, so it adds its
    whole fraction for every x > 0 and nothing at x = 0 -- strictly below, as the name says; ``inclusive=True``
    gives ``P(r² <= x)``, counting it at x = 0 too, the side an ECDF is read on. The moving populations are
    continuous, so ``inclusive`` changes nothing without the immobile one."""
    x = np.asarray(x, dtype=float)
    D = np.asarray(D, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    if D.shape != fractions.shape:
        raise SQDError(f"D and fractions must have the same length, got {D.shape} vs {fractions.shape}")
    point = (D == 0) if immobile else np.zeros(D.shape, dtype=bool)
    if np.any(D[~point] <= 0):
        raise SQDError(f"all D must be > 0, got {D}"
                       + ("" if immobile else f" (a D of 0 is the immobile population of {IMMOBILE_LABEL}: "
                                              f"pass immobile=True)"))
    if tau_lag_s <= 0:
        raise SQDError(f"tau_lag_s must be > 0, got {tau_lag_s}")
    msd = _population_msd(D, tau_lag_s, localization_sigma_um)
    if not point.any():
        return 1.0 - np.sum(fractions[None, :] * np.exp(-x[:, None] / msd[None, :]), axis=1)
    held = (x[:, None] < 0) if inclusive else (x[:, None] <= 0)
    survival = np.where(point[None, :], held.astype(float),
                        np.exp(-x[:, None] / np.where(point, 1.0, msd)[None, :]))
    return 1.0 - np.sum(fractions[None, :] * survival, axis=1)


#: How the optimiser learns which way to move: ``"numeric"`` (finite differences, the validated
#: default every published number was made with) or ``"analytic"`` (the exact derivatives of the
#: model, `_cdf_jacobian` / `_nll_and_gradient`: the same optimum, reached in a fraction of the time).
#: Asked for 15 Oct 2026 as an OPTION, "test it against protein A and other protein normal fits".
GRADIENTS = ("numeric", "analytic")
LIBRARY_GRADIENT = "numeric"

#: PER THREAD, never one module global (found 16 Oct 2026, the full suite): two threads each inside
#: ``with gradient_mode("analytic")`` interleave their enter/exit, and the second exit "restores" the
#: first one's "analytic" -- every later fit in the process that names no gradient then ran the fast
#: way, and a serial bootstrap stopped matching its parallel twin. A thread that never entered a
#: `gradient_mode` sees the library's default.
_GRADIENT_DEFAULT = threading.local()


def default_gradient() -> str:
    """The gradient a fit uses when none is named, in THIS thread (`gradient_mode` sets it for one Run)."""
    return getattr(_GRADIENT_DEFAULT, "mode", LIBRARY_GRADIENT)


class gradient_mode:
    """``with gradient_mode("analytic"): ...``: every fit inside, in this thread, that names no gradient,
    uses this one; the previous default comes back afterwards (a pool worker serves other sections next).
    A fit sent to another process takes the choice along explicitly (`bootstrap_sqd`, the multistart's
    plans), since a worker process starts at the library's default."""

    def __init__(self, mode: str | None):
        self.mode = mode or LIBRARY_GRADIENT
        if self.mode not in GRADIENTS:
            raise SQDError(f"gradient must be one of {GRADIENTS}, got {mode!r}")

    def __enter__(self):
        self._before = default_gradient()
        _GRADIENT_DEFAULT.mode = self.mode
        return self

    def __exit__(self, *exc):
        _GRADIENT_DEFAULT.mode = self._before
        return False


def fit_sqd(
    squared_displacements,
    *,
    n_pop: int,
    tau_lag_s: float | None = None,
    method: str = "ecdf_lsq",
    weighting: str = "step",
    starts: str = "multistart",
    n_starts: int | None = None,
    seed: int | None = 0,
    D_bounds=None,
    ecdf_max_points: int | None = None,
    localization_sigma_um: float = 0.0,
    n_jobs: int | None = 1,
    gradient: str | None = None,
    zero_jumps: str = "auto",
    step_noise_um2=None,
    precision=None,
) -> SQDFit:
    """Fit an ``n_pop``-population SQD model.

    Parameters
    ----------
    squared_displacements
        The observable: either the DataFrame from
        ``observables.squared_displacements`` (τ is taken from its ``.attrs``)
        or a bare array of r² values in µm² (then ``tau_lag_s`` is required).
    n_pop
        Number of populations (``>= 1``; 1 is supported, unlike the legacy tool).
    tau_lag_s
        Lag time in seconds. Required only for a bare array.
    method
        ``"ecdf_lsq"`` (legacy-compatible, for reference validation) or
        ``"mle"`` (statistically preferable).
    starts
        ``"multistart"`` (default): one moment-based start plus ``n_starts - 1``
        random ones, keep-best. ``"legacy"``: the single hand-placed MATLAB seed
        and its bounds (``LEGACY_STARTS``), exactly as the frozen tool ran it.
        Use ``"legacy"`` to reproduce or to quantify what the single seed cost;
        see ``compare_start_strategies``.
    n_starts
        Random multistart count (keep-best). Ignored when ``starts="legacy"``.
    seed
        RNG seed for the starts, so a fit is reproducible.
    D_bounds
        ``(low, high)`` bounds on ``D``, µm²/s; each may be a scalar or a
        per-population sequence. Defaults to ``(1e-4, 10.0)`` for multistart and
        to the legacy bounds when ``starts="legacy"``.
    zero_jumps
        Jumps of length exactly 0 (module head, ruled 4 Nov 2026): ``"auto"`` (default) makes them an immobile
        population (D 0, counted, the other ``n_pop - 1`` fitted to the moving jumps) unless the coordinates sit
        on a lattice, where they are kept as any jump (`zero_jumps`); ``"immobile"`` / ``"kept"`` impose a
        decision made on the whole data set, for a subset fitted alone (`zero_rule_of`). No effect without zeros
        under "auto" or "kept".
    step_noise_um2
        PER-SPOT PRECISION (6 Nov 2026, `precision`): each step's own per-axis localisation variance (um^2,
        calibrated, `precision.step_noise`), one per row of ``squared_displacements``. Population k's mean r^2 for
        step i is then 2 ndim D_k tau + ndim noise_i, so a step whose spots were placed badly is expected long and
        is not read as movement. Maximum likelihood only (``method="mle"``: with a noise per step there is no one
        ECDF curve); not beside ``localization_sigma_um``, lags fitted together, or an immobile population of
        exact-zero jumps. None (default): the model as before. The calibration behind the noise is always GIVEN
        (a stationary set or a stationary measurement), NEVER learned by the fit: learned with the populations it
        invented a bound population where few molecules were bound (SynR's mutants 26-30 % against 6.5; dropped,
        ruled 6 Nov 2026).
    precision
        PER-SPOT PRECISION FROM THE TABLE ITSELF: a GIVEN `precision.Calibration`; the observable DataFrame's steps'
        own spots (`observables.LOC_SIGMA_STEP_COLUMNS`) give each step's noise through it, inside the fit, so
        every subset a caller fits (a replicate, a bootstrap resample, a BIC candidate) carries its own. Steps
        with a spot without a precision are LEFT OUT and counted (``precision_left_out``; refused above
        `precision.MAX_MISSING_SHARE`). Maximum likelihood only, as ``step_noise_um2``; never beside it.

    Raises
    ------
    SQDError
        On bad input, or if no start converged, never returns a zero "fit".
    """
    plan = _plan_fit(
        squared_displacements, n_pop=n_pop, tau_lag_s=tau_lag_s, method=method,
        weighting=weighting, starts=starts, n_starts=n_starts, seed=seed,
        D_bounds=D_bounds, ecdf_max_points=ecdf_max_points,
        localization_sigma_um=localization_sigma_um, gradient=gradient, zero_rule=zero_jumps,
        step_noise_um2=step_noise_um2, precision=precision)
    return _finish_fit(plan, _run_plans([plan], n_jobs)[0])


@dataclass
class NPopSelection:
    """BIC comparison across population counts: a **recommendation**, not a verdict.

    BIC ranks the models; the analyst chooses. A statistically preferred model
    can still be biologically implausible (e.g. an extra state that sits at the
    localization floor and so may be measurement heterogeneity rather than a
    real population). ``choose`` makes that decision explicit and recorded,
    instead of silently overriding a number or silently obeying one.
    """

    fits: list[SQDFit]
    chosen_n_pop: int | None = None
    choice_reason: str | None = None
    #: The exact-zero jumps' rule every candidate was fitted under (`zero_jumps`), or None without a zero.
    zero_jumps: "ZeroJumps | None" = None
    #: What the comparison left out and why (one population, beside an immobile population of exact-zero jumps).
    notes: tuple = ()

    @property
    def recommended(self) -> SQDFit:
        """The lowest-BIC fit. What the data prefers, all else being equal."""
        return min(self.fits, key=lambda f: f.bic)

    @property
    def best(self) -> SQDFit:
        """The fit in force: the analyst's choice if made, else the recommendation."""
        if self.chosen_n_pop is None:
            return self.recommended
        return self.fit_for(self.chosen_n_pop)

    def resolvability(self, params, *, n_pop: int | None = None):
        """Which populations of a candidate fit sit above the localisation noise.

        Defaults to the fit in force, so it answers "can this acquisition separate the
        states I am about to report". BIC ranks models on fit quality alone and cannot
        know that an extra state has landed inside the measurement error: this is the
        check the class docstring above describes in words.
        """
        fit = self.best if n_pop is None else self.fit_for(n_pop)
        return resolvability(fit, params)

    def fit_for(self, n_pop: int) -> SQDFit:
        for fit in self.fits:
            if fit.n_pop == n_pop:
                return fit
        available = sorted(f.n_pop for f in self.fits)
        raise SQDError(f"no fit for n_pop={n_pop}; available: {available}")

    def delta_bic(self, n_pop: int) -> float:
        """How much worse than the recommendation this model is (0 = the best)."""
        return self.fit_for(n_pop).bic - self.recommended.bic

    def evidence_against(self, n_pop: int) -> str:
        """Strength of the BIC evidence against ``n_pop``, in words.

        ``"none (recommended)"`` for the preferred model itself;
        ``"inconclusive"`` when ΔBIC is too small to separate the models: that
        is the case where the choice is genuinely the analyst's.
        """
        delta = self.delta_bic(n_pop)
        if delta <= 0:
            return "none (recommended)"
        if delta < EVIDENCE_THRESHOLDS[0][0]:
            return "inconclusive"
        for threshold, label in EVIDENCE_THRESHOLDS:
            if delta < threshold:
                return label
        return "very strong"

    def is_close_call(self, a: int, b: int) -> bool:
        """True if BIC cannot meaningfully separate these two models."""
        return abs(self.fit_for(a).bic - self.fit_for(b).bic) < EVIDENCE_THRESHOLDS[0][0]

    def choose(self, n_pop: int, *, reason: str) -> SQDFit:
        """Record the analyst's choice of population count, with its reason.

        A ``reason`` is required: overriding the data-preferred model is a
        legitimate scientific judgement, but it must travel with the result
        rather than disappear into a plot.
        """
        fit = self.fit_for(n_pop)
        if not reason or not reason.strip():
            raise SQDError(
                "choosing a population count requires a reason (it is a scientific "
                "judgement and must be recorded alongside the result)."
            )
        self.chosen_n_pop = n_pop
        self.choice_reason = reason.strip()
        return fit

    def table(self) -> pd.DataFrame:
        """One row per candidate: BIC, ΔBIC, evidence, D's and fractions."""
        rows = []
        for fit in sorted(self.fits, key=lambda f: f.n_pop):
            rows.append(
                {
                    "n_pop": fit.n_pop,
                    "bic": fit.bic,
                    "delta_bic": self.delta_bic(fit.n_pop),
                    "evidence_against": self.evidence_against(fit.n_pop),
                    "recommended": fit.n_pop == self.recommended.n_pop,
                    "chosen": fit.n_pop == self.chosen_n_pop,
                    "D": np.round(fit.D, 5).tolist(),
                    "fractions_percent": np.round(fit.fractions_percent, 2).tolist(),
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = [
            f"BIC recommends n_pop = {self.recommended.n_pop} "
            f"(of {sorted(f.n_pop for f in self.fits)})."
        ]
        if self.chosen_n_pop is not None and self.chosen_n_pop != self.recommended.n_pop:
            lines.append(
                f"Analyst chose n_pop = {self.chosen_n_pop} "
                f"(ΔBIC = {self.delta_bic(self.chosen_n_pop):+.1f}): {self.choice_reason}"
            )
        elif self.chosen_n_pop is not None:
            lines.append(f"Analyst confirmed n_pop = {self.chosen_n_pop}: {self.choice_reason}")
        else:
            lines.append("No analyst choice recorded; the recommendation is in force.")
        return " ".join([*lines, *self.notes])


@dataclass
class _FitPlan:
    """Everything one fit needs before its starts run.

    SPLIT OUT so the starts of SEVERAL fits -- every population count BIC compares --
    can run as ONE pool of independent tasks: one process start-up per selection
    rather than one per count, and the 100-start count shares cores with the 50-start
    ones instead of queueing behind them.
    """

    values: np.ndarray
    weights: "np.ndarray | None"
    tau_lag_s: float
    x_ecdf: np.ndarray
    y_ecdf: np.ndarray
    n_pop: int
    method: str
    weighting: str
    starts: str
    D_bounds: tuple
    start_points: list
    localization_sigma_um: float
    #: `GRADIENTS`: resolved when the plan is made, so pool workers use what the caller chose.
    gradient: str = "numeric"
    joint_lags: tuple | None = None
    #: The exact-zero jumps' rule (`zero_jumps`). When ``immobile``, ``values`` / ``weights`` are the MOVING jumps
    #: and ``n_pop`` counts the moving populations; the fit gains the immobile one in `_finish_fit`.
    zero_jumps: "ZeroJumps | None" = None
    #: Immobile only: the zero / non-zero split's log-likelihood (scaled to ``n_effective_all``) and the effective
    #: n of every jump, for the BIC's part from the split.
    immobile_log_likelihood: float = 0.0
    n_effective_all: float = 0.0
    #: Per-spot precision (`fit_sqd`'s ``step_noise_um2``): each value's own per-axis localisation variance, um^2.
    step_noise: "np.ndarray | None" = None
    #: With ``precision``: the given calibration (c in nm, p) and the steps left out for a spot without a precision.
    precision_calibration: "tuple | None" = None
    precision_left_out: int = 0


def _plan_fit(squared_displacements, *, n_pop, tau_lag_s, method, weighting, starts,
              n_starts, seed, D_bounds, ecdf_max_points,
              localization_sigma_um, gradient=None, zero_rule="auto", step_noise_um2=None,
              precision=None) -> _FitPlan:
    """Validate, build the ECDF and draw the start points -- `fit_sqd`'s own first
    half, unchanged, so a planned fit is exactly the fit `fit_sqd` would run."""
    left_out = 0
    if precision is not None:
        squared_displacements, step_noise_um2, left_out = _with_precision(squared_displacements, precision,
                                                                          step_noise_um2)
    values, tau_lag_s, weights = _extract(squared_displacements, tau_lag_s, weighting)
    if n_pop < 1:
        raise SQDError(f"n_pop must be >= 1, got {n_pop}")
    if method not in ("ecdf_lsq", "mle"):
        raise SQDError(f"method must be 'ecdf_lsq' or 'mle', got {method!r}")
    lags = lag_labels(squared_displacements)
    noise = _check_step_noise(step_noise_um2, values, method=method, lags=lags,
                              localization_sigma_um=localization_sigma_um)
    # EXACT-ZERO JUMPS (module head, 4 Nov 2026): where they are an immobile population, it is counted and the
    # other n_pop - 1 are fitted to the moving jumps, everything below on those alone.
    zeros = zero_jumps(values, weights, lags, weighting=weighting, rule=zero_rule)
    immobile_ll, n_eff_all = 0.0, 0.0
    if noise is not None and zeros is not None and zeros.immobile:
        raise SQDError(f"per-spot precision is not combined with an immobile population of exact-zero jumps "
                       f"({zeros.note}): a jump of exactly 0 has no localisation error to model.")
    if zeros is not None and zeros.immobile:
        if n_pop < 2:
            raise SQDError(
                f"one population cannot be fitted to these jumps: {zeros.note} One population cannot hold them "
                f"beside the moving ones; fit two or more (the immobile one and those that move).")
        immobile_ll = _split_log_likelihood(values == 0, weights)
        n_eff_all = _effective_n(values, weights)
        moving = values > 0
        values = values[moving]
        weights = None if weights is None else weights[moving]
        lags = None if lags is None else lags[moving]
        n_pop = n_pop - 1
        if D_bounds is not None and not np.isscalar(D_bounds[0]) and len(D_bounds[0]) == n_pop + 1:
            # per-population bounds name every population, the immobile one first (the lowest D)
            D_bounds = (np.asarray(D_bounds[0])[1:], np.asarray(D_bounds[1])[1:])
        if len(values) == 0 or np.ptp(values) <= 0:
            raise SQDError(
                f"the {len(values)} moving jumps are all the same length; there is no distribution to fit "
                f"beside the {zeros.n_zero} exact-zero jumps.")
    # RESOLVED AFTER n_pop AND method ARE VALIDATED, because the requirement depends on
    # BOTH -- see STARTS_BY_N_POP and MLE_STARTS. Passing `method` here is the whole fix:
    # without it the cheap `ecdf_lsq` counts reach the `mle` path, where the 2-population
    # fit is unconverged at 12 starts by up to 1387 BIC units. An explicit n_starts always
    # wins; None means "use the measured default for this estimator and population count",
    # which is what a caller not thinking about optimiser seeds should get.
    #
    # The method check sits ABOVE this deliberately: resolving first would key the count
    # on a value already known to be nonsense, and quietly hand it the lsq branch.
    if n_starts is None:
        n_starts = default_starts(n_pop, method=method)
    if n_starts < 1:
        raise SQDError(f"n_starts must be >= 1, got {n_starts}")
    if starts not in START_STRATEGIES:
        raise SQDError(f"starts must be one of {START_STRATEGIES}, got {starts!r}")
    if noise is not None and starts == "legacy":
        # the per-spot audit's finding 4 (7 Nov 2026): the legacy bounds start every D at 0.03² / (4 tau), about
        # 0.011 at 20 ms, which holds the bound population near the plain fit's apparent slow D and undoes the
        # correction without a word
        raise SQDError("per-spot precision cannot be fitted from the legacy starts: their D bounds start at the "
                       "legacy localisation floor (0.03² / 4τ), which holds the bound population near the plain "
                       "fit's apparent D and undoes the correction. Use starts='multistart'.")
    if len(values) < 2 * n_pop:
        raise SQDError(
            f"only {len(values)} squared displacements for a {n_pop}-population fit; "
            "refusing to fit an underdetermined model."
        )

    # ECDF SUBSAMPLE (Item 6): fit the LSQ to ~ecdf_max_points evenly-spaced
    # quantile points instead of all N, an estimator approximation, gated by a
    # measured "identical to 3 s.f." bar. Only bites when N exceeds the budget.
    # Lags 1-4 together: one ECDF per lag, stacked (`observables.fit_ecdf`, v2's own sum).
    x_ecdf, y_ecdf = fit_ecdf(values, weights, lags,
                              max_points=ecdf_max_points if method == "ecdf_lsq" else None)
    rng = np.random.default_rng(seed)

    if starts == "legacy":
        theta0, legacy_bounds = legacy_start(n_pop, tau_lag_s)
        D_bounds = D_bounds if D_bounds is not None else legacy_bounds
        start_points = [theta0]
    else:
        D_bounds = D_bounds if D_bounds is not None else D_BOUNDS
        _bounds(n_pop, D_bounds)          # validate before generating starts
        start_points = _starts(values, n_pop, n_starts, D_bounds, tau_lag_s, rng)

    return _FitPlan(values=values, weights=weights, tau_lag_s=tau_lag_s,
                    x_ecdf=x_ecdf, y_ecdf=y_ecdf, n_pop=n_pop, method=method,
                    weighting=weighting, starts=starts, D_bounds=D_bounds,
                    start_points=list(start_points),
                    localization_sigma_um=localization_sigma_um,
                    gradient=_resolve_gradient(gradient),
                    joint_lags=(tuple(squared_displacements.attrs["joint_lags"]) if lags is not None else None),
                    zero_jumps=zeros, immobile_log_likelihood=immobile_ll, n_effective_all=n_eff_all,
                    step_noise=noise,
                    precision_calibration=(None if precision is None
                                           else (float(precision.c_nm), float(precision.p))),
                    precision_left_out=left_out)


def _with_precision(observable, precision, step_noise_um2):
    """``(the observable without steps lacking a spot's precision, each kept step's noise, how many left out)``
    for `fit_sqd`'s ``precision``."""
    from .precision import PrecisionError, step_noise, usable_steps

    if step_noise_um2 is not None:
        raise SQDError("give a calibration (precision) or each step's noise (step_noise_um2), not both")
    if not isinstance(observable, pd.DataFrame):
        raise SQDError("per-spot precision reads each step's spots from the observable table: pass the DataFrame "
                       "from observables.squared_displacements, not bare values")
    try:
        ok = usable_steps(observable)
        kept = observable
        if not ok.all():
            kept = observable.loc[ok]
            kept.attrs = dict(observable.attrs)
        return kept, step_noise(kept, precision), int((~ok).sum())
    except PrecisionError as exc:
        raise SQDError(str(exc)) from exc


def _check_step_noise(step_noise_um2, values, *, method, lags, localization_sigma_um):
    """`fit_sqd`'s ``step_noise_um2`` checked against the values it belongs to; None stays None."""
    if step_noise_um2 is None:
        return None
    if method != "mle":
        raise SQDError("per-spot precision is fitted by maximum likelihood (method='mle'): with a noise per step "
                       "there is no one ECDF curve to fit.")
    if lags is not None:
        raise SQDError("per-spot precision is not built for lags fitted together (they reuse each jump in up to four displacements, so a likelihood over them counts it up to four times); fit one lag.")
    if localization_sigma_um:
        raise SQDError("per-spot precision replaces the one static error (localization_sigma_um): give one, not "
                       "both.")
    noise = np.asarray(step_noise_um2, dtype=float).ravel()
    if noise.shape != values.shape:
        raise SQDError(f"step_noise_um2 has {noise.size} values for {values.size} squared displacements: one per "
                       f"row, in the observable's order.")
    if not np.all(np.isfinite(noise)) or np.any(noise < 0):
        raise SQDError("step_noise_um2 holds a non-finite or negative variance.")
    return noise


def _means(D, tau_lag_s, localization_sigma_um=0.0, step_noise=None):
    """Each population's mean r^2: one per population (`_population_msd`), or with per-spot precision one ROW per
    value (n x K): 2 ndim D_k tau + ndim noise_i, the form `_log_mixture_density` takes."""
    if step_noise is None:
        return _population_msd(D, tau_lag_s, localization_sigma_um)
    return (2.0 * _NDIM * np.asarray(D, dtype=float)[None, :] * tau_lag_s
            + _NDIM * np.asarray(step_noise, dtype=float)[:, None])


def _split_log_likelihood(zero, weights) -> float:
    """The log-likelihood of WHICH jumps are exactly 0 at the fitted share (a Bernoulli), scaled to the effective n
    of every jump as `_log_likelihood` scales a weighted sum: the immobile population's part of a fit's likelihood
    (and of its BIC, `_finish_fit`). ``0 log 0`` is 0, so a share of 0 or 1 is finite."""
    zero = np.asarray(zero, dtype=bool)
    w = np.ones(len(zero)) if weights is None else np.asarray(weights, dtype=float)
    w0, w1 = float(w[zero].sum()), float(w[~zero].sum())
    total = w0 + w1
    value = sum(part * np.log(part / total) for part in (w0, w1) if part > 0)
    scale = 1.0 if weights is None else _effective_n(zero, weights) / total
    return float(value * scale)


def _resolve_gradient(gradient) -> str:
    gradient = gradient or default_gradient()
    if gradient not in GRADIENTS:
        raise SQDError(f"gradient must be one of {GRADIENTS}, got {gradient!r}")
    return gradient


def _run_start(plan: _FitPlan, theta0):
    """One start: ``(cost, fitted)``, or None when it fails or leaves the domain."""
    try:
        if plan.method == "ecdf_lsq":
            fitted, cost = _fit_lsq(theta0, plan.x_ecdf, plan.y_ecdf, plan.n_pop,
                                    plan.tau_lag_s, plan.D_bounds,
                                    plan.localization_sigma_um,
                                    gradient=getattr(plan, "gradient", "numeric"))
        else:
            fitted, cost = _fit_mle(theta0, plan.values, plan.n_pop, plan.tau_lag_s,
                                    plan.D_bounds, plan.weights,
                                    plan.localization_sigma_um,
                                    gradient=getattr(plan, "gradient", "numeric"),
                                    step_noise=getattr(plan, "step_noise", None))
    except (ValueError, FloatingPointError):
        return None
    if fitted is None or not np.all(np.isfinite(fitted)):
        return None
    return (cost, fitted)


def _start_task(i, shared):                        # pragma: no cover - pool worker
    """Module level, so the pool can pickle it. Reads the shipped-once plans."""
    p, k = shared["tasks"][i]
    plan = shared["plans"][p]
    return _run_start(plan, plan.start_points[k])


def _run_plans(plans, n_jobs):
    """Every start of every plan, grouped back per plan IN START ORDER.

    PARALLELISM NEVER CHANGES A NUMBER: the start points are drawn before anything
    runs, `parallel.bootstrap_map` returns results in task order whatever order the
    workers finish in, and the failures are dropped the same way the serial loop
    dropped them -- so the best start, and every field derived from the list, is
    identical to a serial run. Serial when `n_jobs` is 1, under the GUI's
    no-nested-pools guard, or below the pool's task threshold.
    """
    from . import parallel

    tasks = [(p, k) for p, plan in enumerate(plans)
             for k in range(len(plan.start_points))]
    out = parallel.bootstrap_map(_start_task, len(tasks),
                                 shared={"plans": plans, "tasks": tasks},
                                 n_jobs=n_jobs)
    grouped = [[] for _ in plans]
    for (p, _k), result in zip(tasks, out):
        if result is not None:
            grouped[p].append(result)
    return grouped


def _finish_fit(plan: _FitPlan, results) -> SQDFit:
    """Pick the best start and build the fit -- `fit_sqd`'s own second half."""
    if not results:
        raise SQDError(
            f"no {plan.method} start converged for n_pop={plan.n_pop} on {len(plan.values)} "
            f"displacements (starts={plan.starts!r}). The model may not be identifiable "
            "on this data."
        )

    results.sort(key=lambda cr: cr[0])
    best_cost, best = results[0]
    step_noise = getattr(plan, "step_noise", None)
    D, fractions = _unpack(best, plan.n_pop)
    order = np.argsort(D)
    D, fractions = D[order], fractions[order]

    log_likelihood = _log_likelihood(plan.values, D, fractions, plan.tau_lag_s, plan.weights,
                                     plan.localization_sigma_um, step_noise=step_noise)
    n_free = 2 * plan.n_pop - 1                       # n D's + (n − 1) free fractions
    # Kish effective sample size: weighting reduces the information content, and
    # BIC's penalty must reflect that rather than the raw displacement count.
    n_effective = _effective_n(plan.values, plan.weights)
    bic = n_free * np.log(n_effective) - 2.0 * log_likelihood
    rss = float(best_cost * 2.0) if plan.method == "ecdf_lsq" else None
    n_pop, n_observations, spread = plan.n_pop, len(plan.values), _start_spread(results, plan.n_pop)
    lower, upper = _bounds(plan.n_pop, plan.D_bounds)
    sides = at_bound(D, (lower[:plan.n_pop], upper[:plan.n_pop]))

    zeros = getattr(plan, "zero_jumps", None)
    if zeros is not None and zeros.immobile:
        # THE IMMOBILE POPULATION JOINS (module head): D 0 first, its counted share, the moving ones scaled. The
        # likelihood and BIC add the zero / non-zero split's own (one more free parameter, the share, on the effective
        # n of every jump) to the moving jumps', so BIC ranks counts exactly as on the moving jumps alone. `rss` stays
        # the moving jumps' ECDF residual, the curve that was fitted.
        D, fractions = with_immobile(D, fractions, zeros.share)
        log_likelihood = log_likelihood + plan.immobile_log_likelihood
        bic = bic + np.log(plan.n_effective_all) - 2.0 * plan.immobile_log_likelihood
        n_pop, n_observations = n_pop + 1, zeros.n_jumps
        spread = {**spread, **{k: [0.0, *spread[k]] for k in ("D_min", "D_max", "D_relative_range")}}
        sides = ["", *sides]

    return SQDFit(
        n_pop=n_pop,
        D=D,
        fractions=fractions,
        tau_lag_s=plan.tau_lag_s,
        n_observations=n_observations,
        method=plan.method,
        log_likelihood=float(log_likelihood),
        bic=float(bic),
        rss=rss,
        starts_strategy=plan.starts,
        weighting=plan.weighting,
        n_starts=len(plan.start_points),
        n_converged_starts=len(results),
        start_spread=spread,
        localization_sigma_um=float(plan.localization_sigma_um),
        joint_lags=getattr(plan, "joint_lags", None),
        zero_jumps=zeros,
        per_spot_precision=step_noise is not None,
        precision_calibration=getattr(plan, "precision_calibration", None),
        precision_left_out=int(getattr(plan, "precision_left_out", 0)),
        D_at_bound=tuple(sides) if any(sides) else (),
    )


#: Below this many displacements a BIC selection runs its starts serially -- see
#: `select_n_pop`. The break-even from the measured costs is ~3400; 5000 leaves margin.
MIN_PARALLEL_DISPLACEMENTS = 5000


def select_n_pop(
    squared_displacements,
    *,
    max_pop: int = 3,
    min_pop: int = 1,
    tau_lag_s: float | None = None,
    method: str = "mle",
    n_jobs: int | None = 1,
    **fit_kwargs,
) -> NPopSelection:
    """Compare population counts by BIC and return the full comparison.

    BIC comes from the true likelihood in both methods, never from ECDF
    residuals. The result **recommends**; call ``.choose(n, reason=...)`` to
    record a different (or confirming) decision, see ``NPopSelection``.
    """
    if min_pop < 1 or max_pop < min_pop:
        raise SQDError(f"need 1 <= min_pop <= max_pop, got {min_pop}..{max_pop}")
    # fit_sqd's own defaults, read off its signature so they exist once
    import inspect

    params = inspect.signature(fit_sqd).parameters
    planned = ("weighting", "starts", "n_starts", "seed", "D_bounds",
               "ecdf_max_points", "localization_sigma_um", "gradient", "zero_jumps", "precision")
    unknown = set(fit_kwargs) - set(planned)
    if unknown:
        raise TypeError(f"select_n_pop got unexpected arguments {sorted(unknown)}")
    kwargs = {name: fit_kwargs.get(name, params[name].default) for name in planned}
    kwargs["zero_rule"] = kwargs.pop("zero_jumps")
    # EXACT-ZERO JUMPS (module head): where they are an immobile population every candidate holds it, so Best
    # compares the count of MOVING populations; one population cannot hold it beside them and is not a candidate.
    values, _tau, weights = _extract(squared_displacements, tau_lag_s, kwargs["weighting"])
    zeros = zero_jumps(values, weights, lag_labels(squared_displacements), weighting=kwargs["weighting"],
                       rule=kwargs["zero_rule"])
    notes = ()
    if zeros is not None and zeros.immobile and min_pop < 2:
        min_pop = 2
        notes = (f"One population was not compared: {zeros.note} Best compared 2 to {max_pop} populations, "
                 f"the immobile one and 1 to {max_pop - 1} that move.",)
        if max_pop < min_pop:
            raise SQDError(f"Best cannot choose among 1 population here: {zeros.note} Allow two or more.")
    plans = [_plan_fit(squared_displacements, n_pop=n, tau_lag_s=tau_lag_s,
                       method=method, **kwargs)
             for n in range(min_pop, max_pop + 1)]
    # A POOL ONLY WHERE IT PAYS: starting workers costs ~2.5 s on Windows (spawn
    # re-imports the package), and a 1-3 population selection costs ~1.75 ms per
    # displacement serially. Measured: a 2942-step condition 5.3 s serial against 7.7 s in a
    # pool; a 13511-step one 23.0 s against 12.9 s.
    if len(plans[0].values) < MIN_PARALLEL_DISPLACEMENTS:
        n_jobs = 1
    # ONE POOL FOR EVERY COUNT'S STARTS -- see `_run_plans`. The result is identical
    # to fitting each count in turn; only the wall time changes.
    fits = [_finish_fit(plan, results)
            for plan, results in zip(plans, _run_plans(plans, n_jobs))]
    return NPopSelection(fits=fits, zero_jumps=zeros, notes=notes)


def compare_start_strategies(
    squared_displacements,
    *,
    n_pop: int,
    tau_lag_s: float | None = None,
    method: str = "ecdf_lsq",
    n_starts: int = _DEFAULT_STARTS,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Run the legacy single seed and random multistart side by side.

    The legacy tool used one hand-placed start, which can settle in a local
    minimum with nothing in the output to show it. This quantifies what that
    cost on *this* dataset instead of assuming it either way: a row per
    strategy, plus a ``worse_than_best_rss`` column and an ``agree`` flag.

    Where the two agree, the legacy point estimates were fine and can be quoted
    with confidence. Where they differ, the multistart fit is the better one
    (lower cost by construction) and the gap is the seed's cost.
    """
    fits = {}
    for strategy in START_STRATEGIES:
        fits[strategy] = fit_sqd(
            squared_displacements,
            n_pop=n_pop,
            tau_lag_s=tau_lag_s,
            method=method,
            starts=strategy,
            n_starts=n_starts,
            seed=seed,
        )

    key = "rss" if method == "ecdf_lsq" else "log_likelihood"
    best = (
        min(f.rss for f in fits.values()) if key == "rss"
        else max(f.log_likelihood for f in fits.values())
    )
    reference = fits["multistart"]

    rows = []
    for strategy, fit in fits.items():
        row = {
            "starts": strategy,
            "n_starts": fit.n_starts,
            "rss": fit.rss,
            "log_likelihood": fit.log_likelihood,
            "bic": fit.bic,
        }
        row["worse_than_best"] = (
            (fit.rss - best) if key == "rss" else (best - fit.log_likelihood)
        )
        row["agrees_with_multistart"] = bool(
            np.allclose(fit.D, reference.D, rtol=1e-3)
            and np.allclose(fit.fractions, reference.fractions, atol=1e-3)
        )
        for i, (d, f) in enumerate(zip(fit.D, fit.fractions_percent), start=1):
            row[f"D_{i}"] = d
            row[f"fraction_{i}_percent"] = f
        rows.append(row)
    return pd.DataFrame(rows)


def _sqd_boot_worker(replicate: int, shared: dict):
    """One SQD bootstrap replicate: module-level, picklable for a pool."""
    groups = shared["groups"]
    picked = shared["picks"][replicate]
    unit = shared["unit"]
    n_pop = shared["n_pop"]
    tau_lag_s = shared["tau_lag_s"]
    fit_kwargs = shared["fit_kwargs"]

    if shared["needs_frame"]:
        copies = []
        for draw, index in enumerate(picked):
            copy = groups[index].copy()
            suffix = f"#{draw}"
            copy["uid"] = copy["uid"].astype(str) + suffix
            if unit in copy.columns:
                copy[unit] = copy[unit].astype(str) + suffix
            copies.append(copy)
        values = pd.concat(copies, ignore_index=True)
        values.attrs.update(shared.get("attrs") or {})
        values.attrs["lag_time_s"] = tau_lag_s
    else:
        values = np.concatenate([groups[i]["sq_disp_um2"].to_numpy() for i in picked])
    try:
        fit = fit_sqd(values, n_pop=n_pop, tau_lag_s=tau_lag_s, **fit_kwargs)
    except SQDError:
        return None
    return fit.to_dict()


def legacy_parameter_interval(fit: "SQDFit", squared_displacements, *,
                              alpha: float = 0.05, tau_lag_s: float | None = None,
                              weighting: str | None = None,
                              ecdf_max_points: int | None = None) -> dict:
    """MATLAB ``nlparci`` on the ECDF fit: the LEGACY error bar, reproduced exactly.

    ASKED FOR, AND WRONG, AND BOTH MATTER. This project rejects this interval as the
    reported uncertainty -- ``bootstrap_sqd`` over cells or days is the honest one -- and
    the module head says so in terms. It is implemented anyway, and kept beside the
    bootstrap rather than instead of it, for the reason the legacy seeds are kept: a
    number that a previous MATLAB analysis published can be REPRODUCED and COMPARED,
    which is worth more than an assertion that it was too small.

    WHY IT IS TOO SMALL. `nlparci` treats every ECDF point as an independent observation.
    Adjacent points of an empirical CDF differ by 1/N and are almost perfectly
    correlated, so 42 587 of them were never 42 587 independent constraints on three
    parameters. The interval therefore shrinks as sqrt(N) of a number that is not a
    sample size. Measured on this project's data the gap against a cell bootstrap is
    about three orders of magnitude.

    WHAT MATLAB DOES, and what is reproduced here:

        mse = resid' * resid / (n - p)
        cov = mse * inv(J' * J)
        ci  = beta +- tinv(1 - alpha/2, n - p) * sqrt(diag(cov))

    The parameter vector is the legacy one -- ``[D_1..D_n, f_1..f_{n-1}]``, the last
    fraction implied -- which at ``n_pop = 2`` is exactly ``[D_1, D_2, f_1]``, since the
    stick-breaking variable and the free fraction coincide there. Above two populations
    the sticks are NOT the fractions, so only ``D`` and the first fraction are directly
    comparable with a legacy table and the rest are reported on the stick scale, named.

    Returns a dict keyed ``D_1``, ``D_2``, ..., ``fraction_1``, ... each holding
    ``(estimate, low, high, standard_error)``, plus ``n_points``, ``n_parameters``,
    ``t_multiplier`` and a ``caveat``.
    """
    from scipy.optimize._numdiff import approx_derivative
    from scipy.stats import t as _student_t

    if getattr(fit, "immobile", False):
        raise SQDError(
            f"the legacy interval (nlparci on the ECDF) is defined for moving populations only, and this fit holds "
            f"the immobile population of {IMMOBILE_LABEL} (D 0, counted, not fitted); the legacy tool never met one.")
    values, tau, weights = _extract(squared_displacements, tau_lag_s,
                                    weighting or fit.weighting)
    x_ecdf, y_ecdf = fit_ecdf(values, weights, lag_labels(squared_displacements), max_points=ecdf_max_points)

    n_pop = int(fit.n_pop)
    theta = _pack(np.asarray(fit.D, float), np.asarray(fit.fractions, float).ravel())
    sigma_um = float(fit.localization_sigma_um or 0.0)

    def residuals(t_vec):
        D, fractions = _unpack(t_vec, n_pop)
        return sqd_cdf(x_ecdf, D, fractions, tau, sigma_um) - y_ecdf

    resid = residuals(theta)
    n_points, n_par = len(resid), len(theta)
    if n_points <= n_par:
        raise SQDError(
            f"{n_points} ECDF points for {n_par} parameters; nlparci needs more points "
            f"than parameters")

    jac = approx_derivative(residuals, theta, method="2-point")
    mse = float(resid @ resid) / (n_points - n_par)
    try:
        cov = mse * np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError:
        cov = mse * np.linalg.pinv(jac.T @ jac)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    t_mult = float(_student_t.ppf(1.0 - alpha / 2.0, n_points - n_par))

    names = [f"D_{i + 1}" for i in range(n_pop)]
    names += ([f"fraction_{i + 1}" for i in range(n_pop - 1)] if n_pop == 2
              else [f"stick_{i + 1}" for i in range(n_pop - 1)])
    out = {}
    for name, est, err in zip(names, theta, se):
        out[name] = (float(est), float(est - t_mult * err),
                     float(est + t_mult * err), float(err))
    if n_pop == 2:
        # the implied fraction: its interval is the free one's, mirrored
        est, lo, hi, err = out["fraction_1"]
        out["fraction_2"] = (1.0 - est, 1.0 - hi, 1.0 - lo, err)
    out.update({
        "n_points": n_points, "n_parameters": n_par, "t_multiplier": t_mult,
        "mse": mse, "alpha": alpha,
        "caveat": (
            "nlparci treats each ECDF point as independent; adjacent points differ by "
            "1/N and are almost perfectly correlated, so this interval is far narrower "
            "than the data support. Reported for comparison with a legacy MATLAB table "
            "only -- quote bootstrap_sqd over cells or replicates instead"),
    })
    return out


def bootstrap_sqd(
    observable: pd.DataFrame,
    *,
    n_pop: int,
    unit: str = "cell",
    n_boot: int = 200,
    seed: int | None = 0,
    n_jobs: int | None = None,
    **fit_kwargs,
) -> pd.DataFrame:
    """Honest error bars: resample **independent units**, refit, repeat.

    Steps within a track and tracks within a cell are correlated, so resampling
    displacements would badly understate uncertainty, exactly the flaw in the
    legacy ``nlparci`` intervals. Units are resampled with replacement.

    Parameters
    ----------
    observable
        Output of ``observables.squared_displacements`` (needs the unit column).
    unit
        ``"cell"``, ``"movie"`` or ``"day"``. ``"cell"`` needs cell assignment.
    n_boot
        Bootstrap replicates.

    Returns
    -------
    pandas.DataFrame
        One row per successful replicate, columns ``D_k`` / ``fraction_k``.
        Percentile intervals are then just ``.quantile([0.025, 0.975])``.
        ``.attrs`` record how many replicates failed, how many *distinct*
        resamples are possible, and a ``caveat`` string when the unit count is
        small: a shaky interval is visible rather than hidden.

    Choosing the unit is a scientific decision
    ------------------------------------------
    Coarser units are more conservative but scarcer. With three replicate days
    only ``C(5,3) = 10`` distinct resamples exist, so the interval is quantized
    however many replicates you draw; cells are more numerous but share a day's
    batch effects. Report which unit was used: the number is not comparable
    across units.
    """
    if unit not in BOOTSTRAP_UNITS:
        raise SQDError(f"unit must be one of {BOOTSTRAP_UNITS}, got {unit!r}")
    if unit not in observable.columns:
        raise SQDError(f"observable has no '{unit}' column")
    if n_boot < 1:
        raise SQDError(f"n_boot must be >= 1, got {n_boot}")

    units = observable[unit]
    if units.isna().any():
        raise SQDError(
            f"'{unit}' is unassigned for some displacements, so bootstrapping over it "
            "would drop data. Assign it, or bootstrap over another unit."
        )
    groups = [g for _, g in observable.groupby(unit, observed=True)]
    if len(groups) < 2:
        raise SQDError(
            f"only {len(groups)} distinct {unit}(s); a bootstrap over one unit is "
            "meaningless. Use a coarser unit or collect more data."
        )

    # With few units the percentile interval is *quantized*: resampling n units
    # with replacement admits only C(2n-1, n) distinct multisets (n=3 -> 10),
    # so the CI is coarse however many replicates are drawn. Surfaced, not hidden.
    n_units = len(groups)
    distinct = int(comb(2 * n_units - 1, n_units))

    tau_lag_s = observable.attrs.get("lag_time_s")
    # Track/cell weighting needs the frame, not a bare array of displacements,
    # and a unit drawn twice must count as two units, not merge into one, so the
    # copies are given distinct uids. With plain step weighting the array path
    # is equivalent and cheaper.
    weighting = fit_kwargs.get("weighting", "step")
    # Lags 1-4 together: each resample keeps its rows' lags, so it is fitted as the point fit was (one ECDF per lag).
    joint = bool(observable.attrs.get("joint_lags"))
    # per-spot precision reads each resample's own spots from its rows (`fit_sqd`'s ``precision``)
    needs_frame = weighting != "step" or joint or fit_kwargs.get("precision") is not None
    # THE OPTIMISER TRAVELS WITH THE RESAMPLES: a pool worker is a fresh process at the library's
    # default, so a Run's `gradient_mode` would reach a serial bootstrap and not a parallel one --
    # the two would differ in the seventh digit (found 16 Oct 2026, `test_parallel`).
    fit_kwargs["gradient"] = _resolve_gradient(fit_kwargs.get("gradient"))
    # SO DOES THE EXACT-ZERO RULE (module head), decided once on the whole observable: a resample that happened to
    # draw no zero keeps the immobile population (at share 0), so every row's D_1 is the same population.
    if (fit_kwargs.get("zero_jumps", "auto") == "auto"
            and "sq_disp_um2" in observable.columns and (observable["sq_disp_um2"] == 0).any()):
        values, _tau, weights = _extract(observable, tau_lag_s, weighting)
        decided = zero_jumps(values, weights, lag_labels(observable), weighting=weighting)
        if decided is not None:
            fit_kwargs["zero_jumps"] = "immobile" if decided.immobile else "kept"

    # EXACT WHEN THE WHOLE BOOTSTRAP DISTRIBUTION FITS IN FEWER FITS THAN THE DRAW
    # COUNT. Resampling n units admits only C(2n-1, n) distinct multisets, so with three
    # replicate days there are TEN answers and drawing 200 redraws them twenty times over.
    # MEASURED on the reference project (4 units, 35 multisets): two seeds at n_boot 200 agree
    # to four decimals -- the draws have covered the space and the interval has saturated
    # -- while 10 and 35 draws still disagree between seeds. So sampling needs about 200
    # to reach what enumeration gets in 35, and in 10 for three real days.
    #
    # THE MULTISETS ARE NOT EQUALLY LIKELY, which is the trap. A bootstrap draws each of
    # the n^n ORDERED tuples with equal probability, so {0,1,2} arrives 6 times as often
    # as {0,0,0}. Enumerating multisets and treating them alike would give a WRONG
    # interval, quietly. Each fit is therefore repeated by its multinomial count, which
    # makes the returned frame an exact bootstrap sample of n^n rows -- so `np.percentile`
    # over it stays correct and no caller needs to learn about weights. That matters more
    # than the elegance of a weight column: this project has twice shipped a field nothing
    # read.
    exact = distinct <= n_boot
    if exact:
        from itertools import combinations_with_replacement
        from math import factorial

        multisets = list(combinations_with_replacement(range(n_units), n_units))
        picks = [np.asarray(m, dtype=int) for m in multisets]
        repeats = []
        for m in multisets:
            counts = Counter(m)
            ways = factorial(n_units)
            for c in counts.values():
                ways //= factorial(c)
            repeats.append(ways)
        n_tasks = len(picks)
    else:
        # Index sets up front, in order, so parallel and serial give identical draws.
        picks = resample_indices(len(groups), n_boot, seed)
        repeats = None
        n_tasks = n_boot
    shared = {
        "groups": groups, "picks": picks, "unit": unit, "n_pop": n_pop,
        "tau_lag_s": tau_lag_s, "needs_frame": needs_frame, "fit_kwargs": fit_kwargs,
        "attrs": {"joint_lags": tuple(observable.attrs["joint_lags"])} if joint else {},
    }
    results = bootstrap_map(_sqd_boot_worker, n_tasks, shared=shared, n_jobs=n_jobs)
    keep = [i for i, r in enumerate(results) if r is not None]
    rows = [results[i] for i in keep]
    failures = n_tasks - len(rows)

    if not rows:
        raise SQDError(
            f"every one of {n_tasks} bootstrap replicates failed to fit. "
            "The model is not identifiable on this data."
        )
    out = pd.DataFrame(rows)
    if exact:
        # Repeat each multiset by how often a real bootstrap would have drawn it. A fit
        # that FAILED is dropped with its weight, exactly as a failed random draw is.
        weights = [repeats[i] for i in keep]
        out = out.loc[out.index.repeat(weights)].reset_index(drop=True)
    out.attrs["unit"] = unit
    out.attrs["n_units"] = n_units
    out.attrs["n_requested"] = n_boot
    out.attrs["n_failed"] = failures
    out.attrs["n_distinct_resamples"] = distinct
    out.attrs["exact"] = bool(exact)
    out.attrs["n_fits"] = int(len(keep) + failures)
    # THE SEED IS MEANINGLESS ON THE EXACT PATH, and saying so stops someone reporting a
    # seed as though it were a choice that moved the answer.
    out.attrs["seed"] = None if exact else seed
    out.attrs["few_units"] = n_units < FEW_UNITS
    coarse = (
        f"only {n_units} {unit}(s): at most {distinct} distinct resamples exist, "
        "so this percentile interval is coarse and should be read as indicative, "
        "not as a precise confidence bound."
        if n_units < FEW_UNITS
        else ""
    )
    if exact:
        # COARSE AND EXACT ARE DIFFERENT THINGS, and a reader who is told only one of
        # them will infer the other. The interval is quantized because the design has few
        # units; it is not ALSO uncertain from sampling, because every resample was
        # enumerated rather than drawn.
        coarse = (coarse + " " if coarse else "") + (
            f"Computed EXACTLY: all {distinct} distinct resamples were enumerated and "
            f"weighted by how often a draw would produce them, so this interval carries "
            f"no Monte-Carlo error and does not depend on the seed.")
    out.attrs["caveat"] = coarse
    return out


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------

def _extract(observable, tau_lag_s, weighting="step"):
    """Accept the observable DataFrame or a bare array of r² values.

    Returns ``(values, tau_lag_s, weights)``; ``weights`` is ``None`` for the
    default step weighting.
    """
    weights = None
    if isinstance(observable, pd.DataFrame):
        if "sq_disp_um2" not in observable.columns:
            raise SQDError("DataFrame has no 'sq_disp_um2' column")
        values = observable["sq_disp_um2"].to_numpy(dtype=float)
        tau_lag_s = tau_lag_s if tau_lag_s is not None else observable.attrs.get("lag_time_s")
        try:
            weights = displacement_weights(observable, weighting)
        except ObservableError as exc:
            raise SQDError(str(exc)) from exc
    else:
        values = np.asarray(observable, dtype=float).ravel()
        if weighting != "step":
            raise SQDError(
                f"weighting={weighting!r} needs the observable DataFrame (it must "
                "know which track/cell each displacement came from); a bare array "
                "carries no such information."
            )
    if tau_lag_s is None:
        raise SQDError(
            "tau_lag_s is unknown: pass it explicitly, or pass the DataFrame from "
            "observables.squared_displacements (which carries it in .attrs). "
            "Every D scales with it, so it must not be guessed."
        )
    if len(values) == 0:
        raise SQDError("no squared displacements to fit")
    if not np.isfinite(values).all():
        raise SQDError("squared displacements contain non-finite values")
    if np.any(values < 0):
        raise SQDError("squared displacements cannot be negative")
    if np.ptp(values) <= 0:
        raise SQDError(
            f"all {len(values)} squared displacements are identical ({values[0]}); "
            "there is no distribution to fit. An all-zero input points to a broken "
            "pipeline."
        )
    return values, float(tau_lag_s), weights


def _pack(D: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """[D_1..D_n, u_1..u_{n-1}] with stick-breaking ``u`` in (0, 1)."""
    sticks, remaining = [], 1.0
    for f in fractions[:-1]:
        sticks.append(np.clip(f / remaining, 1e-6, 1 - 1e-6) if remaining > 0 else 0.5)
        remaining -= f
    return np.concatenate([D, sticks])


def _unpack(theta: np.ndarray, n_pop: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of ``_pack``: fractions always form a valid simplex."""
    D = np.asarray(theta[:n_pop], dtype=float)
    sticks = np.asarray(theta[n_pop:], dtype=float)
    fractions, remaining = [], 1.0
    for u in sticks:
        f = remaining * float(np.clip(u, 0.0, 1.0))
        fractions.append(f)
        remaining -= f
    fractions.append(max(remaining, _MIN_FRACTION))
    fractions = np.asarray(fractions, dtype=float)
    total = fractions.sum()
    if total <= 0:
        raise SQDError("degenerate fractions (sum <= 0)")
    return D, fractions / total


def _bounds(n_pop: int, D_bounds):
    """Parameter bounds. ``D_bounds`` is ``(lo, hi)`` or per-population arrays.

    The legacy code bounds each D differently (n_pop = 2 uses ub = [3, 10]), so
    per-population bounds are supported to reproduce it exactly.
    """
    lo, hi = D_bounds
    lo = np.full(n_pop, lo, dtype=float) if np.isscalar(lo) else np.asarray(lo, dtype=float)
    hi = np.full(n_pop, hi, dtype=float) if np.isscalar(hi) else np.asarray(hi, dtype=float)
    if lo.shape != (n_pop,) or hi.shape != (n_pop,):
        raise SQDError(
            f"D_bounds must be scalars or length-{n_pop} sequences, got "
            f"{lo.shape} and {hi.shape}"
        )
    if np.any(lo <= 0) or np.any(hi <= lo):
        raise SQDError(f"D_bounds must satisfy 0 < low < high, got {D_bounds}")
    return (
        np.concatenate([lo, np.zeros(n_pop - 1)]),
        np.concatenate([hi, np.ones(n_pop - 1)]),
    )


def legacy_start(n_pop: int, tau_lag_s: float):
    """The legacy MATLAB seed and bounds for ``n_pop``, in our parameterization.

    Returns ``(theta0, D_bounds)``. The lower bound on every D is the legacy
    localization floor ``0.03²/(4·τ)``; the upper bounds are per-population, as
    in ``initializeVars.m``.
    """
    if n_pop not in LEGACY_STARTS:
        raise SQDError(
            f"the legacy tool defined seeds only for n_pop in {sorted(LEGACY_STARTS)}, "
            f"got {n_pop}. Use starts='multistart'."
        )
    spec = LEGACY_STARTS[n_pop]
    floor = LEGACY_PRECISION_UM**2 / (2.0 * _NDIM * tau_lag_s)
    D = np.asarray(spec["D"], dtype=float)
    upper = np.asarray(spec["D_upper"], dtype=float)
    lower = np.full(n_pop, floor, dtype=float)
    # Keep the seed inside its own bounds (the legacy seed 0.01 sits below the
    # floor when tau is small; MATLAB's lsqnonlin would clip it identically).
    D = np.clip(D, lower, upper)
    fractions = np.asarray(spec["fractions"], dtype=float)
    return _pack(D, fractions), (lower, upper)


def _starts(values, n_pop, n_starts, D_bounds, tau_lag_s, rng) -> list[np.ndarray]:
    """One physically-motivated start plus random ones (log-uniform in D).

    Bounds may be per-population, so the widest common window is used for the
    moment-based start and each D is drawn within its own bounds.
    """
    lower, upper = _bounds(n_pop, D_bounds)
    lo = lower[:n_pop]
    hi = upper[:n_pop]

    # A sensible first guess: D spread around the moment estimate <r²>/(4τ).
    d_hat = float(np.mean(values)) / (2 * _NDIM * tau_lag_s)
    d_hat = float(np.clip(d_hat, lo.max() * 2, hi.min() / 2))
    spread = np.geomspace(max(d_hat / 10, lo.max()), min(d_hat * 5, hi.min()), n_pop)
    starts = [_pack(np.clip(spread, lo, hi), np.full(n_pop, 1.0 / n_pop))]

    for _ in range(max(n_starts - 1, 0)):
        D = np.exp(rng.uniform(np.log(lo), np.log(hi), size=n_pop))
        weights = rng.dirichlet(np.ones(n_pop))
        starts.append(_pack(np.clip(np.sort(D), lo, hi), weights))
    return starts


def _fraction_jacobian(theta, n_pop) -> tuple[np.ndarray, np.ndarray]:
    """``(fractions, dF)``: the stick-breaking fractions of `_unpack` and their derivatives,
    ``dF[k, m] = d f_k / d u_m`` (``u`` = ``theta[n_pop:]``), exactly -- products, no division,
    so a stick at 0 or 1 is no special case. f_j = u_j * prod_{i<j}(1 - u_i), f_n = prod(1 - u)."""
    u = np.clip(np.asarray(theta[n_pop:], dtype=float), 0.0, 1.0)
    m = len(u)
    fractions = np.empty(n_pop)
    dF = np.zeros((n_pop, m))
    for j in range(n_pop):
        lead = u[j] if j < m else 1.0                     # f_n has no stick of its own
        before = np.prod(1.0 - u[:j]) if j else 1.0
        fractions[j] = lead * before
        if j < m:
            dF[j, j] = before
        for i in range(min(j, m)):
            others = np.prod(np.delete(1.0 - u[:j], i)) if j > 1 else 1.0
            dF[j, i] = -lead * others
    return fractions, dF


def _cdf_jacobian(theta, x, n_pop, tau_lag_s, localization_sigma_um=0.0) -> np.ndarray:
    """d P(r^2 < x) / d theta, exactly: P = 1 - sum_k f_k exp(-x / m_k), m_k = 4 (D_k tau + sigma^2).
    dP/dD_k = -f_k exp(-x/m_k) x / m_k^2 * 4 tau;  dP/du = -sum_k (df_k/du) exp(-x/m_k)."""
    D = np.asarray(theta[:n_pop], dtype=float)
    fractions, dF = _fraction_jacobian(theta, n_pop)
    msd = _population_msd(D, tau_lag_s, localization_sigma_um)
    x = np.asarray(x, dtype=float)[:, None]
    e = np.exp(-x / msd[None, :])
    dD = -fractions[None, :] * e * x / msd[None, :] ** 2 * (2.0 * _NDIM * tau_lag_s)
    du = -e @ dF
    return np.hstack([dD, du])


def _nll_and_gradient(theta, values, n_pop, tau_lag_s, weights, localization_sigma_um, penalty, step_noise=None):
    """The negative log-likelihood of `_fit_mle` and its exact gradient. g_i = sum_k f_k/m_k
    exp(-x_i/m_k); d g_i/d D_k = f_k exp(-x_i/m_k) (x_i - m_k)/m_k^3 * 4 tau; d g_i/d u = sum_k
    (df_k/du) exp(-x_i/m_k)/m_k; weights as `_log_likelihood` (renormalised to the effective n). With per-spot
    precision m_ik depends on the step too (`_means`), and every formula holds with m_ik in place of m_k."""
    D = np.asarray(theta[:n_pop], dtype=float)
    zero = np.zeros(len(theta))
    if np.any(D <= 0):
        return penalty, zero
    fractions, dF = _fraction_jacobian(theta, n_pop)
    msd = _means(D, tau_lag_s, localization_sigma_um, step_noise)
    log_g, a = _log_mixture_density(values, msd, fractions)
    if log_g is None:
        return penalty, zero
    if weights is None:
        w = np.ones(len(log_g))
        scale = 1.0
    else:
        w = np.asarray(weights, dtype=float)
        scale = _effective_n(values, weights) / w.sum()
    value = -float(np.sum(w * log_g) * scale)
    if not np.isfinite(value):
        return penalty, zero
    # IN LOG SPACE, as the value is: q_ik = (exp(-x_i/m_k)/m_k) / g_i and r_ik = f_k q_ik (each step's share in
    # population k, in [0, 1]). Formed as g'/g with g in linear space, a step far beyond every population's
    # mean square displacement made g ~ 1e-310: finite, so it passed the g > 0 guard, while w / g overflowed and
    # inf x 0 put NaN into the gradient (reported 20 Oct 2026 from the app: "overflow encountered in divide").
    x = np.asarray(values, dtype=float)[:, None]
    q = np.exp(np.minimum(a - log_g[:, None], 700.0))
    r = q * fractions[None, :]
    m = msd if msd.ndim == 2 else msd[None, :]
    dlog_dD = r * (x - m) / m ** 2 * (2.0 * _NDIM * tau_lag_s)
    dlog_du = q @ dF
    wv = w[:, None]
    grad = -scale * np.concatenate([np.sum(wv * dlog_dD, axis=0), np.sum(wv * dlog_du, axis=0)])
    return value, grad


def _log_mixture_density(values, means, fractions):
    """``(log g, a)``: ``log g_i = log sum_k f_k exp(-x_i/m_k)/m_k`` for every r^2 value, computed in LOG SPACE
    (``a_ik = -x_i/m_k - log m_k``), so a value far out in the tail is a large negative number rather than an
    underflow to 0 -- for the likelihood (`_log_likelihood`) and the fit's gradient (`_nll_and_gradient`) alike,
    one computation. ``(None, a)`` where the density is not positive for some value (a zero or negative mixture,
    as a fraction below 0 can make it): the same refusal the linear sum made at ``total <= 0``.

    ``means`` and ``fractions`` are one value per population, or one ROW per value (n x K) for a mixture that
    changes along the values -- `induction`'s, whose D's and shares move with the time since an induction.

    A mean of exactly 0 is the immobile population of exact-zero jumps (module head), a POINT at 0: the mixture is
    then a density against "a point at 0 plus the line", where an exact zero scores the immobile fraction (a moving
    population holds no mass AT 0) and a moving jump the moving populations' density (the point holds none there)."""
    from scipy.special import logsumexp

    x = np.asarray(values, dtype=float)[:, None]
    means = np.asarray(means, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    m = means if means.ndim == 2 else means[None, :]
    point = m == 0
    if point.any():
        safe = np.where(point, 1.0, m)
        at_zero = x == 0
        atom = point.any(axis=1, keepdims=True)              # this row's mixture holds the point at 0
        a = np.where(point, np.where(at_zero, 0.0, -np.inf),
                     np.where(at_zero & atom, -np.inf, -x / safe - np.log(safe)))
    else:
        a = -x / m - np.log(m)
    log_g, sign = logsumexp(a, b=fractions if fractions.ndim == 2 else fractions[None, :], axis=1,
                            return_sign=True)
    if np.any(sign <= 0) or not np.all(np.isfinite(log_g)):
        return None, a
    return log_g, a


def _fit_lsq(theta0, x_ecdf, y_ecdf, n_pop, tau_lag_s, D_bounds,
             localization_sigma_um=0.0, gradient="numeric"):
    """Least squares on ECDF points: the legacy estimator. ``gradient="analytic"`` hands the optimiser
    the exact Jacobian (`_cdf_jacobian`) instead of finite differences."""
    def residuals(theta):
        D, fractions = _unpack(theta, n_pop)
        return sqd_cdf(x_ecdf, D, fractions, tau_lag_s,
                       localization_sigma_um) - y_ecdf

    jac = "2-point"
    if gradient == "analytic":
        def jac(theta):
            return _cdf_jacobian(theta, x_ecdf, n_pop, tau_lag_s, localization_sigma_um)

    result = least_squares(
        residuals, theta0, jac=jac, bounds=_bounds(n_pop, D_bounds),
        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=5000,
    )
    return (result.x, float(result.cost)) if result.success else (None, np.inf)


def _fit_mle(theta0, values, n_pop, tau_lag_s, D_bounds, weights=None,
             localization_sigma_um=0.0, gradient="numeric", step_noise=None):
    """Maximum likelihood on the r² values (a mixture of exponentials). ``gradient="analytic"``
    hands L-BFGS-B the exact gradient (`_nll_and_gradient`) instead of finite differences.
    ``step_noise``: per-spot precision, each value's own localisation variance (`_means`)."""
    # A large *finite* penalty, not inf: the optimizer's finite differences
    # would otherwise evaluate inf - inf = NaN and lose the gradient.
    penalty = _PENALTY_PER_POINT * max(len(values), 1)
    if gradient == "analytic":
        lower, upper = _bounds(n_pop, D_bounds)
        result = minimize(
            _nll_and_gradient, theta0, method="L-BFGS-B", jac=True,
            args=(values, n_pop, tau_lag_s, weights, localization_sigma_um, penalty, step_noise),
            bounds=list(zip(lower, upper)),
            options={"maxiter": 5000, "ftol": 1e-12},
        )
        return (result.x, float(result.fun)) if result.success else (None, np.inf)

    def negative_log_likelihood(theta):
        D, fractions = _unpack(theta, n_pop)
        if np.any(D <= 0):
            return penalty
        value = -_log_likelihood(values, D, fractions, tau_lag_s, weights,
                                 localization_sigma_um, step_noise=step_noise)
        return penalty if not np.isfinite(value) else value

    lower, upper = _bounds(n_pop, D_bounds)
    result = minimize(
        negative_log_likelihood, theta0, method="L-BFGS-B",
        bounds=list(zip(lower, upper)),
        options={"maxiter": 5000, "ftol": 1e-12},
    )
    return (result.x, float(result.fun)) if result.success else (None, np.inf)


def _effective_n(values, weights) -> float:
    """Kish effective sample size: the raw count when unweighted."""
    if weights is None:
        return float(len(values))
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.sum(w**2))


def posteriors(values, D, fractions, tau_lag_s: float, localization_sigma_um: float = 0.0,
               step_noise=None) -> np.ndarray:
    """Each population's posterior for each squared jump (n x K): its share of the jump's likelihood, the soft label
    every map and per-step assignment reads. THE ONE FORMULA (`maps.labels_from_sqd`, `spatialsqd`), computed as
    ``f_k / m_k exp(-x / m_k)`` over its row sum, the exact arithmetic those modules used before.

    With the immobile population of exact-zero jumps (a D of 0, module head) an exact zero belongs to it (posterior
    1, or the moving populations' share where its fraction is 0) and a moving jump to the moving populations alone.
    A row whose likelihood is 0 everywhere comes back NaN, as before; the caller decides what that means.
    ``step_noise``: a per-spot precision fit's own per-value noise (`_means`); a fit made with it needs it here."""
    x = np.asarray(values, dtype=float).ravel()
    D = np.asarray(D, dtype=float).ravel()
    f = np.asarray(fractions, dtype=float).ravel()
    point = D == 0
    if step_noise is not None:
        if point.any():
            raise SQDError("per-spot precision has no immobile population of exact-zero jumps")
        means = _means(D, tau_lag_s, step_noise=np.asarray(step_noise, dtype=float).ravel())
        density = f[None, :] / means * np.exp(-x[:, None] / means)
        return density / density.sum(axis=1, keepdims=True)
    if not point.any():
        means = _population_msd(D, tau_lag_s, localization_sigma_um)
        density = f[None, :] / means[None, :] * np.exp(-x[:, None] / means[None, :])
        return density / density.sum(axis=1, keepdims=True)
    if point.all():
        raise SQDError(f"posteriors need a moving population beside the immobile one of {IMMOBILE_LABEL}")
    moving = posteriors(x, D[~point], f[~point], tau_lag_s, localization_sigma_um)
    out = np.zeros((len(x), len(D)))
    out[:, ~point] = moving
    zero = x == 0
    held = float(f[point].sum())
    if held > 0 and zero.any():
        rows = np.flatnonzero(zero)
        out[rows] = 0.0
        out[np.ix_(rows, np.flatnonzero(point))] = f[point] / held
    return out


def log_likelihood_of(values, fit: SQDFit, *, step_noise_um2=None) -> float:
    """The exact log-likelihood of ``values`` (r² in µm²) under a fitted model -- the
    likelihood the fit's own BIC uses, applied to data it may never have seen. What a
    held-out (cross-validated) comparison of state counts reads. A per-spot precision fit needs the values' own
    noise (``step_noise_um2``, `precision.step_noise`), and is refused without it."""
    if getattr(fit, "per_spot_precision", False) and step_noise_um2 is None:
        raise SQDError("a per-spot precision fit scores values with their own noise: pass step_noise_um2")
    noise = None if step_noise_um2 is None else np.asarray(step_noise_um2, dtype=float).ravel()
    return _log_likelihood(np.asarray(values, dtype=float), fit.D, fit.fractions,
                           fit.tau_lag_s, localization_sigma_um=fit.localization_sigma_um, step_noise=noise)


def _log_likelihood(values, D, fractions, tau_lag_s, weights=None,
                    localization_sigma_um: float = 0.0, step_noise=None) -> float:
    """r² ~ Σ f_k Exponential(mean = 4·D_k·τ + 4·σ²). Exact, no binning.

    ``weights`` rescales each observation's contribution and is renormalised to
    the effective sample size, so the likelihood stays on a comparable scale.
    ``localization_sigma_um = 0`` (default) is the pure-diffusion likelihood.

    A population with D = 0 is the immobile one of exact-zero jumps, a point at 0 with no localisation error
    whatever ``localization_sigma_um`` (`_log_mixture_density`).

    ``step_noise``: per-spot precision, each value's own localisation variance; the mean is then
    4·D_k·τ + 2·noise_i (`_means`), one row per value.
    """
    D = np.asarray(D, dtype=float)
    if step_noise is not None:
        means = _means(D, tau_lag_s, step_noise=step_noise)
        if np.any(D <= 0) or np.any(means <= 0):
            return -np.inf
        log_density, _a = _log_mixture_density(values, means, fractions)
        if log_density is None:
            return -np.inf
        if weights is None:
            return float(np.sum(log_density))
        w = np.asarray(weights, dtype=float)
        return float(np.sum(w * log_density) * _effective_n(values, weights) / w.sum())
    means = _population_msd(D, tau_lag_s, localization_sigma_um)
    point = D == 0
    if point.any():
        if np.any(D < 0) or np.any(means[~point] <= 0):
            return -np.inf
        means = np.where(point, 0.0, means)
    elif np.any(means <= 0):
        return -np.inf
    log_density, _a = _log_mixture_density(values, means, fractions)
    if log_density is None:
        return -np.inf
    if weights is None:
        return float(np.sum(log_density))
    w = np.asarray(weights, dtype=float)
    scale = _effective_n(values, weights) / w.sum()
    return float(np.sum(w * log_density) * scale)


def _start_spread(results, n_pop) -> dict:
    """Identifiability report: how much do near-best starts disagree?

    A tight spread means the optimum is well determined; a wide one means the
    likelihood surface is flat and the D's should not be over-interpreted.
    """
    costs = np.array([c for c, _ in results])
    best = costs.min()
    tolerance = abs(best) * 1e-3 + 1e-12
    near_best = [theta for c, theta in results if c <= best + tolerance]
    D_values = np.array([np.sort(_unpack(t, n_pop)[0]) for t in near_best])
    return {
        "n_near_best": len(near_best),
        "D_min": D_values.min(axis=0).tolist(),
        "D_max": D_values.max(axis=0).tolist(),
        "D_relative_range": (
            (D_values.max(axis=0) - D_values.min(axis=0)) / np.abs(D_values.mean(axis=0))
        ).tolist(),
    }


@dataclass(frozen=True)
class Resolvability:
    """Which fitted populations sit above the localisation noise, and which do not.

    THE COMPARATOR IS DECIDED, NOT CHOSEN BY TASTE. `canonical.D_bound_legacy` records
    that two sigma-to-D conversions exist here and that they differ by 4x, and which one
    applies depends on whether the D being compared is apparent or sigma-corrected. A
    fit made with ``localization_sigma_um = 0`` returns an APPARENT D, which carries
    ``+sigma^2/tau``, so its comparator is ``D_static_error_offset``. A sigma-corrected
    fit has had that removed, and is read against the legacy ``sigma^2/(4 tau)`` bound.
    Both are stamped in ``comparator`` so a number is never quoted against the wrong one.

    A population BELOW its comparator is not a refutation of the fit: it says the
    acquisition cannot separate that population from immobile molecules jittering inside
    their own localisation error. What to do about it is the analyst's call, which is why
    this reports rather than dropping a state.
    """

    D: tuple[float, ...]
    resolvable: tuple[bool, ...]
    ratio: tuple[float, ...]
    threshold: float
    comparator: str
    sigma_nm: float
    sigma_provenance: str
    #: Per population, the sigma at which it stops being resolvable. THE VERDICT IS NOT
    #: ONE NUMBER, because the sigma it rests on is itself a range: a tracker-reported
    #: sigma is a fit's formal covariance and therefore a LOWER bound, so a population
    #: clearing the threshold by a small factor is not established, it is merely
    #: unrefuted at the most optimistic sigma available. Reporting the crossover instead
    #: of a bare yes lets a reader apply their own sigma without refitting.
    sigma_crossover_nm: tuple[float, ...] = ()

    @property
    def n_below(self) -> int:
        return sum(1 for ok in self.resolvable if not ok)

    def caption(self) -> str:
        from .chart_text import fold_words

        parts = ", ".join(
            f"D{i + 1}={d:.4f} ({fold_words(r, 2, of=True)} the comparator, {'above' if ok else 'BELOW'})"
            for i, (d, r, ok) in enumerate(zip(self.D, self.ratio, self.resolvable)))
        cross = ", ".join(f"{c:.1f}" for c in self.sigma_crossover_nm)
        return (f"{parts}; comparator {self.comparator} = {self.threshold:.4f} um2/s "
                f"at sigma {self.sigma_nm:.1f} nm ({self.sigma_provenance}); "
                f"resolvable only while sigma stays below {cross} nm respectively")


def resolvability(fit: SQDFit, params) -> Resolvability:
    """Flag the populations of ``fit`` that fall below the localisation noise.

    ``params`` is an ``AcquisitionParams``. Taken duck-typed rather than imported so
    this module keeps no dependency on `canonical`.

    Fails loud on a mixed-rate project, because ``params.tau_s`` refuses to hand out a
    scalar there and a floor computed from one movie's tau would not describe the rest.
    """
    apparent = not fit.localization_sigma_um
    threshold = (params.D_static_error_offset if apparent else params.D_bound_legacy)
    comparator = "D_static_error_offset" if apparent else "D_bound_legacy"
    D = tuple(float(d) for d in np.sort(np.asarray(fit.D, dtype=float)))
    ratio = tuple(d / threshold if threshold > 0 else float("inf") for d in D)
    # Invert the comparator: the sigma at which each D would meet its own floor.
    scale = 1.0 if apparent else 4.0
    cross = tuple(float(np.sqrt(scale * d * params.tau_s) * 1000.0) for d in D)
    stamp = params.sigma_stamp()
    return Resolvability(
        D=D,
        resolvable=tuple(d > threshold for d in D),
        ratio=ratio,
        threshold=float(threshold),
        comparator=comparator,
        sigma_nm=float(stamp["sigma_nm"]),
        sigma_provenance=str(stamp["sigma_provenance"]),
        sigma_crossover_nm=cross,
    )
