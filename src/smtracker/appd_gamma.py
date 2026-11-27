"""The gamma mixture fit on per-track apparent D. Phase 2's last piece.

READ ``SMT_GUI/Help_fcn/calcAppD_gamma.m``, ``gammaCDF.m``, ``gammafit_one/two/three.m``
and ``calculatesBIC.m`` beside this. Every statement below was confirmed in those files,
and the two numerical claims were measured rather than reasoned.

THE MODEL, reproduced. A per-track apparent D is the mean of ``n_steps`` squared
displacements, so under one Brownian population it is **gamma distributed** with shape
``n_steps`` and mean ``D`` (Stracy et al., PNAS 2015). A mixture of ``n_pop`` populations
is ``sum_k A_k Gamma(x; n_steps, D_k / n_steps)``. That is the legacy's model and it is
unchanged here. Only the arithmetic differs.

The shape is the DATA'S, not a free parameter. ``n_steps`` has no default: it is how
many steps each track's apparent D averaged, and inventing it would change the width of
every component without changing any fitted D.

EACH TRACK ITS OWN SHAPE (16 Oct 2026, the Apparent Diffusion section's definition). ``n_steps`` may be ONE number
(fixed-length pieces, v2's construction) or ONE PER VALUE: a track's apparent D over ALL its ``n_i`` single-frame steps
is Gamma(``n_i``, D / ``n_i``) under one Brownian state, so a mixture fitted with each track's own shape uses every step
and counts every molecule once (`appd.track_appd`). The EM step for D becomes the step-weighted mean
``sum_i r_ik n_i x_i / sum_i r_ik n_i``, which is the old one exactly when every ``n_i`` is equal.

Three defects, each confirmed.

**1. The BIC multiplies densities, and it overflows.** ``calcAppD_gamma.m:60`` is
``L_ = prod(pdf_handle(xData, paramEsts))`` followed by ``BIC = log(n)*k - 2*log(L_)``.
The product fails in both directions, and which one depends on the data. It follows
the mean LOG density of the fitted mixture:

* mean log density **below zero**: the product runs DOWN, reaches ``0``, ``log(0)`` is
  ``-Inf``, and the BIC is **plus infinity** for every model;
* mean log density **above zero**: the product runs UP, reaches ``Inf``, and the BIC is
  **minus infinity** for every model.

**On this project's real data it underflows, exactly as originally recorded.** Measured on
the fitted parameters of all twelve groups, the mean log density is negative everywhere
because the fast component
near D = 1 spreads its mass thinly. **The product came back exactly 0.0 in all 24 fits, so
the legacy criterion was usable in ZERO of them.**

An intermediate version of this docstring claimed the mechanism was an overflow. That came
from synthetic cases built with a narrow distribution (one population at D = 0.5, or two
close together at 0.05 and 0.20), whose densities exceed one. Those overflow; they are the
other direction of the same failure and are not what this data does. Both are reproduced by
``legacy_bic``, which reports the product it actually got.

Either way the fix is the same and the consequence is the one always recorded: the
criterion returns an identical value for every model and cannot choose a state count. Here
the log density is SUMMED and never exponentiated.

**2. A failed fit returns zeros.** ``calcAppD_gamma.m:50-54`` sets ``paramEsts`` and
``paramSD`` to zeros and announces it with a bare ``disp``, so a caller receives a D of
zero with a standard deviation of zero and cannot tell it from a result. Here a fit that
does not converge RAISES.

**3. The Gaussian variant's complex seed is not inherited.** ``sqrt(var(x) - .25*muStart^2)``
lives in ``calcAppD_gau.m`` only. The gamma path never had it, and the Gaussian variant is
retired because it places mass below zero for a positive quantity.

Two things checked that came back clean, recorded so the next reader does not repeat them.

* **The legacy's parameter count is right.** ``k = numel(paramEsts)`` gives 1, 3 and 5 for
  one, two and three populations, which is exactly ``2 * n_pop - 1``. The WRONG count is in
  the OTHER BIC, ``calculatesBIC.m``, where ``k`` is the population count and the penalty
  ignores ``ncond`` while the number of fitted fractions grows with it. Two different BICs
  live in this codebase and only one of them is the one described.
* **The unstable PDF form is unreachable on this data.** ``gammafit_two.m`` evaluates
  ``(n/D)^n`` and ``factorial(n-1)`` separately rather than in logs. Measured, it agrees
  with the stable log form to full precision to ``n = 80`` and overflows only above about
  ``n = 170``; a track here has 4 to 19 steps. Real in form, unreachable in practice.

And the legacy is kept runnable. ``starts="legacy"`` reproduces the legacy's own seeding
(a quantile-based start plus twenty uniform random draws) and ``legacy_bic`` reproduces
the multiplied-density BIC, so "the old criterion could not choose" is a MEASURED claim
per dataset rather than an assumed one. Same treatment as ``sqd.compare_start_strategies``.

EXACT-ZERO TRACKS: AN IMMOBILE POPULATION, COUNTED, NOT FITTED (ruled 4 Nov 2026, "the same counted-population rule
as SQD"). A track whose every jump is exactly 0 has an apparent D of exactly 0: a molecule found at the same point all
along its track, which only a synthetic TRUTH condition holds (no localisation error). A gamma of shape 2 or more has
density 0 there, so until this ruling the fit DROPPED them without a word (166 of 2,022 tracks on SynG truth).
Now, as `sqd`'s module head says for jumps (one decision, `sqd.zero_jumps`, read here through `zero_tracks`):

* the zeros are their own population, D 0, its fraction their share of the tracks (each track once), COUNTED;
* the other ``n_pop - 1`` populations are the gamma mixture fitted to the tracks that move, scaled by the share that
  moves (`sqd.with_immobile`); ``n_pop`` is the TOTAL, the immobile population first;
* the likelihood is "a point at 0 plus the line": the zero / non-zero split's Bernoulli plus the moving tracks' own,
  and BIC adds the split's BIC (one free parameter on every track) to the moving fit's, so Best (BIC) ranks the counts
  of MOVING populations exactly as BIC on the moving tracks alone; one population is refused by name;
* the quantization guard is SQD's: the tracks' summed squared jumps (``appd * n_steps * 4 tau``, µm²) on a lattice
  mean the zeros may be rounding, and they are then left out of the fit as before, said (a gamma cannot hold an
  apparent D of exactly 0, so "kept" means LEFT OUT here); so are too few moving tracks to tell;
* NO FLOOR (a 1 % floor was ruled and withdrawn on 4 Nov 2026, as in `sqd`): any exact-zero tracks that pass the
  guard form the counted population, however few, and the note says how many (count and share);
* ``zeros=`` ("auto", "immobile", "kept") imposes the whole data set's decision on a part of it (a replicate, a fold,
  a resample), as `sqd.fit_sqd`'s ``zero_jumps``; ``tau_s`` gives the lattice its size in nm;
* the fit says so: ``GammaFit.zero_tracks`` (an `sqd.ZeroJumps`), ``.immobile``, ``.notes``.

Data without an exact zero takes none of these paths: every number is what it was.

PER-SPOT PRECISION (6 Nov 2026, `smtracker.precision`; asked: "why not appd?", then "yes"). A track's apparent D
averages its steps' squared jumps, and each step's mean is 4 D tau + 2 noise_i, noise_i its two spots' per-axis
localisation variance through a calibration measured on immobile spots (never learned). So the track's value has mean
D + o, ``o`` = mean(noise_i) / (2 tau), the track's OWN OFFSET (`appd.NOISE_OFFSET_COLUMN`), and the model is
a gamma of mean D_k + o_i: every function here takes ``offset`` (one per value, um^2/s; None is the model as
before, exactly). With an offset the EM's D step has no closed form: a few Newton-like steps on log D from the last
D, each kept only where it raises the likelihood (a generalised EM, still monotone; `_d_step`); a D may go down to
`D_FLOOR` (a bound population).

THE SHAPE UNDER LOCALISATION ERROR IS NOT THE STEP COUNT (10 Nov 2026; `shape_and_scale`, the one place it is
computed). Shape ``n`` says a track's ``n`` squared jumps are independent. Localisation error makes them dependent:
two consecutive jumps share a localisation, so along each axis they are correlated with
``rho = -0.5 o / (D + o)`` (minus one half for a bound molecule, 0 for a fast one). The squares of two zero-mean
Gaussians with correlation rho have covariance ``2 rho^2 s^4``, so the track's mean squared jump has variance
``(D + o)^2 [n + 2 (n - 1) rho^2] / n^2`` in place of ``(D + o)^2 / n``, and the gamma of the same mean and variance
has shape

    k = n / (1 + 2 rho^2 (n - 1) / n),        scale (D + o) / k

which is ``n`` exactly where there is no noise (so the model without an offset is untouched) and 27 % below ``n``
(a variance 1.375 times wider) for a bound molecule on a 4-step track. Until this date the per-spot model kept shape
``n``: a bound population's values then scatter wider than its one gamma allows, and Best (BIC) bought a second
population to cover it (3,000 simulated bound tracks at 30 nm and 20 ms read two, 96.8 / 3.2 %, by a BIC difference
of 17.5; one with this shape). "Not modelled" reads the same tracks as two populations of similar D, 45 / 55 % at
0.035 and 0.053 um^2/s, and goes on doing so (`docs/claude/sqd_and_statistics.md`, "THE APPARENT D'S GAMMA SHAPE
UNDER LOCALISATION ERROR"). Three
approximations, all said: the gamma is MOMENT-MATCHED (the true law of a sum of dependent squares is not a gamma;
mean and variance are exact); motion blur, which adds a positive correlation, is not modelled (as in the mean); and
a track whose noise varies from step to step is given its mean noise. The model WITHOUT an offset ("Not modelled")
cannot make this correction, because it has no noise to compute rho from: there the shape stays ``n`` and the
over-count stays, warned about where it shows (`appd_fits`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

__all__ = ["AppDGammaError", "GammaFit", "gamma_logpdf", "mixture_logpdf", "refit",
           "fit_appd_gamma", "fit_appd_gamma_simultaneous", "legacy_bic",
           "select_n_pop_gamma", "n_parameters", "LEGACY_N_RANDOM_STARTS",
           "MIN_VALUES_PER_POPULATION", "UNIT_KINDS", "zero_tracks", "zero_rule_of", "posteriors",
           "fractions_at", "D_FLOOR", "d_at_floor", "shape_and_scale"]


class AppDGammaError(RuntimeError):
    """Raised when a gamma mixture fit cannot be produced honestly."""


#: ``calcAppD_gamma.m``: ``p00 = [p0 ; rand(20, numel(p0))]``. The legacy already does a
#: multistart, unlike the SQD fit's single hand-placed seed -- worth stating, because the
#: two legacy paths are not equally exposed to the seeding problem.
LEGACY_N_RANDOM_STARTS = 20

#: The fewest usable values a fit accepts per population, for one fit and for each unit of
#: the simultaneous fit. One constant so a caller deciding which replicate days can enter a
#: fit (`appd_view`) applies the same rule the fit enforces.
MIN_VALUES_PER_POPULATION = 10

#: The lowest D a per-spot fit reaches (um^2/s): with each track's own noise offset a bound population's D goes to about
#: 0, and the search stops here rather than at 0, where log D has no value.
D_FLOOR = 1e-8


def d_at_floor(D, *, rtol: float = 1e-3) -> bool:
    """Whether a population's D sits on `D_FLOOR`: the M-step's stopping point, not an estimate, so it is said ("not
    resolved"), never printed as a value (10 Nov 2026; as SQD's `sqd.at_bound`, the per-spot audit of 7 Nov 2026)."""
    try:
        D = float(D)
    except (TypeError, ValueError):
        return False
    return bool(0.0 < D <= D_FLOOR * (1.0 + rtol))


def n_parameters(n_pop: int) -> int:
    """Free parameters in an ``n_pop`` gamma mixture: ``n_pop`` D's and ``n_pop - 1``
    fractions, the last fraction being determined by the others.

    The legacy's own count agrees (1, 3, 5). It is ``calculatesBIC.m`` that is wrong.
    """
    if n_pop < 1:
        raise AppDGammaError("n_pop must be at least 1")
    return 2 * n_pop - 1


def shape_and_scale(D, n_steps, offset=None):
    """``(shape, scale)`` of one population's gamma for values averaging ``n_steps`` steps: THE ONE PLACE the model's
    shape is decided, read by the density here and by every CDF drawn from a fit (`appd_view`).

    Without ``offset`` (no localisation error modelled): shape ``n_steps``, scale ``D / n_steps``, as always. With
    each value's noise offset (per-spot precision): mean ``D + offset`` and the EFFECTIVE shape
    ``n / (1 + 2 rho^2 (n - 1) / n)``, ``rho = -0.5 offset / (D + offset)`` the correlation of consecutive jumps
    that share a localisation (module head). An offset of 0 gives shape ``n`` exactly."""
    n = np.asarray(n_steps, dtype=float)
    if offset is None:
        return n, D / n
    o = np.asarray(offset, dtype=float)
    m = D + o
    with np.errstate(divide="ignore", invalid="ignore"):
        rho2 = np.where(m > 0, (0.5 * o / m) ** 2, 0.0)
    shape = n / (1.0 + 2.0 * (n - 1.0) / n * rho2)
    return shape, m / shape


def gamma_logpdf(x, D: float, n_steps, offset=None) -> np.ndarray:
    """Log density of a per-track apparent D under one population.

    Shape ``n_steps``, scale ``D / n_steps``, so the mean is ``D``. ``n_steps`` is one number or one per value of
    ``x`` (each track its own). Computed in LOGS: the legacy's algebraic form evaluates an enormous numerator
    against an enormous factorial and is only safe below about 170 steps. ``offset``: each value's own noise
    offset (per-spot precision, module head), the mean then ``D + offset`` and the shape the effective one
    (`shape_and_scale`: consecutive jumps share a localisation).
    """
    x = np.asarray(x, dtype=float)
    n = np.asarray(n_steps, dtype=float)
    if D <= 0:
        raise AppDGammaError(f"D must be positive, got {D}")
    if np.any(n < 1):
        raise AppDGammaError(f"n_steps must be at least 1, got {np.min(n)}")
    shape, scale = shape_and_scale(D, n, offset)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (special.xlogy(shape - 1, x) - x / scale
               - special.gammaln(shape) - shape * np.log(scale))
    return np.where(x > 0, out, -np.inf)


def _values_and_shapes(appd, n_steps):
    """The usable values (finite, above zero) and the shape for each: ``n_steps`` broadcast when it is one number,
    filtered alongside when it is one per value."""
    x, n, _o = _kept(appd, n_steps, None)
    return x, n


def _offsets_of(appd, offset):
    """``offset`` as one float per value of ``appd`` (None stays None), refused when it does not match or holds a
    negative or non-finite noise."""
    if offset is None:
        return None
    x = np.asarray(appd, dtype=float)
    o = np.asarray(offset, dtype=float)
    if o.shape != x.shape:
        raise AppDGammaError(f"{o.size} noise offsets for {x.size} apparent-D values: one each (per-spot precision)")
    if not np.all(np.isfinite(o) & (o >= 0)):
        raise AppDGammaError("a noise offset that is negative or not finite: refused, never filled in")
    return o


def _kept(appd, n_steps, offset, *, zeros: bool = False):
    """`_values_and_shapes` with each value's noise offset filtered alongside (None without per-spot precision);
    ``zeros`` keeps the exact zeros (`_with_zeros`)."""
    x = np.asarray(appd, dtype=float)
    n = np.asarray(n_steps, dtype=float)
    if n.ndim == 0:
        n = np.full(x.shape, float(n))
    elif n.shape != x.shape:
        raise AppDGammaError(f"{n.size} step counts for {x.size} apparent-D values: one each, or one for all")
    o = _offsets_of(x, offset)
    keep = np.isfinite(x) & ((x >= 0) if zeros else (x > 0)) & np.isfinite(n)
    return x[keep], n[keep], (None if o is None else o[keep])


def _with_zeros(appd, n_steps):
    """The values a fit holds or counts (finite, at or above zero) and the shape for each: `_values_and_shapes`
    with the exact zeros kept, for the zero rule (`zero_tracks`) to count."""
    x, n, _o = _kept(appd, n_steps, None, zeros=True)
    return x, n


#: The weighting a value of each kind counts once, named in the zero rule's note (`sqd.zero_jumps`).
_NATURAL_WEIGHTING = {"track": "track", "sub-track": "step"}


def zero_tracks(appd, n_steps, *, tau_s=None, rule: str = "auto", noun: str = "track"):
    """THE EXACT-ZERO RULE FOR THE APPARENT D (module head): `sqd.zero_jumps` -- the one decision, its
    quantization guard and its notes -- over the values a gamma fit receives. None without a zero.

    ``appd`` one apparent D per track (``noun`` "track", `appd.track_appd`) or per 4-step sub-track ("sub-track",
    the Spatial Map's fit), ``n_steps`` the steps each averages, ``tau_s`` the frame interval (one, or one per value)
    so that ``appd * n_steps * 4 tau`` is each value's summed squared jumps in µm², what the lattice test reads.
    Without ``tau_s`` the test is the same (it reads whole multiples of a unit) and the note gives no size. ``rule``
    "immobile" / "kept" imposes a decision made on the whole data set (`zero_rule_of`)."""
    from .sqd import SQDError, zero_jumps

    if noun not in _NATURAL_WEIGHTING:
        raise AppDGammaError(f"noun must be one of {tuple(_NATURAL_WEIGHTING)}, got {noun!r}")
    raw = np.asarray(appd, dtype=float)
    x, n = _with_zeros(raw, n_steps)
    lags = n
    if tau_s is not None:
        tau = np.asarray(tau_s, dtype=float)
        if tau.ndim:
            if tau.shape != raw.shape:
                raise AppDGammaError(f"{tau.size} frame intervals for {raw.size} apparent-D values")
            tau = _with_zeros(raw, tau)[1]
        lags = n * 4.0 * tau
    try:
        return zero_jumps(x, None, lags, weighting=_NATURAL_WEIGHTING[noun], rule=rule, noun=noun,
                          lattice_in_um=tau_s is not None)
    except SQDError as exc:
        raise AppDGammaError(str(exc)) from exc


def zero_rule_of(found) -> str:
    """The ``zeros`` rule that gives a part of a data set the same populations (`sqd.zero_rule_of` for the apparent
    D): "immobile" when it holds the immobile population, "kept" when it left its zeros out, "auto" without.
    ``found`` is a gamma fit (its ``zero_tracks``) or the decision itself (`zero_tracks`' result, or None)."""
    zeros = getattr(found, "zero_tracks", found)
    if zeros is None:
        return "auto"
    return "immobile" if zeros.immobile else "kept"


def _split_log_likelihood(n_zero: int, n_all: int) -> float:
    """The zero / non-zero split's Bernoulli log-likelihood at its own share (each value once): the immobile
    population's part of a fit's likelihood (`sqd._split_log_likelihood` for unweighted values)."""
    n_moving = n_all - n_zero
    if n_all <= 0:
        return 0.0
    return float(special.xlogy(n_zero, n_zero / n_all) + special.xlogy(n_moving, n_moving / n_all))


def _refused_one(zeros) -> str:
    return (f"one population cannot be fitted to these values: {zeros.note} One population cannot hold them beside "
            f"the moving ones; fit two or more (the immobile one and those that move).")


def _shape_record(n):
    """What a fit records as its shape: the one number, or "per track" with the spread of step counts."""
    n = np.asarray(n, dtype=float)
    if n.size and np.all(n == n[0]):
        return int(n[0])
    return "per track"


def mixture_logpdf(x, D, fractions, n_steps, *, immobile: bool = False, offset=None) -> np.ndarray:
    """Log density of the mixture, via ``logsumexp`` so nothing is exponentiated early.

    A D of 0 is refused unless ``immobile`` says the model holds the immobile population of exact-zero tracks
    (module head; pass ``fit.immobile``): it is then a point mass at 0, and the density is taken against "a point at
    0 plus the line" -- an exact zero scores the immobile fraction, a moving value the gamma mixture."""
    D = np.atleast_1d(np.asarray(D, dtype=float))
    fractions = np.atleast_1d(np.asarray(fractions, dtype=float))
    if len(D) != len(fractions):
        raise AppDGammaError(f"{len(D)} D values against {len(fractions)} fractions")
    if not np.isclose(fractions.sum(), 1.0, atol=1e-6):
        raise AppDGammaError(f"fractions sum to {fractions.sum()}, not 1")
    if immobile and np.any(D == 0):
        x = np.asarray(x, dtype=float)
        stack = np.stack([np.log(max(f, 1e-300)) + (np.where(x == 0, 0.0, -np.inf) if d == 0
                                                     else gamma_logpdf(x, d, n_steps, offset))
                          for d, f in zip(D, fractions)])
        return special.logsumexp(stack, axis=0)
    stack = np.stack([np.log(max(f, 1e-300)) + gamma_logpdf(x, d, n_steps, offset)
                      for d, f in zip(D, fractions)])
    return special.logsumexp(stack, axis=0)


def posteriors(x, n_steps, D, fractions, offset=None) -> np.ndarray:
    """Each value's posterior for each population (rows: values), each with its own shape: the one formula every
    reader of a fit's per-track membership uses (the section's slow share and gallery, the cells).

    The plain arithmetic exactly, wherever no value and no D is 0. An exact zero belongs to the immobile population
    when the fit holds one (D 0, a point mass there; it has no density anywhere else); where it does not (zeros left
    out of the fit: a lattice, too few moving tracks), a zero takes the limit of the posteriors as the value goes
    to 0 -- the ``(n - 1) log x`` term every population shares cancels, leaving ``log f_k - n log(D_k / n)`` --
    rather than the 0 / 0 of the plain form. ``offset``: each value's noise offset (per-spot precision); the
    populations' shapes then differ (`shape_and_scale`), ``(k - 1) log x`` no longer cancels, and the limit goes to
    the population(s) of the SMALLEST shape for that value (the slowest: its jumps are the most correlated)."""
    x = np.asarray(x, dtype=float)
    n = np.broadcast_to(np.asarray(n_steps, dtype=float), x.shape)
    o = np.zeros(x.shape) if offset is None else _offsets_of(x, offset)
    D = np.asarray(D, dtype=float)
    f = np.asarray(fractions, dtype=float)
    zero, point = x == 0, D == 0
    if not zero.any() and not point.any():
        comp = np.stack([np.log(max(float(fk), 1e-300)) + gamma_logpdf(x, float(dk), n, o) for dk, fk in zip(D, f)])
        return np.exp(comp - special.logsumexp(comp, axis=0)).T
    out = np.zeros((len(x), len(D)))
    moving = np.flatnonzero(~point)
    on = np.flatnonzero(~zero)
    if on.size and moving.size:
        comp = np.stack([np.log(max(float(f[k]), 1e-300)) + gamma_logpdf(x[on], float(D[k]), n[on], o[on])
                         for k in moving])
        out[np.ix_(on, moving)] = np.exp(comp - special.logsumexp(comp, axis=0)).T
    at_zero = np.flatnonzero(zero)
    if at_zero.size:
        if point.any():
            out[at_zero, int(np.flatnonzero(point)[0])] = 1.0
        elif moving.size and offset is None:
            n0, o0 = n[at_zero], o[at_zero]
            comp = np.stack([np.log(max(float(f[k]), 1e-300)) - n0 * np.log((float(D[k]) + o0) / n0)
                             for k in moving])
            out[np.ix_(at_zero, moving)] = np.exp(comp - special.logsumexp(comp, axis=0)).T
        elif moving.size:
            pairs = [shape_and_scale(float(D[k]), n[at_zero], o[at_zero]) for k in moving]
            shapes = np.stack([shape for shape, _scale in pairs])
            comp = np.stack([np.log(max(float(f[k]), 1e-300)) - special.gammaln(shape) - shape * np.log(scale)
                             for k, (shape, scale) in zip(moving, pairs)])
            comp = np.where(shapes <= shapes.min(axis=0) + 1e-12, comp, -np.inf)
            out[np.ix_(at_zero, moving)] = np.exp(comp - special.logsumexp(comp, axis=0)).T
    return out


class GammaFit:
    """One gamma mixture fit. Every field is a measurement or a recorded choice."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return (f"GammaFit(n_pop={self.n_pop}, D={np.round(self.D, 4).tolist()}, "
                f"fraction={np.round(self.fraction, 4).tolist()}, "
                f"BIC={self.bic:.1f}, n_tracks={self.n_tracks})")

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "population": np.arange(1, self.n_pop + 1),
            "D_um2_per_s": self.D,
            "fraction": self.fraction,
        })


def _unpack(theta, n_pop):
    """Parameters are stored unconstrained: log D, and a softmax over fractions."""
    log_D = theta[:n_pop]
    D = np.exp(log_D)
    if n_pop == 1:
        return D, np.ones(1)
    logits = np.concatenate([[0.0], theta[n_pop:]])
    w = np.exp(logits - logits.max())
    return D, w / w.sum()


def _pack(D, fractions, n_pop):
    log_D = np.log(np.asarray(D, dtype=float))
    if n_pop == 1:
        return log_D
    f = np.maximum(np.asarray(fractions, dtype=float), 1e-12)
    return np.concatenate([log_D, np.log(f[1:] / f[0])])


def _d_step(D: float, r, n, x, o, steps: int = 4) -> float:
    """One population's D, raised in likelihood from ``D`` with each value's noise offset ``o`` (module head): the
    population's own term of the expected log-likelihood, ``h(D) = -sum r log Gamma(x; k, (D + o) / k)`` with ``r``
    the responsibilities, ``n`` the step counts and ``k`` the effective shape (`shape_and_scale`, which moves with
    D). Steps on log D along the exact gradient, scaled by the curvature the fixed-shape problem has (a
    quasi-Newton step: the shape's own second derivative is left out), each halved until ``h`` does not rise, at
    most ``steps`` of them; never below `D_FLOOR`. Without offsets the minimum is the weighted mean, closed form
    (`_em`); here it has none."""
    c = 2.0 * (n - 1.0) / n
    log_x = np.log(x)

    def h(d):
        m = d + o
        k = n / (1.0 + c * (0.5 * o / m) ** 2)
        return m, k, -float(np.sum(r * ((k - 1.0) * log_x - x * k / m - special.gammaln(k) - k * np.log(m / k))))

    D = max(float(D), D_FLOOR)
    m, k, current = h(D)
    for _ in range(steps):
        dk = k * k * c * o * o / (2.0 * n * m ** 3)                 # dk/dD: the shape rises with D, towards n
        g = -float(np.sum(r * (k * (x - m) / m ** 2                 # dh/dD at fixed shape ...
                               + (log_x - x / m - special.digamma(k) - np.log(m / k) + 1.0) * dk)))  # ... and its own
        c2 = float(np.sum(r * k * (2.0 * x - m) / m ** 3))          # d2h/dD2 at fixed shape
        gu, cu = D * g, D * g + D * D * c2                          # in log D
        step = -gu / cu if cu > 0 else -float(np.sign(gu))
        step = float(np.clip(step, -2.0, 2.0))
        for _ in range(40):
            trial = max(D * float(np.exp(step)), D_FLOOR)
            m_trial, k_trial, value = h(trial)
            if value <= current:
                break
            step /= 2.0
        else:
            return D
        moved = abs(np.log(trial / D))
        D, current, m, k = trial, value, m_trial, k_trial
        # a step this small leaves less than the next iteration's own step to gain: the gamma functions are the
        # fit's cost, so they are not evaluated again for it (the EM's tolerance is on the likelihood, unchanged)
        if moved < 1e-5:
            break
    return D


def _em(x, D0, f0, n_steps, max_iter=2000, tol=1e-9, offset=None):
    """Expectation-maximisation for a gamma mixture of KNOWN shape.

    Why this is the right optimiser, and why it is fast. With the shape fixed the M-step
    is closed form: a gamma of shape ``a`` and scale ``theta`` has mean ``a*theta``, and
    the weighted maximum-likelihood estimate of that mean is simply the weighted mean of
    the data. So each iteration is

        responsibility  r_ik  proportional to  A_k * Gamma(x_i; n, D_k/n)
        A_k = mean_i r_ik
        D_k = sum_i r_ik x_i / sum_i r_ik

    with no line search and no gradient. Every step is a numpy reduction over the whole
    data array, so it is vectorised over tracks AND over populations.

    Three outcomes, not two. ``"converged"`` met the tolerance; ``"max_iter"`` ran out of
    iterations but returned a finite, monotonically improving likelihood; ``"failed"``
    produced something unusable. **The middle one is a fit and must not be thrown away.**
    Discarding it made a three-population fit on two-population data raise, because EM
    crawls when two components are nearly identical -- which is exactly the situation
    where slow convergence is itself the answer.

    Returns ``(D, fractions, loglik, n_iter, status)``. ``offset``: each value's noise offset (per-spot
    precision); the D step is then `_d_step`, a generalised EM.
    """
    D = np.array(D0, dtype=float)
    f = np.array(f0, dtype=float)
    n_pop = len(D)
    # each value weighs by its step count in the D update: a track averaging more steps says more about its D
    nw = np.broadcast_to(np.asarray(n_steps, dtype=float), np.shape(x))
    last = -np.inf
    for it in range(max_iter):
        # E-step, in logs so nothing is exponentiated before it is normalised.
        comp = np.stack([np.log(max(fk, 1e-300)) + gamma_logpdf(x, dk, n_steps, offset)
                         for dk, fk in zip(D, f)])
        total = special.logsumexp(comp, axis=0)
        loglik = float(np.sum(total))
        if not np.isfinite(loglik):
            return D, f, -np.inf, it, "failed"
        r = np.exp(comp - total)                       # (n_pop, n_data)
        # M-step, closed form.
        weight = r.sum(axis=1)
        # A COMPONENT THAT LOSES ALL ITS WEIGHT IS INFORMATION, NOT A FAILURE. It means
        # the data does not support this many populations. Failing the whole fit here
        # made the result depend on how many starts happened to be tried; instead the
        # dead component keeps its last D, its fraction goes to zero, and the caller is
        # told. The BIC then penalises the extra parameters on its own.
        alive = weight > 0
        if not alive.any():
            return D, f, loglik, it, "failed"
        f = weight / weight.sum()
        if offset is None:
            step_weight = r @ nw
            D_new = np.where(alive, (r @ (nw * x)) / np.where(alive, step_weight, 1.0), D)
        else:
            D_new = np.array([_d_step(D[k], r[k], nw, x, offset) if alive[k] else D[k] for k in range(n_pop)])
        if np.any(~np.isfinite(D_new)) or np.any(D_new <= 0):
            return D, f, loglik, it, "failed"
        D = D_new
        if loglik - last < tol * max(1.0, abs(loglik)):
            return D, f, loglik, it + 1, "converged"
        last = loglik
    return D, f, last, max_iter, "max_iter"


def fit_appd_gamma(appd, *, n_steps, n_pop: int, n_starts: int = 24,
                   seed: int = 0, starts: str = "multistart",
                   optimiser: str = "em", zeros: str = "auto", tau_s=None,
                   noun: str = "track", offset=None) -> GammaFit:
    """Fit an ``n_pop`` gamma mixture to per-track apparent D by maximum likelihood.

    ``appd`` is one apparent D per TRACK, or per fixed-length SUB-TRACK (``appd.apparent_D``
    with ``mode="split"``, the legacy's own construction, which the workspace's bubble plot
    passes). Both are the model's exact unit: a value averaging ``n_steps`` steps of one
    Brownian state is Gamma(``n_steps``, D / ``n_steps``). Passing single steps is not.

    What sub-tracks change is what the numbers MEAN, recorded so no caller reads them as
    per-track (14 Oct 2026, the v2 -> v3 settings audit):

    * a fraction is the share of SUB-TRACKS. Each holds exactly ``n_steps`` steps, so that is
      the share of the steps used: STEP weighting, as the SQD fit's default, in which a long
      (and so preferentially slow) track weighs more than a short one;
    * ``n`` in the BIC counts sub-tracks. Those of one track share its molecule's state, so
      they are not independent draws from the mixture and the BIC leans towards more
      populations, as the SQD fit's BIC over pooled steps does.

    Neither licenses an interval from this fit. Pooled values are never the unit of an error
    bar or a test (that is the pseudoreplication this project refuses); the spread comes
    from one fit per independent unit, ``fit_appd_gamma_simultaneous`` over replicate days.

    ``starts`` is ``"multistart"`` (the fix) or ``"legacy"``, which reproduces the
    legacy's own seeding so the difference can be MEASURED rather than assumed.

    ``optimiser`` is ``"em"`` (default) or ``"nelder-mead"``. EM is exact for this model
    because the shape is fixed, so the M-step is a weighted mean with no search in it.
    Nelder-Mead is kept so the two can be compared on the same data rather than one being
    trusted.

    RAISES when no start converges. The legacy returns zeros with a console line, which a
    caller cannot distinguish from a fit.

    EXACT ZEROS (module head, ruled 4 Nov 2026): ``zeros`` "auto" decides by `zero_tracks` (``tau_s``
    the frame interval, ``noun`` what a value is), "immobile" / "kept" impose a whole data set's
    decision. Where they are an immobile population the fit holds ``n_pop`` populations, the
    immobile one first (D 0, its share counted), the rest this fit of the moving values at
    ``n_pop - 1``; one population is refused by name. Where they are not, they stay out of the fit
    as before and ``notes`` says so. No zero: the fit is exactly what it was.

    ``offset``: each value's noise offset, per-spot precision (module head); the fit records ``per_spot``.
    """
    found = zero_tracks(appd, n_steps, tau_s=tau_s, rule=zeros, noun=noun)
    x, n_shape, o = _kept(appd, n_steps, offset)
    if found is not None and found.immobile:
        if n_pop < 2:
            raise AppDGammaError(_refused_one(found))
        if not len(x):
            raise AppDGammaError(f"every value is exactly zero: {found.note} Nothing moves to fit.")
        moving = _fit_gamma(x, n_shape, n_pop - 1, n_starts=n_starts, seed=seed, starts=starts,
                            optimiser=optimiser, offset=o)
        return _with_immobile_fit(moving, found)
    fit = _fit_gamma(x, n_shape, n_pop, n_starts=n_starts, seed=seed, starts=starts, optimiser=optimiser, offset=o)
    fit.zero_tracks = found
    fit.notes = () if found is None else (found.note,)
    return fit


def _with_immobile_fit(moving: GammaFit, zeros) -> GammaFit:
    """``moving`` (the gamma fit of the moving values) with the immobile population of exact-zero values put first
    (`sqd.with_immobile`): its share counted, the likelihood and BIC gaining the zero / non-zero split's (one free
    parameter on every value), so BIC ranks the counts of moving populations as on the moving values alone."""
    from .sqd import with_immobile

    D, fractions = with_immobile(moving.D, moving.fraction, zeros.share)
    split = _split_log_likelihood(zeros.n_zero, zeros.n_jumps)
    n_all = int(zeros.n_jumps)
    out = GammaFit(**{**moving.__dict__})
    out.n_pop = int(moving.n_pop) + 1
    out.D, out.fraction = D, fractions
    out.n_tracks = n_all
    out.n_moving = int(moving.n_tracks)
    out.loglik = float(moving.loglik + split)
    out.n_parameters = int(moving.n_parameters) + 1
    out.bic = float(moving.bic + np.log(max(n_all, 1)) - 2.0 * split)
    out.aic = float(moving.aic + 2.0 - 2.0 * split)
    out.min_fraction = float(np.min(fractions[1:]))
    out.zero_tracks = zeros
    out.immobile = True
    out.notes = (zeros.note,)
    return out


def _start_range(x, o) -> tuple:
    """The 5th and 95th percentiles a start's D is drawn between: of the values, or with per-spot precision of the
    values less their own noise offsets (a bound track's value IS about its offset), kept positive."""
    if o is None:
        return tuple(float(v) for v in np.quantile(x, [0.05, 0.95]))
    lo, hi = np.quantile(x - o, [0.05, 0.95])
    hi = max(float(hi), 1e-3)
    return max(float(lo), hi * 1e-3), hi


def _fit_gamma(x, n_shape, n_pop: int, *, n_starts: int, seed: int, starts: str, optimiser: str,
               offset=None) -> GammaFit:
    """`fit_appd_gamma`'s fit itself, of the usable (positive) values ``x`` with their shapes (and, with per-spot
    precision, their noise offsets)."""
    if len(x) < MIN_VALUES_PER_POPULATION * n_pop:
        raise AppDGammaError(
            f"{len(x)} usable apparent-D values for a {n_pop}-population fit; that is too "
            f"few to estimate {n_parameters(n_pop)} parameters")
    if starts not in ("multistart", "legacy"):
        raise AppDGammaError(f"starts must be 'multistart' or 'legacy', got {starts!r}")

    rng = np.random.default_rng(seed)

    def negloglik(theta):
        D, f = _unpack(theta, n_pop)
        if not np.all(np.isfinite(D)) or np.any(D <= 0):
            return np.inf
        ll = mixture_logpdf(x, D, f, n_shape, offset=offset)
        return -float(np.sum(ll)) if np.all(np.isfinite(ll)) else np.inf

    seeds = []
    if starts == "legacy":
        # calcAppD_gamma: quantiles of the data, then twenty uniform random draws.
        q = np.quantile(x, np.linspace(0.25, 0.75, n_pop)) if n_pop > 1 else \
            np.array([np.quantile(x, 0.5)])
        seeds.append(_pack(q, np.full(n_pop, 1.0 / n_pop), n_pop))
        for _ in range(LEGACY_N_RANDOM_STARTS):
            seeds.append(_pack(np.maximum(rng.random(n_pop), 1e-3),
                               np.full(n_pop, 1.0 / n_pop), n_pop))
    else:
        lo, hi = _start_range(x, offset)
        for _ in range(n_starts):
            D0 = np.sort(np.exp(rng.uniform(np.log(lo), np.log(hi), n_pop)))
            f0 = rng.dirichlet(np.ones(n_pop))
            seeds.append(_pack(D0, f0, n_pop))
        q = (np.quantile(x, np.linspace(0.2, 0.8, n_pop)) if offset is None
             else np.exp(np.linspace(np.log(lo), np.log(hi), n_pop)))
        seeds.append(_pack(q, np.full(n_pop, 1.0 / n_pop), n_pop))

    if optimiser not in ("em", "nelder-mead"):
        raise AppDGammaError(f"optimiser must be 'em' or 'nelder-mead', got {optimiser!r}")

    best_theta, best_nll, n_converged, n_max_iter = None, np.inf, 0, 0
    for theta0 in seeds:
        if optimiser == "em":
            D0, f0 = _unpack(theta0, n_pop)
            D1, f1, ll, _, status = _em(x, D0, f0, n_shape, offset=offset)
            if status != "failed" and np.isfinite(ll):
                n_converged += 1
                if status == "max_iter":
                    n_max_iter += 1
                if -ll < best_nll:
                    best_theta, best_nll = _pack(D1, f1, n_pop), -ll
            continue
        try:
            res = optimize.minimize(negloglik, theta0, method="Nelder-Mead",
                                    options={"maxiter": 4000, "xatol": 1e-8,
                                             "fatol": 1e-8})
        except Exception:
            continue
        if res.success and np.isfinite(res.fun):
            n_converged += 1
            if res.fun < best_nll:
                best_theta, best_nll = res.x, res.fun
    if best_theta is None:
        raise AppDGammaError(
            f"no start converged for {n_pop} population(s) on {len(x)} tracks. "
            f"Try more starts, or a smaller n_pop.")

    D, fractions = _unpack(best_theta, n_pop)
    order = np.argsort(D)
    D, fractions = D[order], fractions[order]

    k = n_parameters(n_pop)
    n = len(x)
    loglik = -best_nll
    # A COLLAPSED COMPONENT IS REPORTED, NOT HIDDEN. If a fitted fraction is negligible
    # the data did not support this many populations, and a reader comparing state counts
    # needs to know that before reading the BIC.
    min_fraction = float(np.min(fractions))
    collapsed = int(np.sum(fractions < 1e-3))
    return GammaFit(
        n_pop=n_pop, D=D, fraction=fractions, n_steps=_shape_record(n_shape),
        n_tracks=n, loglik=float(loglik), n_parameters=k,
        # SUMMED LOG DENSITY, never a product. See the module docstring.
        bic=float(k * np.log(n) - 2.0 * loglik),
        aic=float(2 * k - 2.0 * loglik),
        starts=starts, optimiser=optimiser, n_starts_tried=len(seeds),
        n_starts_converged=n_converged, seed=int(seed),
        min_fraction=min_fraction, n_collapsed_components=collapsed,
        n_starts_hit_iteration_limit=n_max_iter,
        slow_convergence_note=(
            f"{n_max_iter} of {n_converged} usable starts ran out of iterations rather "
            f"than meeting the tolerance. EM crawls when two components are nearly "
            f"identical, so this is a sign the data may not support {n_pop} populations"
            if n_max_iter else ""),
        collapsed_note=(
            f"{collapsed} of {n_pop} components carry a fraction below 0.1 per cent: the "
            f"data does not support this many populations" if collapsed else ""),
        zero_tracks=None, immobile=False, notes=(), per_spot=offset is not None,
    )


def refit(appd, n_steps, D0, f0, offset=None):
    """ONE EM run from a given D and fractions (a bootstrap resample refitted from the full fit, so it cannot wander
    to another optimum): ``(D, fractions)`` sorted by D, or None when it fails or a fraction collapses to nothing.
    A D of 0 in ``D0`` is the immobile population of exact-zero values (module head): its share is the resample's
    zeros, counted, and the EM runs on the moving values from the other populations. ``offset``: per-spot
    precision's noise offset per value."""
    D0 = np.asarray(D0, dtype=float)
    f0 = np.asarray(f0, dtype=float)
    if np.any(D0 == 0):
        from .sqd import with_immobile

        x_all, _n = _with_zeros(appd, n_steps)
        if not len(x_all):
            return None
        share = float(np.mean(x_all == 0))
        keep = D0 != 0
        rest = f0[keep] / max(float(f0[keep].sum()), 1e-300)
        got = refit(appd, n_steps, D0[keep], rest, offset)
        if got is None:
            return None
        D, f = with_immobile(got[0], got[1], share)
        return D, f
    x, n, o = _kept(appd, n_steps, offset)
    if len(x) < MIN_VALUES_PER_POPULATION * len(np.atleast_1d(D0)):
        return None
    D, f, ll, _, status = _em(x, np.asarray(D0, dtype=float), np.asarray(f0, dtype=float), n, offset=o)
    if status == "failed" or not np.isfinite(ll):
        return None
    order = np.argsort(D)
    return D[order], f[order]


def fractions_at(appd, n_steps, D, f0=None, max_iter=2000, tol=1e-9, offset=None):
    """The fractions of ``appd`` (each value its own shape ``n_steps``) with the populations' D HELD at ``D``: EM on
    the fractions alone, from ``f0`` (equal shares when None). What a replicate, a chunk or a cell reads at a D shared
    with others -- SQD's `aggregate._fit_fractions` for the apparent D (3 Nov 2026, the section's Force same D across).
    Returns the fractions in ``D``'s order, or None when no value is usable.

    A D of 0 is the immobile population of exact-zero values (module head): its fraction is their share, counted
    (0 where this part has none), and the others are this EM on the moving values, scaled by the share that moves
    -- the EM against "a point at 0 plus the line", as `aggregate._fit_fractions` does for jumps. ``offset``: per-spot
    precision's noise offset per value."""
    D = np.asarray(D, dtype=float)
    if np.any(D == 0):
        x_all, _n = _with_zeros(appd, n_steps)
        if not len(x_all):
            return None
        point = D == 0
        if int(point.sum()) != 1:
            raise AppDGammaError(f"one immobile population (D 0) beside moving ones, got D = {D.tolist()}")
        share = float(np.mean(x_all == 0))
        out = np.zeros(len(D))
        out[point] = share
        if share < 1.0:
            rest = None if f0 is None else np.asarray(f0, dtype=float)[~point]
            moving = fractions_at(appd, n_steps, D[~point], None if rest is None or rest.sum() <= 0 else rest,
                                  max_iter=max_iter, tol=tol, offset=offset)
            if moving is None:
                return None
            out[~point] = (1.0 - share) * moving
        return out
    x, n, o = _kept(appd, n_steps, offset)
    if not len(x) or not len(D):
        return None
    f = np.full(len(D), 1.0 / len(D)) if f0 is None else np.maximum(np.asarray(f0, dtype=float), 1e-12)
    f = f / f.sum()
    if len(D) == 1:
        return np.ones(1)
    dens = np.stack([gamma_logpdf(x, float(d), n, o) for d in D])       # (n_pop, n_values), fixed: computed once
    last = -np.inf
    for _ in range(max_iter):
        comp = np.log(np.maximum(f, 1e-300))[:, None] + dens
        total = special.logsumexp(comp, axis=0)
        loglik = float(np.sum(total))
        if not np.isfinite(loglik):
            return None
        f = np.exp(comp - total).mean(axis=1)
        if loglik - last < tol * max(1.0, abs(loglik)):
            break
        last = loglik
    return f / f.sum()


def legacy_bic(fit: GammaFit, appd) -> dict:
    """Reproduce the legacy's multiplied-density BIC beside the summed-log one.

    So "the old criterion could not choose a state count" is a MEASURED claim on the
    dataset in hand, not an assumed one. Both are returned and neither is called the
    difference.
    """
    if getattr(fit, "immobile", False):
        raise AppDGammaError(
            "the legacy BIC is not defined for a fit holding the immobile population of exact-zero tracks (D 0, "
            "counted, not fitted); the legacy tool never met one")
    x = np.asarray(appd, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    logpdf = mixture_logpdf(x, fit.D, fit.fraction, fit.n_steps)
    with np.errstate(over="ignore"):
        product = float(np.prod(np.exp(logpdf)))
    n, k = len(x), fit.n_parameters
    legacy = float(np.log(n) * k - 2.0 * np.log(product)) if product > 0 else np.nan
    return {
        "legacy_product_of_densities": product,
        "legacy_bic": legacy,
        "legacy_bic_is_usable": bool(np.isfinite(legacy)),
        "honest_bic": fit.bic,
        "n_tracks": n,
        "why": (
            "the legacy multiplies the density over every support point. The densities mostly "
            "exceed one, so above roughly a thousand tracks the product reaches infinity and "
            "the criterion is minus infinity for every model."),
    }


def select_n_pop_gamma(appd, *, n_steps, n_pop_values=(1, 2, 3),
                       n_starts: int = 24, seed: int = 0, zeros: str = "auto", tau_s=None,
                       noun: str = "track", offset=None) -> pd.DataFrame:
    """BIC across state counts, on the summed log likelihood.

    RECOMMENDS, never decides. The rule in this project is that the data chooses the
    count in the sense that it is never asserted -- not that a number overrules the
    analyst. A gap below 2 is reported as inconclusive rather than silently resolved.

    EXACT ZEROS (module head): decided ONCE (`zero_tracks`) and imposed on every count, so
    every candidate holds the same populations; where they are an immobile population one
    population is no candidate (its row says why) and ``attrs["notes"]`` says so.
    ``attrs["zero_tracks"]`` is the decision (`sqd.ZeroJumps.to_dict`), where zeros were met.
    """
    found = zero_tracks(appd, n_steps, tau_s=tau_s, rule=zeros, noun=noun)
    rule = "auto" if found is None else ("immobile" if found.immobile else "kept")
    rows = []
    for n_pop in n_pop_values:
        try:
            fit = fit_appd_gamma(appd, n_steps=n_steps, n_pop=n_pop,
                                 n_starts=n_starts, seed=seed, zeros=rule, noun=noun, offset=offset)
        except AppDGammaError as exc:
            rows.append({"n_pop": n_pop, "bic": np.nan, "loglik": np.nan,
                         "n_parameters": n_parameters(n_pop), "collapsed": np.nan,
                         "failed": str(exc)})
            continue
        rows.append({"n_pop": n_pop, "bic": fit.bic, "loglik": fit.loglik,
                     "n_parameters": fit.n_parameters,
                     "D": np.round(fit.D, 4).tolist(),
                     "fraction": np.round(fit.fraction, 4).tolist(),
                     "collapsed": fit.n_collapsed_components,
                     "n_starts_converged": fit.n_starts_converged, "failed": ""})
    table = pd.DataFrame(rows)
    if table["bic"].notna().any():
        best = table.loc[table["bic"].idxmin(), "n_pop"]
        table["delta_bic"] = table["bic"] - table["bic"].min()
        table["evidence"] = np.where(
            table["delta_bic"] < 2, "inconclusive against the best",
            np.where(table["delta_bic"] < 10, "worse", "much worse"))
        table.attrs["recommended"] = int(best)
    if found is not None:
        table.attrs["zero_tracks"] = found.to_dict()
        table.attrs["notes"] = [found.note] + ([one_population_note(found, max(n_pop_values))]
                                               if found.immobile and 1 in tuple(n_pop_values) else [])
    return table


def one_population_note(zeros, max_pop: int) -> str:
    """What Best says when the immobile population of exact-zero values keeps one population out of the comparison
    (`sqd.select_n_pop`'s note, for the apparent D)."""
    return (f"One population was not compared: {zeros.note} Best compared 2 to {max_pop} populations, the immobile "
            f"one and 1 to {max_pop - 1} that move.")


# =====================================================================================
# THE SIMULTANEOUS FIT (``AppDFit_simul.m`` / ``fit_appD_simultaneously.m``)
# =====================================================================================
#
# ONE SET OF D VALUES SHARED ACROSS UNITS, ONE SET OF FRACTIONS PER UNIT. That is the
# whole design, and it is the only thing that makes per-unit fractions comparable:
# fractions from fits with different D are not the same quantity.
#
# **WHAT A "UNIT" IS DECIDES WHETHER THIS IS SOUND, AND THE CODE CANNOT TELL YOU.** v2
# loops over ``obj.conditions``, which is whatever subfolders were loaded. In the standard
# workflow those are REPLICATE DAYS of one protein and one carbon source, and sharing D
# across days is correct -- measured on the SQD fit, it changes differences by at most
# 1.2 percentage points and does not degrade the fit. Load two carbon sources as separate
# "conditions" and the same control shares D across THEM, which is a different and much
# stronger assumption: measured, it grew the between-condition difference in 6 of 6
# enzymes while the fit got WORSE in 6 of 6 by a median factor of 3.9, and the negative
# control's difference tripled. ``unit_kind`` is therefore required and is recorded on the
# result, so a fit can never be read without knowing what was shared.
#
# TWO DEAD COMPUTATIONS IN THE ORIGINAL, recorded rather than reproduced.
#
# * ``AppDFit_simul.m:38-52`` assigns ``initialValues``, ``lb``, ``ub`` and ``xgrid``
#   INSIDE the per-condition loop, and none of them depends on the condition. Every
#   iteration overwrites the last, so only the final draw reaches the fit. Same shape as
#   the ``1:5:end`` subsampling in ``sqdFit_``: computed, then thrown away.
# * The per-condition ``BIC`` is computed and **is not among the function's return
#   values**, so nothing can read it. It also uses the multiplied-density form, which on
#   this project's data returns exactly zero.
#
# ONE THING CHECKED AND CLEAN: the parameter slicing is correct here. ``pEsts`` and
# ``paramSD`` are indexed with the same per-condition offset, unlike
# ``sqdFit_Simultaneous.m:80`` where the standard-deviation vector grows inside the loop
# and is never re-sliced. The simultaneous SQD fit has that defect; the simultaneous appD
# fit does not.

UNIT_KINDS = ("day", "cell", "condition", "movie")


def _em_shared_D(xs, D0, f0, n_steps, max_iter=2000, tol=1e-9, offsets=None):
    """EM with D SHARED across units and fractions PER unit.

    The M-step is still closed form, which is what makes this cheap: each unit's
    fractions are its own mean responsibilities, and each shared D is the responsibility
    weighted mean POOLED over every unit. ``offsets``: each unit's values' noise offsets (per-spot precision), the
    shared D step then `_d_step` on every unit's values together.
    """
    D = np.array(D0, dtype=float)
    f = [np.array(fu, dtype=float) for fu in f0]
    n_pop = len(D)
    # one shape per value of each unit (`_values_and_shapes`), or one for all
    ns = (list(n_steps) if isinstance(n_steps, (list, tuple))
          else [np.broadcast_to(np.asarray(n_steps, dtype=float), np.shape(x)) for x in xs])
    last = -np.inf
    for it in range(max_iter):
        loglik = 0.0
        resp, ok = [], True
        for i, (x, fu, nu) in enumerate(zip(xs, f, ns)):
            ou = None if offsets is None else offsets[i]
            comp = np.stack([np.log(max(fk, 1e-300)) + gamma_logpdf(x, dk, nu, ou)
                             for dk, fk in zip(D, fu)])
            total = special.logsumexp(comp, axis=0)
            if not np.all(np.isfinite(total)):
                ok = False
                break
            loglik += float(np.sum(total))
            resp.append(np.exp(comp - total))
        if not ok or not np.isfinite(loglik):
            return D, f, -np.inf, it, "failed"

        # Per-unit fractions.
        new_f = []
        for r in resp:
            w = r.sum(axis=1)
            if w.sum() <= 0:
                return D, f, loglik, it, "failed"
            new_f.append(w / w.sum())
        # Shared D: pooled over every unit, each value weighed by its step count.
        num = np.zeros(n_pop)
        den = np.zeros(n_pop)
        for r, x, nu in zip(resp, xs, ns):
            num += r @ (nu * x)
            den += r @ nu
        alive = den > 0
        if offsets is None:
            D_new = np.where(alive, num / np.where(alive, den, 1.0), D)
        else:
            x_all, o_all, n_all = np.concatenate(xs), np.concatenate(offsets), np.concatenate(ns)
            D_new = np.array([_d_step(D[k], np.concatenate([r[k] for r in resp]), n_all, x_all, o_all)
                              if alive[k] else D[k] for k in range(n_pop)])
        if np.any(~np.isfinite(D_new)) or np.any(D_new <= 0):
            return D, f, loglik, it, "failed"
        D, f = D_new, new_f
        if loglik - last < tol * max(1.0, abs(loglik)):
            return D, f, loglik, it + 1, "converged"
        last = loglik
    return D, f, last, max_iter, "max_iter"


def fit_appd_gamma_simultaneous(appd_by_unit, *, n_steps, n_pop, unit_kind,
                                n_starts=24, seed=0, zeros="auto", tau_s=None, offset=None):
    """Shared D across units, fractions per unit. ``AppDFit_simul`` reproduced.

    ``appd_by_unit`` is ``{unit_label: array of per-track apparent D}``; ``n_steps`` one number for all, or
    ``{unit_label: array}`` with each track's own step count (`appd.track_appd`).

    ``unit_kind`` has NO DEFAULT and must be one of :data:`UNIT_KINDS`. Sharing D across
    replicate DAYS is the sound version; sharing it across CONDITIONS is a much stronger
    assumption that this project has measured and does not recommend. The code cannot tell
    which you meant, so it asks.

    EXACT ZEROS (module head): decided ONCE on every unit's values together (``zeros`` "auto", or
    a whole data set's "immobile" / "kept"). Where they are an immobile population every unit
    holds it, at its OWN share of exact zeros (counted; 0 in a unit without any), and the
    ``n_pop - 1`` moving D are shared across the units' moving values -- `aggregate.fit_simultaneous`'s
    design for jumps. One population is refused by name.

    ``offset``: per-spot precision, ``{unit_label: array}`` of each value's noise offset (module head).
    """
    if unit_kind not in UNIT_KINDS:
        raise AppDGammaError(
            f"unit_kind must be one of {UNIT_KINDS}, got {unit_kind!r}. What is shared "
            f"decides what the fit means and cannot be inferred from the data.")
    labels = list(appd_by_unit)
    if len(labels) < 2:
        raise AppDGammaError(
            f"{len(labels)} unit(s): a simultaneous fit shares D ACROSS units, so it "
            f"needs at least two. Use fit_appd_gamma for one.")
    def shapes_of(label):
        return n_steps[label] if isinstance(n_steps, dict) else n_steps

    every = [_with_zeros(appd_by_unit[label], shapes_of(label)) for label in labels]
    # the frame interval: one, or ``{unit: one}`` (conditions fitted together may differ), for the lattice's size
    taus = (np.concatenate([np.full(len(x), float(tau_s[label])) for label, (x, _n) in zip(labels, every)])
            if isinstance(tau_s, dict) else tau_s)
    found = zero_tracks(np.concatenate([x for x, _n in every]), np.concatenate([n for _x, n in every]),
                        tau_s=taus, rule=zeros)
    immobile = found is not None and found.immobile
    if immobile and n_pop < 2:
        raise AppDGammaError(_refused_one(found))
    shares = [float(np.mean(x == 0)) if len(x) else 0.0 for x, _n in every]
    fitted = n_pop - 1 if immobile else n_pop
    xs, ns, os_ = [], [], []
    for label in labels:
        x, n_shape, o = _kept(appd_by_unit[label], shapes_of(label), None if offset is None else offset[label])
        if len(x) < MIN_VALUES_PER_POPULATION * fitted:
            raise AppDGammaError(
                f"unit {label!r} has {len(x)} usable values for a {n_pop}-population "
                f"fit; that is too few" + (" (the moving ones: its exact zeros are the immobile population)"
                                           if immobile else ""))
        xs.append(x)
        ns.append(n_shape)
        os_.append(o)
    offsets = None if offset is None else os_
    if immobile:
        return _simultaneous_with_immobile(xs, ns, labels, every, shares, found, n_pop, unit_kind,
                                           n_starts=n_starts, seed=seed, offsets=offsets)

    rng = np.random.default_rng(seed)
    pooled = np.concatenate(xs)
    lo, hi = _start_range(pooled, None if offsets is None else np.concatenate(offsets))
    best, best_ll, n_conv, n_max = None, -np.inf, 0, 0
    for _ in range(n_starts):
        D0 = np.sort(np.exp(rng.uniform(np.log(lo), np.log(hi), n_pop)))
        f0 = [rng.dirichlet(np.ones(n_pop)) for _ in xs]
        D1, f1, ll, _, status = _em_shared_D(xs, D0, f0, ns, offsets=offsets)
        if status != "failed" and np.isfinite(ll):
            n_conv += 1
            if status == "max_iter":
                n_max += 1
            if ll > best_ll:
                best, best_ll = (D1, f1), ll
    if best is None:
        raise AppDGammaError(
            f"no start converged for {n_pop} population(s) across {len(labels)} units")

    D, fractions = best
    order = np.argsort(D)
    D = D[order]
    fractions = [fu[order] for fu in fractions]

    # PARAMETERS: n_pop shared D values, plus (n_pop - 1) free fractions per unit.
    k = n_pop + len(labels) * (n_pop - 1)
    n = int(sum(len(x) for x in xs))
    per_unit = pd.DataFrame(
        [{"unit": lab, "n_tracks": len(x),
          **{f"fraction_{i + 1}": float(fu[i]) for i in range(n_pop)}}
         for lab, x, fu in zip(labels, xs, fractions)])
    return GammaFit(
        n_pop=n_pop, D=D, fraction=np.mean(fractions, axis=0), per_unit=per_unit,
        fractions_by_unit=fractions, unit_labels=labels, unit_kind=unit_kind,
        n_units=len(labels), n_steps=_shape_record(np.concatenate(ns)), n_tracks=n,
        loglik=float(best_ll),
        n_parameters=k, bic=float(k * np.log(n) - 2.0 * best_ll),
        aic=float(2 * k - 2.0 * best_ll), starts="multistart", optimiser="em",
        n_starts_tried=n_starts, n_starts_converged=n_conv,
        n_starts_hit_iteration_limit=n_max, seed=int(seed),
        min_fraction=float(np.min([fu.min() for fu in fractions])),
        n_collapsed_components=int(sum(int((fu < 1e-3).sum()) for fu in fractions)),
        collapsed_note="", slow_convergence_note="",
        shared_note=_shared_note(len(labels), unit_kind),
        zero_tracks=found, immobile=False, notes=() if found is None else (found.note,), per_spot=offsets is not None,
    )


def _shared_note(n_units: int, unit_kind: str) -> str:
    return (f"D is shared across {n_units} {unit_kind}(s) and the fractions are "
            f"fitted per {unit_kind}. Sharing across replicate DAYS is the sound version; "
            f"sharing across CONDITIONS grew the between-condition difference in 6 of 6 "
            f"enzymes while the fit got worse in 6 of 6, and tripled the negative "
            f"control's difference")


def _simultaneous_with_immobile(xs, ns, labels, every, shares, found, n_pop, unit_kind, *, n_starts, seed,
                                offsets=None):
    """`fit_appd_gamma_simultaneous` where the exact zeros are an immobile population: the ``n_pop - 1`` moving D
    shared across the units' moving values (the same EM, from the same seeded starts), each unit's own share of
    zeros counted and its moving fractions scaled by the rest (`sqd.with_immobile`)."""
    from .sqd import with_immobile

    k_moving = n_pop - 1
    rng = np.random.default_rng(seed)
    pooled = np.concatenate(xs)
    lo, hi = _start_range(pooled, None if offsets is None else np.concatenate(offsets))
    best, best_ll, n_conv, n_max = None, -np.inf, 0, 0
    for _ in range(n_starts):
        D0 = np.sort(np.exp(rng.uniform(np.log(lo), np.log(hi), k_moving)))
        f0 = [rng.dirichlet(np.ones(k_moving)) for _ in xs]
        D1, f1, ll, _, status = _em_shared_D(xs, D0, f0, ns, offsets=offsets)
        if status != "failed" and np.isfinite(ll):
            n_conv += 1
            if status == "max_iter":
                n_max += 1
            if ll > best_ll:
                best, best_ll = (D1, f1), ll
    if best is None:
        raise AppDGammaError(
            f"no start converged for {k_moving} moving population(s) across {len(labels)} units")
    D_moving, moving_f = best
    order = np.argsort(D_moving)
    D_moving = D_moving[order]
    D = with_immobile(D_moving, np.ones(k_moving), 0.0)[0]
    fractions = [with_immobile(D_moving, fu[order], share)[1] for fu, share in zip(moving_f, shares)]
    split = sum(_split_log_likelihood(int(np.sum(x == 0)), len(x)) for x, _n in every)
    # PARAMETERS: the shared moving D, each unit's (n_pop - 2) free moving fractions, and each unit's zero share
    k = k_moving + len(labels) * (k_moving - 1) + len(labels)
    n = int(sum(len(x) for x, _n in every))
    loglik = float(best_ll + split)
    per_unit = pd.DataFrame(
        [{"unit": lab, "n_tracks": len(x_all),
          **{f"fraction_{i + 1}": float(fu[i]) for i in range(n_pop)}}
         for lab, (x_all, _n), fu in zip(labels, every, fractions)])
    return GammaFit(
        n_pop=n_pop, D=D, fraction=np.mean(fractions, axis=0), per_unit=per_unit,
        fractions_by_unit=fractions, unit_labels=labels, unit_kind=unit_kind,
        n_units=len(labels), n_steps=_shape_record(np.concatenate(ns)), n_tracks=n,
        loglik=loglik, n_parameters=k, bic=float(k * np.log(n) - 2.0 * loglik),
        aic=float(2 * k - 2.0 * loglik), starts="multistart", optimiser="em",
        n_starts_tried=n_starts, n_starts_converged=n_conv,
        n_starts_hit_iteration_limit=n_max, seed=int(seed),
        min_fraction=float(np.min([fu[1:].min() for fu in fractions])),
        n_collapsed_components=int(sum(int((fu[1:] < 1e-3).sum()) for fu in fractions)),
        collapsed_note="", slow_convergence_note="",
        shared_note=_shared_note(len(labels), unit_kind),
        zero_tracks=found, immobile=True, notes=(found.note,), per_spot=offsets is not None,
    )
