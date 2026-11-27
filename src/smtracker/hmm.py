"""Hidden Markov model of diffusive states: the estimator that measures switching.

Why this module exists
----------------------
SQD and appD both assume a molecule's state is *fixed* while it is observed:
SQD fits the pooled distribution of single steps, appD averages several steps
into one D. Neither can express "this molecule was slow, then bound, then fast".
On the reference data those two estimators disagree in exactly the way
state-switching predicts (see ``smtracker.appd``), which makes the switching
itself the thing worth measuring rather than a nuisance to average over.

An HMM models it directly:

- ``K`` diffusive states with apparent coefficients ``D_1..D_K``;
- a Markov **transition matrix** ``A`` giving the per-frame probability of
  moving between them;
- per-step posterior occupancies, so fractions are **switching-corrected**
  rather than blurred.

This is the vbSPT model class (Persson et al. 2013). vbSPT itself is MATLAB-only
and so cannot ship inside a Python installer: the model is therefore
implemented natively here. A canonical-table export for an external vbSPT
cross-check is **not implemented**: this docstring previously cited a
``smtracker.adapters`` module that has never existed (AP2). The legacy
``exp2SMTrack.m`` export is recorded as a genuine parity gap in
``docs/analysis_inventory.md``; do not go looking for the module.

Emission model
--------------
For a single step of one 2-D state, the squared displacement is exponential:

    r²  ~  Exponential(mean = 4·D_k·τ + 4·σ_loc²)

exact for Brownian motion with isotropic localization error ``σ_loc``. Fitting
uses that likelihood directly: no binning, no CDF.

What it answers that the static fits cannot
-------------------------------------------
- **Switching rate**: ``k_ij = A_ij / τ`` per second, and dwell time
  ``τ_k = τ / (1 − A_kk)``.
- **Switching-corrected fractions**: the time-averaged posterior occupancy,
  which is what SQD's fractions approximate and appD's averaging distorts.
- **Whether an extra state is real**: a state that appears only because
  molecules are caught mid-transition shows up as a state with *no dwell time*,
  it is entered and left within a frame or two. An HMM separates that from a
  genuine long-lived population, which the static fits cannot.

Modeling assumptions (what this model IS and IS NOT)
----------------------------------------------------
- **Free diffusion per state.** Each state's emission is the Gaussian-displacement
  (exponential-r²) likelihood of free Brownian motion with localization noise:
  ``mean r² = 4·D·τ + 4·σ²``. σ is folded into the emission likelihood here, not
  corrected post hoc: vbSPT-correct, and necessary because at D_slow ≈ 0.05 /
  20 ms the slow step (~63 nm) ≈ σ, so without it the slow state and the noise
  floor blur. σ = 0 (apparent D) is the DEFAULT for cross-method comparability;
  σ-aware is the option.
- **The slow state is immobile-plus-localization-noise, NOT confined diffusion.**
  A molecule genuinely bouncing inside a small cage has a different displacement
  distribution; this model does not fit that. Geometric confinement is handled
  by the SEPARATE detector in ``smtracker.confinement``, deliberately, so the
  two questions ("what fraction is slow" vs "where does it dwell") do not
  contaminate each other.

Honest limits
-------------
- Short tracks carry little transition information; a 5-localization track has
  4 steps, so per-track evidence about ``A`` is weak and ``A`` is only estimable
  by pooling many tracks.
- ``D`` values remain **apparent** and bounded by the measurement limits.
- EM finds a local optimum: multistart is mandatory here, and the spread across
  starts is reported.
- Localization error is a *fixed input*, not fitted, and per ``CLAUDE.md`` it is
  itself uncertain (30–76 nm), so vary it and check the conclusions hold.
- Rates are only identifiable where switching is observed. Occupancy /
  fractions are well determined regardless: a fraction is "what share of steps
  look slow" and needs no transitions. A transition RATE needs observed
  transitions, so a low-switching condition yields honestly *unbounded* rate
  intervals (as one lab condition did: an off-rate CI reaching down to ~0.001 s⁻¹, some resamples with
  no transition at all), NOT point estimates. This is a known, expected limit of
  the method, reproduced from known truth in ``smtracker.recovery``'s
  ``rate_identifiability_map``, not a defect. Quote fractions freely; quote a
  rate only with its bootstrap interval, and never quote a dwell for a condition
  that barely switches.
- Fail loud: no convergence, no result.
"""

from __future__ import annotations

from .chart_text import unit_word  # the word shown for a grouping unit

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .canonical import CANONICAL_COLUMNS
from .parallel import bootstrap_map, resample_indices
from .canonical import without_attrs

__all__ = ["NOISE_D_FLOOR", "d_at_floor", 
    "HMMError",
    "HMMFit",
    "HMMPlan",
    "step_sequences",
    "fit_hmm",
    "plan_hmm",
    "run_hmm_start",
    "finish_hmm",
    "select_n_states",
    "decode_states",
    "bootstrap_hmm",
    "BOOTSTRAP_UNITS",
]

_NDIM = 2

#: Below this, a stationary probability is a point mass and decoding is
#: decided by the prior rather than the data. See ``decode_states``.
_DEGENERATE_STATIONARY = 1e-8
#: Multistart count per state count, measured, the same treatment `sqd.default_starts`
#: gives the SQD fit, and for the same reason: a flat number applied to every state count
#: is a guess, and this one cost 9x for nothing at K = 2 (17.3 s against 1.9 s for one
#: start, with the answers identical to every digit reported).
#:
#: METHOD, and it is the SQD sweep's logic rather than a new one. Many single-start fits
#: from different seeds, recording each one's log-likelihood; the fraction `p` reaching
#: the best optimum gives the requirement directly, since P(all n starts miss) =
#: (1 - p)^n. EVERY START REACHED THE OPTIMUM in every group and state count measured
#: (`docs/hmm_starts.md`), so the measured requirement is ONE.
#:
#: WHY THAT DIFFERS FROM SQD, which genuinely needs 16 starts at 3 populations on 4 of 18
#: groups: Baum-Welch maximises a likelihood by a monotone EM update, where the SQD fit
#: is a least-squares descent on an ECDF whose surface carries local minima exactly where
#: a population is small (a few per cent occupancy). Better conditioning, not luck.
#:
#: SO WHY NOT SHIP 1? The measurement is one, not zero, and this file records four
#: separate occasions where a constant measured in one regime was applied to another and
#: was wrong. Four is a margin over a measured requirement, in the SQD tradition of
#: shipping 2-8x it, and still a THIRD of the old flat 12.
#:
#: ABOVE THREE STATES IS UNMEASURED and keeps the old number. The SQD sweep found the
#: requirement climbing sharply with the count (1 -> 24 -> 100 under `mle`), so carrying
#: a cheap number upward is the mistake this comment exists to prevent.
STARTS_BY_N_STATES = {1: 4, 2: 4, 3: 4}
_STARTS_ABOVE = 12


def default_starts(n_states: int) -> int:
    """Multistart count for ``n_states``, from the measurements above.

    Exposed rather than inlined so a caller can report which count it used, and so
    lowering one is a documented edit to a named constant, not a call-site literal.
    """
    if n_states < 1:
        raise HMMError(f"n_states must be >= 1, got {n_states}")
    return STARTS_BY_N_STATES.get(int(n_states), _STARTS_ABOVE)


_DEFAULT_STARTS = 12                # the pre-measurement flat count, kept for reference only
_MAX_ITER = 300
_TOLERANCE = 1e-7


class HMMError(RuntimeError):
    """Raised when the HMM cannot be built or fitted."""


@dataclass
class HMMFit:
    """A fitted diffusive-state HMM. States are sorted by ascending ``D``."""

    n_states: int
    D: np.ndarray                       # µm²/s, ascending
    transition_matrix: np.ndarray       # per-frame probabilities, rows sum to 1
    occupancy: np.ndarray               # time-averaged posterior fractions
    stationary: np.ndarray              # stationary distribution of A
    tau_s: float
    n_steps: int
    n_sequences: int
    log_likelihood: float
    bic: float
    localization_sigma_um: float = 0.0
    n_starts: int = 1
    n_converged_starts: int = 0
    start_spread: dict = field(default_factory=dict)
    #: Fitted distribution over states at the FIRST step of a track. Kept because
    #: ``occupancy`` and ``stationary`` are DIFFERENT quantities and the gap between
    #: them is explained by this one: a track observed for 5-6 steps does not reach
    #: stationarity, so its time-average sits between the initial distribution and
    #: the stationary one. Reporting occupancy alone invites the reader to check it
    #: against ``k_hi_lo/(k_hi_lo + k_lo_hi)`` and find a discrepancy that is not an
    #: error. ``None`` for fits built before this field existed.
    initial_distribution: np.ndarray | None = None
    #: PER-SPOT PRECISION (6 Nov 2026, `smtracker.precision`): the calibration (c in nm, p) each step's noise was made
    #: with, sigma_true = c reported^p, GIVEN to the fit, never learned by it; `decode_states` rebuilds the same noise
    #: from it. None: the model as before (one ``localization_sigma_um`` or none). The D's are then corrected, not
    #: apparent.
    precision_calibration: tuple | None = None

    @property
    def per_spot_precision(self) -> bool:
        return self.precision_calibration is not None

    @property
    def occupancy_percent(self) -> np.ndarray:
        return self.occupancy * 100.0

    @property
    def dwell_times_s(self) -> np.ndarray:
        """Mean time in each state: ``τ / (1 − A_kk)``. ``inf`` if never left."""
        stay = np.diag(self.transition_matrix)
        with np.errstate(divide="ignore"):
            return np.where(stay >= 1.0, np.inf, self.tau_s / (1.0 - stay))

    @property
    def rates_per_s(self) -> np.ndarray:
        """Transition rates ``A_ij / τ`` (off-diagonal), s⁻¹."""
        rates = self.transition_matrix / self.tau_s
        np.fill_diagonal(rates, 0.0)
        return rates

    @property
    def is_absorbing(self) -> np.ndarray:
        """States the chain can enter but never leave (``A_kk`` = 1).

        A fit like this means **no transition out of that state was observed**,
        which happens when the data contains no *within-track* switching: every
        track is entirely slow or entirely fast. The fit is then not wrong about
        D or occupancy, but its chain is reducible, and its stationary
        distribution collapses onto the absorbing state. ``decode_states``
        decodes it under the fitted initial distribution, each track in one
        state for its whole length (``attrs["no_switching"]``; it refused before
        4 Nov 2026, when the stationary distribution was its prior).
        """
        return np.isclose(np.diag(self.transition_matrix), 1.0)

    @property
    def has_degenerate_chain(self) -> bool:
        """Is the stationary distribution a point mass on one state?"""
        return bool(np.any(self.is_absorbing)) or bool(
            np.min(self.stationary) < _DEGENERATE_STATIONARY
        )

    @property
    def no_switching(self) -> bool:
        """No WITHIN-TRACK switching was fitted (a degenerate chain of two or more states): `decode_states` then
        decodes each track in one state for its whole length, and an HMM map says so."""
        return bool(self.has_degenerate_chain and int(self.n_states) > 1)

    def is_transient(self, threshold_frames: float = 2.0) -> np.ndarray:
        """States whose dwell time is shorter than ``threshold_frames`` frames.

        A state that is entered and left within a frame or two is the signature
        of molecules caught mid-transition rather than a real population, the
        check that distinguishes a genuine extra state from a switching artifact.
        """
        return self.dwell_times_s < threshold_frames * self.tau_s

    def summary(self) -> str:
        parts = [
            f"state{i + 1}: {occ * 100:.1f} %, D = {d:.4g} µm²/s, "
            f"dwell = {dwell * 1000:.0f} ms"
            for i, (occ, d, dwell) in enumerate(
                zip(self.occupancy, self.D, self.dwell_times_s)
            )
        ]
        return (
            f"HMM {self.n_states}-state fit ({self.n_steps} steps, "
            f"{self.n_sequences} tracks): " + "; ".join(parts)
            + f" | BIC = {self.bic:.1f}"
        )

    def transition_table(self) -> pd.DataFrame:
        """Per-frame transition probabilities and per-second rates."""
        rows = []
        for i in range(self.n_states):
            for j in range(self.n_states):
                if i == j:
                    continue
                rows.append(
                    {
                        "from_state": i + 1,
                        "to_state": j + 1,
                        "probability_per_frame": self.transition_matrix[i, j],
                        "rate_per_s": self.transition_matrix[i, j] / self.tau_s,
                    }
                )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        out = {
            "n_states": self.n_states,
            "n_steps": self.n_steps,
            "log_likelihood": self.log_likelihood,
            "bic": self.bic,
        }
        for i, (d, occ, dwell) in enumerate(
            zip(self.D, self.occupancy, self.dwell_times_s), start=1
        ):
            out[f"D_{i}"] = d
            out[f"occupancy_{i}"] = occ
            out[f"dwell_s_{i}"] = dwell
        return out

    # ``NPopSelection`` compatibility (BIC recommends, the analyst decides).
    @property
    def n_pop(self) -> int:
        return self.n_states

    @property
    def fractions(self) -> np.ndarray:
        return self.occupancy

    @property
    def fractions_percent(self) -> np.ndarray:
        return self.occupancy * 100.0


def step_sequences(canonical: pd.DataFrame, *, min_steps: int = 2, calibration=None):
    """Ordered squared-displacement sequences, one per contiguous track run.

    A track is split wherever frames are not consecutive: a bridged gap breaks
    the Markov chain, so the two sides must not be treated as adjacent steps.
    Sequences shorter than ``min_steps`` carry no transition information and are
    dropped.

    With a ``calibration`` (`precision.Calibration`, per-spot precision) it returns ``(sequences, noises)``: each
    step's own per-axis localisation variance, s_a^2 + s_b^2 through the calibration (um^2), in the same runs; a table
    without the tracker's per-spot precision, or a spot without one, is refused by name, never filled in.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise HMMError(f"not a canonical table (missing {missing})")
    if min_steps < 1:
        raise HMMError(f"min_steps must be >= 1, got {min_steps}")
    if calibration is not None:
        from .canonical import LOC_SIGMA_COLUMN

        if LOC_SIGMA_COLUMN not in canonical.columns:
            raise HMMError("these tracks carry no per-spot precision: the tracker wrote none (u-track does; TrackMate "
                           "does when its spots were fitted, `spotfit`; track tables do not). Per-spot precision "
                           "cannot be used on them.")
        from .precision import MAX_MISSING_SHARE

        values = canonical[LOC_SIGMA_COLUMN].to_numpy(dtype=float)
        lacking = int((~np.isfinite(values)).sum())
        if lacking > MAX_MISSING_SHARE * len(values):
            raise HMMError(f"{lacking} of {len(values)} spots have no precision, more than the "
                           f"{100 * MAX_MISSING_SHARE:.0f} % a failed fit here and there explains: refused, never filled in")

    sequences, noises = [], []
    ordered = canonical.sort_values(["uid", "frame"], kind="stable")
    for _, track in without_attrs(ordered).groupby("uid", sort=False, observed=True):
        frames = track["frame"].to_numpy()
        x = track["x_um"].to_numpy()
        y = track["y_um"].to_numpy()
        squared = np.diff(x) ** 2 + np.diff(y) ** 2
        contiguous = np.diff(frames) == 1
        if calibration is not None:
            sigma = calibration.true_sigma_um(track[LOC_SIGMA_COLUMN].to_numpy(dtype=float))
            noise = sigma[:-1] ** 2 + sigma[1:] ** 2
            # a step touching a spot without precision (a failed fit, at most `MAX_MISSING_SHARE`) ends the run
            # there, as a gap does: never filled in
            contiguous = contiguous & np.isfinite(noise)
        else:
            noise = np.zeros(len(squared))
        run: list[int] = []
        for index, ok in enumerate(contiguous):
            if ok:
                run.append(index)
            else:
                if len(run) >= min_steps:
                    sequences.append(squared[run])
                    noises.append(noise[run])
                run = []
        if len(run) >= min_steps:
            sequences.append(squared[run])
            noises.append(noise[run])

    if not sequences:
        raise HMMError(
            f"no contiguous run of {min_steps} steps could be formed. Tracks may "
            "be too short or entirely gapped."
        )
    return (sequences, noises) if calibration is not None else sequences


def fit_hmm(
    sequences,
    *,
    n_states: int,
    tau_s: float,
    localization_sigma_um: float = 0.0,
    n_starts: int | None = None,
    seed: int | None = 0,   # STOCHASTIC: random multistart init. Capture as key material.
    max_iter: int = _MAX_ITER,
    tolerance: float = _TOLERANCE,
    init: "tuple | None" = None,   # (means, A, pi) warm-start for start 0 (bootstrap)
    fixed_D=None,
    noise_sequences=None,
    calibration=None,
) -> HMMFit:
    """Fit the diffusive-state HMM by Baum-Welch (EM), keeping the best start.

    ``sequences`` is a list of squared-displacement arrays (from
    ``step_sequences``) or a canonical table, which is converted first.

    PER-SPOT PRECISION (6 Nov 2026, `smtracker.precision`): with a ``calibration`` (a canonical table is then turned
    into steps and each step's own noise; a list of sequences needs its ``noise_sequences`` beside it, from
    `step_sequences(..., calibration=...)`), state k's mean r^2 for step i is 4 D_k tau + 2 noise_i, so a bound step
    whose spots were placed badly is expected long. Each state's D is then fitted by a bounded search in every EM
    round (`_baum_welch_noise`; no closed form). The calibration is GIVEN, never learned (ruled 6 Nov 2026); the fit
    keeps it (``precision_calibration``) for `decode_states`. Not beside ``localization_sigma_um``.

    ``fixed_D`` HOLDS the states' D (µm²/s, one per state; 3 Nov 2026, asked: "can we do a simultanous fit over
    conditions for HMM as well?"): only the transitions and the initial distribution are fitted, so replicates or
    conditions fitted at one set of D are compared on the SAME states -- a rate "into the middle state" then means
    the same state in each. The D come from a fit of the pooled data (`hmm_view`'s Force same D across); the BIC
    counts no D as free.

    THREE PARTS, run here one after another (4 Nov 2026): `plan_hmm` checks the data and draws every start's initial
    guess, `run_hmm_start` is one start's EM, `finish_hmm` keeps the best. The starts are independent, so a caller
    with a pool may run them as jobs of their own (`hmm_run.joint_D`) and get exactly this fit.
    """
    plan = plan_hmm(sequences, n_states=n_states, tau_s=tau_s, localization_sigma_um=localization_sigma_um,
                    n_starts=n_starts, seed=seed, max_iter=max_iter, tolerance=tolerance, init=init,
                    fixed_D=fixed_D, noise_sequences=noise_sequences, calibration=calibration)
    return finish_hmm(plan, [run_hmm_start(plan, start) for start in range(plan.n_starts)])


def _start_draws(n_states: int, n_starts: int, seed) -> list:
    """Every start's random initial guess: ``None`` for start 0 (the fixed spread, or a warm start), else ``(factors,
    A, pi)``, DRAWN IN START ORDER FROM ONE GENERATOR, as the starts drew them one after another when they ran in one
    loop. The EM itself draws nothing, so a start given its draw here begins exactly where it began in the loop,
    whichever process runs it and whenever."""
    rng = np.random.default_rng(seed)
    draws: list = [None]
    for _start in range(1, int(n_starts)):
        factors = np.sort(np.exp(rng.uniform(np.log(0.05), np.log(5.0), size=n_states)))
        A = rng.dirichlet(np.ones(n_states) * 5.0, size=n_states)
        pi = rng.dirichlet(np.ones(n_states))
        draws.append((factors, A, pi))
    return draws


class HMMPlan:
    """One HMM fit before any start has run: the checked steps, the settings, and every start's initial guess
    (`_start_draws`). Made by `plan_hmm`; `run_hmm_start(plan, s)` runs start ``s``, `finish_hmm` keeps the best.

    SPLIT OUT (4 Nov 2026) so the starts can run as independent POOL JOBS: the joint fit over conditions ran Best's
    twelve starts (four per state count) one after another in one job, over ten minutes on one user's protein F with one
    worker of seven busy. PICKLES SMALL: the sequences travel end to end as one array with their lengths
    (``__getstate__``), never as thousands of little arrays."""

    def __init__(self, sequences, *, n_states, tau_s, localization_sigma_um, n_starts, seed, max_iter, tolerance,
                 init, fixed_D, fixed_means, offset, n_steps, noises=None, precision_calibration=None):
        self._sequences = list(sequences)
        #: per-spot precision: each sequence's per-step noise (um^2), or None; and the calibration's (c in nm, p)
        self._noises = None if noises is None else list(noises)
        self.precision_calibration = precision_calibration
        self.n_states = int(n_states)
        self.tau_s = float(tau_s)
        self.localization_sigma_um = float(localization_sigma_um)
        self.n_starts = int(n_starts)
        self.seed = seed
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.init = init
        self.fixed_D = fixed_D
        self.fixed_means = fixed_means
        self.offset = offset
        self.n_steps = int(n_steps)
        self.n_sequences = len(self._sequences)
        self.draws = _start_draws(self.n_states, self.n_starts, seed)

    @property
    def sequences(self) -> list:
        return self._sequences

    @property
    def noises(self) -> "list | None":
        return self._noises

    def __getstate__(self):
        state = dict(self.__dict__)
        sequences = state.pop("_sequences")
        noises = state.pop("_noises")
        state["_lengths"] = np.array([len(s) for s in sequences], dtype=np.int64)
        state["_values"] = np.concatenate(sequences) if sequences else np.zeros(0)
        state["_noise_values"] = None if noises is None else (np.concatenate(noises) if noises else np.zeros(0))
        return state

    def __setstate__(self, state):
        state = dict(state)
        lengths, values = state.pop("_lengths"), state.pop("_values")
        noise_values = state.pop("_noise_values", None)
        cuts = np.cumsum(lengths)[:-1]
        state["_sequences"] = np.split(values, cuts) if len(lengths) else []
        state["_noises"] = None if noise_values is None else (np.split(noise_values, cuts) if len(lengths) else [])
        state.setdefault("precision_calibration", None)
        self.__dict__.update(state)


def plan_hmm(
    sequences,
    *,
    n_states: int,
    tau_s: float,
    localization_sigma_um: float = 0.0,
    n_starts: int | None = None,
    seed: int | None = 0,
    max_iter: int = _MAX_ITER,
    tolerance: float = _TOLERANCE,
    init: "tuple | None" = None,
    fixed_D=None,
    noise_sequences=None,
    calibration=None,
) -> HMMPlan:
    """`fit_hmm`'s first part: check the data and the settings and draw every start's initial guess. RAISES
    `HMMError` on what `fit_hmm` refuses, before any start runs. Same keywords as `fit_hmm`."""
    if isinstance(sequences, pd.DataFrame):
        params = sequences.attrs.get("params")
        if params is not None and tau_s is None:
            tau_s = float(params.tau_s)
        if calibration is not None:
            if noise_sequences is not None:
                raise HMMError("a canonical table makes its own noise from the calibration: give no noise_sequences")
            sequences, noise_sequences = step_sequences(sequences, calibration=calibration)
        else:
            sequences = step_sequences(sequences)
    sequences = [np.asarray(s, dtype=float).ravel() for s in sequences]
    noises = None
    if noise_sequences is not None:
        if calibration is None:
            raise HMMError("per-spot precision needs the calibration its noise was made with (it is kept with the "
                           "fit, for decoding)")
        if localization_sigma_um:
            raise HMMError("per-spot precision replaces the one static error (localization_sigma_um): give one, not "
                           "both")
        noises = [np.asarray(n, dtype=float).ravel() for n in noise_sequences]
        if len(noises) != len(sequences) or any(len(n) != len(s) for n, s in zip(noises, sequences)):
            raise HMMError("noise_sequences must hold one noise per step, sequence for sequence")
        if not all(np.all(np.isfinite(n)) and np.all(n >= 0) for n in noises):
            raise HMMError("noise_sequences hold a non-finite or negative variance")
    elif calibration is not None:
        raise HMMError("a calibration without noise: pass a canonical table, or its noise_sequences")

    if n_states < 1:
        raise HMMError(f"n_states must be >= 1, got {n_states}")
    if tau_s is None or tau_s <= 0:
        raise HMMError(f"tau_s must be > 0, got {tau_s}")
    # RESOLVED AFTER `n_states` IS VALIDATED, so a bad state count is refused before it
    # can index the table. An explicit count always wins: `None` means "use the
    # measured default", the same contract `sqd.fit_sqd` uses for `n_starts`.
    if n_starts is None:
        n_starts = default_starts(n_states)
    elif n_starts < 1:
        raise HMMError(f"n_starts must be >= 1, got {n_starts}")
    if not sequences:
        raise HMMError("no sequences to fit")
    all_steps = np.concatenate(sequences)
    if not np.isfinite(all_steps).all():
        raise HMMError("sequences contain non-finite squared displacements")
    if np.any(all_steps < 0):
        raise HMMError("squared displacements cannot be negative")
    if len(all_steps) < 4 * n_states:
        raise HMMError(
            f"only {len(all_steps)} steps for a {n_states}-state HMM; refusing to "
            "fit an underdetermined model."
        )
    if np.ptp(all_steps) <= 0:
        raise HMMError("all squared displacements are identical; nothing to fit")

    offset = 2 * _NDIM * localization_sigma_um**2
    fixed_means = None
    if fixed_D is not None:
        fixed_D = np.sort(np.asarray(fixed_D, dtype=float).ravel())
        if len(fixed_D) != n_states:
            raise HMMError(f"{len(fixed_D)} fixed D values for a {n_states}-state HMM")
        if not np.all(np.isfinite(fixed_D)) or np.any(fixed_D <= 0):
            raise HMMError(f"fixed D must be positive and finite, got {fixed_D.tolist()}")
        fixed_means = fixed_D * (2 * _NDIM * tau_s) + offset

    return HMMPlan(sequences, n_states=n_states, tau_s=tau_s, localization_sigma_um=localization_sigma_um,
                   n_starts=n_starts, seed=seed, max_iter=max_iter, tolerance=tolerance, init=init,
                   fixed_D=fixed_D, fixed_means=fixed_means, offset=offset, n_steps=len(all_steps),
                   noises=noises,
                   precision_calibration=(None if calibration is None
                                          else (float(calibration.c_nm), float(calibration.p))))


def run_hmm_start(plan: HMMPlan, start: int):
    """Start ``start`` of ``plan``: its EM outcome, or ``None`` when it failed (`finish_hmm` drops those). Module
    level, so a process pool can run it as a job of its own."""
    start = int(start)
    try:
        if getattr(plan, "noises", None) is not None:
            return _baum_welch_noise(
                plan.sequences, plan.noises, plan.n_states, plan.tau_s, plan.draws[start], plan.max_iter,
                plan.tolerance, init=(plan.init if start == 0 else None), fixed_means=plan.fixed_means)
        # start 0 warm-starts from `init` when given (bootstrap); later starts
        # fall back to random spread, so a poor warm start is still recoverable.
        return _baum_welch(
            plan.sequences, plan.n_states, plan.tau_s, plan.offset, plan.draws[start],
            plan.max_iter, plan.tolerance,
            init=(plan.init if start == 0 else None),
            fixed_means=plan.fixed_means,
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return None


def finish_hmm(plan: HMMPlan, outcomes) -> HMMFit:
    """`fit_hmm`'s last part: the best of ``outcomes`` -- every start's `run_hmm_start` result IN START ORDER, so a
    tie goes to the earlier start as in the loop -- as the fit. RAISES `HMMError` when no start converged."""
    results = [outcome for outcome in outcomes if outcome is not None]
    n_states = plan.n_states
    if not results:
        raise HMMError(
            f"no EM start converged for {n_states} states on {plan.n_steps} "
            "steps. Inspect the data; no fallback result is given."
        )

    results.sort(key=lambda r: -r["log_likelihood"])
    best = results[0]

    order = np.argsort(best["D"])
    D = best["D"][order]
    A = best["A"][np.ix_(order, order)]
    occupancy = best["occupancy"][order]

    # n D's + K(K-1) free transition entries + (K-1) free initial probabilities; no D free when they are held
    n_free = (0 if plan.fixed_means is not None else n_states) + n_states * (n_states - 1) + (n_states - 1)
    bic = n_free * np.log(plan.n_steps) - 2.0 * best["log_likelihood"]

    return HMMFit(
        n_states=n_states,
        D=D,
        transition_matrix=A,
        occupancy=occupancy,
        stationary=_stationary(A),
        tau_s=float(plan.tau_s),
        n_steps=plan.n_steps,
        n_sequences=plan.n_sequences,
        log_likelihood=float(best["log_likelihood"]),
        bic=float(bic),
        localization_sigma_um=float(plan.localization_sigma_um),
        n_starts=plan.n_starts,
        n_converged_starts=len(results),
        start_spread=_start_spread(results, n_states),
        initial_distribution=best["pi"][order],
        precision_calibration=getattr(plan, "precision_calibration", None),
    )


def decode_states(canonical: pd.DataFrame, fit: "HMMFit") -> pd.DataFrame:
    """Assign a state to every STEP of every track, the flagship spatial input.

    Posterior (forward-backward) decoding, so each step carries both its most
    likely state and the confidence in it. This is what makes a *spatial* map of
    binding possible: a per-track average D is switching-blurred and puts a
    binding molecule at some muddy middle value, whereas this localises the
    individual bound segments: where binding actually happens, not where slow
    molecules are on average.

    Returns one row per step with the step's start position (so it can be
    mapped), its assigned ``state``, and ``posterior`` (max state probability).
    Steps whose confidence is low should be filtered before drawing conclusions.
    Decoded under the FITTED model, its initial distribution included
    (``HMMFit.initial_distribution``), so the posteriors of the steps it was
    fitted to average to its ``occupancy`` (since 4 Nov 2026; the stationary
    distribution was the first step's prior before, and a map of these
    posteriors read up to 1.4 pp off the fit's own occupancy).
    Since 3 Nov 2026 each row also carries EVERY state's posterior, ``p_state_1`` ...
    ``p_state_K`` (the HMM maps' soft weights at any state count; before, only the
    winner's was kept, which fixed those maps at two states). Since 22 Nov 2026 also
    ``p_stay_1`` ... ``p_stay_K``: the probability that this step AND the next one in its
    run are both in state k (the jump angles' weights, `smtracker.hmm_angles`); NaN at a
    run's last step. One minus their sum is the chance the pair straddles a switch. A ONE-state fit has
    nothing to decode: every step is that state, posterior 1 (its "absorbing" chain is
    no degenerate fit), so its map of every jump can be drawn.
    A fit with NO within-track switching (an absorbing chain) is decoded too, each track
    in one state for its whole length, and ``attrs["no_switching"]`` says so (since 4 Nov
    2026; it was refused while the stationary distribution was the prior).
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise HMMError(f"not a canonical table (missing {missing})")

    # A CHAIN WITH NO WITHIN-TRACK SWITCHING IS DECODED (ruled 4 Nov 2026: "LIFT the HMM maps' refusal of a condition
    # whose tracks never switch state"). The refusal existed because the STATIONARY distribution was the first step's
    # prior: an absorbing chain makes it a point mass, which decided every step (a fit reporting 87.6 % occupancy for
    # state 1 decoded 100 % of steps as state 2). Under the FITTED initial distribution (below, tester C2) the
    # posteriors are the fit's own and average to its occupancy, so each track is simply decoded in one state for its
    # whole length: a per-TRACK classification, said by ``attrs["no_switching"]`` for a map to say so. Refused only
    # for a fit made before pi was kept, which would still decode from that point mass.
    no_switching = fit.no_switching
    if no_switching and fit.initial_distribution is None:
        absorbing = np.flatnonzero(fit.is_absorbing) + 1
        raise HMMError(
            "cannot decode: the fitted chain is degenerate: "
            f"state(s) {absorbing.tolist() or 'unknown'} are absorbing and the "
            f"stationary distribution is {np.round(fit.stationary, 6).tolist()}, "
            "and this fit keeps no initial distribution. Fit again; a new fit keeps it."
        )

    log_A = np.log(np.maximum(fit.transition_matrix, 1e-300))
    # THE FIRST STEP'S PRIOR IS THE FITTED INITIAL DISTRIBUTION (pi), the model the EM fitted and its occupancy was
    # computed under -- never the stationary one (found 4 Nov 2026, tester C2: the HMM map's state shares sat 0.4-1.4
    # pp above the HMM section's occupancy on real data, equal on synthetic tracks that start at stationarity). Real
    # tracks are 5-6 steps against dwells of tens of frames and do not reach stationarity (`HMMFit.initial_distribution`,
    # docs/claude/states_and_kinetics.md), so a stationary prior pulled every track's first steps, and through the
    # chain the rest, towards the stationary shares. The stationary one stays only for a fit made before pi was kept.
    prior = fit.initial_distribution if fit.initial_distribution is not None else fit.stationary
    log_pi = np.log(np.maximum(np.asarray(prior, dtype=float), 1e-300))
    offset = 2 * _NDIM * fit.localization_sigma_um**2
    means = 2 * _NDIM * fit.D * fit.tau_s + offset
    if np.any(means <= 0):
        raise HMMError("fitted state means are non-positive; cannot decode")
    # PER-SPOT PRECISION: each step's own noise, rebuilt from the calibration the fit was made with (never another)
    calibration = None
    if getattr(fit, "precision_calibration", None) is not None:
        from .canonical import LOC_SIGMA_COLUMN
        from .precision import Calibration

        if LOC_SIGMA_COLUMN not in canonical.columns:
            raise HMMError("this fit used per-spot precision, and these tracks carry none: decode the tracks it was "
                           "fitted to")
        calibration = Calibration(c_nm=fit.precision_calibration[0], p=fit.precision_calibration[1],
                                  source="the fit's own")

    extra = [c for c in ("L_um", "D_um", "l_norm", "d_norm") if c in canonical.columns]
    rows = []
    ordered = canonical.sort_values(["uid", "frame"], kind="stable")
    for uid, track in without_attrs(ordered).groupby("uid", sort=False, observed=True):
        frames = track["frame"].to_numpy()
        x = track["x_um"].to_numpy()
        y = track["y_um"].to_numpy()
        squared = np.diff(x) ** 2 + np.diff(y) ** 2
        contiguous = np.diff(frames) == 1
        if not contiguous.any():
            continue
        # Columns read ONCE per track, never as one row object per step: a row
        # (`track.iloc[i]`) is a new Series every time.
        source = {c: track[c].to_numpy() for c in ("condition", "cell", "movie", "day", *extra)}
        if calibration is not None:
            sigma = calibration.true_sigma_um(track[LOC_SIGMA_COLUMN].to_numpy(dtype=float))
            track_noise = sigma[:-1] ** 2 + sigma[1:] ** 2
            # a step touching a spot without precision is not decoded: it ends the run, as in the fit
            contiguous = contiguous & np.isfinite(track_noise)

        # Decode each contiguous run separately: a gap breaks the chain.
        start = 0
        while start < len(squared):
            if not contiguous[start]:
                start += 1
                continue
            stop = start
            while stop < len(contiguous) and contiguous[stop]:
                stop += 1
            steps = squared[start:stop]
            if calibration is not None:
                run_means = means[None, :] + _NDIM * track_noise[start:stop][:, None]
            else:
                run_means = means[None, :]
            log_b = -np.log(run_means) - steps[:, None] / run_means
            log_alpha, log_beta, log_lik = _forward_backward(log_b, log_A, log_pi)
            gamma = np.exp(log_alpha + log_beta - log_lik)
            gamma = gamma / gamma.sum(axis=1, keepdims=True)
            # THE PAIR'S OWN POSTERIOR (22 Nov 2026, for the jump angles, `smtracker.hmm_angles`): P(this step AND the
            # next one in state k), the two-slice marginal xi_t(k, k) of the same forward-backward. The product of the
            # two steps' own posteriors is not it: at a switch both read about one half for each state and the product
            # gives each a quarter, where the chain says the pair straddles the switch. NaN at a run's last step,
            # which has no next step in the run.
            stay = np.full_like(gamma, np.nan)
            if len(steps) > 1:
                stay[:-1] = np.exp(log_alpha[:-1] + np.diag(log_A)[None, :] + log_b[1:] + log_beta[1:] - log_lik)

            for offset_index, step_index in enumerate(range(start, stop)):
                row = {
                    "uid": uid,
                    "frame": int(frames[step_index]),
                    "state": int(np.argmax(gamma[offset_index])) + 1,
                    "posterior": float(gamma[offset_index].max()),
                    **{f"p_state_{k + 1}": float(gamma[offset_index, k]) for k in range(gamma.shape[1])},
                    **{f"p_stay_{k + 1}": float(stay[offset_index, k]) for k in range(gamma.shape[1])},
                    "sq_disp_um2": float(steps[offset_index]),
                    "x_um": float(x[step_index]),
                    "y_um": float(y[step_index]),
                    "condition": source["condition"][step_index],
                    "cell": source["cell"][step_index],
                    "movie": source["movie"][step_index],
                    "day": source["day"][step_index],
                }
                for column in extra:
                    row[column] = source[column][step_index]
                rows.append(row)
            start = stop

    if not rows:
        raise HMMError("no contiguous step could be decoded")
    out = pd.DataFrame(rows)
    out.attrs["n_states"] = fit.n_states
    out.attrs["D"] = fit.D
    # no within-track switching was fitted: every track decoded in one state for its whole length
    out.attrs["no_switching"] = no_switching
    return out


def select_n_states(
    sequences,
    *,
    max_states: int = 3,
    min_states: int = 1,
    tau_s: float,
    **fit_kwargs,
):
    """BIC across state counts. Reuses ``sqd.NPopSelection``: BIC recommends,
    the analyst decides."""
    from .sqd import NPopSelection

    if min_states < 1 or max_states < min_states:
        raise HMMError(f"need 1 <= min_states <= max_states, got {min_states}..{max_states}")
    if isinstance(sequences, pd.DataFrame):
        sequences = step_sequences(sequences)
    fits = [
        fit_hmm(sequences, n_states=k, tau_s=tau_s, **fit_kwargs)
        for k in range(min_states, max_states + 1)
    ]
    return NPopSelection(fits=fits)


# --------------------------------------------------------------------------
# Baum-Welch
# --------------------------------------------------------------------------

def _log_emission(steps: np.ndarray, means: np.ndarray) -> np.ndarray:
    """log p(r² | state): Exponential with the given means. Shape (T, K)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log(means)[None, :] - steps[:, None] / means[None, :]


def _pad(sequences):
    """Pack ragged sequences into ``(steps, mask)`` arrays for a batched E-step.

    Real datasets hold thousands of short tracks; looping over them in Python
    makes EM unusably slow, so every sequence is padded to the longest and the
    padding is masked out of every sum.
    """
    lengths = np.array([len(s) for s in sequences], dtype=int)
    n, t_max = len(sequences), int(lengths.max())
    steps = np.zeros((n, t_max))
    mask = np.zeros((n, t_max), dtype=bool)
    for i, sequence in enumerate(sequences):
        steps[i, : len(sequence)] = sequence
        mask[i, : len(sequence)] = True
    return steps, mask, lengths


#: What ONE TIME STEP of the E-step's loop costs, in padded cells (sequence x time) of its arithmetic: the price
#: `_length_buckets` sets on a group's width. Each group runs its own Python loop over time (forward and backward,
#: about a dozen numpy calls a step), so a group costs ``n_sequences x longest`` cells plus this many per step.
#: MEASURED (4 Nov 2026, `_estep` on synthetic shapes, least squares over sizes): a step costs as much as 63 cells at
#: two states and 18 at three; 50 is between them and nearer the dearer end.
_STEP_COST_CELLS = 50


def _length_buckets(lengths):
    """Indices grouped by length so each group pads to near its OWN longest sequence: the grouping of the sorted
    lengths that costs least, ``sum(n_in_group x longest + _STEP_COST_CELLS x longest)``, over every way of cutting
    the DISTINCT lengths into runs (exact, by dynamic programming: a few ms at a few hundred distinct lengths).

    Returns a list of index arrays, shortest group first. One group means "do not bucket": equal-length tracks.

    UNTIL 4 NOV 2026 SIX GROUPS OF EQUAL COUNT. On the reference data (longest 44 steps) that recovered most of the
    padding; on a lab project's pooled tracks (9,604, median 6 steps, longest 394) the last sixth -- 1,600 tracks from
    11 steps to 394 -- was padded to 394, and a start did 8.5 times the arithmetic its 80,742 steps needed (683,222
    cells; 1.15 times now, 92,614 cells in 13 groups). The sums the M-step reads are added over the groups, so a
    different grouping changes only their float rounding: last digits, not the fit
    (`tests/test_hmm.py::TestTheEStepIsBucketedByLength`).
    """
    lengths = np.asarray(lengths)
    if len(lengths) == 0:
        return [np.arange(0)]
    values, counts = np.unique(lengths, return_counts=True)
    if len(values) == 1:
        return [np.arange(len(lengths))]
    cumulative = np.concatenate([[0], np.cumsum(counts)])
    best = np.zeros(len(values) + 1)
    cut = np.zeros(len(values) + 1, dtype=int)
    for k in range(1, len(values) + 1):          # the cheapest grouping of the first k distinct lengths
        width = float(values[k - 1])
        costs = best[:k] + (cumulative[k] - cumulative[:k]) * width + _STEP_COST_CELLS * width
        cut[k] = int(np.argmin(costs))
        best[k] = costs[cut[k]]
    edges, k = [], len(values)
    while k > 0:
        edges.append((values[cut[k]], values[k - 1]))
        k = cut[k]
    order = np.argsort(lengths, kind="stable")
    ordered = lengths[order]
    return [order[(ordered >= low) & (ordered <= high)] for low, high in reversed(edges)]


def _estep(steps, mask, rows, last_index, log_A, log_pi, means, n_states, keep_gamma=False):
    """One batch's forward-backward. Returns the sufficient statistics, or None.

    Every quantity here is a SUM over sequences and time, which is what makes bucketing
    exact rather than approximate: run this on any partition of the data and add the
    results, and the M-step below cannot tell the difference.

    ``means`` is one per state, or with per-spot precision one per step and state (n_seq x t_max x K); then
    ``keep_gamma`` also returns every step's posterior, which that M-step needs (`_baum_welch_noise`).
    """
    n_seq, t_max = steps.shape
    m = means if np.ndim(means) == 3 else np.asarray(means)[None, None, :]
    log_b = -np.log(m) - steps[:, :, None] / m
    if not np.isfinite(log_b[mask]).all():
        return None
    log_b = np.where(mask[:, :, None], log_b, 0.0)

    log_alpha = np.empty((n_seq, t_max, n_states))
    log_alpha[:, 0] = log_pi[None, :] + log_b[:, 0]
    for t in range(1, t_max):
        step = _logsumexp(log_alpha[:, t - 1][:, :, None] + log_A[None], axis=1)
        log_alpha[:, t] = np.where(
            mask[:, t][:, None], step + log_b[:, t], log_alpha[:, t - 1]
        )
    log_lik = log_alpha[rows, last_index]
    total_ll = float(_logsumexp(log_lik, axis=1).sum())
    if not np.isfinite(total_ll):
        return None

    log_beta = np.zeros((n_seq, t_max, n_states))
    for t in range(t_max - 2, -1, -1):
        candidate = _logsumexp(
            log_A[None] + (log_b[:, t + 1] + log_beta[:, t + 1])[:, None, :], axis=2
        )
        log_beta[:, t] = np.where(mask[:, t + 1][:, None], candidate, 0.0)

    sequence_ll = _logsumexp(log_lik, axis=1)[:, None, None]
    gamma = np.exp(log_alpha + log_beta - sequence_ll)
    gamma = np.where(mask[:, :, None], gamma, 0.0)

    gamma_sum = gamma.sum(axis=(0, 1))
    gamma_weighted = (gamma * steps[:, :, None]).sum(axis=(0, 1))
    gamma_first = gamma[:, 0].sum(axis=0)

    pair = mask[:, 1:]
    log_xi = (
        log_alpha[:, :-1, :, None]
        + log_A[None, None]
        + (log_b[:, 1:] + log_beta[:, 1:])[:, :, None, :]
        - sequence_ll[:, :, :, None]
    )
    xi = np.where(pair[:, :, None, None], np.exp(log_xi), 0.0)
    if keep_gamma:
        return total_ll, gamma_sum, gamma_weighted, gamma_first, xi.sum(axis=(0, 1)), gamma
    return total_ll, gamma_sum, gamma_weighted, gamma_first, xi.sum(axis=(0, 1))


#: Per-spot precision's bounds on a state's D (um^2/s): above 0, for a bound state (its steps are then noise alone).
_NOISE_D_BOUNDS = (1e-6, 50.0)
#: The floor a per-spot state's D stops at: a D there is NOT RESOLVED (`d_at_floor`, 10 Nov 2026).
NOISE_D_FLOOR = _NOISE_D_BOUNDS[0]


def d_at_floor(D, *, rtol: float = 1e-3) -> bool:
    """Whether a state's D sits on the per-spot M-step's floor (`NOISE_D_FLOOR`): the optimiser's stopping point, not
    an estimate, so it is said ("not resolved"), never printed as a value (10 Nov 2026; as SQD's `sqd.at_bound`, the
    per-spot audit of 7 Nov 2026). SynR's bound state sat there, 1.0000005e-06, in every condition."""
    try:
        D = float(D)
    except (TypeError, ValueError):
        return False
    return bool(0.0 < D <= NOISE_D_FLOOR * (1.0 + rtol))


def _core_mstep(gamma_k, steps, noise, tau_s, current):
    """Per-spot precision's M-step for ONE state: its noise-free mean c = 4 D tau maximising sum_i gamma_ik log
    Exp(r_i | c + 2 noise_i), by a bounded search on log c (no closed form once every step has its own noise)."""
    from scipy.optimize import minimize_scalar

    weight = float(gamma_k.sum())
    if weight <= 0:
        return current
    lo, hi = (np.log(2 * _NDIM * tau_s * d) for d in _NOISE_D_BOUNDS)
    extra = _NDIM * noise

    def negative(log_c):
        m = np.exp(log_c) + extra
        return float(np.sum(gamma_k * (np.log(m) + steps / m)))

    result = minimize_scalar(negative, bounds=(lo, hi), method="bounded", options={"xatol": 1e-7})
    return float(np.exp(result.x)) if np.isfinite(result.fun) else current


def _baum_welch_noise(sequences, noises, n_states, tau_s, draw, max_iter, tolerance, init=None, fixed_means=None):
    """One EM run with PER-SPOT PRECISION: state k's mean for step i is c_k + 2 noise_i, c_k = 4 D_k tau. As
    `_baum_welch` (the same draws, buckets, E-step, A and pi updates), except the means' M-step (`_core_mstep`).
    ``init`` / ``fixed_means`` hold the noise-free means c (4 D tau). ``None`` if it fails."""
    lengths = np.array([len(s) for s in sequences], dtype=int)
    all_steps = np.concatenate([np.asarray(s, dtype=float) for s in sequences])
    scale = float(all_steps.mean())
    if scale <= 0:
        return None
    if init is not None:
        core = np.maximum(np.asarray(init[0], dtype=float), 1e-12)
        A, pi = np.asarray(init[1], dtype=float), np.asarray(init[2], dtype=float)
    elif draw is not None:
        factors, A, pi = draw
        core = np.maximum(scale * factors, 1e-12)
        A, pi = np.array(A, dtype=float), np.array(pi, dtype=float)
    else:
        core = scale * np.geomspace(0.2, 3.0, n_states) if n_states > 1 else np.array([scale])
        A = np.full((n_states, n_states), 0.05 / max(n_states - 1, 1))
        np.fill_diagonal(A, 0.95)
        if n_states == 1:
            A = np.ones((1, 1))
        pi = np.full(n_states, 1.0 / n_states)
    if fixed_means is not None:
        core = np.asarray(fixed_means, dtype=float).copy()

    batches = []
    for index in _length_buckets(lengths):
        b_steps, b_mask, b_lengths = _pad([sequences[i] for i in index])
        b_noise, _mask, _lengths = _pad([noises[i] for i in index])
        batches.append((b_steps, b_noise, b_mask, np.arange(len(index)), b_lengths - 1))
    flat_steps = np.concatenate([b[0][b[2]] for b in batches])
    flat_noise = np.concatenate([b[1][b[2]] for b in batches])
    previous = -np.inf

    for _ in range(max_iter):
        log_A = np.log(np.maximum(A, 1e-300))
        log_pi = np.log(np.maximum(pi, 1e-300))
        total_ll = 0.0
        gamma_sum = np.zeros(n_states)
        gamma_first = np.zeros(n_states)
        xi_sum = np.zeros((n_states, n_states))
        gammas = []
        for b_steps, b_noise, b_mask, b_rows, b_last in batches:
            means = core[None, None, :] + _NDIM * b_noise[:, :, None]
            got = _estep(b_steps, b_mask, b_rows, b_last, log_A, log_pi, means, n_states, keep_gamma=True)
            if got is None:
                return None
            total_ll += got[0]
            gamma_sum += got[1]
            gamma_first += got[3]
            xi_sum += got[4]
            gammas.append(got[5][b_mask])
        pi = gamma_first / max(gamma_first.sum(), 1e-300)
        row_totals = xi_sum.sum(axis=1, keepdims=True)
        A = np.where(row_totals > 0, xi_sum / np.maximum(row_totals, 1e-300), A)
        if fixed_means is None:
            gamma = np.concatenate(gammas)
            core = np.array([_core_mstep(gamma[:, k], flat_steps, flat_noise, tau_s, core[k])
                             for k in range(n_states)])
        if abs(total_ll - previous) < tolerance * max(abs(previous), 1.0):
            break
        previous = total_ll

    occupancy = gamma_sum / max(gamma_sum.sum(), 1e-300)
    D = core / (2 * _NDIM * tau_s)
    if np.any(~np.isfinite(D)) or np.any(D <= 0):
        return None
    if len(np.unique(np.round(D, 12))) < n_states:
        return None
    return {"D": D, "A": A, "occupancy": occupancy, "pi": pi, "log_likelihood": float(total_ll)}


def _baum_welch(sequences, n_states, tau_s, offset, draw, max_iter, tolerance,
                init=None, fixed_means=None):
    """One EM run, batched over all sequences. ``None`` if it fails.

    ``init`` (means, A, pi) WARM-STARTS the EM from a known solution, the bootstrap
    passes the full-data point estimate, so each resample converges in a few
    iterations from an already-good place instead of exploring from scratch. When
    None, the start's random guess ``draw`` (`_start_draws`: factors on the mean step, A, pi) is used, or the fixed
    spread when there is none (start 0), so the point-fit path is unchanged. ``fixed_means`` holds the emission means
    (`fit_hmm`'s ``fixed_D``): only A and pi are updated.
    """
    # the steps end to end, in the order padding read them back (`_pad`'s rows): the same mean, bit for bit, without
    # the array of EVERY sequence padded to the longest (9,604 x 394 on protein F's pooled tracks) made only to take it
    lengths = np.array([len(s) for s in sequences], dtype=int)
    all_steps = np.concatenate([np.asarray(s, dtype=float) for s in sequences])
    scale = float(all_steps.mean())
    if scale <= 0:
        return None

    if init is not None:
        means = np.maximum(np.asarray(init[0], dtype=float), 1e-12)
        A = np.asarray(init[1], dtype=float)
        pi = np.asarray(init[2], dtype=float)
    elif draw is not None:
        factors, A, pi = draw
        means = np.maximum(scale * factors, 1e-12)
        A, pi = np.array(A, dtype=float), np.array(pi, dtype=float)
    else:
        means = scale * np.geomspace(0.2, 3.0, n_states) if n_states > 1 else np.array([scale])
        A = np.full((n_states, n_states), 0.05 / max(n_states - 1, 1))
        np.fill_diagonal(A, 0.95)
        if n_states == 1:
            A = np.ones((1, 1))
        pi = np.full(n_states, 1.0 / n_states)
    if fixed_means is not None:
        means = np.asarray(fixed_means, dtype=float).copy()

    # ONE PADDED BATCH PER LENGTH BUCKET, built once and reused every iteration.
    batches = []
    for index in _length_buckets(lengths):
        group = [sequences[i] for i in index]
        b_steps, b_mask, b_lengths = _pad(group)
        batches.append((b_steps, b_mask, np.arange(len(group)), b_lengths - 1))
    previous = -np.inf

    for _ in range(max_iter):
        log_A = np.log(np.maximum(A, 1e-300))
        log_pi = np.log(np.maximum(pi, 1e-300))

        # THE STATISTICS ADD ACROSS BUCKETS, which is why this is exact and not an
        # approximation: each is a sum over sequences and time, so any partition gives
        # the same totals.
        total_ll = 0.0
        gamma_sum = np.zeros(n_states)
        gamma_weighted = np.zeros(n_states)
        gamma_first = np.zeros(n_states)
        xi_sum = np.zeros((n_states, n_states))
        for b_steps, b_mask, b_rows, b_last in batches:
            got = _estep(b_steps, b_mask, b_rows, b_last, log_A, log_pi, means,
                         n_states)
            if got is None:
                return None
            total_ll += got[0]
            gamma_sum += got[1]
            gamma_weighted += got[2]
            gamma_first += got[3]
            xi_sum += got[4]

        pi = gamma_first / max(gamma_first.sum(), 1e-300)
        row_totals = xi_sum.sum(axis=1, keepdims=True)
        A = np.where(row_totals > 0, xi_sum / np.maximum(row_totals, 1e-300), A)
        if fixed_means is None:
            means = gamma_weighted / np.maximum(gamma_sum, 1e-300)
            if np.any(means <= offset) or not np.isfinite(means).all():
                means = np.maximum(means, offset + 1e-9)

        if abs(total_ll - previous) < tolerance * max(abs(previous), 1.0):
            break
        previous = total_ll

    occupancy = gamma_sum / max(gamma_sum.sum(), 1e-300)
    D = (means - offset) / (2 * _NDIM * tau_s)
    if np.any(~np.isfinite(D)) or np.any(D <= 0):
        return None
    if len(np.unique(np.round(D, 12))) < n_states:
        return None
    return {"D": D, "A": A, "occupancy": occupancy, "pi": pi,
            "log_likelihood": float(total_ll)}


def _forward_backward(log_b, log_A, log_pi):
    """Log-space forward-backward for ONE sequence. Used by ``decode_states``.

    (The EM fit uses a batched version over padded sequences; this is the plain
    single-sequence form, which is what decoding needs.)
    """
    T, K = log_b.shape
    log_alpha = np.empty((T, K))
    log_beta = np.zeros((T, K))

    log_alpha[0] = log_pi + log_b[0]
    for t in range(1, T):
        log_alpha[t] = log_b[t] + _logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)
    for t in range(T - 2, -1, -1):
        log_beta[t] = _logsumexp(log_A + (log_b[t + 1] + log_beta[t + 1])[None, :], axis=1)

    return log_alpha, log_beta, float(_logsumexp(log_alpha[-1], axis=0))


def _logsumexp(values, axis):
    peak = np.max(values, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    return (peak + np.log(np.sum(np.exp(values - peak), axis=axis, keepdims=True))).squeeze(
        axis=axis
    )


def _stationary(A: np.ndarray) -> np.ndarray:
    """Stationary distribution of the transition matrix."""
    if A.shape[0] == 1:
        return np.array([1.0])
    values, vectors = np.linalg.eig(A.T)
    index = int(np.argmin(np.abs(values - 1.0)))
    vector = np.real(vectors[:, index])
    total = vector.sum()
    if not np.isfinite(total) or abs(total) < 1e-12:
        return np.full(A.shape[0], 1.0 / A.shape[0])
    vector = vector / total
    return np.clip(vector, 0.0, 1.0) / np.clip(vector, 0.0, 1.0).sum()


def _start_spread(results, n_states) -> dict:
    best = max(r["log_likelihood"] for r in results)
    near = [r for r in results if r["log_likelihood"] >= best - 1e-3 * abs(best)]
    D_values = np.array([np.sort(r["D"]) for r in near])
    return {
        "n_near_best": len(near),
        "D_relative_range": (
            (D_values.max(axis=0) - D_values.min(axis=0)) / np.abs(D_values.mean(axis=0))
        ).tolist(),
    }


#: Units a rate bootstrap may resample over, the same independent units the
#: fraction bootstrap uses, so the two carry comparable uncertainty.
BOOTSTRAP_UNITS = ("cell", "movie", "day")


def _hmm_boot_worker(replicate: int, shared: dict):
    """One bootstrap replicate: module-level so it is picklable for a pool.

    Reads the shared read-only payload; builds the resample from the pre-drawn
    index set (so it is deterministic and order-free); refits the HMM. Returns a
    result row, or ``None`` if this resample could not be fitted.
    """
    unit_sequences = shared["unit_sequences"]
    picked = shared["picks"][replicate]
    # gather the drawn units' PRE-BUILT sequences; a unit drawn twice appears twice
    # (its arrays are repeated in the list, never merged into one sequence).
    sequences = [s for index in picked for s in unit_sequences[index]]
    unit_noises = shared.get("unit_noises")
    noises = None if unit_noises is None else [n for index in picked for n in unit_noises[index]]

    n_states = shared["n_states"]
    try:
        fit = fit_hmm(
            sequences, n_states=n_states, tau_s=shared["tau_s"],
            localization_sigma_um=shared["localization_sigma_um"],
            n_starts=shared["n_starts"], seed=shared["seed"] + replicate,
            init=shared["init"], fixed_D=shared.get("fixed_D"),
            noise_sequences=noises, calibration=shared.get("calibration"),
        )
    except HMMError:
        return None

    row = {"replicate": replicate, "degenerate": bool(fit.has_degenerate_chain)}
    rates = fit.rates_per_s
    for i in range(n_states):
        row[f"D_{i + 1}"] = float(fit.D[i])
        row[f"occupancy_{i + 1}"] = float(fit.occupancy[i])
        row[f"dwell_{i + 1}_s"] = float(fit.dwell_times_s[i])
        for j in range(n_states):
            if i != j:
                row[f"rate_{i + 1}_to_{j + 1}"] = float(rates[i, j])
    return row


def bootstrap_hmm(
    canonical: pd.DataFrame,
    *,
    n_states: int,
    tau_s: float | None = None,
    unit: str = "cell",
    n_boot: int = 200,
    n_starts: int = 3,
    localization_sigma_um: float = 0.0,
    seed: int | None = 0,
    min_steps: int = 2,
    n_jobs: int | None = None,
    warm_start: bool = True,
    fixed_D=None,
    calibration=None,
) -> pd.DataFrame:
    """Bootstrap the fitted HMM over independent units, CIs for the RATES. ``fixed_D``: every resample at those D
    (`fit_hmm`), the interval then on the rates and occupancy alone. ``calibration``: per-spot precision (`fit_hmm`),
    every resample with its steps' own noise.

    Why this exists
    ---------------
    Fractions in this project always carry a cell/day bootstrap interval. The
    switching **rates** did not, and they had become a headline claim ("binding
    events are ~26x more frequent"). A rate ratio with no interval is exactly
    the kind of naked point estimate the rest of the project refuses, and the
    rates are the *more* exposed number: a condition that rarely switches
    supplies few transitions to estimate them from.

    Method: resample ``unit`` with replacement, refit the whole HMM on each
    resample, and return one row per replicate. A unit drawn twice has its
    ``uid`` suffixed so its tracks count twice rather than merging into one
    sequence: the same care ``bootstrap_sqd`` takes.

    Reading the result
    ------------------
    Use **percentile** intervals, not mean ± sd: a rate is bounded below by zero
    and the bootstrap distribution of a low-event rate is right-skewed, so a
    symmetric interval would misstate it and can even reach below zero.

    ``degenerate`` marks replicates whose fitted chain is absorbing (no
    transition out of a state was observed in that resample). Those are kept and
    counted rather than dropped: for a rarely-switching condition, *how often*
    the resample contains no transitions is itself the honest statement of how
    weakly the rates are determined.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise HMMError(f"not a canonical table (missing {missing})")
    if unit not in BOOTSTRAP_UNITS:
        raise HMMError(f"unit must be one of {BOOTSTRAP_UNITS}, got {unit!r}")
    if unit not in canonical.columns:
        raise HMMError(f"table has no {unit!r} column")
    if canonical[unit].isna().any():
        raise HMMError(
            f"{unit!r} is unassigned for some localizations; resampling over it "
            "would silently drop data"
        )
    if n_boot < 1:
        raise HMMError(f"n_boot must be >= 1, got {n_boot}")

    if tau_s is None:
        params = canonical.attrs.get("params")
        if params is None:
            raise HMMError("tau_s not given and the table carries no params")
        tau_s = float(params.tau_s)

    groups = [g for _, g in canonical.groupby(unit, observed=True)]
    if len(groups) < 2:
        raise HMMError(
            f"only {len(groups)} distinct {unit_word(unit)}(s); a bootstrap over one unit is "
            "meaningless"
        )

    # PRECOMPUTE the per-unit step sequences ONCE (the pandas concat + step_sequences
    # were being redone every resample, pure waste). A resample now just gathers the
    # drawn units' sequence lists; a unit drawn twice contributes its sequences twice,
    # never merged, the same care the old uid-suffix path took.
    unit_noises = None
    if calibration is not None:
        pairs = [step_sequences(g, min_steps=min_steps, calibration=calibration) for g in groups]
        unit_sequences = [p[0] for p in pairs]
        unit_noises = [p[1] for p in pairs]
    else:
        unit_sequences = [step_sequences(g, min_steps=min_steps) for g in groups]

    # WARM-START each resample from the full-data point estimate: EM then converges in
    # a few iterations from an already-good place, so one start per resample suffices
    # instead of `n_starts` random explorations. Falls back to random on any failure.
    init = None
    if warm_start:
        try:
            full = [s for us in unit_sequences for s in us]
            full_noise = None if unit_noises is None else [n for un in unit_noises for n in un]
            point = fit_hmm(full, n_states=n_states, tau_s=tau_s,
                            localization_sigma_um=localization_sigma_um,
                            n_starts=max(n_starts, 6), seed=0 if seed is None else seed, fixed_D=fixed_D,
                            noise_sequences=full_noise, calibration=calibration)
            # with per-spot precision the warm start holds the noise-free means, 4 D tau (`_baum_welch_noise`)
            offset = 2 * _NDIM * localization_sigma_um ** 2
            means0 = np.asarray(point.D) * (2 * _NDIM * tau_s) + offset
            init = (means0, np.asarray(point.transition_matrix),
                    np.asarray(point.occupancy))
        except HMMError:
            init = None                                # no warm start; workers go random

    # Resample index sets drawn up front, in order, so the result is identical
    # whatever order the workers finish in (see smtracker.parallel).
    picks = resample_indices(len(groups), n_boot, seed)
    shared = {
        "unit_sequences": unit_sequences, "unit_noises": unit_noises, "calibration": calibration,
        "n_states": n_states, "tau_s": tau_s,
        "localization_sigma_um": localization_sigma_um,
        "n_starts": 1 if init is not None else n_starts, "init": init,
        "fixed_D": None if fixed_D is None else [float(d) for d in fixed_D],
        "seed": 0 if seed is None else seed, "min_steps": min_steps, "picks": picks,
    }
    results = bootstrap_map(_hmm_boot_worker, n_boot, shared=shared, n_jobs=n_jobs)
    rows = [r for r in results if r is not None]
    failures = n_boot - len(rows)

    if not rows:
        raise HMMError(
            f"every one of {n_boot} bootstrap replicates failed to fit; the "
            "model is not identifiable on this data"
        )

    out = pd.DataFrame(rows)
    out.attrs["unit"] = unit
    out.attrs["n_units"] = len(groups)
    out.attrs["n_boot"] = int(n_boot)
    out.attrs["n_ok"] = int(len(out))
    out.attrs["n_failed"] = int(failures)
    out.attrs["n_degenerate"] = int(out["degenerate"].sum())
    out.attrs["n_states"] = int(n_states)
    out.attrs["tau_s"] = float(tau_s)
    if len(groups) < 5:
        out.attrs["caveat"] = (
            f"only {len(groups)} {unit_word(unit)}s: the percentile interval is quantized "
            "and should be read as indicative"
        )
    return out
