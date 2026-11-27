"""How each population's D and share change with the time since an INDUCTION.

THE DESIGN. the experimenter's triggered-kinetics design (`docs/claude/states_and_kinetics.md`, "TRIGGERED KINETICS"), asked
again on 21 Oct 2026: measure the kinetic, plot the D's over time, read how fast they change (linear, exponential,
...), knowing the event starts at different times for different molecules, cells and movies. Built for every
direction and rate and shown on SynK's ground truth in the paper (`docs/synthetic_synr.md`, "SynK").

ONE CLOCK. Every step sits at its first frame's start, in seconds since the induction:
``t = start + (frame - 1) * tau``, ``start`` being the movie's first frame on that clock -- ``-(trigger_frame - 1)
* tau`` when the induction falls inside the movie (the convention of `kinetics.response_curve`: a step starting on
the trigger frame is at 0), or the movie's own time since the induction (movies across a session, data sets hours
apart). Steps before the induction are KEPT: they are the baseline the change is measured from.

ONE MODEL, FITTED TO EVERY STEP AT ITS OWN TIME. The SQD mixture (`sqd`'s density, one definition) whose D's and
shares move from a before to an after value by one population response ``H(t)`` in [0, 1]::

    p(r2 | t) = sum_k f_k(t) exp(-r2 / m_k(t)) / m_k(t),   m_k(t) = 4 D_k(t) tau_lag
    D_k(t) = D_k,before + (D_k,after - D_k,before) H(t),   f(t) the same way

``change`` says what moves -- "none" (the constant mixture: the null), "fractions", "D" or "both" -- and ``shape``
how H rises: "linear" (a ramp from the onset, complete after ``time_s``), "exponential" (from the onset, half-life
``time_s``) or "sigmoid" (Hill from the induction, half at ``time_s``, steepness ``hill``). Every direction is
allowed: an after value may lie above or below its before. The populations are numbered by their D BEFORE.

WHY ONE MODEL AND NOT A FIT PER TIME BIN. A bin holds too few steps to fit D's and shares freely, and two free fits
side by side trade a D change for a share change from bin to bin. A D that changes, read through shares fitted with
D held fixed (`kinetics`' slow fraction, `timecourse`), comes out as a SHARE change; that confusion is what this
model settles, by fitting both and letting units held out choose (`compare_models`). The per-bin fits
(`binned_fits`) are drawn beside the model as what the data look like, never as the estimate.

THE POPULATION RESPONSE IS NOT EACH MOLECULE'S. Where molecules, cells or movies start at different times, H is
their responses averaged over those onsets -- a step for every molecule reads as a smooth rise -- so ``time_s``
describes the population. Each unit's own onset is `kinetics.detect_onsets`' question, read against its null.

THE UNIT IS THE MOVIE OR THE CELL, never the bin or the step: the model is chosen on whole units held out, and every
interval resamples whole units (`bootstrap`). Steps weigh equally, the SQD standard. One condition at a time
(Rule 0: units of different conditions share nothing).

INVARIANTS: no Qt; the gradient is exact (checked against differences in the tests); a fit that cannot be made is
refused, never returned empty.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "InductionError",
    "InductionFit",
    "RESPONSE_SHAPES",
    "SHAPES",
    "SHAPE_PREFERENCE",
    "CHANGES",
    "response_shape",
    "induction_steps",
    "fit_model",
    "compare_models",
    "bootstrap",
    "binned_fits",
    "replicate_binned_fits",
    "OnsetSpread",
    "fit_onset_spread",
    "shuffle_members",
    "onset_spread_null",
    "UNIT_COLUMNS",
    "unit_column",
    "PERMUTATIONS",
    "shuffle_clock",
    "permutation_arrangements",
    "changing_models",
    "permutation_gains",
    "permutation_seeds",
    "permutation_summary",
    "run_permutation_chunks",
    "permutation_test",
]

#: Every shape a response can take; `synthetic.Induction` simulates all of them.
RESPONSE_SHAPES = ("step", "linear", "exponential", "sigmoid")
#: The shapes the model fits (a step is the limit of each; a fit cannot place a jump by gradient).
SHAPES = ("linear", "exponential", "sigmoid")
#: The DECLARED default order among shapes the data cannot tell apart (8 Nov 2026): exponential first (first-order
#: kinetics), then sigmoid, then linear. They tie on parameters, so without it a linear ramp won "by a hair" and drew
#: a kinked curve that looked sudden (SynK's binding rise, truth exponential).
SHAPE_PREFERENCE = ("exponential", "sigmoid", "linear")
#: What the model lets move, simplest first.
CHANGES = ("none", "fractions", "D", "both")
#: The clock's column on a step table.
CLOCK = "t_induction_s"
#: A response this far run where the data end is complete; below it the after values are extrapolated and said so.
COMPLETE = 0.9
#: The independent units a model may be chosen on and resampled by, and the step-table column of each. A "replicate"
#: is the table's `day` column (a user reads replicate; the column keeps its name).
UNIT_COLUMNS = {"movie": "movie", "cell": "cell", "replicate": "day"}
D_BOUNDS_UM2_S = (1e-4, 20.0)


def d_at_floor(D, *, rtol: float = 1e-3) -> bool:
    """Whether a population's D sits on the fit's floor (`D_BOUNDS_UM2_S`'s lower end): where the optimiser stopped,
    not an estimate, so it is said ("not resolved"), never printed as a value (10 Nov 2026; as SQD's `sqd.at_bound`
    and the HMM's `hmm.d_at_floor`). Per spot, SynK course's bound population sat there in every condition."""
    try:
        D = float(D)
    except (TypeError, ValueError):
        return False
    return bool(0.0 < D <= D_BOUNDS_UM2_S[0] * (1.0 + rtol))
_LOGIT = 12.0
_LN2 = math.log(2.0)


class InductionError(RuntimeError):
    """Raised when a time since the induction or a fit to it cannot be formed honestly."""


def unit_column(unit: str) -> str:
    """The step table's column holding ``unit``'s identity (`UNIT_COLUMNS`); refused by name otherwise."""
    if unit not in UNIT_COLUMNS:
        raise InductionError(f"unit {unit!r}: choose one of {tuple(UNIT_COLUMNS)}")
    return UNIT_COLUMNS[unit]


# --------------------------------------------------------------------------- #
# The response and the clock
# --------------------------------------------------------------------------- #


def response_shape(shape: str, since_s, time_s: float | None = None, hill: float | None = None) -> np.ndarray:
    """``H`` in [0, 1] at ``since_s`` seconds after an onset (0 at and before it), by `RESPONSE_SHAPES`: step
    (instant), linear (complete after ``time_s``), exponential (half-life ``time_s``), sigmoid (Hill, half at
    ``time_s``, steepness ``hill``). One definition: SynK's molecules and this module's population model."""
    if shape not in RESPONSE_SHAPES:
        raise InductionError(f"shape {shape!r}: choose one of {RESPONSE_SHAPES}")
    t = np.asarray(since_s, float)
    if shape == "step":
        return (t > 0).astype(float)
    if time_s is None or not time_s > 0:
        raise InductionError(f"a {shape} response needs time_s > 0, got {time_s}")
    pos = np.clip(t, 0.0, None)
    if shape == "linear":
        h = np.clip(pos / time_s, 0.0, 1.0)
    elif shape == "exponential":
        h = 1.0 - np.power(0.5, pos / time_s)
    else:
        if hill is None or not hill > 0:
            raise InductionError(f"a sigmoid response needs hill > 0, got {hill}")
        p = np.power(pos / time_s, hill)
        h = p / (1.0 + p)
    return np.where(t > 0, h, 0.0)


#: PER-SPOT PRECISION (6 Nov 2026; asked: "pls build where its possible with yes"): each step's own per-axis
#: localisation variance (um^2), `with_step_noise`'s column. Where a step table carries it, every likelihood here --
#: the model, its held-out scores, the free per-bin fits, the onset spread -- puts it in each population's mean,
#: 4 D(t) tau + 2 noise_i (`sqd._means`); its derivatives keep their form, the noise being given. Absent, nothing moves.
NOISE_COLUMN = "step_noise_um2"


def with_step_noise(steps: pd.DataFrame, calibration) -> pd.DataFrame:
    """``steps`` (`observables.squared_displacements`' table, on the clock or not) with `NOISE_COLUMN`: each step's
    noise through ``calibration`` (`precision.Calibration`, measured on immobile spots), the steps lacking a spot's
    precision left out (`precision.usable_steps`; above 1 % refused). Every subset the model fits -- a fold, a
    resample, a unit -- then carries its own steps' noise."""
    from .precision import PrecisionError, step_noise, usable_steps

    try:
        ok = usable_steps(steps)
        kept = steps.loc[ok].copy() if not ok.all() else steps.copy()
        kept[NOISE_COLUMN] = step_noise(kept, calibration)
    except PrecisionError as exc:
        raise InductionError(str(exc)) from exc
    kept.attrs = dict(steps.attrs)
    kept.attrs["precision_calibration"] = (float(calibration.c_nm), float(calibration.p))
    kept.attrs["precision_left_out"] = int((~ok).sum())
    return kept


def induction_steps(sd: pd.DataFrame, *, trigger_frame: int | None = None,
                    movie_start_s: dict | None = None, tau_s: float | None = None) -> pd.DataFrame:
    """The steps of ONE condition on the induction clock: ``sd`` (`observables.squared_displacements`) plus
    `CLOCK`, seconds since the induction at each step's first frame.

    Give exactly one of ``trigger_frame`` (the induction falls on that frame of every movie) and ``movie_start_s``
    ({movie: its first frame's time since the induction, s}; every movie present must be named). ``tau_s`` defaults
    to the table's own (``sd.attrs["tau_s"]``)."""
    if (trigger_frame is None) == (movie_start_s is None):
        raise InductionError("give either trigger_frame (the induction inside every movie) or movie_start_s (each "
                             "movie's own time since the induction), but not both")
    for column in ("sq_disp_um2", "frame", "movie", "condition"):
        if column not in sd.columns:
            raise InductionError(f"missing '{column}': pass `observables.squared_displacements`' table")
    conditions = pd.unique(sd["condition"])
    if len(conditions) != 1:
        raise InductionError(f"one condition at a time (their replicates share nothing); got "
                             f"{sorted(map(str, conditions))}")
    tau = float(tau_s if tau_s is not None else sd.attrs.get("tau_s", np.nan))
    if not tau > 0:
        raise InductionError("no frame interval: the table carries no tau_s and none was given")
    frame = sd["frame"].to_numpy(float)
    if trigger_frame is not None:
        start = np.full(len(sd), -(float(trigger_frame) - 1.0) * tau)
        how = f"the induction on frame {int(trigger_frame)} of every movie"
    else:
        movies = pd.unique(sd["movie"].astype(str))
        missing = sorted(m for m in movies if m not in movie_start_s)
        if missing:
            raise InductionError(f"no time since the induction for {missing}: every movie needs its own")
        start = sd["movie"].astype(str).map({str(k): float(v) for k, v in movie_start_s.items()}).to_numpy(float)
        how = "each movie's own time since the induction"
    out = sd.assign(**{CLOCK: start + (frame - 1.0) * tau})
    out.attrs.update(sd.attrs)
    out.attrs.update(tau_s=tau, clock=how)
    return out


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InductionFit:
    """One fitted model; populations numbered by their D BEFORE the induction."""

    change: str
    shape: str | None
    n_pop: int
    tau_lag_s: float
    D_before: tuple
    D_after: tuple
    fractions_before: tuple
    fractions_after: tuple
    onset_s: float | None
    time_s: float | None
    hill: float | None
    log_likelihood: float
    n_params: int
    n_steps: int
    theta: tuple
    #: The last step's time: where the data end. ``D_after`` / ``fractions_after`` lie beyond it unless the response
    #: is complete there (`completion`); `D_end` / `fractions_end` are what the data reach.
    t_end_s: float = math.nan
    #: Fitted with each step's own localisation noise (`NOISE_COLUMN`): its D have the error taken out.
    per_spot: bool = False

    @property
    def completion(self) -> float:
        """How far the response has run where the data end (1 = complete; below `COMPLETE` the after values are
        an extrapolation of the shape, found on SynK: a sigmoid half done at the end called a 13 % share 0 %)."""
        return 1.0 if self.change == "none" else float(self.response([self.t_end_s])[0])

    @property
    def D_end(self) -> tuple:
        return tuple(float(v) for v in self.D_at([self.t_end_s])[0])

    @property
    def fractions_end(self) -> tuple:
        return tuple(float(v) for v in self.fractions_at([self.t_end_s])[0])

    @property
    def bic(self) -> float:
        return float(self.n_params * math.log(self.n_steps) - 2.0 * self.log_likelihood)

    @property
    def half_time_s(self) -> float | None:
        """When the population response is half complete, seconds since the induction."""
        if self.change == "none":
            return None
        if self.shape == "exponential":
            return float(self.onset_s + self.time_s)
        if self.shape == "linear":
            return float(self.onset_s + 0.5 * self.time_s)
        return float(self.time_s)

    def response(self, t) -> np.ndarray:
        if self.change == "none":
            return np.zeros_like(np.asarray(t, float))
        onset = 0.0 if self.shape == "sigmoid" else self.onset_s
        return response_shape(self.shape, np.asarray(t, float) - onset, self.time_s, self.hill)

    def D_at(self, t) -> np.ndarray:
        """Each population's D at ``t`` (len(t) x n_pop)."""
        h = self.response(t)[:, None]
        b, a = np.asarray(self.D_before), np.asarray(self.D_after)
        return b[None, :] + (a - b)[None, :] * h

    def fractions_at(self, t) -> np.ndarray:
        h = self.response(t)[:, None]
        b, a = np.asarray(self.fractions_before), np.asarray(self.fractions_after)
        return b[None, :] + (a - b)[None, :] * h


@dataclass(frozen=True)
class _Layout:
    n_pop: int
    change: str
    shape: str | None

    @property
    def moves_D(self) -> bool:
        return self.change in ("D", "both")

    @property
    def moves_f(self) -> bool:
        return self.change in ("fractions", "both")

    @property
    def n_timing(self) -> int:
        return 0 if self.change == "none" else 2

    @property
    def size(self) -> int:
        k = self.n_pop
        return k * (2 if self.moves_D else 1) + (k - 1) * (2 if self.moves_f else 1) + self.n_timing

    def unpack(self, theta) -> dict:
        k = self.n_pop
        theta = np.asarray(theta, float)
        i = 0
        log_db = theta[i:i + k]
        i += k
        log_da = theta[i:i + k] if self.moves_D else log_db
        i += k if self.moves_D else 0
        ub = theta[i:i + k - 1]
        i += k - 1
        ua = theta[i:i + k - 1] if self.moves_f else ub
        i += k - 1 if self.moves_f else 0
        timing = theta[i:i + self.n_timing]
        return {"log_db": log_db, "log_da": log_da, "ub": ub, "ua": ua, "timing": timing}


def _softmax(u) -> np.ndarray:
    """Shares from K - 1 logits, the last population's logit fixed at 0."""
    z = np.concatenate([np.asarray(u, float), [0.0]])
    z = np.exp(z - z.max())
    return z / z.sum()


def _response_and_partials(shape: str, t: np.ndarray, timing) -> tuple:
    """H(t) and its derivatives by the two timing parameters: (onset, log time) for linear and exponential, (log
    time, log hill) for the sigmoid (whose onset is the induction)."""
    if shape == "sigmoid":
        T, n = math.exp(timing[0]), math.exp(timing[1])
        pos = t > 0
        q = np.where(pos, t, T) / T
        p = np.where(pos, np.power(q, n), 0.0)
        H = p / (1.0 + p)
        dH_dp = 1.0 / (1.0 + p) ** 2
        return H, (dH_dp * (-n * p), dH_dp * np.where(pos, n * p * np.log(q), 0.0))
    t0, T = float(timing[0]), math.exp(timing[1])
    s = t - t0
    if shape == "exponential":
        pos = s > 0
        e = np.where(pos, np.power(0.5, np.where(pos, s, 0.0) / T), 1.0)
        H = np.where(pos, 1.0 - e, 0.0)
        return H, (np.where(pos, -_LN2 / T * e, 0.0), np.where(pos, -_LN2 * (s / T) * e, 0.0))
    inside = (s > 0) & (s < T)
    H = np.clip(s / T, 0.0, 1.0)
    return H, (np.where(inside, -1.0 / T, 0.0), np.where(inside, -s / T, 0.0))


def _nll_and_gradient(theta, r2: np.ndarray, t: np.ndarray, layout: _Layout, tau_lag: float,
                      weights: np.ndarray | None = None, noise: np.ndarray | None = None) -> tuple:
    """The negative log-likelihood of every step at its own time, and its exact gradient; with ``weights``, of the
    weighted sum (each quadrature node of `_spread_nll` weighs its steps by their posterior); ``noise``: each step's
    own localisation noise (`NOISE_COLUMN`), in every population's mean."""
    from .sqd import _log_mixture_density

    p = layout.unpack(theta)
    k = layout.n_pop
    db, da = np.exp(p["log_db"]), np.exp(p["log_da"])
    fb, fa = _softmax(p["ub"]), _softmax(p["ua"])
    if layout.change == "none":
        H, partials = np.zeros_like(t), ()
    else:
        H, partials = _response_and_partials(layout.shape, t, p["timing"])
    h = H[:, None]
    D = db[None, :] + (da - db)[None, :] * h
    f = fb[None, :] + (fa - fb)[None, :] * h
    m = 4.0 * tau_lag * D
    if noise is not None:
        m = m + 2.0 * np.asarray(noise, float)[:, None]
    log_g, a = _log_mixture_density(r2, m, f)
    if log_g is None:
        return np.inf, np.zeros_like(np.asarray(theta, float))
    phi_g = np.exp(a - log_g[:, None])                       # phi_ik / g_i
    if weights is not None:
        phi_g = phi_g * np.asarray(weights, float)[:, None]
        log_g = log_g * np.asarray(weights, float)
    dl_dm = f * phi_g * (r2[:, None] - m) / (m * m)
    dl_df = phi_g
    grad = []
    if layout.moves_D:
        grad.append(np.sum(dl_dm * (4.0 * tau_lag) * db[None, :] * (1.0 - h), axis=0))
        grad_da = np.sum(dl_dm * (4.0 * tau_lag) * da[None, :] * h, axis=0)
    else:
        grad.append(np.sum(dl_dm * (4.0 * tau_lag) * db[None, :], axis=0))

    def through_softmax(g_f, shares):
        return (shares * (g_f - np.dot(g_f, shares)))[:k - 1]

    if layout.moves_f:
        g_ub = through_softmax(np.sum(dl_df * (1.0 - h), axis=0), fb)
        g_ua = through_softmax(np.sum(dl_df * h, axis=0), fa)
    else:
        g_ub = through_softmax(np.sum(dl_df, axis=0), fb)
    if layout.moves_D:
        grad.append(grad_da)
    grad.append(g_ub)
    if layout.moves_f:
        grad.append(g_ua)
    if partials:
        dl_dh = np.sum(dl_df * (fa - fb)[None, :] + dl_dm * (4.0 * tau_lag) * (da - db)[None, :], axis=1)
        grad.append(np.array([np.sum(dl_dh * partials[0]), np.sum(dl_dh * partials[1])]))
    return -float(np.sum(log_g)), -np.concatenate(grad)


def _bounds(layout: _Layout, t: np.ndarray, tau: float) -> list:
    k = layout.n_pop
    lo, hi = (math.log(v) for v in D_BOUNDS_UM2_S)
    b = [(lo, hi)] * (k * (2 if layout.moves_D else 1))
    b += [(-_LOGIT, _LOGIT)] * ((k - 1) * (2 if layout.moves_f else 1))
    if layout.change != "none":
        span = float(max(t.max(), tau))
        long = math.log(10.0 * span)
        if layout.shape == "sigmoid":
            b += [(math.log(tau), long), (math.log(0.5), math.log(12.0))]
        else:
            b += [(0.0, span), (math.log(tau), long)]
    return b


def _seed_theta(layout: _Layout, D, fractions, t: np.ndarray, tau: float, rng, *, jitter: bool) -> np.ndarray:
    """A start: the constant mixture's D's and shares (perturbed unless it is the first start) and a timing spread
    over the window after the induction."""
    k = layout.n_pop
    log_d = np.log(np.clip(np.sort(np.asarray(D, float)), *D_BOUNDS_UM2_S))
    shares = np.clip(np.asarray(fractions, float), 1e-4, None)
    u = np.log(shares[:-1] / shares[-1])
    parts = [log_d + (rng.normal(0.0, 0.3, k) if jitter else 0.0)]
    if layout.moves_D:
        parts.append(log_d + (rng.normal(0.0, 1.0, k) if jitter else 0.0))
    parts.append(np.clip(u + (rng.normal(0.0, 0.3, k - 1) if jitter else 0.0), -_LOGIT, _LOGIT))
    if layout.moves_f:
        parts.append(np.clip(u + (rng.normal(0.0, 1.0, k - 1) if jitter else 0.0), -_LOGIT, _LOGIT))
    if layout.change != "none":
        span = float(max(t.max(), tau))
        T = span / 4.0 if not jitter else math.exp(rng.uniform(math.log(tau * 5), math.log(2.0 * span)))
        if layout.shape == "sigmoid":
            parts.append(np.array([math.log(T), math.log(2.0 if not jitter else rng.uniform(1.0, 5.0))]))
        else:
            parts.append(np.array([0.0 if not jitter else rng.uniform(0.0, 0.5 * span), math.log(T)]))
    return np.concatenate(parts)


def _arrays(steps: pd.DataFrame) -> tuple:
    """``(r2, t)`` of the steps with a finite r^2 and time."""
    r2, t, _noise = _arrays_with_noise(steps)
    return r2, t


def _arrays_with_noise(steps: pd.DataFrame) -> tuple:
    """`_arrays` and each step's noise (`NOISE_COLUMN`, per-spot precision; 6 Nov 2026): ``(r2, t, noise)``, the noise
    None where the table carries none."""
    if CLOCK not in steps.columns:
        raise InductionError(f"no '{CLOCK}' column: place the steps on the clock with `induction_steps` first")
    r2 = steps["sq_disp_um2"].to_numpy(float)
    t = steps[CLOCK].to_numpy(float)
    ok = np.isfinite(r2) & np.isfinite(t)
    noise = steps[NOISE_COLUMN].to_numpy(float)[ok] if NOISE_COLUMN in steps.columns else None
    return r2[ok], t[ok], noise


def _tau_lag(steps: pd.DataFrame, tau_lag_s: float | None) -> float:
    tau = float(tau_lag_s if tau_lag_s is not None else steps.attrs.get("lag_time_s", np.nan))
    if not tau > 0:
        raise InductionError("no lag time: the step table carries no lag_time_s and none was given")
    return tau


def fit_model(steps: pd.DataFrame, *, n_pop: int, change: str, shape: str | None = "exponential",
              tau_lag_s: float | None = None, n_starts: int = 8, seed: int = 0, starts=None,
              constant: InductionFit | None = None) -> InductionFit:
    """The model (module docstring) by maximum likelihood over every step at its own time.

    ``n_starts`` starts. The constant mixture ("none") starts from the steps' quantiles; a changing model from the
    constant mixture's fit (``constant``, made here when not given), then perturbations of it with timings spread
    over the window. ``starts`` -- parameter vectors, e.g. a fit's ``theta`` -- replace them, as the held-out folds
    and the bootstrap do. The best optimum is kept."""
    from scipy.optimize import minimize

    if change not in CHANGES:
        raise InductionError(f"change {change!r}: choose one of {CHANGES}")
    if change != "none" and shape not in SHAPES:
        raise InductionError(f"shape {shape!r}: choose one of {SHAPES}")
    if n_pop < 1:
        raise InductionError(f"n_pop must be at least 1, got {n_pop}")
    if change in ("fractions", "both") and n_pop < 2:
        raise InductionError("one population has no shares to change")
    layout = _Layout(n_pop, change, None if change == "none" else shape)
    r2, t, noise = _arrays_with_noise(steps)
    tau = _tau_lag(steps, tau_lag_s)
    if len(r2) < 20 * layout.size:
        raise InductionError(f"{len(r2)} steps for {layout.size} parameters: too few to fit")
    if change != "none" and not np.any(t > 0):
        raise InductionError("no step after the induction: nothing can change")
    bounds = _bounds(layout, t, tau)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    if starts is None:
        if change == "none":
            # the populations spread over the steps' own quantiles, equal shares
            q = np.quantile(r2 if noise is None else np.maximum(r2 - 2.0 * noise, 1e-8), (np.arange(n_pop) + 0.5)
                            / n_pop)
            D, shares = np.maximum(q / (4.0 * tau), 1e-4), np.full(n_pop, 1.0 / n_pop)
        else:
            if constant is None:
                constant = fit_model(steps, n_pop=n_pop, change="none", tau_lag_s=tau, n_starts=n_starts, seed=seed)
            D, shares = constant.D_before, constant.fractions_before
        rng = np.random.default_rng(seed)
        starts = [_seed_theta(layout, D, shares, t, tau, rng, jitter=i > 0) for i in range(max(1, int(n_starts)))]
    best = None
    for theta0 in starts:
        theta0 = np.clip(np.asarray(theta0, float), lo, hi)
        res = minimize(_nll_and_gradient, theta0, args=(r2, t, layout, tau, None, noise), jac=True,
                       method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000})
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise InductionError(f"no start reached a finite likelihood ({change}, {shape})")
    return _as_fit(best.x, -float(best.fun), layout, tau, len(r2), float(t.max()), per_spot=noise is not None)


def _as_fit(theta, log_likelihood: float, layout: _Layout, tau: float, n_steps: int,
            t_end: float = math.nan, per_spot: bool = False) -> InductionFit:
    p = layout.unpack(theta)
    db, da = np.exp(p["log_db"]), np.exp(p["log_da"])
    fb, fa = _softmax(p["ub"]), _softmax(p["ua"])
    order = np.argsort(db, kind="stable")
    onset = time = hill = None
    if layout.change != "none":
        if layout.shape == "sigmoid":
            onset, time, hill = 0.0, math.exp(p["timing"][0]), math.exp(p["timing"][1])
        else:
            onset, time = float(p["timing"][0]), math.exp(p["timing"][1])
    return InductionFit(
        change=layout.change, shape=layout.shape, n_pop=layout.n_pop, tau_lag_s=tau,
        D_before=tuple(float(v) for v in db[order]), D_after=tuple(float(v) for v in da[order]),
        fractions_before=tuple(float(v) for v in fb[order]), fractions_after=tuple(float(v) for v in fa[order]),
        onset_s=onset, time_s=time, hill=hill, log_likelihood=log_likelihood, n_params=layout.size,
        n_steps=n_steps, theta=tuple(float(v) for v in theta), t_end_s=t_end, per_spot=bool(per_spot))


def _log_density(fit: InductionFit, r2: np.ndarray, t: np.ndarray, noise: np.ndarray | None = None) -> np.ndarray:
    from .sqd import _log_mixture_density

    m = 4.0 * fit.tau_lag_s * fit.D_at(t)
    if noise is not None:
        m = m + 2.0 * np.asarray(noise, float)[:, None]
    log_g, _a = _log_mixture_density(r2, m, fit.fractions_at(t))
    if log_g is None:
        raise InductionError("the fitted mixture has no density for some held-out step")
    return log_g


# --------------------------------------------------------------------------- #
# Choosing the model on units held out, and the intervals
# --------------------------------------------------------------------------- #


def _folds(units: np.ndarray, seed: int) -> list:
    """Leave one unit out up to 20 units, ten folds above (as `kinetics.fit_response`)."""
    if len(units) <= 20:
        return [[u] for u in units]
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    return [list(part) for part in np.array_split(shuffled, 10)]


def _label(model: tuple) -> str:
    return "none" if model[0] == "none" else f"{model[0]} / {model[1]}"


def compare_models(steps: pd.DataFrame, *, n_pop: int, unit: str = "movie", changes=CHANGES, shapes=SHAPES,
                   tau_lag_s: float | None = None, n_starts: int = 8, seed: int = 0) -> dict:
    """Every model in ``changes`` x ``shapes`` ("none" once), each scored by how well it predicts whole units HELD
    OUT, and the choice.

    A model is preferred over another only where it predicts held-out units better by `kinetics.HELDOUT_Z` standard
    errors of the per-unit difference. The answer is "no change" unless a changing model beats "none" so; otherwise
    the SIMPLEST model within that margin of the best (fewest parameters, then the better held-out score: SynK's D
    change scored "both" a hair above "D", 21 Oct 2026) -- the CHANGE is chosen so, and then, among the shapes of that
    change within the margin, the fewest parameters and the first of `SHAPE_PREFERENCE` (exponential, sigmoid, linear:
    8 Nov 2026; the held-out score only decides between changes, never between shapes it cannot tell apart) -- with
    what it could not be told from: another CHANGE
    (fractions or D?) within the margin leaves the change undetermined, another SHAPE only the shape -- and then the
    half-time is reported as the range over those shapes (``half_time_range_s``), since each shape places it
    differently. BIC on all steps is reported beside, not used."""
    from .kinetics import HELDOUT_Z

    column = unit_column(unit)
    if column not in steps.columns:
        raise InductionError(f"no '{column}' column to hold {unit}s out by")
    units = np.asarray(pd.unique(steps[column]))
    if len(units) < 3:
        raise InductionError(f"{len(units)} {unit}(s): a choice made on units held out needs at least three")
    models = [("none", None)] + [(c, s) for c in changes if c != "none" for s in shapes]
    constant = fit_model(steps, n_pop=n_pop, change="none", tau_lag_s=tau_lag_s, n_starts=n_starts, seed=seed)
    full = {m: constant if m[0] == "none" else
            fit_model(steps, n_pop=n_pop, change=m[0], shape=m[1], tau_lag_s=tau_lag_s, n_starts=n_starts,
                      seed=seed, constant=constant) for m in models}
    held = pd.DataFrame(0.0, index=pd.Index(units, name=column), columns=[_label(m) for m in models])
    key = steps[column].to_numpy()
    for fold in _folds(units, seed):
        out = np.isin(key, fold)
        train, test = steps[~out], steps[out]
        train.attrs.update(steps.attrs)
        r2, t, noise = _arrays_with_noise(test)
        test_units = test[column].to_numpy()[np.isfinite(test["sq_disp_um2"].to_numpy(float))
                                           & np.isfinite(test[CLOCK].to_numpy(float))]
        for m in models:
            fit = fit_model(train, n_pop=n_pop, change=m[0], shape=m[1], tau_lag_s=tau_lag_s,
                            starts=[full[m].theta])
            ll = _log_density(fit, r2, t, noise)
            for u in fold:
                held.loc[u, _label(m)] = float(ll[test_units == u].sum())
    totals = held.sum(axis=0)
    best = models[[_label(m) for m in models].index(totals.idxmax())]
    n = len(units)

    def z_against(a, b) -> float:
        """How many standard errors of the per-unit difference ``a`` predicts better than ``b``."""
        d = held[_label(a)].to_numpy() - held[_label(b)].to_numpy()
        sd = float(np.std(d, ddof=1))
        return float(np.mean(d) / (sd / math.sqrt(n))) if sd > 0 else (math.inf if np.mean(d) > 0 else 0.0)

    rows = []
    won = held.idxmax(axis=1).value_counts()
    for m in models:
        rows.append({"model": _label(m), "change": m[0], "shape": m[1] or "",
                     "heldout_log_likelihood": float(totals[_label(m)]),
                     "against_best": float(totals[_label(m)] - totals[_label(best)]),
                     "z_behind_best": 0.0 if m == best else z_against(best, m),
                     "units_won": int(won.get(_label(m), 0)), "bic": full[m].bic, "n_params": full[m].n_params})
    table = pd.DataFrame(rows)
    none = ("none", None)
    if best == none or z_against(best, none) < HELDOUT_Z:
        chosen, change_open, shape_open = none, [], []
        verdict = ("no change shown: no changing model predicts held-out units better than the constant mixture "
                   f"by {HELDOUT_Z:g} standard errors")
    else:
        within = [m for m in models if m != none and (m == best or z_against(best, m) < HELDOUT_Z)]
        by_change = {c: [m for m in within if m[0] == c] for c in dict.fromkeys(m[0] for m in within)}
        change = min(by_change, key=lambda c: min((full[m].n_params, -float(totals[_label(m)]))
                                                  for m in by_change[c]))
        # The DRAWN shape must itself beat "none" by the margin (review, 8 Nov 2026): the declared order picks among
        # the shapes that do; where none of this change's shapes does on its own, its best-scoring one is drawn.
        beating = [m for m in by_change[change] if z_against(m, none) >= HELDOUT_Z]
        chosen = (min(beating, key=lambda m: (full[m].n_params, SHAPE_PREFERENCE.index(m[1]))) if beating else
                  max(by_change[change], key=lambda m: float(totals[_label(m)])))
        change_open = sorted({m[0] for m in within if m[0] != chosen[0]})
        shape_open = sorted({m[1] for m in within if m[0] == chosen[0] and m != chosen})
        verdict = f"{chosen[0]} change, {chosen[1]} shape"
        if change_open:
            verdict += f"; the change cannot be told from: {', '.join(change_open)}"
        if shape_open:
            verdict += f"; the shape cannot be told from: {', '.join(shape_open)}"
        done = full[chosen].completion
        if done < COMPLETE:
            verdict += (f"; the response is {100 * done:.0f} % complete where the data end, so its 'after' values "
                        f"are the shape's extrapolation, so read the values at the end of the data")
    # THE MODEL THE PERMUTATION TEST REFITS: the verdict's own where it found a change, else the changing model the
    # held-out units liked best (never "none")
    changing = [m for m in models if m != none]
    tested = chosen if chosen != none else max(changing, key=lambda m: float(totals[_label(m)]))
    same_change = [m for m in models if m[0] == chosen[0] and (m == chosen or m[1] in shape_open)]
    halves = [full[m].half_time_s for m in same_change if full[m].half_time_s is not None]
    return {"table": table, "recommended": chosen, "fit": full[chosen], "fits": full,
            "change_undetermined": change_open, "shape_undetermined": shape_open, "verdict": verdict,
            "half_time_range_s": (min(halves), max(halves)) if halves else None,
            "tested": tested, "constant": constant, "unit": unit, "n_units": n, "heldout": held}


def bootstrap(steps: pd.DataFrame, fit: InductionFit, *, unit: str = "movie", n_boot: int | None = None,
              seed: int = 0, grid=None, stratify: bool = True) -> dict:
    """``fit``'s parameters refitted on ``n_boot`` resamples of whole units (`kinetics.N_BOOT` by default): the
    draws and a 95 % percentile interval per parameter. Below five units the interval is coarse and says so.

    ``grid`` (seconds since the induction) also keeps every refit's D's and shares there (``curves``: ``"D"`` and
    ``"fractions"``, each n_boot x len(grid) x n_pop), so a band is the percentile of whole refits at each time --
    the Kinetics page's band, from the same resamples as the parameters' intervals."""
    from .kinetics import N_BOOT

    n_boot = N_BOOT if n_boot is None else int(n_boot)
    column = unit_column(unit)
    if column not in steps.columns:
        raise InductionError(f"no '{column}' column to resample {unit}s by")
    units = np.asarray(pd.unique(steps[column]))
    if len(units) < 2:
        raise InductionError(f"{len(units)} {unit}(s): an interval over units needs at least two")
    rng = np.random.default_rng(seed)
    parts = {u: steps[steps[column].to_numpy() == u] for u in units}
    # STRATIFIED BY REPLICATE: movies (or cells) are resampled WITHIN their replicate, every replicate keeping its
    # number of them. The replicates themselves are then held fixed, so the interval leaves out the variation
    # BETWEEN replicates; resampling by replicate (unit "replicate") puts it back.
    strata = None
    if stratify and unit != "replicate" and UNIT_COLUMNS["replicate"] in steps.columns:
        owner = steps.groupby(column)[UNIT_COLUMNS["replicate"]].nunique()
        if (owner == 1).all() and steps[UNIT_COLUMNS["replicate"]].nunique() >= 2:
            home = steps.groupby(column)[UNIT_COLUMNS["replicate"]].first()
            strata = [np.asarray([u for u in units if home[u] == r]) for r in pd.unique(home)]
    draws, curve_D, curve_f = [], [], []
    grid = None if grid is None else np.asarray(grid, float)
    for _b in range(n_boot):
        if strata is None:
            pick = rng.choice(units, size=len(units), replace=True)
        else:
            pick = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        sample = pd.concat([parts[u] for u in pick], ignore_index=True)
        sample.attrs.update(steps.attrs)
        f = fit_model(sample, n_pop=fit.n_pop, change=fit.change, shape=fit.shape, tau_lag_s=fit.tau_lag_s,
                      starts=[fit.theta])
        row = {f"D_before_{k + 1}": f.D_before[k] for k in range(f.n_pop)}
        row.update({f"D_after_{k + 1}": f.D_after[k] for k in range(f.n_pop)})
        row.update({f"fraction_before_{k + 1}": f.fractions_before[k] for k in range(f.n_pop)})
        row.update({f"fraction_after_{k + 1}": f.fractions_after[k] for k in range(f.n_pop)})
        if f.change != "none":
            row.update(onset_s=f.onset_s, time_s=f.time_s, half_time_s=f.half_time_s)
        draws.append(row)
        if grid is not None:
            curve_D.append(f.D_at(grid))
            curve_f.append(f.fractions_at(grid))
    draws = pd.DataFrame(draws)
    interval = {c: (float(np.percentile(draws[c], 2.5)), float(np.percentile(draws[c], 97.5)))
                for c in draws.columns}
    few = len(units) < 5
    return {"draws": draws, "interval": interval, "unit": unit, "n_units": len(units), "n_boot": n_boot,
            "few_units": few, "stratified": strata is not None,
            "n_strata": 0 if strata is None else len(strata),
            "curves": None if grid is None else {"t_s": grid, "D": np.asarray(curve_D),
                                                 "fractions": np.asarray(curve_f)},
            "caveat": (f"{len(units)} {unit}s: a percentile interval over so few units is coarse" if few else "")}


def binned_fits(steps: pd.DataFrame, *, n_pop: int, bin_s: float, tau_lag_s: float | None = None,
                min_steps: int = 300, seed: int = 0, origin: float | None = None) -> pd.DataFrame:
    """The D's and shares fitted FREELY in each time bin (the constant mixture, "none", by maximum likelihood;
    populations by D): what the data look like, drawn beside the model, never the estimate (module docstring). A
    bin under ``min_steps`` steps is NaN, never zero. ``origin`` (seconds) puts the bins' edges on a grid that starts
    at or below it, so fits of parts of one table (`replicate_binned_fits`) share their bins."""
    if not bin_s > 0:
        raise InductionError(f"bin_s must be positive, got {bin_s}")
    r2, t, noise = _arrays_with_noise(steps)
    tau = _tau_lag(steps, tau_lag_s)
    lo = math.floor((t.min() if origin is None else min(origin, t.min())) / bin_s) * bin_s
    edges = np.arange(lo, t.max() + bin_s, bin_s)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        inside = (t >= a) & (t < b)
        row = {"t_from_s": float(a), "t_to_s": float(b), "t_s": float(0.5 * (a + b)), "n_steps": int(inside.sum())}
        if row["n_steps"] >= min_steps:
            part = pd.DataFrame({"sq_disp_um2": r2[inside], CLOCK: t[inside],
                                 **({NOISE_COLUMN: noise[inside]} if noise is not None else {})})
            fit = fit_model(part, n_pop=n_pop, change="none", tau_lag_s=tau, n_starts=4, seed=seed)
            row.update({f"D_{k + 1}": fit.D_before[k] for k in range(n_pop)})
            row.update({f"fraction_{k + 1}": fit.fractions_before[k] for k in range(n_pop)})
        else:
            row.update({f"D_{k + 1}": np.nan for k in range(n_pop)})
            row.update({f"fraction_{k + 1}": np.nan for k in range(n_pop)})
        rows.append(row)
    return pd.DataFrame(rows)


def replicate_binned_fits(steps: pd.DataFrame, *, n_pop: int, bin_s: float, tau_lag_s: float | None = None,
                          min_steps: int = 300, seed: int = 0) -> pd.DataFrame:
    """`binned_fits` made on each REPLICATE's steps alone (the table's ``day`` column), on the bins of the whole
    table: the pooled fits' rows with a ``replicate`` column, a bin the replicate holds under ``min_steps`` steps of
    NaN. Replicates are independent (Rule 0): nothing is shared between them, and nothing matches replicate n of one
    condition to replicate n of another. CHOSEN OVER A BOOTSTRAP INTERVAL PER BIN because the fit is cheap, a bin of a
    replicate usually holds the steps of ONE movie (a movie lasts about a bin), so a resampled interval over its units
    would have one unit to resample; the replicate's own value is the honest picture of that. An empty table (no
    ``day`` column, or one replicate) returns an empty frame: there is nothing to separate."""
    column = UNIT_COLUMNS["replicate"]
    if column not in steps.columns or steps[column].nunique() < 2:
        return pd.DataFrame()
    _r2, t = _arrays(steps)
    parts = []
    for rep, part in steps.groupby(column, sort=True):
        part = part.copy()
        part.attrs.update(steps.attrs)
        if not np.isfinite(part[CLOCK].to_numpy(float)).any():
            continue
        fits = binned_fits(part, n_pop=n_pop, bin_s=bin_s, tau_lag_s=tau_lag_s, min_steps=min_steps, seed=seed,
                           origin=float(t.min()))
        fits.insert(0, "replicate", rep)
        parts.append(fits)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Onsets that differ between movies and between cells
# --------------------------------------------------------------------------- #
#
# THE CASE (the experimenter's design: "the event might trigger at different timepoints for different molecules of the same
# cell, for different cells of the same movies and so on"). Each MOVIE and each CELL in it starts responding at its
# own time: the population response H is shifted by a normal offset per movie and one per cell, and the spreads
# (their SDs) are estimated with the response, the offsets integrated out (Gauss-Hermite, `GH_NODES` nodes a level)
# so a unit's offset is never fitted to its own noise -- the trap `kinetics.onset_alignment_null` guards against.
#
# THE MOLECULE LEVEL IS PART OF H. A track lasts a fraction of a second against a response of seconds, so a molecule
# whose onset is late looks exactly like a slower response: molecules' onsets cannot be told from H's shape and are
# folded into it.
#
# THE LIKELIHOOD RATIO IS NOT A TEST ON TRACKS, AND THE SD IS NOT A MEASUREMENT (SynK, 22 Oct 2026,
# `docs/claude/states_and_kinetics.md`). A random onset per group of tracks buys back whatever the model leaves
# unexplained between tracks: on SynK it found 1.7-2 s between cells where every cell started together, also after
# the tracks were shuffled between cells, with ratios of 150-400. What decides is `onset_spread_null`, the same fit
# on the units SHUFFLED (tracks moved between the cells of a movie; whole cells between movies): there 30 movies
# whose onsets were spread 2.7 s gave a ratio of 957 against 371-576 shuffled, 30 that started together 145 against
# 41-168, while cells (~27 tracks of ~7 steps each) were not told apart at all (a 1.2 s spread scored BELOW its
# shuffles). The SD it reports carries the shuffles' own 0.7-2 s and is a sign, not a size.

GH_NODES = 7


@dataclass(frozen=True)
class OnsetSpread:
    """The model with onsets spread between units: its re-fitted parameters and each level's SD. The ratio against
    no spread is reported but is NOT a test (module comment above): `onset_spread_null` is."""

    fit: InductionFit
    sd_s: dict
    log_likelihood: float
    log_likelihood_no_spread: float
    n_units: dict

    @property
    def likelihood_ratio(self) -> float:
        return float(2.0 * (self.log_likelihood - self.log_likelihood_no_spread))


def _dH_dt(shape: str, t: np.ndarray, timing) -> np.ndarray:
    if shape == "sigmoid":
        T, n = math.exp(timing[0]), math.exp(timing[1])
        pos = t > 0
        q = np.where(pos, t, T) / T
        p = np.where(pos, np.power(q, n), 0.0)
        return np.where(pos, (n * p / np.where(pos, t, 1.0)) / (1.0 + p) ** 2, 0.0)
    _H, partials = _response_and_partials(shape, t, timing)
    return -partials[0]                                      # H depends on t - onset


def _track_terms(theta, ctx: dict, shift: float, weights: np.ndarray | None = None):
    """Each TRACK's log-likelihood with every time moved by ``shift``: the track in ONE state throughout, that
    state's D following each step's own time, the shares read at the track's first step. With ``weights`` (one per
    track) also the gradient of the weighted sum by every parameter, and its derivative by the steps' time.

    WHY TRACKS AND NOT STEPS. Steps of one track mostly come from one molecule in one state, so slow tracks cluster
    within cells; scored as independent steps, that clustering reads as cells starting at different times. On SynK's
    tracks the step version found 3 s between cells where every cell started together (p ~ 1e-69, 22 Oct 2026)."""
    from scipy.special import logsumexp

    layout, r2, tau = ctx["layout"], ctx["r2"], ctx["tau"]
    t, tt = ctx["t"] - shift, ctx["t_track"] - shift
    k = layout.n_pop
    p = layout.unpack(theta)
    db, da = np.exp(p["log_db"]), np.exp(p["log_da"])
    fb, fa = _softmax(p["ub"]), _softmax(p["ua"])
    Hs, ps = _response_and_partials(layout.shape, t, p["timing"])
    Ht, pt = _response_and_partials(layout.shape, tt, p["timing"])
    m = 4.0 * tau * (db[None, :] + (da - db)[None, :] * Hs[:, None])
    if ctx.get("noise") is not None:
        m = m + 2.0 * ctx["noise"][:, None]
    a = -r2[:, None] / m - np.log(m)
    A = np.add.reduceat(a, ctx["track_starts"], axis=0)
    f = fb[None, :] + (fa - fb)[None, :] * Ht[:, None]
    L = np.log(f) + A
    log_p = logsumexp(L, axis=1)
    if weights is None:
        return log_p
    w = np.exp(L - log_p[:, None]) * np.asarray(weights, float)[:, None]      # (tracks, K), weighted posterior
    dl_dm = w[ctx["step_track"]] * (r2[:, None] - m) / (m * m)
    g_f = w / f
    hs, ht = Hs[:, None], Ht[:, None]

    def through_softmax(g, shares):
        return (shares * (g - np.dot(g, shares)))[:k - 1]

    grad = []
    if layout.moves_D:
        grad.append(np.sum(dl_dm * (4.0 * tau) * db[None, :] * (1.0 - hs), axis=0))
        grad.append(np.sum(dl_dm * (4.0 * tau) * da[None, :] * hs, axis=0))
    else:
        grad.append(np.sum(dl_dm * (4.0 * tau) * db[None, :], axis=0))
    if layout.moves_f:
        grad.append(through_softmax(np.sum(g_f * (1.0 - ht), axis=0), fb))
        grad.append(through_softmax(np.sum(g_f * ht, axis=0), fa))
    else:
        grad.append(through_softmax(np.sum(g_f, axis=0), fb))
    per_step = np.sum(dl_dm * (4.0 * tau) * (da - db)[None, :], axis=1)        # d / dH at each step
    per_track = np.sum(g_f * (fa - fb)[None, :], axis=1)                        # d / dH at each track's start
    grad.append(np.array([np.sum(per_step * ps[0]) + np.sum(per_track * pt[0]),
                          np.sum(per_step * ps[1]) + np.sum(per_track * pt[1])]))
    dt = (np.sum(per_step * _dH_dt(layout.shape, t, p["timing"]))
          + np.sum(per_track * _dH_dt(layout.shape, tt, p["timing"])))
    return log_p, np.concatenate(grad), float(dt)


def _spread_nll(phi, ctx: dict) -> tuple:
    """Negative marginal log-likelihood of the units' TRACKS, every level's onset offset integrated out, and its
    exact gradient by every model parameter and the log SD of each level (none: no spread). Two passes over the
    nodes: the likelihoods give each node's posterior weight on each track; each node's gradient is then the track
    gradient weighted by it."""
    from scipy.special import logsumexp

    layout = ctx["layout"]
    x, lw = ctx["nodes"]
    P = layout.size
    theta, sd = np.asarray(phi[:P], float), np.exp(np.asarray(phi[P:], float))
    n_levels = len(sd)
    if n_levels == 0:
        ones = np.ones(len(ctx["t_track"]))
        log_p, g, _dt = _track_terms(theta, ctx, 0.0, ones)
        return -float(np.sum(log_p)), -g
    q = len(x)
    root2 = math.sqrt(2.0)
    if n_levels == 2:
        grid_a, grid_b = np.meshgrid(x, x, indexing="ij")
        node_a, node_b = root2 * grid_a.ravel(), root2 * grid_b.ravel()
        shifts = sd[0] * node_a + sd[1] * node_b
    else:
        node_a = root2 * x
        shifts = sd[0] * node_a
    LP = np.stack([_track_terms(theta, ctx, s) for s in shifts])           # (J, tracks)
    S = np.add.reduceat(LP, ctx["inner_starts"], axis=1)                   # (J, inner units)
    if n_levels == 2:
        S = S.reshape(q, q, -1)
        C = logsumexp(S + lw[None, :, None], axis=1)                       # (q, cells): each cell's offset out
        M = np.add.reduceat(C, ctx["outer_starts"], axis=1)                # (q, movies)
        LL_m = logsumexp(M + lw[:, None], axis=0)
        post_a = np.exp(M + lw[:, None] - LL_m[None, :])
        post_b = np.exp(S + lw[None, :, None] - C[:, None, :])
        W = (post_a[:, ctx["inner_outer"]][:, None, :] * post_b).reshape(len(shifts), -1)[:, ctx["track_inner"]]
    else:
        LL_m = logsumexp(S + lw[:, None], axis=0)
        W = np.exp(S + lw[:, None] - LL_m[None, :])[:, ctx["track_inner"]]
    g_theta = np.zeros(P)
    d_shift = np.empty(len(shifts))
    for j, s in enumerate(shifts):
        _lp, g, dt = _track_terms(theta, ctx, s, W[j])
        g_theta += g
        d_shift[j] = -dt                                                    # every time moves by -shift
    g_sd = [np.sum(d_shift * node_a) * sd[0]] + ([np.sum(d_shift * node_b) * sd[1]] if n_levels == 2 else [])
    return -float(np.sum(LL_m)), -np.concatenate([g_theta, g_sd])


def _spread_context(steps: pd.DataFrame, fit: InductionFit, levels: tuple, n_nodes: int) -> tuple:
    for column in (*levels, "uid"):
        if column not in steps.columns:
            raise InductionError(f"no '{column}' column: the spread is read per track, in its units")
    r2_all, t_all = steps["sq_disp_um2"].to_numpy(float), steps[CLOCK].to_numpy(float)
    ok = np.isfinite(r2_all) & np.isfinite(t_all)
    frame = steps.loc[ok, [*levels, "uid"]].astype(str).reset_index(drop=True)
    frame["_r2"], frame["_t"] = r2_all[ok], t_all[ok]
    if NOISE_COLUMN in steps.columns:
        frame["_noise"] = steps[NOISE_COLUMN].to_numpy(float)[ok]
    frame = frame.sort_values([*levels, "uid", "_t"], kind="stable").reset_index(drop=True)
    track_key = frame[[*levels, "uid"]].agg("\x1f".join, axis=1).to_numpy()
    new_track = np.r_[True, track_key[1:] != track_key[:-1]]
    track_starts = np.flatnonzero(new_track)
    inner_key = frame[list(levels)].agg("\x1f".join, axis=1).to_numpy()[track_starts]
    new_inner = np.r_[True, inner_key[1:] != inner_key[:-1]]
    x, w = np.polynomial.hermite.hermgauss(n_nodes)
    t = frame["_t"].to_numpy()
    ctx = {"layout": _Layout(fit.n_pop, fit.change, fit.shape), "tau": fit.tau_lag_s, "r2": frame["_r2"].to_numpy(),
           "t": t, "t_track": t[track_starts], "track_starts": track_starts,
           "noise": frame["_noise"].to_numpy() if "_noise" in frame.columns else None,
           "step_track": np.cumsum(new_track) - 1, "nodes": (x, np.log(w / math.sqrt(math.pi))),
           "inner_starts": np.flatnonzero(new_inner), "track_inner": np.cumsum(new_inner) - 1}
    n_units = {levels[-1]: int(new_inner.sum()), "track": len(track_starts)}
    if len(levels) == 2:
        outer_of_inner = frame[levels[0]].to_numpy()[track_starts][ctx["inner_starts"]]
        new_outer = np.r_[True, outer_of_inner[1:] != outer_of_inner[:-1]]
        ctx["outer_starts"] = np.flatnonzero(new_outer)
        ctx["inner_outer"] = np.cumsum(new_outer) - 1
        n_units[levels[0]] = int(new_outer.sum())
    return ctx, n_units


#: The spread fits' stopping rule, RELATIVE to the likelihood (L-BFGS-B's ``ftol``): 1e-13 of a likelihood of ~1e5 is
#: 1e-8. At scipy's default (2.2e-9, i.e. 2e-4 here) a fit on a FRACTIONS change stopped after one iteration at its
#: 0.5 s start, the likelihood being nearly flat in the SD there, and every shuffle with it (SynK's grid, 23 Oct 2026).
SPREAD_FTOL = 1e-13


def fit_onset_spread(steps: pd.DataFrame, fit: InductionFit, *, levels=("movie", "cell"), n_nodes: int = GH_NODES,
                     n_starts: int = 2, seed: int = 0) -> OnsetSpread:
    """``fit``'s model, scored per TRACK (`_track_terms`), with onsets spread between units of ``levels`` (outer
    first; at most two, the second nested in the first): each level's SD and EVERY model parameter re-fitted with
    it. Holding the D's and shares at the unspread fit is wrong: a smeared rise is fitted there by a slower approach
    to a further plateau (a slow D 0.02 -> 0.25 read 0.39 with a 14 s time for 3 s). The same per-track model is
    fitted without spread for the ratio; both start from ``fit``. Whether the spread is real is
    `onset_spread_null`'s answer, never this ratio's."""
    from scipy.optimize import minimize

    if fit.change == "none":
        raise InductionError("a fit without a change has no onset to spread")
    levels = tuple(levels)
    if not 1 <= len(levels) <= 2:
        raise InductionError(f"one or two levels (the second inside the first), got {levels}")
    ctx, n_units = _spread_context(steps, fit, levels, n_nodes)
    span = float(max(ctx["t"].max(), fit.tau_lag_s))
    layout = ctx["layout"]
    P = layout.size
    model_bounds = _bounds(layout, ctx["t"], fit.tau_lag_s)
    theta0 = np.clip(np.asarray(fit.theta, float), [b[0] for b in model_bounds], [b[1] for b in model_bounds])
    flat = minimize(_spread_nll, theta0, args=(ctx,), jac=True, method="L-BFGS-B", bounds=model_bounds,
                    options={"maxiter": 1000, "ftol": SPREAD_FTOL})
    if not np.isfinite(flat.fun):
        raise InductionError("the per-track model without spread reached no finite likelihood")
    bounds = model_bounds + [(math.log(0.01), math.log(span))] * len(levels)
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    rng = np.random.default_rng(seed)
    best = None
    # ALWAYS ALSO FROM NO SPREAD (23 Oct 2026): the model at the SD's floor IS the no-spread model, so a fit started
    # there can never end worse than it. From 0.5 s alone, SynK's grid read ratios down to -2.6 -- fits settled
    # worse than no spread, a noise the shuffle test then had to carry on both sides.
    starts = [np.full(len(levels), 0.5)] + [
        np.exp(rng.uniform(math.log(0.1), math.log(0.5 * span), len(levels))) for _ in range(max(1, int(n_starts)) - 1)]
    starts.append(np.exp(lo[P:]))
    for sd0 in starts:
        phi0 = np.clip(np.concatenate([flat.x, np.log(sd0)]), lo, hi)
        res = minimize(_spread_nll, phi0, args=(ctx,), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "ftol": SPREAD_FTOL})
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise InductionError("no start reached a finite likelihood with spread onsets")
    refit = _as_fit(best.x[:P], -float(best.fun), layout, fit.tau_lag_s, fit.n_steps, fit.t_end_s,
                    per_spot=fit.per_spot)
    return OnsetSpread(fit=refit, sd_s={lv: float(math.exp(v)) for lv, v in zip(levels, best.x[P:])},
                       log_likelihood=-float(best.fun), log_likelihood_no_spread=-float(flat.fun),
                       n_units=n_units)


#: What a unit of each level is made of, moved whole by the shuffle: a movie's CELLS, a cell's TRACKS.
SPREAD_LEVELS = {"movie": ("cell", None), "cell": ("uid", "movie")}


def shuffle_members(steps: pd.DataFrame, level: str, rng) -> pd.DataFrame:
    """``steps`` with the members of every ``level`` unit (`SPREAD_LEVELS`: a movie's cells, a cell's tracks) dealt
    out again at random among the units -- within the same movie for cells -- each member moved WHOLE, with its
    steps and their times; every unit keeps its number of members. A cell moved to another movie keeps a name of its
    own (``<movie>/<cell>``)."""
    if level not in SPREAD_LEVELS:
        raise InductionError(f"level {level!r}: choose one of {sorted(SPREAD_LEVELS)}")
    member, parent = SPREAD_LEVELS[level]
    for column in [c for c in (level, member, parent) if c]:
        if column not in steps.columns:
            raise InductionError(f"no '{column}' column to shuffle {level}s by")
    out = steps.copy()
    member_key = (steps["movie"].astype(str) + "/" + steps[member].astype(str)) if level == "movie" else \
        steps[member].astype(str)
    if level == "movie":
        out["cell"] = member_key
    owners = pd.DataFrame({"key": member_key, level: steps[level].astype(str),
                           "parent": steps[parent].astype(str) if parent else ""}).drop_duplicates("key")
    if owners["key"].duplicated().any():
        raise InductionError(f"a {member} belongs to two {level}s: it cannot be moved whole")
    new_owner = {}
    for _p, group in owners.groupby("parent", sort=False):
        new_owner.update(zip(group["key"], rng.permutation(group[level].to_numpy())))
    out[level] = member_key.map(new_owner).to_numpy()
    out.attrs.update(steps.attrs)
    return out


def onset_spread_null(steps: pd.DataFrame, fit: InductionFit, *, level: str = "movie", n_shuffles: int = 19,
                      n_nodes: int = GH_NODES, seed: int = 0) -> dict:
    """Do onsets differ between units of ``level`` beyond chance? The spread fitted on the data against the same
    fit on ``n_shuffles`` copies with the units' members dealt out at random (`shuffle_members`): ``p`` is the rank
    of the observed ratio among them (``(1 + shuffles at least as high) / (1 + shuffles)``; 19 reach 0.05). The SD
    is reported beside the shuffles' own, as a sign and not a size (module comment)."""
    observed = fit_onset_spread(steps, fit, levels=(level,), n_nodes=n_nodes, n_starts=1)
    rng = np.random.default_rng(seed)
    null = [fit_onset_spread(shuffle_members(steps, level, rng), fit, levels=(level,), n_nodes=n_nodes,
                             n_starts=1) for _ in range(int(n_shuffles))]
    ratios = np.array([n.likelihood_ratio for n in null])
    sds = np.array([n.sd_s[level] for n in null])
    p = float((1 + np.sum(ratios >= observed.likelihood_ratio)) / (1 + len(ratios)))
    shown = p <= 0.05
    verdict = (f"onsets differ between {level}s beyond chance (ratio {observed.likelihood_ratio:.0f} against "
               f"{ratios.min():.0f}-{ratios.max():.0f} shuffled, p = {p:.2g}); the SD ({observed.sd_s[level]:.2f} s) "
               f"is a sign, not a size: shuffling alone gives {sds.min():.2f}-{sds.max():.2f} s"
               if shown else
               f"no spread of onsets between {level}s shown (ratio {observed.likelihood_ratio:.0f} against "
               f"{ratios.min():.0f}-{ratios.max():.0f} shuffled, p = {p:.2g})")
    return {"observed": observed, "null_ratios": ratios, "null_sd_s": sds, "p": p, "shown": shown,
            "verdict": verdict, "level": level, "n_shuffles": int(n_shuffles)}


# --------------------------------------------------------------------------- #
# The time-label permutation test (7 Nov 2026)
# --------------------------------------------------------------------------- #
#
# THE QUESTION. Does the order of the clock matter at all? The held-out verdict (`compare_models`) asks whether a
# changing model predicts NEW units better than the constant mixture, by two standard errors of the per-unit
# difference; with a handful of movies a replicate that is 278 log-likelihood units ahead can still sit at 1 SE
# (SynK_binding-up, 26 Oct 2026, `docs/synthetic_synr.md`), because the movie-to-movie scatter is in the
# denominator. This test asks the other question: of the arrangements of the clock that a world WITHOUT change could
# have produced, how many fit a changing model as well as the one observed?
#
# THE NULL, AND WHAT IS SHUFFLED. Under no change, which time a step is labelled with does not matter, but WHICH MOVIE
# it belongs to does (movies differ in their own D's and shares, and that scatter must travel with the movie). So
# the clock moves as whole movies, never step by step:
#   * movies of one replicate that start at different times (the addition clock, movies minutes apart): their start
#     times are DEALT AMONG THE MOVIES of that replicate, every movie's steps shifted by the same amount, so a
#     movie's own scatter, its tracks and their order inside it are untouched;
#   * movies that start together (the trigger clock: the induction falls on one frame of every movie): a movie has
#     nothing to swap with, so its TRACKS' start times are dealt among its tracks, every track shifted whole.
# The replicate stays the unit (Rule 0): nothing is ever moved between replicates, and a replicate's movies never
# borrow a time from another's.
#
# THE STATISTIC (EXACT since 7 Nov 2026, asked "yes"): the LARGEST log-likelihood any changing model of the Run
# (every change x shape compared, `compare_models`' models but "none") gains over the constant mixture, every model
# refitted on every shuffle with the same starts as on the observed clock (the constant mixture does not look at the
# clock, so it is the same on every shuffle). The real clock and each shuffle are scored ALIKE, each by its own best
# model, so the choice among models is part of the test and the p is exact for "does the clock matter at all"; which
# KIND of change it is stays the held-out comparison's answer. (Until then the one model the held-out folds chose on
# the observed data was refitted, which gave the real clock a head start: a slightly small p.) p is the rank of the
# observed statistic among the shuffles' (observed counted in): ``(1 + shuffles at least as large) / (1 + shuffles)``;
# 199 shuffles reach 0.005. A model NAMED IN ADVANCE may still be tested alone (``models`` of one); a model picked
# from these same data must not be.

#: Shuffles of the standard test (the smallest p they reach is 1/200).
PERMUTATIONS = 199
#: Starts of every refit, the observed clock's too (same effort on both sides).
PERM_STARTS = 4
#: Shuffles per pool job: a job is a few seconds, which is how long a Cancel waits (every shuffle refits every
#: changing model since the test became exact, about nine fits: two shuffles a job).
PERM_CHUNK = 2
#: The fewest distinct arrangements a test needs to be able to reach 0.05 at all.
PERM_MIN_ARRANGEMENTS = 20


@dataclass(frozen=True)
class _Start:
    """What `fit_model` reads of a constant fit (``constant=``): the mixture the changing fits start from."""

    D_before: tuple
    fractions_before: tuple


def _movie_table(steps: pd.DataFrame, replicate: str) -> tuple:
    """``(movie names, their replicate, their start)`` as aligned arrays, a movie in one replicate only. A movie's
    START is the clock of its first frame, ``clock - (frame - 1) * tau`` (`induction_steps`): the same for every
    movie under the trigger clock, each movie's own time since the addition under the other. (The first STEP's clock
    would not do: it is wherever the movie's first track begins.)"""
    for column in ("movie", replicate, CLOCK, "frame"):
        if column not in steps.columns:
            raise InductionError(f"no '{column}' column to shuffle the clock by")
    tau = float(steps.attrs.get("tau_s", math.nan))
    if not tau > 0:
        raise InductionError("no frame interval (tau_s) in the step table to find each movie's start by")
    movie = steps["movie"].astype(str).to_numpy()
    t = steps[CLOCK].to_numpy(float) - (steps["frame"].to_numpy(float) - 1.0) * tau
    first = pd.DataFrame({"movie": movie, "rep": steps[replicate].astype(str).to_numpy(), "t": t})
    owners = first.groupby("movie")["rep"].nunique()
    if (owners > 1).any():
        raise InductionError(f"movie {owners[owners > 1].index[0]!r} is in two replicates: a clock cannot be dealt "
                             f"among movies of a replicate")
    table = first.groupby("movie").agg(rep=("rep", "first"), start=("t", "min"))
    return table.index.to_numpy(), table["rep"].to_numpy(), table["start"].to_numpy(float)


def permutation_arrangements(steps: pd.DataFrame, replicate: str = UNIT_COLUMNS["replicate"]) -> float:
    """How many different clocks `shuffle_clock` can deal (the product, over replicates whose movies start at
    different times, of the factorial of the number of distinct starts; infinite where a replicate's movies start
    together, whose tracks are dealt instead). Below `PERM_MIN_ARRANGEMENTS` no test can reach 0.05."""
    samples = steps.attrs.get("samples")
    if samples:
        return _sample_arrangements(steps, samples, replicate)
    _names, reps, starts = _movie_table(steps, replicate)
    total = 1.0
    for r in pd.unique(reps):
        s = np.round(starts[reps == r], 6)
        if len(np.unique(s)) <= 1:
            return math.inf
        total *= math.factorial(len(s))
    return total


def _samples_by_replicate(steps: pd.DataFrame, samples: dict, replicate: str) -> dict:
    """``{replicate: {sample: addition time (s)}}`` for the design "Added to each sample" (``steps.attrs["samples"]``:
    ``{"movie_sample": {movie: sample}, "addition_s": {sample: the addition on an absolute clock, s}}``). RAISES where
    a movie has no sample or a sample spans two replicates (the culture is the unit; a sample is inside one)."""
    names, reps, _starts = _movie_table(steps, replicate)
    movie_sample, addition = samples["movie_sample"], samples["addition_s"]
    owner: dict = {}
    out: dict = {}
    for m, r in zip(names, reps):
        if str(m) not in movie_sample:
            raise InductionError(f"movie {m!r} is in no sample: every movie needs its sample and its addition time")
        s = movie_sample[str(m)]
        if owner.setdefault(s, r) != r:
            raise InductionError(f"sample {s!r} holds movies of two replicates ({owner[s]} and {r}): a sample is "
                                 f"inside one culture")
        out.setdefault(r, {})[s] = float(addition[s])
    return out


def _sample_arrangements(steps: pd.DataFrame, samples: dict, replicate: str) -> float:
    """Distinct ways to deal the samples' addition times among a replicate's samples (the product over replicates of
    n! over the repeats of equal times); a replicate whose samples were all added together deals nothing (1)."""
    total = 1.0
    for _r, adds in _samples_by_replicate(steps, samples, replicate).items():
        values = np.round(np.array(list(adds.values())), 3)
        ways = math.factorial(len(values))
        for v in np.unique(values):
            ways //= math.factorial(int(np.sum(values == v)))
        total *= ways
    return total


def _shuffle_samples(steps: pd.DataFrame, rng, samples: dict, replicate: str) -> pd.DataFrame:
    """The design "Added to each sample": within each replicate the samples' ADDITION TIMES are permuted among its
    samples; every movie of a sample moves by the change in its sample's addition (``clock = camera start - addition``,
    so a later addition moves the clock down). Nothing crosses a replicate. RAISES where there is nothing to deal."""
    movie = steps["movie"].astype(str).to_numpy()
    movie_sample = samples["movie_sample"]
    shift = np.zeros(len(steps))
    dealt = False
    for _r, adds in _samples_by_replicate(steps, samples, replicate).items():
        names = list(adds)
        values = np.array([adds[s] for s in names])
        if len(np.unique(np.round(values, 3))) <= 1:
            continue
        moved = rng.permutation(values)
        delta = {s: float(a - b) for s, a, b in zip(names, values, moved)}
        sample_of = np.array([movie_sample.get(m, "") for m in movie])
        for s, d in delta.items():
            if d:
                shift[sample_of == s] += d
        dealt = True
    if not dealt:
        raise InductionError("nothing to shuffle: every replicate holds one sample, or its samples were all added "
                             "at one time")
    out = steps.copy()
    out[CLOCK] = steps[CLOCK].to_numpy(float) + shift
    out.attrs.update(steps.attrs)
    return out


def shuffle_clock(steps: pd.DataFrame, rng, replicate: str = UNIT_COLUMNS["replicate"]) -> pd.DataFrame:
    """``steps`` with the clock dealt out again at random, WHOLE MOVIES at a time (see the section's comment above):
    within each replicate the movies' start times are permuted among its movies; a replicate whose movies all start
    together has the start times of each movie's TRACKS (``uid``) permuted among that movie's tracks instead. Every
    movie keeps its steps, tracks and their order; nothing crosses a replicate. RAISES where there is nothing to
    deal: a single movie per replicate on the addition clock, or no ``uid`` column on the trigger clock."""
    samples = steps.attrs.get("samples")
    if samples:
        return _shuffle_samples(steps, rng, samples, replicate)
    names, reps, starts = _movie_table(steps, replicate)
    t = steps[CLOCK].to_numpy(float).copy()
    movie = steps["movie"].astype(str).to_numpy()
    shift = np.zeros(len(steps))
    dealt = False
    for r in pd.unique(reps):
        here = reps == r
        s = starts[here]
        if len(np.unique(np.round(s, 6))) > 1:
            delta = dict(zip(names[here], rng.permutation(s) - s))
            shift += np.fromiter((delta.get(m, 0.0) for m in movie), float, len(movie))
            dealt = True
            continue
        if "uid" not in steps.columns:
            raise InductionError("the movies of a replicate start together (the trigger clock) and there is no 'uid' "
                                 "column to deal their tracks' times by")
        in_rep = np.isin(movie, names[here])
        frame = pd.DataFrame({"movie": movie[in_rep], "uid": steps["uid"].to_numpy()[in_rep], "t": t[in_rep]})
        firsts = frame.groupby(["movie", "uid"])["t"].min().reset_index()
        new = np.empty(len(firsts))
        for _m, idx in firsts.groupby("movie").indices.items():
            new[idx] = rng.permutation(firsts["t"].to_numpy()[idx])
        firsts["delta"] = new - firsts["t"].to_numpy()
        merged = frame.merge(firsts[["movie", "uid", "delta"]], on=["movie", "uid"], how="left")
        shift[in_rep] += merged["delta"].to_numpy()
        dealt = dealt or len(firsts) > len(np.unique(movie[in_rep]))
    if not dealt:
        raise InductionError("nothing to shuffle: every replicate holds one movie, or one track a movie")
    out = steps.copy()
    out[CLOCK] = t + shift
    out.attrs.update(steps.attrs)
    return out


def permutation_seeds(n_shuffles: int, seed: int = 0) -> list:
    """The seeds of a test: ``None`` (the clock as observed) then one per shuffle. A shuffle's seed does not depend
    on how the test is split into jobs, so the answer does not either."""
    if int(n_shuffles) < 1:
        raise InductionError(f"{n_shuffles} shuffles: a permutation test needs at least one")
    return [None] + [int(seed) * 1_000_003 + i + 1 for i in range(int(n_shuffles))]


def changing_models(changes=CHANGES, shapes=SHAPES) -> list:
    """Every changing model (``change``, ``shape``) of a Run's ``changes`` x ``shapes``, "none" left out: what the
    exact permutation test scores each clock by the best of."""
    return [(c, sh) for c in changes if c != "none" for sh in shapes]


def permutation_gains(steps: pd.DataFrame, *, n_pop: int, models, tau_lag_s: float, D, fractions,
                      constant_ll: float, seeds, n_starts: int = PERM_STARTS,
                      replicate: str = UNIT_COLUMNS["replicate"]) -> list:
    """For each seed (``None``: the clock as observed) ``[gain, index]``: the largest log-likelihood any of ``models``
    (``(change, shape)`` pairs) gains over the constant mixture (``constant_ll``, with its ``D`` and ``fractions``,
    where every changing fit starts) on that clock, and which model gave it. A pool job: picklable, and the same seeds
    give the same gains wherever they run."""
    models = [tuple(m) for m in models]
    if not models or any(m[0] == "none" for m in models):
        raise InductionError("the permutation test refits changing models; 'none' has nothing to gain")
    start = _Start(tuple(D), tuple(fractions))
    out = []
    for seed in seeds:
        data = steps if seed is None else shuffle_clock(steps, np.random.default_rng(int(seed)), replicate)
        gains = [float(fit_model(data, n_pop=n_pop, change=c, shape=sh, tau_lag_s=tau_lag_s, n_starts=n_starts,
                                 seed=0, constant=start).log_likelihood - constant_ll) for c, sh in models]
        best = int(np.argmax(gains))
        out.append([gains[best], best])
    return out


def permutation_summary(seeds, gains, *, models, replicate_count: int = 0) -> dict:
    """The test from its seeds and their results (aligned; each ``[gain, model index]`` as `permutation_gains`
    gives, or a bare gain where one model was tested): ``p`` the rank of the observed statistic among the
    shuffles', counted in (``(1 + at least as large) / (1 + shuffles)``), the gains, and the model that gained most
    on the observed clock (``change``, ``shape``)."""
    models = [tuple(m) for m in models]
    seeds = list(seeds)
    pairs = [g if isinstance(g, (list, tuple)) else [g, 0] for g in gains]
    values = np.asarray([float(g[0]) for g in pairs], float)
    if seeds[0] is not None or len(seeds) != len(values) or len(values) < 2:
        raise InductionError("a permutation summary needs the observed clock's gain first, then the shuffles'")
    observed, null = float(values[0]), values[1:]
    change, shape = models[int(pairs[0][1])]
    p = float((1 + np.sum(null >= observed)) / (1 + len(null)))
    return {"p": p, "observed_gain": observed, "null_gains": [float(g) for g in null], "n_shuffles": int(len(null)),
            "change": change, "shape": shape, "n_models": len(models), "exact": len(models) > 1,
            "shown": bool(p <= 0.05), "smallest_p": 1.0 / (1 + len(null)), "n_replicates": int(replicate_count)}


def run_permutation_chunks(submit_chunk, seeds, *, chunk: int = PERM_CHUNK, progress=None, cancelled=None) -> list:
    """The gains of ``seeds`` (`permutation_seeds`), computed in jobs of ``chunk`` seeds: ``submit_chunk(seeds) ->
    Future`` of their gains (a pool job; or `_run_now`'s finished future to run here), in the order of ``seeds``.

    ``progress(done, total)`` is called once at the start and whenever more chunks have landed, ALWAYS FROM THIS
    THREAD. A set ``cancelled`` (a `threading.Event`) stops the submitting, cancels what has not started and raises
    `concurrent.futures.CancelledError`; a chunk already on a worker runs to its end (a few seconds). A chunk that
    raised raises here, the rest cancelled."""
    from concurrent.futures import CancelledError, FIRST_COMPLETED, wait

    parts = [list(seeds[i:i + chunk]) for i in range(0, len(seeds), chunk)]
    futures = []

    def stop() -> None:
        if cancelled is not None and cancelled.is_set():
            raise CancelledError("the permutation test was stopped")

    said = [-1]

    def say() -> None:
        done = sum(1 for f in futures if f.done() and not f.cancelled())
        if progress is not None and done != said[0]:
            said[0] = done
            progress(done, len(parts))

    try:
        say()
        for part in parts:
            stop()
            futures.append(submit_chunk(part))
            say()
        pending = set(futures)
        while pending:
            stop()
            landed, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            for f in landed:
                f.result()                                  # a failed chunk raises now, not at the end
            say()
        gains = [g for f in futures for g in f.result()]
    except BaseException:
        for f in futures:
            f.cancel()
        raise
    return gains


def _run_now(function, *args, **kwargs):
    """A finished future holding ``function(*args, **kwargs)``: the jobs run here, one after another."""
    from concurrent.futures import Future

    future = Future()
    try:
        future.set_result(function(*args, **kwargs))
    except BaseException as exc:                            # noqa: BLE001 - handed on, as a pool does
        future.set_exception(exc)
    return future


def permutation_test(steps: pd.DataFrame, *, n_pop: int, models=None, change: str | None = None,
                     shape: str | None = None, tau_lag_s: float | None = None, n_shuffles: int = PERMUTATIONS,
                     seed: int = 0, constant: InductionFit | None = None, submit=None, progress=None, cancelled=None,
                     chunk: int = PERM_CHUNK) -> dict:
    """The time-label permutation test (the section's comment above) against the constant mixture, on ``steps``
    (one condition, with ``day`` and ``movie``, ``uid`` on the trigger clock): EXACT over ``models`` (each clock
    scored by the best of them; None: every changing model, `changing_models`), or of ONE model named in advance
    (``change``, ``shape``). ``constant`` is the constant mixture's fit (made here when not given).

    ``submit(function, *args, **kwargs) -> Future`` runs each job of ``chunk`` shuffles on a pool; None runs them
    here. The numbers do not depend on it. REFUSES where fewer than `PERM_MIN_ARRANGEMENTS` clocks exist, so a test
    that could never reach 0.05 is not run and called a failure to find change."""
    if change == "none":
        raise InductionError("the permutation test refits a changing model; 'none' has nothing to gain")
    if models is None:
        models = [(change, shape)] if change is not None else changing_models()
    tau = _tau_lag(steps, tau_lag_s)
    n_arr = permutation_arrangements(steps)
    if n_arr < PERM_MIN_ARRANGEMENTS:
        raise InductionError(f"only {n_arr:.0f} different clocks can be dealt among the movies of these replicates: "
                             f"fewer than {PERM_MIN_ARRANGEMENTS}, so no permutation test can reach p = 0.05")
    constant = constant or fit_model(steps, n_pop=n_pop, change="none", tau_lag_s=tau, n_starts=8, seed=seed)
    submit = submit or _run_now
    seeds = permutation_seeds(n_shuffles, seed)

    def chunk_job(part):
        return submit(permutation_gains, steps, n_pop=n_pop, models=models, tau_lag_s=tau,
                      D=constant.D_before, fractions=constant.fractions_before,
                      constant_ll=constant.log_likelihood, seeds=part)

    gains = run_permutation_chunks(chunk_job, seeds, chunk=chunk, progress=progress, cancelled=cancelled)
    out = permutation_summary(seeds, gains, models=models,
                              replicate_count=int(steps[UNIT_COLUMNS["replicate"]].nunique()))
    out["arrangements"] = n_arr
    return out
