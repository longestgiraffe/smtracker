"""Synthetic tracks: the null model every pattern detector must be judged against.

Why this module exists
----------------------
Several classical single-molecule analyses *always* return a pattern, whether or
not one is there:

- a confinement detector run on **free Brownian** motion still finds "dwell
  events", because a random walk revisits its own neighbourhood by chance;
- k-means on per-track MSDs always returns clusters, because the per-track MSD
  of a *single* population is broad and unimodal, not point-like.

The only honest way to read such a number is against what the *same detector*
returns on data with the effect deliberately absent. This module generates that
comparison data: free Brownian tracks matched to the real data in track-length
distribution, D, τ and localization noise. One source of truth, shared by
``smtracker.confinement`` and ``smtracker.clustering``.

What is deliberately NOT simulated
----------------------------------
Photobleaching, defocalization, and the tracker's own linking failures. The null
therefore reproduces the *length distribution* of the real data (by resampling
it) rather than generating it from a photophysics model: matching the observed
lengths is what matters for a length-sensitive detector, and inventing a
bleaching model would smuggle in unmeasured constants (see the dZ note in
CLAUDE.md).

Invariants
----------
- ``D``, ``tau_s`` > 0; ``sigma_um`` >= 0.
- Output is a canonical table: µm, seconds, ``CANONICAL_COLUMNS``, with
  ``attrs["params"]`` set, so every downstream module accepts it unchanged.
- Localization noise is added to *positions*, so it inflates apparent D exactly
  as it does in real data (r² gains 4σ²).
- Fails loud on empty or non-positive lengths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .canonical import CANONICAL_COLUMNS, AcquisitionParams

__all__ = [
    "SimulationError",
    "brownian_tracks",
    "track_lengths",
    "matched_brownian_null",
    "mixture_tracks",
    "matched_mixture_null",
    "StateSpec",
    "CellSpec",
    "ImagingSpec",
    "ConditionSpec",
    "simulate_condition",
    "simulate_experiment",
    "rates_for_occupancy",
    "tracks_for_visible",
]


class SimulationError(RuntimeError):
    """Raised when a simulation is asked for something impossible."""


def track_lengths(canonical: pd.DataFrame) -> np.ndarray:
    """Number of localizations per track, in the order the tracks appear."""
    if "uid" not in canonical.columns:
        raise SimulationError("not a canonical table (no 'uid')")
    return canonical.groupby("uid", sort=False, observed=True).size().to_numpy()


def brownian_tracks(
    lengths,
    *,
    D_um2_s: float,
    params: AcquisitionParams,
    sigma_um: float = 0.0,
    field_um: float = 10.0,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Free 2-D Brownian tracks with the given per-track lengths.

    Each step is drawn from N(0, 2·D·τ) per axis, so ⟨r²⟩ = 4·D·τ + 4·σ².
    Start positions are scattered over ``field_um`` so that tracks do not
    overlap in a way that would confuse a spatial method; the field is
    otherwise unbounded: this is *free* diffusion, which is the point.
    """
    lengths = np.asarray(lengths, dtype=int)
    if lengths.size == 0:
        raise SimulationError("no track lengths given")
    if np.any(lengths < 2):
        raise SimulationError("every track needs at least 2 localizations")
    if D_um2_s <= 0:
        raise SimulationError(f"D must be > 0, got {D_um2_s}")
    if sigma_um < 0:
        raise SimulationError(f"sigma_um must be >= 0, got {sigma_um}")

    rng = np.random.default_rng(seed)
    step_sd = np.full(len(lengths), np.sqrt(2.0 * D_um2_s * params.tau_s))
    out = _walk_tracks(lengths, step_sd, params, sigma_um, field_um, rng)
    out.attrs["true_D"] = float(D_um2_s)
    out.attrs["true_sigma_um"] = float(sigma_um)
    return out


def _walk_tracks(lengths, step_sd, params: AcquisitionParams, sigma_um: float, field_um: float, rng) -> pd.DataFrame:
    """The one random-walk builder: track ``i`` has ``lengths[i]`` localizations and a per-axis step SD
    ``step_sd[i]`` (µm). Draws, per track and in this order, the origin, the steps and the noise, so a single-D call
    is bit-identical to what `brownian_tracks` always produced."""
    tau = params.tau_s
    frames, xs, ys, uids = [], [], [], []
    for i, n in enumerate(lengths):
        origin = rng.uniform(0.0, field_um, size=2)
        steps = rng.normal(0.0, step_sd[i], size=(n - 1, 2))
        path = np.vstack([np.zeros(2), np.cumsum(steps, axis=0)]) + origin
        if sigma_um > 0:
            path = path + rng.normal(0.0, sigma_um, size=path.shape)
        frames.append(np.arange(1, n + 1))
        xs.append(path[:, 0])
        ys.append(path[:, 1])
        uids.append(np.full(n, f"sim:{i}", dtype=object))

    frame = np.concatenate(frames)
    out = pd.DataFrame(
        {
            "protein": "sim",
            "condition": "null",
            "day": 1,
            "movie": "sim",
            "cell": "sim:1",
            "uid": np.concatenate(uids),
            "track_id": np.concatenate(uids),
            "frame": frame,
            "t_s": frame * tau,
            "x_um": np.concatenate(xs),
            "y_um": np.concatenate(ys),
            "amp": 1.0,
        }
    )
    out = out[CANONICAL_COLUMNS]
    out.attrs["params"] = params
    out.attrs["simulated"] = True
    return out


def mixture_tracks(
    lengths,
    *,
    D,
    fractions,
    params: AcquisitionParams,
    sigma_um: float = 0.0,
    field_um: float = 10.0,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Brownian tracks drawn from an SQD MIXTURE: each track takes one population (probability ``fractions``) and keeps
    its D for its whole length (no switching). ``D`` may hold 0 (an immobile population: the tracks then move by
    localization noise alone). Each coordinate gains N(0, sigma_um), so r² gains 4σ² as in `brownian_tracks`.

    This is what an SQD fit CLAIMS the data is, simulated: a null for anything that asks whether the data holds more
    than its fitted populations (`clustering`). The fractions are SQD's, a share of STEPS; a track here takes a
    population independently of its length, so they are also the expected share of tracks."""
    lengths = np.asarray(lengths, dtype=int)
    D = np.asarray(D, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    if lengths.size == 0:
        raise SimulationError("no track lengths given")
    if np.any(lengths < 2):
        raise SimulationError("every track needs at least 2 localizations")
    if D.ndim != 1 or D.size == 0 or D.size != fractions.size:
        raise SimulationError("D and fractions must be equal-length, non-empty 1-D lists")
    if np.any(D < 0):
        raise SimulationError(f"D must be >= 0, got {D.tolist()}")
    if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0, atol=1e-6):
        raise SimulationError(f"fractions must be >= 0 and sum to 1, got {fractions.tolist()}")
    if sigma_um < 0:
        raise SimulationError(f"sigma_um must be >= 0, got {sigma_um}")
    rng = np.random.default_rng(seed)
    population = rng.choice(D.size, size=len(lengths), p=fractions / fractions.sum())
    step_sd = np.sqrt(2.0 * D[population] * params.tau_s)
    out = _walk_tracks(lengths, step_sd, params, sigma_um, field_um, rng)
    out.attrs["true_D"] = D.tolist()
    out.attrs["true_fractions"] = fractions.tolist()
    out.attrs["true_sigma_um"] = float(sigma_um)
    return out


def matched_mixture_null(
    canonical: pd.DataFrame,
    *,
    D,
    fractions,
    sigma_um: float = 0.0,
    n_tracks: int | None = None,
    seed: int | None = 0,
) -> pd.DataFrame:
    """`mixture_tracks` with ``canonical``'s track-length distribution (resampled with replacement, as
    `matched_brownian_null` does)."""
    params = canonical.attrs.get("params")
    if not isinstance(params, AcquisitionParams):
        raise SimulationError("canonical table carries no AcquisitionParams")
    lengths = track_lengths(canonical)
    rng = np.random.default_rng(seed)
    n = len(lengths) if n_tracks is None else int(n_tracks)
    if n < 1:
        raise SimulationError(f"n_tracks must be >= 1, got {n}")
    drawn = rng.choice(lengths, size=n, replace=True)
    return mixture_tracks(drawn, D=D, fractions=fractions, params=params, sigma_um=sigma_um, seed=seed)


def matched_brownian_null(
    canonical: pd.DataFrame,
    *,
    D_um2_s: float,
    sigma_um: float = 0.0,
    n_tracks: int | None = None,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Free Brownian tracks matched to ``canonical``'s track-length distribution.

    Lengths are **resampled with replacement** from the real data rather than
    fitted, so the null inherits the real length distribution (including its
    tail) without a photobleaching model.
    """
    params = canonical.attrs.get("params")
    if not isinstance(params, AcquisitionParams):
        raise SimulationError("canonical table carries no AcquisitionParams")
    lengths = track_lengths(canonical)
    rng = np.random.default_rng(seed)
    n = len(lengths) if n_tracks is None else int(n_tracks)
    if n < 1:
        raise SimulationError(f"n_tracks must be >= 1, got {n}")
    drawn = rng.choice(lengths, size=n, replace=True)
    return brownian_tracks(
        drawn, D_um2_s=D_um2_s, params=params, sigma_um=sigma_um, seed=seed
    )


# =========================================================================== #
# The ground-truth FORWARD MODEL
# =========================================================================== #
#
# Everything above generates null models. Everything below is the general
# forward simulator: it emits observed localizations the way a microscope would,
# carrying the TRUE per-step state alongside, so every estimator can be
# validated by recovery-of-truth rather than by agreement with another
# estimator.
#
# INDEPENDENCE IS THE POINT. Nothing here imports or shares code with any
# fitter's likelihood. Kinetics are specified as physical **rates** (s^-1) and
# converted to a per-frame transition matrix by matrix exponential; the HMM
# independently estimates a per-frame matrix. If the two shared machinery,
# recovery would be circular and would prove nothing. `brownian_tracks` above is
# the single-state, no-noise preset of this model and is regression-tested to be
# unchanged.


@dataclass(frozen=True)
class StateSpec:
    """One diffusive state.

    ``confinement_radius_um`` makes the state confined: a step that would leave a
    disc of that radius around the molecule's anchor is reflected back inside.
    That is a *geometric* confinement, deliberately unlike the Brownian
    likelihood any estimator fits, so "confined" is a genuinely different truth
    to recover.

    ``membrane_affinity`` in [0, 1] biases where molecules in this state sit
    across the cell: 0 uniform, 1 hugging the membrane. It shapes *position*
    only, never the step-size distribution, so a spatial estimator can be tested
    without contaminating a kinetic one.
    """

    D_um2_s: float
    confinement_radius_um: float | None = None
    membrane_affinity: float = 0.0

    def __post_init__(self) -> None:
        if self.D_um2_s < 0:
            raise SimulationError(f"D must be >= 0, got {self.D_um2_s}")
        if self.confinement_radius_um is not None and self.confinement_radius_um <= 0:
            raise SimulationError("confinement_radius_um must be > 0 when given")
        if not 0.0 <= self.membrane_affinity <= 1.0:
            raise SimulationError(
                f"membrane_affinity must be in [0, 1], got {self.membrane_affinity}"
            )


@dataclass(frozen=True)
class CellSpec:
    """A rod-shaped cell. Molecules are reflected at its boundary."""

    length_um: float = 3.0
    width_um: float = 1.0

    def __post_init__(self) -> None:
        if self.length_um <= 0 or self.width_um <= 0:
            raise SimulationError("cell dimensions must be > 0")

    @property
    def half_width(self) -> float:
        return self.width_um / 2.0


@dataclass(frozen=True)
class ImagingSpec:
    """How the microscope turns motion into localizations.

    Each knob is independent, so a test can turn exactly one on:

    - ``sigma_um`` -- localization error added to the observed position.
    - ``n_subframes`` -- motion blur. The molecule is propagated in this many
      sub-steps per frame and the *average* position over the exposure is
      reported, which is what a camera integrating over tau actually records.
      1 disables blur.
    - ``bleach_lifetime_frames`` -- photobleaching, giving the realistic
      exponential track-length distribution instead of a fixed length.
    - ``miss_probability`` -- per-frame detection failure, producing gaps.
    - ``dZ_um`` -- axial detection slab. The molecule also diffuses in z and is
      undetected while |z| exceeds half the slab, so defocalization removes
      *fast* molecules preferentially, exactly as it does in reality.
    """

    params: AcquisitionParams
    sigma_um: float = 0.0
    n_subframes: int = 1
    bleach_lifetime_frames: float | None = None
    miss_probability: float = 0.0
    dZ_um: float | None = None
    max_frames: int = 200

    def __post_init__(self) -> None:
        if self.sigma_um < 0:
            raise SimulationError(f"sigma_um must be >= 0, got {self.sigma_um}")
        if self.n_subframes < 1:
            raise SimulationError(f"n_subframes must be >= 1, got {self.n_subframes}")
        if not 0.0 <= self.miss_probability < 1.0:
            raise SimulationError("miss_probability must be in [0, 1)")
        if self.bleach_lifetime_frames is not None and self.bleach_lifetime_frames <= 0:
            raise SimulationError("bleach_lifetime_frames must be > 0 when given")
        if self.dZ_um is not None and self.dZ_um <= 0:
            raise SimulationError("dZ_um must be > 0 when given")
        if self.max_frames < 2:
            raise SimulationError("max_frames must be >= 2")


@dataclass(frozen=True)
class ConditionSpec:
    """A condition: its states, its switching rates, and its name.

    ``rates_per_s[i, j]`` is the true rate of going from state ``i`` to state
    ``j`` (off-diagonal only; the diagonal is ignored and recomputed). Rates are
    physical, in s^-1 -- the per-frame matrix an HMM estimates is derived from
    them here by matrix exponential, so the two never share machinery.
    """

    name: str
    states: tuple
    rates_per_s: object = None
    initial_occupancy: tuple | None = None

    def __post_init__(self) -> None:
        if not self.states:
            raise SimulationError("a condition needs at least one state")
        k = len(self.states)
        if self.rates_per_s is not None:
            rates = np.asarray(self.rates_per_s, dtype=float)
            if rates.shape != (k, k):
                raise SimulationError(f"rates_per_s must be {k}x{k}, got {rates.shape}")
            if np.any(rates < 0):
                raise SimulationError("rates must be >= 0")
        elif k > 1:
            raise SimulationError("a multi-state condition needs rates_per_s")

    @property
    def n_states(self) -> int:
        return len(self.states)

    def generator(self) -> np.ndarray:
        """The continuous-time generator Q (s^-1)."""
        k = self.n_states
        if k == 1:
            return np.zeros((1, 1))
        Q = np.asarray(self.rates_per_s, dtype=float).copy()
        np.fill_diagonal(Q, 0.0)
        np.fill_diagonal(Q, -Q.sum(axis=1))
        return Q

    def transition_matrix(self, dt_s: float) -> np.ndarray:
        """Per-step transition matrix ``expm(Q*dt)`` -- exact, not first-order.

        A first-order approximation ``I + Q*dt`` would break at high rates and
        silently bias any recovery test toward the estimator's own assumptions.
        """
        from scipy.linalg import expm

        A = expm(self.generator() * float(dt_s))
        A = np.clip(A, 0.0, None)
        return A / A.sum(axis=1, keepdims=True)

    def stationary(self) -> np.ndarray:
        """Stationary occupancy of the true chain -- the fraction to recover.

        Solved from the generator directly: π satisfies ``π·Q = 0`` with
        ``Σπ = 1``. Taking an eigenvector of the transition matrix and dividing
        by its sum fails when that eigenvector happens to sum to ~0 (it is only
        defined up to sign and scale), so the normalisation constraint is
        imposed explicitly instead.
        """
        if self.n_states == 1:
            return np.ones(1)
        k = self.n_states
        # [Q^T ; 1...1] · π = [0...0 ; 1] , least-squares for robustness.
        A = np.vstack([self.generator().T, np.ones(k)])
        b = np.append(np.zeros(k), 1.0)
        pi, *_ = np.linalg.lstsq(A, b, rcond=None)
        pi = np.clip(pi, 0.0, None)
        total = pi.sum()
        if total <= 0:
            raise SimulationError("stationary distribution is degenerate")
        return pi / total



def rates_for_occupancy(occupancy, exchange_per_s: float) -> np.ndarray:
    """A rate matrix whose STATIONARY occupancy is the one you asked for.

    WRITTEN BECAUSE THE OBVIOUS MATRIX IS WRONG, twice over in this project's own use.
    Filling the off-diagonal with a single rate -- the natural way to say "they exchange
    at k per second" -- gives every state the same total inflow, so the chain converges to
    a UNIFORM occupancy no matter what `initial_occupancy` says. A planted 35/65 then
    drifts to 50/50 and the recovered fraction reads 15 points wrong, which looks exactly
    like an estimator bias and is not one.

    Rate i -> j is ``exchange_per_s * occupancy[j]``, which satisfies detailed balance
    against ``occupancy`` and therefore has it as the stationary distribution. The
    per-state exit rate is then ``exchange_per_s * (1 - occupancy[i])``, so a rare state
    is left quickly and a common one is left slowly -- which is what makes the occupancy
    hold.

    ``exchange_per_s`` sets how FAST the exchange is, not where it lands; use it to move
    between the frozen limit (0) and rapid switching without touching the fractions.
    """
    f = np.asarray(occupancy, dtype=float)
    if f.ndim != 1 or len(f) < 1:
        raise SimulationError("occupancy must be a one-dimensional vector")
    if np.any(f < 0) or f.sum() <= 0:
        raise SimulationError("occupancy must be non-negative and sum above zero")
    f = f / f.sum()
    if exchange_per_s < 0:
        raise SimulationError(f"exchange_per_s must be >= 0, got {exchange_per_s}")
    k = len(f)
    rates = float(exchange_per_s) * np.tile(f, (k, 1))
    np.fill_diagonal(rates, 0.0)
    return rates


def tracks_for_visible(*, molecules_visible: float, n_frames: int,
                       mean_lifetime_frames: float) -> int:
    """How many molecules to simulate so that ``molecules_visible`` are on at once.

    THE NUMBER THAT CONTROLS MERGING AND MISLINKING IS THE SIMULTANEOUS ONE, and it is
    not a parameter anywhere: `simulate_condition` takes `tracks_per_cell`, which is the
    TOTAL over the movie. In a real acquisition molecules bleach and new ones appear, so
    a thousand molecules can pass through a cell while only one is ever lit.

    At steady state, visible = total * lifetime / frames, so the total needed is
    ``visible * frames / lifetime``. Feed the result to `simulate_condition` and then
    spread the starts with `moviesim.stagger_starts`, which reports the density it
    actually achieved -- the two together are the realistic regime, and neither alone is.
    """
    if molecules_visible <= 0:
        raise SimulationError("molecules_visible must be > 0")
    if n_frames < 1:
        raise SimulationError("n_frames must be >= 1")
    if mean_lifetime_frames <= 0:
        raise SimulationError("mean_lifetime_frames must be > 0")
    return max(int(round(molecules_visible * n_frames / mean_lifetime_frames)), 1)


def _sample_position(rng, cell: "CellSpec", state: "StateSpec") -> np.ndarray:
    """Start position honouring the state's spatial rule."""
    x = rng.uniform(0.0, cell.length_um)
    if state.membrane_affinity <= 0:
        y = rng.uniform(-cell.half_width, cell.half_width)
    else:
        shape = max(1e-3, 1.0 - state.membrane_affinity)
        magnitude = rng.beta(1.0, shape) if shape < 1.0 else rng.uniform(0.0, 1.0)
        y = cell.half_width * magnitude * rng.choice([-1.0, 1.0])
    return np.array([x, y], dtype=float)


def _reflect(position: np.ndarray, cell: "CellSpec") -> np.ndarray:
    """Keep a molecule inside the rod by reflecting at the boundary."""
    x, y = position
    length, half = cell.length_um, cell.half_width
    if length > 0:
        x = abs(x) % (2 * length)
        x = x if x <= length else 2 * length - x
    if half > 0:
        y = abs(y + half) % (4 * half)
        y = y if y <= 2 * half else 4 * half - y
        y -= half
    return np.array([x, y], dtype=float)


def simulate_condition(
    condition: "ConditionSpec",
    imaging: "ImagingSpec",
    *,
    n_cells: int = 10,
    tracks_per_cell: int = 20,
    cell: "CellSpec" = None,
    between_cell_sd: float = 0.0,
    day: int = 1,
    movie: str | None = None,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Simulate one condition, returning a raw table plus truth columns.

    ``between_cell_sd`` perturbs each cell's *initial* state occupancy on the
    logit scale, giving the real cell-to-cell spread that makes a cell-level
    bootstrap meaningful. It does not touch the rates, so the truth stays well
    defined. Extra columns: ``true_state`` (1-based per localization) and
    ``true_D``; estimators read only the canonical columns.
    """
    if cell is None:
        cell = CellSpec()
    if n_cells < 1 or tracks_per_cell < 1:
        raise SimulationError("n_cells and tracks_per_cell must be >= 1")
    if between_cell_sd < 0:
        raise SimulationError("between_cell_sd must be >= 0")

    rng = np.random.default_rng(seed)
    tau = imaging.params.tau_s
    dt = tau / imaging.n_subframes
    A_sub = condition.transition_matrix(dt)
    base_occupancy = (
        np.asarray(condition.initial_occupancy, dtype=float)
        if condition.initial_occupancy is not None
        else condition.stationary()
    )
    base_occupancy = base_occupancy / base_occupancy.sum()
    survival = (
        np.exp(-1.0 / imaging.bleach_lifetime_frames)
        if imaging.bleach_lifetime_frames else 1.0
    )

    rows = []
    for cell_index in range(n_cells):
        occupancy = base_occupancy
        if between_cell_sd > 0 and condition.n_states > 1:
            logit = np.log(np.clip(base_occupancy, 1e-9, 1 - 1e-9))
            logit = logit + rng.normal(0.0, between_cell_sd, condition.n_states)
            occupancy = np.exp(logit)
            occupancy = occupancy / occupancy.sum()

        movie_name = movie or f"{condition.name}_movie1"
        cell_id = f"{movie_name}:{cell_index + 1}"

        for track_index in range(tracks_per_cell):
            state = int(rng.choice(condition.n_states, p=occupancy))
            position = _sample_position(rng, cell, condition.states[state])
            anchor = position.copy()
            z = 0.0 if imaging.dZ_um is None else rng.uniform(
                -imaging.dZ_um / 2, imaging.dZ_um / 2
            )
            track_id = f"{cell_index + 1}.{track_index + 1}"

            for frame in range(1, imaging.max_frames + 1):
                visited = []
                for _ in range(imaging.n_subframes):
                    state = int(rng.choice(condition.n_states, p=A_sub[state]))
                    spec = condition.states[state]
                    step_sd = np.sqrt(2.0 * spec.D_um2_s * dt)
                    proposal = position + rng.normal(0.0, step_sd, 2)
                    if spec.confinement_radius_um is not None:
                        offset = proposal - anchor
                        distance = float(np.hypot(*offset))
                        if distance > spec.confinement_radius_um:
                            proposal = anchor + offset * (
                                (2 * spec.confinement_radius_um - distance) / distance
                            )
                    else:
                        anchor = proposal
                    position = _reflect(proposal, cell)
                    if imaging.dZ_um is not None:
                        z += rng.normal(0.0, step_sd)
                    visited.append(position.copy())

                # A camera integrates over the exposure: the reported position is
                # the average of where the molecule was -- that is blur.
                observed = np.mean(visited, axis=0)

                detected = True
                if imaging.dZ_um is not None and abs(z) > imaging.dZ_um / 2:
                    detected = False
                if imaging.miss_probability and rng.random() < imaging.miss_probability:
                    detected = False

                if detected:
                    noisy = observed + rng.normal(0.0, imaging.sigma_um, 2)
                    rows.append({
                        "protein": "sim",
                        "condition": condition.name,
                        "day": day,
                        "movie": movie_name,
                        "cell": cell_id,
                        "track_id": track_id,
                        "frame": frame,
                        "x": noisy[0] / imaging.params.um_per_pixel,
                        "y": noisy[1] / imaging.params.um_per_pixel,
                        "amp": 1.0,
                        "true_state": state + 1,
                        "true_D": condition.states[state].D_um2_s,
                    })

                if survival < 1.0 and rng.random() > survival:
                    break

    if not rows:
        raise SimulationError(
            f"condition {condition.name!r} produced no localizations; check "
            "bleaching, dZ and miss_probability"
        )
    out = pd.DataFrame(rows)
    # THE OCCUPANCY THE CHAIN ACTUALLY HOLDS, beside the one that was asked for. They
    # differ whenever the rate matrix was not built for the occupancy -- a uniform
    # off-diagonal drives any chain to uniform, whatever `initial_occupancy` says -- and
    # the difference reads exactly like an estimator bias. `rates_for_occupancy` builds a
    # matrix that agrees; this states it either way rather than leaving it to be
    # discovered in a recovered fraction.
    stationary = condition.stationary()
    asked = (np.asarray(condition.initial_occupancy, float)
             if condition.initial_occupancy is not None else stationary)
    asked = asked / asked.sum()
    drift = float(np.abs(stationary - asked).max())
    out.attrs.update({
        "initial_occupancy": asked.tolist(),
        "stationary_occupancy": stationary.tolist(),
        "occupancy_drift_pp": 100.0 * drift,
        "occupancy_note": (
            "initial and stationary occupancy agree" if drift < 0.005 else
            f"the chain converges to {np.round(100 * stationary, 1).tolist()} %, not the "
            f"{np.round(100 * asked, 1).tolist()} % asked for -- tracks long against "
            f"1/rate will show the stationary value. Use rates_for_occupancy() to make "
            f"them agree, or keep the difference if you are modelling a relaxation."),
    })
    return out


def simulate_experiment(
    conditions,
    imaging: "ImagingSpec",
    *,
    n_cells: int = 10,
    tracks_per_cell: int = 20,
    cell: "CellSpec" = None,
    between_cell_sd: float = 0.0,
    min_localizations: int = 5,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Simulate several conditions and return ONE canonical table with truth.

    Canonical (um, seconds, filtered), plus ``true_state`` / ``true_D`` columns
    and a ground-truth manifest in ``.attrs["truth"]`` -- the per-condition
    rates, stationary occupancy and D's an estimator should recover.
    """
    if cell is None:
        cell = CellSpec()
    conditions = list(conditions)
    if not conditions:
        raise SimulationError("no conditions to simulate")

    frames, truth = [], {}
    for index, condition in enumerate(conditions):
        frames.append(
            simulate_condition(
                condition, imaging, n_cells=n_cells, tracks_per_cell=tracks_per_cell,
                cell=cell, between_cell_sd=between_cell_sd,
                seed=None if seed is None else seed + 1000 * index,
            )
        )
        truth[condition.name] = {
            "D": [s.D_um2_s for s in condition.states],
            "occupancy": condition.stationary().tolist(),
            "rates_per_s": (
                None if condition.rates_per_s is None
                else np.asarray(condition.rates_per_s, dtype=float).tolist()
            ),
            "n_states": condition.n_states,
            "confinement_radius_um": [s.confinement_radius_um for s in condition.states],
            "membrane_affinity": [s.membrane_affinity for s in condition.states],
        }

    raw = pd.concat(frames, ignore_index=True)
    truth_columns = raw[["track_id", "movie", "frame", "true_state", "true_D"]].copy()

    from .canonical import to_canonical

    canonical = to_canonical(
        raw.drop(columns=["true_state", "true_D"]),
        imaging.params,
        min_localizations=min_localizations,
    )
    # Re-attach the truth per (movie, track_id, frame) so it survives filtering.
    canonical = canonical.merge(
        truth_columns, on=["movie", "track_id", "frame"], how="left"
    )
    canonical.attrs["params"] = imaging.params
    canonical.attrs["truth"] = truth
    canonical.attrs["imaging"] = {
        "sigma_um": imaging.sigma_um,
        "n_subframes": imaging.n_subframes,
        "bleach_lifetime_frames": imaging.bleach_lifetime_frames,
        "miss_probability": imaging.miss_probability,
        "dZ_um": imaging.dZ_um,
        "max_frames": imaging.max_frames,
        "tau_s": imaging.params.tau_s,
    }
    canonical.attrs["simulated"] = True
    return canonical
