"""Spot-On: multi-lag jump-length fit with a defocalization correction.

The field-standard kinetic-modelling method (Hansen et al. 2018), implemented
natively. Two things distinguish it from the estimators already here:

1. **Several lag times fitted jointly.** SQD uses one lag; Spot-On fits
   Δt = 1, 2, 3… frames together. The *lag dependence* is itself information,
   a bound fraction that changes with Δt is the fingerprint of axial loss.
2. **Defocalization correction.** Fast molecules diffuse out of the focal slab
   and stop being observed, so a naive fit over-counts the slow ones and
   increasingly so at longer Δt. Spot-On corrects for it explicitly, given the
   axial detection depth ``dZ``.

That correction addresses, in the axial direction, the same survivorship bias
this project has already met twice: the MSD tail flattening, and long tracks
being preferentially slow (``observables.displacement_weights``). Here it is
modelled rather than reweighted.

Model
-----
For a 2-D state ``k`` the jump length ``r`` at lag ``Δt`` is Rayleigh with

    p_k(r | Δt) = r / (2·s_k²) · exp( −r² / (2·s_k²) ),
    s_k² = 2·(D_k·Δt + σ_loc²)

and the states are mixed with **lag-dependent** weights, because a fast state
loses more molecules from focus by time ``Δt``:

    w_k(Δt) = f_k · Z_k(Δt) / Σ_j f_j · Z_j(Δt)

``f_k`` is therefore the *corrected* fraction (what you would have measured
with an infinitely deep focus) which is the number Spot-On exists to produce.

The Z correction
----------------
``Z_k(Δt)`` is the probability a molecule of state ``k``, starting uniformly in
a slab of thickness ``dZ``, is still inside it after ``Δt``. Using absorbing
boundaries this has the classical series solution

    Z(D, t) = Σ_{n odd} 8/(n²π²) · exp( −n²π²·D·t / dZ² )

**Divergence from the published implementation, stated plainly:** Spot-On adds
an empirical term for molecules that leave and *re-enter* the slab. Absorbing
boundaries forbid re-entry, so this ``Z`` under-estimates retention, and the
correction here is therefore conservative: it removes somewhat less of the
fast-state loss than Spot-On's own does. ``dZ`` is an input and is not fitted;
vary it and check the conclusions hold.

Honest limits
-------------
- ``dZ`` must come from the optics. A wrong ``dZ`` moves the corrected fraction,
  so ``fit_spoton`` refuses to guess one.
- Like SQD, appD and saSPT, Spot-On assumes each molecule holds one state: it
  is **switching-blind**. On data with switching at ~6–8 s⁻¹ read it as a
  defocalization-corrected static fit, not a corrected-for-everything answer.
- Fitting is by maximum likelihood on the raw jump lengths rather than on binned
  histograms as the original does, which avoids an arbitrary bin width. The
  model is identical; only the estimator differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .canonical import CANONICAL_COLUMNS
from .sqd import _pack, _unpack

__all__ = [
    "SpotOnError",
    "SpotOnFit",
    "z_correction",
    "jump_lengths",
    "fit_spoton",
    "dz_sensitivity",
]

#: Odd terms used in the slab-survival series; it converges fast.
_Z_TERMS = 25
_DEFAULT_LAGS = (1, 2, 3)
_DEFAULT_STARTS = 30
_PENALTY = 1e12


class SpotOnError(RuntimeError):
    """Raised when a Spot-On fit cannot be performed."""


@dataclass
class SpotOnFit:
    """A defocalization-corrected multi-lag fit. States sorted by ascending D."""

    n_states: int
    D: np.ndarray                       # µm²/s, ascending
    fractions: np.ndarray               # CORRECTED for axial loss
    lags: tuple[int, ...]
    lag_times_s: np.ndarray
    dZ_um: float
    localization_sigma_um: float
    n_jumps: int
    log_likelihood: float
    bic: float
    z_corrections: np.ndarray = field(default_factory=lambda: np.array([]))
    n_starts: int = 1
    n_converged_starts: int = 0

    @property
    def fractions_percent(self) -> np.ndarray:
        return self.fractions * 100.0

    def apparent_fractions(self, lag: int) -> np.ndarray:
        """What an *uncorrected* fit at this lag would have reported.

        The gap between this and ``fractions`` is exactly what the
        defocalization correction is worth on this data.
        """
        if lag not in self.lags:
            raise SpotOnError(f"lag {lag} was not fitted; have {self.lags}")
        index = self.lags.index(lag)
        weights = self.fractions * self.z_corrections[index]
        return weights / weights.sum()

    def summary(self) -> str:
        parts = [
            f"state{i + 1}: {f * 100:.1f} %, D = {d:.4g} µm²/s"
            for i, (f, d) in enumerate(zip(self.fractions, self.D))
        ]
        return (
            f"Spot-On {self.n_states}-state fit (lags {list(self.lags)}, "
            f"{self.n_jumps} jumps, dZ = {self.dZ_um:g} µm): "
            + "; ".join(parts)
            + " | fractions are defocalization-CORRECTED"
        )

    # ``NPopSelection`` compatibility.
    @property
    def n_pop(self) -> int:
        return self.n_states


def z_correction(D, lag_time_s, dZ_um: float) -> np.ndarray:
    """Fraction of molecules still inside the focal slab after ``lag_time_s``.

    Absorbing-slab survival: ``Σ_{n odd} 8/(n²π²) exp(−n²π²Dt/dZ²)``. Returns 1
    for an immobile state and falls toward 0 for a fast one, which is why a
    naive fit over-counts the slow state.
    """
    D = np.atleast_1d(np.asarray(D, dtype=float))
    if dZ_um <= 0:
        raise SpotOnError(f"dZ_um must be > 0, got {dZ_um}")
    if lag_time_s < 0:
        raise SpotOnError(f"lag_time_s must be >= 0, got {lag_time_s}")
    if np.any(D < 0):
        raise SpotOnError("D must be >= 0")

    n = np.arange(1, 2 * _Z_TERMS, 2)          # odd terms only
    coefficient = 8.0 / (n**2 * np.pi**2)
    exponent = -(n[None, :] ** 2) * np.pi**2 * D[:, None] * lag_time_s / dZ_um**2
    series = (coefficient[None, :] * np.exp(exponent)).sum(axis=1)
    # The exact coefficient sum is 1 (Sum_{n odd} 1/n^2 = pi^2/8), but the
    # truncated one is not, which would make even an IMMOBILE state appear to
    # escape. Renormalising by the truncated sum enforces Z(D=0) = 1 exactly and
    # improves accuracy at short lag times, where the tail terms matter most.
    return np.clip(series / coefficient.sum(), 0.0, 1.0)


def jump_lengths(
    canonical: pd.DataFrame, *, lags=_DEFAULT_LAGS, pairing: str = "frame"
) -> dict[int, np.ndarray]:
    """Jump lengths (µm, not squared) per lag, built on the shared observable."""
    from .observables import ObservableError, squared_displacements

    lags = tuple(int(lag) for lag in lags)
    if not lags or any(lag < 1 for lag in lags):
        raise SpotOnError(f"lags must be positive integers, got {lags}")

    out: dict[int, np.ndarray] = {}
    for lag in lags:
        try:
            displacements = squared_displacements(
                canonical, lag_frames=lag, pairing=pairing
            )
        except ObservableError as exc:
            raise SpotOnError(f"no jumps at lag {lag}: {exc}") from exc
        out[lag] = np.sqrt(displacements["sq_disp_um2"].to_numpy(dtype=float))
    return out


def fit_spoton(
    canonical: pd.DataFrame,
    *,
    n_states: int = 2,
    dZ_um: float,
    localization_sigma_um: float = 0.035,
    lags=_DEFAULT_LAGS,
    n_starts: int = _DEFAULT_STARTS,
    seed: int | None = 0,
    D_bounds: tuple[float, float] = (1e-4, 10.0),
) -> SpotOnFit:
    """Fit the Spot-On model jointly across lags, correcting for axial loss.

    Parameters
    ----------
    canonical
        Canonical table (µm, seconds) with ``AcquisitionParams`` in ``.attrs``.
    n_states
        Number of diffusive states (2 or 3 in normal use).
    dZ_um
        **Required.** Axial detection depth in µm, the thickness of the slab in
        which a molecule is detectable. It comes from the optics, not from the
        data, and there is no sensible default: it directly scales how much of
        the fast state is deemed lost.
    localization_sigma_um
        Localization precision. Enters as ``s² = 2(D·Δt + σ²)``. Note this
        project treats σ as uncertain in the 30–76 nm band, so vary it.
    lags
        Frame lags to fit jointly (default 1, 2, 3).

    Raises
    ------
    SpotOnError
        On bad input or if no start converged. It never returns zeros.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise SpotOnError(f"not a canonical table (missing {missing})")
    params = canonical.attrs.get("params")
    if params is None:
        raise SpotOnError(
            "canonical table carries no AcquisitionParams in .attrs['params']; τ "
            "is required."
        )
    if n_states < 1:
        raise SpotOnError(f"n_states must be >= 1, got {n_states}")
    if dZ_um is None or dZ_um <= 0:
        raise SpotOnError(
            "dZ_um (axial detection depth, µm) is required and must be > 0. It "
            "comes from the optics and rescales the defocalization correction, and "
            "so the corrected fractions."
        )
    if localization_sigma_um < 0:
        raise SpotOnError(f"localization_sigma_um must be >= 0, got {localization_sigma_um}")
    low, high = D_bounds
    if not 0 < low < high:
        raise SpotOnError(f"D_bounds must satisfy 0 < low < high, got {D_bounds}")

    tau = float(params.tau_s)
    by_lag = jump_lengths(canonical, lags=lags)
    lags = tuple(sorted(by_lag))
    lag_times = np.array([lag * tau for lag in lags], dtype=float)
    counts = np.array([len(by_lag[lag]) for lag in lags])
    total = int(counts.sum())
    if total < 4 * n_states:
        raise SpotOnError(
            f"only {total} jumps for a {n_states}-state fit; refusing to fit an "
            "underdetermined model."
        )

    variance_sigma = localization_sigma_um**2
    rng = np.random.default_rng(seed)
    lower = np.concatenate([np.full(n_states, low), np.zeros(n_states - 1)])
    upper = np.concatenate([np.full(n_states, high), np.ones(n_states - 1)])

    def negative_log_likelihood(theta):
        D, fractions = _unpack(theta, n_states)
        if np.any(D <= 0):
            return _PENALTY
        total_ll = 0.0
        for lag_time, lag in zip(lag_times, lags):
            r = by_lag[lag]
            scale_squared = 2.0 * (D * lag_time + variance_sigma)
            if np.any(scale_squared <= 0):
                return _PENALTY
            z = z_correction(D, lag_time, dZ_um)
            weights = fractions * z
            weight_sum = weights.sum()
            if weight_sum <= 0:
                return _PENALTY
            weights = weights / weight_sum
            density = (
                r[:, None] / scale_squared[None, :]
                * np.exp(-(r[:, None] ** 2) / (2.0 * scale_squared[None, :]))
            ) @ weights
            if np.any(density <= 0) or not np.isfinite(density).all():
                return _PENALTY
            total_ll += float(np.log(density).sum())
        return -total_ll if np.isfinite(total_ll) else _PENALTY

    starts = _starts(by_lag, lag_times, n_states, n_starts, D_bounds, variance_sigma, rng)
    results = []
    for theta0 in starts:
        try:
            outcome = minimize(
                negative_log_likelihood, theta0, method="L-BFGS-B",
                bounds=list(zip(lower, upper)),
                options={"maxiter": 3000, "ftol": 1e-12},
            )
        except (ValueError, FloatingPointError):
            continue
        if outcome.success and np.all(np.isfinite(outcome.x)):
            results.append((float(outcome.fun), outcome.x))

    if not results:
        raise SpotOnError(
            f"no start converged for {n_states} states on {total} jumps. Check "
            "the data."
        )

    results.sort(key=lambda cr: cr[0])
    best_cost, best = results[0]
    D, fractions = _unpack(best, n_states)
    order = np.argsort(D)
    D, fractions = D[order], fractions[order]

    n_free = 2 * n_states - 1
    log_likelihood = -best_cost
    bic = n_free * np.log(total) - 2.0 * log_likelihood

    return SpotOnFit(
        n_states=n_states,
        D=D,
        fractions=fractions,
        lags=lags,
        lag_times_s=lag_times,
        dZ_um=float(dZ_um),
        localization_sigma_um=float(localization_sigma_um),
        n_jumps=total,
        log_likelihood=float(log_likelihood),
        bic=float(bic),
        z_corrections=np.array(
            [z_correction(D, lag_time, dZ_um) for lag_time in lag_times]
        ),
        n_starts=len(starts),
        n_converged_starts=len(results),
    )


def dz_sensitivity(
    canonical: pd.DataFrame,
    *,
    n_states: int = 2,
    dZ_values=(0.4, 0.7, 1.0, 1.5, 3.0),
    **fit_kwargs,
) -> pd.DataFrame:
    """How much does the corrected fraction depend on ``dZ``? Usually: a lot.

    ``dZ`` is an optical constant, not a fitted parameter, and the correction
    scales directly with it. On the reference dataset the corrected slow
    fraction swings by tens of points between a corrected ``dZ`` and no correction at
    all, driven entirely by a number the data cannot supply.

    **Run this before quoting any defocalization-corrected fraction.** If the
    spread across plausible ``dZ`` exceeds the effect being claimed, the
    correction is not yet usable and the uncorrected fit should be reported
    instead, labelled as uncorrected.
    """
    rows = []
    for dZ in dZ_values:
        fit = fit_spoton(canonical, n_states=n_states, dZ_um=float(dZ), **fit_kwargs)
        row = {
            "dZ_um": float(dZ),
            "slow_fraction_percent": float(fit.fractions_percent[0]),
            "n_jumps": fit.n_jumps,
        }
        for i, d in enumerate(fit.D, start=1):
            row[f"D_{i}"] = float(d)
        rows.append(row)
    out = pd.DataFrame(rows)
    span = float(out["slow_fraction_percent"].max() - out["slow_fraction_percent"].min())
    out.attrs["span_percentage_points"] = span
    out.attrs["verdict"] = (
        f"the corrected slow fraction varies by {span:.1f} percentage points across "
        f"dZ = {min(dZ_values):g}-{max(dZ_values):g} µm. "
        + (
            "dZ must be measured before this number is quoted."
            if span > 5
            else "the correction is robust to dZ over this range."
        )
    )
    return out


def _starts(by_lag, lag_times, n_states, n_starts, D_bounds, variance_sigma, rng):
    """A moment-based start plus random ones (log-uniform in D)."""
    low, high = D_bounds
    shortest = by_lag[min(by_lag)]
    lag_time = float(lag_times[0])
    # <r²> = 4·D·Δt + 4σ² for 2-D, so D ≈ (<r²>/4 − σ²)/Δt.
    mean_squared = float(np.mean(shortest**2))
    d_hat = max((mean_squared / 4.0 - variance_sigma) / max(lag_time, 1e-12), low * 2)
    d_hat = float(np.clip(d_hat, low * 2, high / 2))
    spread = np.geomspace(max(d_hat / 20, low), min(d_hat * 3, high), n_states)
    starts = [_pack(spread, np.full(n_states, 1.0 / n_states))]
    for _ in range(max(n_starts - 1, 0)):
        D = np.sort(np.exp(rng.uniform(np.log(low), np.log(high), size=n_states)))
        starts.append(_pack(D, rng.dirichlet(np.ones(n_states))))
    return starts
