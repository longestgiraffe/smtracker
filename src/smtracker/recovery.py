"""Recovery-of-truth harness, validate estimators against the forward model.

The principle
-------------
An estimator is only trustworthy to the extent it recovers a *known* answer.
`smtracker.simulate` emits observed trajectories from specified truth (rates,
D's, fractions), carrying the true per-step state; this module runs each
estimator on that output (**blind**, reading only the canonical columns) and
tabulates truth vs estimate. Nothing here shares code with a fitter's
likelihood, so agreement is evidence, not tautology.

What it validates, and why these axes
-------------------------------------
One lab condition made one axis non-negotiable: **rate identifiability degrades
honestly with switching**. Occupancy needs no transitions to estimate, so it
should be recovered even when a molecule almost never switches (a rarely switching
regime); a rate needs observed transitions, so its interval must widen and
eventually become unidentifiable as switching falls. `rate_identifiability_map`
sweeps switch-rate × track-length and reports exactly that, reproducing from
known truth that condition's wide CI: confirming it was correct behaviour, not a bug.

Invariants
----------
- Estimators receive only ``CANONICAL_COLUMNS``; ``true_state`` / ``true_D`` are
  read here for scoring, never passed to a fitter.
- Deterministic under a fixed seed.
- Fails loud if asked to score a table that carries no truth manifest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .canonical import CANONICAL_COLUMNS
from .hmm import HMMError, bootstrap_hmm, fit_hmm, step_sequences
from .observables import squared_displacements
from .simulate import (
    ConditionSpec,
    ImagingSpec,
    SimulationError,
    StateSpec,
    simulate_experiment,
)
from .sqd import fit_sqd

__all__ = [
    "RecoveryError",
    "recover_sqd",
    "recover_hmm",
    "recovery_table",
    "rate_identifiability_map",
    "recover_spatial_geometry",
    "recover_ci_coverage",
]


class RecoveryError(RuntimeError):
    """Raised when a recovery run is malformed."""


def _truth_for(canonical: pd.DataFrame, condition: str) -> dict:
    truth = canonical.attrs.get("truth")
    if not truth:
        raise RecoveryError(
            "table carries no truth manifest; recovery needs a simulated table "
            "from simulate_experiment"
        )
    if condition not in truth:
        raise RecoveryError(f"no truth for condition {condition!r}")
    return truth[condition]


def _slow_index(D) -> int:
    return int(np.argmin(D))


def recover_sqd(canonical: pd.DataFrame, condition: str, *, weighting="track") -> dict:
    """SQD slow-fraction recovery for one condition."""
    truth = _truth_for(canonical, condition)
    part = canonical[canonical["condition"] == condition]
    observable = squared_displacements(part)
    fit = fit_sqd(observable, n_pop=truth["n_states"], weighting=weighting)

    true_slow = truth["occupancy"][_slow_index(truth["D"])]
    est_slow = float(fit.fractions[_slow_index(fit.D)])
    return {
        "condition": condition,
        "estimator": f"SQD ({weighting})",
        "true_slow_frac": float(true_slow),
        "est_slow_frac": est_slow,
        "slow_frac_error": est_slow - float(true_slow),
        "true_D_slow": float(min(truth["D"])),
        "est_D_slow": float(min(fit.D)),
    }


def recover_hmm(
    canonical: pd.DataFrame,
    condition: str,
    *,
    localization_sigma_um: float = 0.0,
    n_starts: int = 6,
    n_boot: int = 0,
) -> dict:
    """HMM recovery for one condition: occupancy, D, and rates against truth.

    With ``n_boot > 0`` the rates carry a cell-bootstrap percentile interval and
    the row reports whether that interval is *identifiable*, a rate whose
    interval spans more than two orders of magnitude, or a run where resamples
    go degenerate, is flagged, which is the honest report for a low-switching
    condition.
    """
    truth = _truth_for(canonical, condition)
    part = canonical[canonical["condition"] == condition]
    tau = canonical.attrs["params"].tau_s

    row = {"condition": condition, "estimator": "HMM"}
    try:
        fit = fit_hmm(
            step_sequences(part), n_states=truth["n_states"], tau_s=tau,
            localization_sigma_um=localization_sigma_um, n_starts=n_starts,
        )
    except HMMError as exc:
        row["error"] = str(exc)
        return row

    slow = _slow_index(fit.D)
    fast = 1 - slow if fit.n_states == 2 else int(np.argmax(fit.D))
    true_slow = truth["occupancy"][_slow_index(truth["D"])]
    row.update({
        "true_slow_frac": float(true_slow),
        "est_slow_frac": float(fit.occupancy[slow]),
        "slow_frac_error": float(fit.occupancy[slow]) - float(true_slow),
        "true_D_slow": float(min(truth["D"])),
        "est_D_slow": float(min(fit.D)),
        "true_D_fast": float(max(truth["D"])),
        "est_D_fast": float(max(fit.D)),
        "degenerate": bool(fit.has_degenerate_chain),
    })

    if truth["rates_per_s"] is not None and fit.n_states == 2:
        rates = np.asarray(truth["rates_per_s"], dtype=float)
        # state indices in truth follow the StateSpec order (slow first here)
        row["true_on_rate"] = float(rates[1, 0])       # fast -> slow
        row["true_off_rate"] = float(rates[0, 1])      # slow -> fast
        est = fit.rates_per_s
        row["est_on_rate"] = float(est[fast, slow])
        row["est_off_rate"] = float(est[slow, fast])

    if n_boot > 0 and truth["n_states"] == 2:
        try:
            boot = bootstrap_hmm(
                part, n_states=2, tau_s=tau, unit="cell", n_boot=n_boot,
                localization_sigma_um=localization_sigma_um, n_starts=3,
            )
            on = boot["rate_2_to_1"].to_numpy()
            on = on[np.isfinite(on) & (on > 0)]
            lo, hi = np.percentile(on, [2.5, 97.5]) if on.size else (np.nan, np.nan)
            row["on_rate_ci_low"] = float(lo)
            row["on_rate_ci_high"] = float(hi)
            row["on_rate_ci_decades"] = (
                float(np.log10(hi / lo)) if lo > 0 and np.isfinite(hi) else np.inf
            )
            row["n_degenerate_boot"] = int(boot.attrs["n_degenerate"])
            row["rates_identifiable"] = bool(
                np.isfinite(row["on_rate_ci_decades"])
                and row["on_rate_ci_decades"] < 1.0
                and boot.attrs["n_degenerate"] == 0
            )
        except HMMError as exc:
            row["boot_error"] = str(exc)
    return row


def recovery_table(
    canonical: pd.DataFrame,
    *,
    localization_sigma_um: float = 0.0,
    n_boot: int = 0,
) -> pd.DataFrame:
    """Truth-vs-estimate for SQD and HMM across every simulated condition."""
    truth = canonical.attrs.get("truth")
    if not truth:
        raise RecoveryError("table carries no truth manifest")
    rows = []
    for condition in truth:
        rows.append(recover_sqd(canonical, condition))
        rows.append(recover_hmm(
            canonical, condition, localization_sigma_um=localization_sigma_um,
            n_boot=n_boot,
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# The rate-identifiability map
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RateIdentifiabilityCell:
    """One (switch-rate, track-length) point in the identifiability sweep."""

    total_switch_rate_s: float
    bleach_lifetime_frames: float
    true_slow_frac: float
    est_slow_frac: float
    slow_frac_error: float
    on_rate_ci_decades: float
    rates_identifiable: bool
    n_degenerate_boot: int
    mean_transitions_per_track: float


def rate_identifiability_map(
    params,
    *,
    switch_rates_s=(0.5, 2.0, 8.0, 32.0),
    bleach_lifetimes=(5.0, 10.0, 30.0),
    slow_frac: float = 0.35,
    D_slow: float = 0.05,
    D_fast: float = 0.9,
    sigma_um: float = 0.0,
    n_cells: int = 12,
    tracks_per_cell: int = 40,
    n_boot: int = 60,
    seed: int = 0,
) -> pd.DataFrame:
    """Sweep switch-rate × track-length, reporting occupancy AND rate recovery.

    The mechanism claim's validation. At each grid point a two-state condition
    is simulated with a **fixed** stationary slow fraction (so occupancy recovery
    can be judged independently of switching) and a chosen total switch rate and
    bleaching lifetime; the HMM is run with a cell bootstrap and the row records
    whether the rate came back identifiable.

    Expectation this is built to demonstrate:

    - ``slow_frac_error`` stays small everywhere, *including* the slowest
      switching: occupancy does not need transitions;
    - ``rates_identifiable`` is True where switching is adequate and False in the
      rarely-switching corner (low rate, short tracks, few transitions per track).
    """
    # Rates that give the target stationary slow fraction at the chosen total
    # switch rate. π_slow = k_on / (k_on + k_off); total = k_on + k_off.
    rows = []
    for total in switch_rates_s:
        k_on = total * slow_frac          # fast -> slow
        k_off = total * (1.0 - slow_frac)  # slow -> fast
        rates = np.array([[0.0, k_off], [k_on, 0.0]])
        condition = ConditionSpec(
            "sim", (StateSpec(D_slow), StateSpec(D_fast)), rates_per_s=rates,
        )
        for bleach in bleach_lifetimes:
            imaging = ImagingSpec(
                params, sigma_um=sigma_um, bleach_lifetime_frames=bleach,
                max_frames=200,
            )
            try:
                canonical = simulate_experiment(
                    [condition], imaging, n_cells=n_cells,
                    tracks_per_cell=tracks_per_cell, seed=seed,
                )
            except SimulationError:
                continue

            hmm = recover_hmm(canonical, "sim", n_boot=n_boot)
            if "error" in hmm:
                continue

            # Expected transitions actually observed per track: total rate × the
            # mean observed track duration in seconds.
            lengths = canonical.groupby("uid", observed=True).size()
            mean_duration_s = float((lengths.mean() - 1) * params.tau_s)
            rows.append(
                {
                    "total_switch_rate_s": float(total),
                    "bleach_lifetime_frames": float(bleach),
                    "true_slow_frac": hmm.get("true_slow_frac", np.nan),
                    "est_slow_frac": hmm.get("est_slow_frac", np.nan),
                    "slow_frac_error": hmm.get("slow_frac_error", np.nan),
                    "on_rate_ci_decades": hmm.get("on_rate_ci_decades", np.inf),
                    "rates_identifiable": hmm.get("rates_identifiable", False),
                    "n_degenerate_boot": hmm.get("n_degenerate_boot", 0),
                    "mean_transitions_per_track": float(total * mean_duration_s),
                }
            )
    if not rows:
        raise RecoveryError("no grid point produced a usable fit")
    table = pd.DataFrame(rows)
    table.attrs["slow_frac_target"] = slow_frac
    table.attrs["sigma_um"] = sigma_um
    return table


# --------------------------------------------------------------------------- #
# Recovery panel, the two remaining rows
# --------------------------------------------------------------------------- #


def recover_spatial_geometry(
    params, *, membrane_affinity: float = 0.9, slow_frac: float = 0.5,
    n_cells: int = 14, tracks_per_cell: int = 40, sigma_um: float = 0.0,
    seed: int = 0,
) -> dict:
    """Does the spatial map recover a PLANTED geometry?

    A two-state condition is simulated where the slow state is given a strong
    membrane affinity and the fast state none, then the HMM per-step radial map
    is built (blind) and asked whether the slow state comes back more peripheral
    than the fast: recovering, from known truth, the peripheral-slow finding.
    """
    from .cellcoords import project_tracks
    from .cellmesh import CellMesh
    from .hmm import fit_hmm, step_sequences
    from .maps import fold_map, labels_from_hmm, state_enrichment

    slow = StateSpec(0.05, membrane_affinity=membrane_affinity)
    fast = StateSpec(0.9, membrane_affinity=0.0)
    k_on = 8.0 * slow_frac
    k_off = 8.0 * (1 - slow_frac)
    condition = ConditionSpec(
        "geo", (slow, fast), rates_per_s=np.array([[0.0, k_off], [k_on, 0.0]])
    )
    imaging = ImagingSpec(params, sigma_um=sigma_um, max_frames=200)
    canonical = simulate_experiment(
        [condition], imaging, n_cells=n_cells, tracks_per_cell=tracks_per_cell,
        seed=seed,
    )

    # A synthetic straight-cell mesh matching the simulator's rod, one per cell.
    cell_len, half = 3.0, 0.5
    meshes = {}
    for cell_id in canonical["cell"].unique():
        movie = str(cell_id).split(":")[0]
        xs = np.linspace(0, cell_len, 11)
        mesh = np.column_stack([xs, np.full(11, -half), xs, np.full(11, half)])
        left = mesh[:, :2]
        right = mesh[::-1, 2:]
        polygon = np.vstack([left, right]) / params.um_per_pixel
        meshes.setdefault(movie, []).append(
            CellMesh(movie, str(cell_id).split(":")[1], polygon,
                     mesh=mesh / params.um_per_pixel)
        )

    projected = project_tracks(canonical, meshes)
    fit = fit_hmm(step_sequences(projected), n_states=2, tau_s=params.tau_s,
                  n_starts=6)
    labelled = fold_map(labels_from_hmm(projected, fit, min_posterior=0.6))
    table = state_enrichment(labelled, axis="d_norm", bins=5)
    wide = table.pivot(index="d_norm", columns="state", values="enrichment")
    slow_profile = wide[1].to_numpy() if 1 in wide.columns else np.array([np.nan])
    fast_profile = wide[2].to_numpy() if 2 in wide.columns else np.array([np.nan])
    slow_more_peripheral = bool(slow_profile[-1] > slow_profile[0])
    return {
        "planted": "slow-at-membrane",
        "slow_profile": slow_profile.tolist(),
        "fast_profile": fast_profile.tolist(),
        "slow_more_peripheral_than_fast": bool(
            slow_profile[-1] / max(slow_profile[0], 1e-9)
            > fast_profile[-1] / max(fast_profile[0], 1e-9)
        ),
        "slow_rises_to_membrane": slow_more_peripheral,
        "recovered": slow_more_peripheral,
    }


def recover_ci_coverage(
    params, *, true_slow: float = 0.35, n_datasets: int = 12, n_cells: int = 10,
    tracks_per_cell: int = 25, between_cell_sd: float = 0.6, n_boot: int = 40,
    seed: int = 0,
) -> dict:
    """Do the cell-bootstrap CIs actually COVER the truth (and does pooling lie)?

    Many datasets are simulated from the SAME known slow fraction, each with real
    cell-to-cell spread. For each, a cell-level bootstrap CI is built and checked
    for coverage of the truth. The honest CI should cover ~95 % of the time.

    Alongside, the pseudoreplication footgun is exercised: a *pooled-step*
    interval (steps as if independent) is far too narrow, so its coverage
    collapses well below nominal: the quantitative reason the project bans
    significance on pooled steps.
    """
    from .observables import squared_displacements
    from .sqd import bootstrap_sqd, fit_sqd

    k_on = 8.0 * true_slow
    k_off = 8.0 * (1 - true_slow)
    condition = ConditionSpec(
        "cov", (StateSpec(0.05), StateSpec(0.9)),
        rates_per_s=np.array([[0.0, k_off], [k_on, 0.0]]),
    )
    imaging = ImagingSpec(params, max_frames=150)

    cell_hits = 0
    pooled_hits = 0
    for d in range(n_datasets):
        canonical = simulate_experiment(
            [condition], imaging, n_cells=n_cells, tracks_per_cell=tracks_per_cell,
            between_cell_sd=between_cell_sd, seed=seed + d,
        )
        obs = squared_displacements(canonical)
        # Honest: cell-level bootstrap interval.
        cell_boot = bootstrap_sqd(obs, n_pop=2, unit="cell", n_boot=n_boot, n_jobs=1)
        lo, hi = cell_boot["fraction_1"].quantile([0.025, 0.975])
        if lo <= true_slow <= hi:
            cell_hits += 1
        # Footgun: pooled-step interval, resample STEPS, ignoring correlation.
        vals = obs["sq_disp_um2"].to_numpy()
        rng = np.random.default_rng(seed + 1000 + d)
        pooled = []
        for _ in range(n_boot):
            draw = rng.choice(vals, size=len(vals), replace=True)
            try:
                pooled.append(
                    fit_sqd(draw, n_pop=2, tau_lag_s=obs.attrs["lag_time_s"]).fractions[0]
                )
            except Exception:                          # noqa: BLE001
                continue
        if pooled:
            plo, phi = np.percentile(pooled, [2.5, 97.5])
            if plo <= true_slow <= phi:
                pooled_hits += 1

    return {
        "true_slow": true_slow,
        "n_datasets": n_datasets,
        "cell_ci_coverage": cell_hits / n_datasets,
        "pooled_step_ci_coverage": pooled_hits / n_datasets,
    }
