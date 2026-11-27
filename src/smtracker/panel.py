"""Cross-method consistency panel, every estimator, side by side, with limits.

Five estimators of the same quantity now exist, and they differ in what they
assume. The panel runs them on one dataset and puts the answers next to each
other, because **agreement across methods with different assumptions is the
evidence**, and disagreement is a finding, not a nuisance to average away.

| method | assumes a state count? | switching-aware? | uses |
|---|---|---|---|
| ``sqd``   | yes (BIC) | no  | single steps |
| ``appd``  | yes (BIC) | no  | short sub-tracks (averaged) |
| ``msd``   | n/a       | no  | ensemble curve (slope only) |
| ``hmm``   | yes (BIC) | **yes** | step sequences + transitions |
| ``saspt`` | **no**    | no  | single steps, nonparametric |
| ``spoton``| yes       | no  | jump lengths at several lags, **defocalization-corrected** |

What the panel is for
---------------------
- **Triangulation.** On the reference dataset SQD (track-weighted), the HMM and
  saSPT independently give ~35 % slow. Three assumptions, one answer.
- **Diagnosis.** A method that disagrees says something specific: appD below the
  others means switching inside its averaging window; MSD between the two states
  means it is a mixture average and not a state at all.
- **Refusing false precision.** Every row carries the measurement limits, so a
  ``D`` at or below the localization floor is visibly a bound.

What the panel is NOT
---------------------
It does not average methods together, vote, or pick a winner. Those numbers are
not interchangeable (they answer subtly different questions) and combining
them would manufacture a precision none of them has. The panel reports; the
analyst reads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .canonical import CANONICAL_COLUMNS, AcquisitionParams

__all__ = ["PanelError", "consistency_panel", "limits_report"]

#: saSPT returns a *continuous* occupancy profile, so turning it into a "slow
#: fraction" needs a boundary, and the answer is sensitive to it (on the
#: reference data, 26 % at D < 0.15 versus 37 % at D < 0.3). The panel therefore
#: uses the geometric mean of the parametric methods' own two D's, which is the
#: boundary those methods already imply, and reports the sensitivity alongside.
_SLOW_CUTOFF_SENSITIVITY = (0.5, 2.0)


class PanelError(RuntimeError):
    """Raised when the panel cannot be built."""


def limits_report(params: AcquisitionParams, D_values) -> pd.DataFrame:
    """Flag every ``D`` against the localization floor and detection ceiling.

    A ``D`` at or below the floor is a **bound**, not a measurement; a ``D`` near
    the ceiling is truncated by the tracker's search radius. This attaches that
    judgement to each number so it cannot be quoted without it.
    """
    floor = params.localization_floor_D
    ceiling = params.detection_ceiling_D
    rows = []
    for D in np.atleast_1d(np.asarray(D_values, dtype=float)):
        if floor is not None and D <= floor:
            verdict = "AT OR BELOW FLOOR: a bound, do not quantify"
        elif floor is not None and D < 2 * floor:
            verdict = "within twofold of the floor: treat as a bound"
        elif ceiling is not None and D > ceiling:
            verdict = "ABOVE CEILING: truncated by the search radius"
        elif ceiling is not None and D > ceiling / 2:
            verdict = "within twofold of the ceiling: fast molecules are being lost"
        else:
            verdict = "resolvable"
        rows.append(
            {
                "D_um2_per_s": float(D),
                "localization_floor_D": floor,
                "detection_ceiling_D": ceiling,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def consistency_panel(
    canonical: pd.DataFrame,
    *,
    n_pop: int = 2,
    weighting: str = "track",
    lag_frames: int = 1,
    include=("sqd", "appd", "msd", "hmm", "saspt"),
    n_starts: int = 20,
    localization_sigma_um: float | None = None,
    dZ_um: float | None = None,
) -> pd.DataFrame:
    """Run every requested estimator on one canonical table.

    Parameters
    ----------
    canonical
        Canonical table (µm, seconds) with ``AcquisitionParams`` in ``.attrs``.
    n_pop
        State count for the methods that need one. Shared deliberately: methods
        fitted at different counts are not comparable.
    weighting
        Displacement weighting for SQD/appD. ``"track"`` by default here (not
        ``"step"``), because the panel is for interpretation rather than for
        reproducing the frozen reference.
    include
        Which estimators to run. A method that fails is reported in the row with
        its error, never silently skipped.
    localization_sigma_um
        Passed to the HMM only. Defaults to ``0`` (apparent D) so the HMM stays
        comparable with the others: σ correction changes what "slow" means and
        is a sensitivity check, not a default.

    Returns
    -------
    pandas.DataFrame
        One row per method: ``slow_fraction_percent``, ``D_slow``, ``D_fast``,
        assumptions, notes, and the limits verdict for each ``D``.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise PanelError(f"not a canonical table (missing {missing})")
    params = canonical.attrs.get("params")
    if params is None:
        raise PanelError(
            "canonical table carries no AcquisitionParams in .attrs['params']"
        )
    # THE TABLE'S OWN SEARCH RADIUS (tester D10, 4 Nov 2026): the limits verdict is drawn at the radius its
    # condition was linked at, and a table pooling conditions linked at different radii has no one ceiling
    from .canonical import CanonicalError, table_params

    if isinstance(params, AcquisitionParams):
        try:
            params = table_params(canonical)
        except CanonicalError as exc:
            raise PanelError(f"{exc}. Run the panel on one condition at a time.") from exc
    unknown = set(include) - {"sqd", "appd", "msd", "hmm", "saspt", "spoton"}
    if unknown:
        raise PanelError(f"unknown method(s): {sorted(unknown)}")
    if "spoton" in include and (dZ_um is None or dZ_um <= 0):
        raise PanelError(
            "spoton needs dZ_um (axial detection depth, µm): it comes from the "
            "optics and directly scales the defocalization correction, so it is "
            "not guessed. Pass dZ_um, or drop 'spoton' from include."
        )

    from .observables import squared_displacements

    observable = squared_displacements(canonical, lag_frames=lag_frames)
    rows = []

    if "sqd" in include:
        rows.append(
            _row(
                "sqd", "yes (BIC)", "no", "single steps",
                lambda: _sqd_row(observable, n_pop, weighting, n_starts),
            )
        )
    if "appd" in include:
        rows.append(
            _row(
                "appd", "yes (BIC)", "no", "sub-tracks (averaged)",
                lambda: _appd_row(canonical, n_pop, weighting, n_starts),
            )
        )
    if "msd" in include:
        rows.append(
            _row(
                "msd", "n/a", "no", "ensemble curve",
                lambda: _msd_row(canonical),
            )
        )
    if "hmm" in include:
        sigma = 0.0 if localization_sigma_um is None else localization_sigma_um
        rows.append(
            _row(
                "hmm", "yes (BIC)", "YES", "step sequences",
                lambda: _hmm_row(canonical, n_pop, params, sigma, n_starts),
            )
        )
    if "saspt" in include:
        # Use the boundary the parametric fits already imply, so the comparison
        # is not decided by an arbitrary cutoff.
        reference = next(
            (r for r in rows if r["method"] == "sqd" and r["ok"] and r["D_slow"]), None
        )
        cutoff = (
            float(np.sqrt(reference["D_slow"] * reference["D_fast"]))
            if reference else None
        )
        rows.append(
            _row(
                "saspt", "NO (nonparametric)", "no", "single steps",
                lambda: _saspt_row(canonical, cutoff),
            )
        )

    if "spoton" in include:
        rows.append(
            _row(
                "spoton", "yes", "no", "jump lengths, several lags",
                lambda: _spoton_row(
                    canonical, n_pop, dZ_um,
                    0.035 if localization_sigma_um is None else localization_sigma_um,
                    n_starts,
                ),
            )
        )

    panel = pd.DataFrame(rows)
    verdicts = []
    for _, row in panel.iterrows():
        values = [v for v in (row.get("D_slow"), row.get("D_fast")) if v is not None and np.isfinite(v)]
        if not values:
            verdicts.append("")
            continue
        report = limits_report(params, values)
        verdicts.append(" | ".join(report["verdict"]))
    panel["limits"] = verdicts
    panel.attrs["params"] = params
    panel.attrs["n_pop"] = n_pop
    panel.attrs["weighting"] = weighting
    return panel


def _row(method, assumes_count, switching_aware, uses, run):
    base = {
        "method": method,
        "assumes_state_count": assumes_count,
        "switching_aware": switching_aware,
        "uses": uses,
        "slow_fraction_percent": None,
        "D_slow": None,
        "D_fast": None,
        "note": "",
        "ok": True,
    }
    try:
        base.update(run())
    except Exception as exc:  # noqa: BLE001 - reported in the row, never hidden
        base["ok"] = False
        base["note"] = f"FAILED: {type(exc).__name__}: {exc}"
    return base


def _sqd_row(observable, n_pop, weighting, n_starts):
    from .sqd import fit_sqd

    fit = fit_sqd(observable, n_pop=n_pop, weighting=weighting, n_starts=n_starts)
    return {
        "slow_fraction_percent": float(fit.fractions_percent[0]),
        "D_slow": float(fit.D[0]),
        "D_fast": float(fit.D[-1]),
        "note": f"{weighting}-weighted",
    }


def _appd_row(canonical, n_pop, weighting, n_starts):
    from .appd import apparent_D, fit_appd

    table = apparent_D(canonical, n_steps=4, mode="split")
    fit = fit_appd(table, n_pop=n_pop, weighting=weighting, n_starts=n_starts)
    return {
        "slow_fraction_percent": float(fit.fractions_percent[0]),
        "D_slow": float(fit.D[0]),
        "D_fast": float(fit.D[-1]),
        "note": "averages 4 steps; biased low by switching inside that window",
    }


def _msd_row(canonical):
    from .msd import ensemble_msd, fit_msd, msd_per_track

    fit = fit_msd(ensemble_msd(msd_per_track(canonical, max_lag=8)), n_lags=4)
    return {
        "slow_fraction_percent": None,
        "D_slow": None,
        "D_fast": float(fit.D_um2_per_s),
        "note": "single mixture-averaged slope: not a state; no fractions",
    }


def _hmm_row(canonical, n_pop, params, sigma, n_starts):
    from .hmm import fit_hmm, step_sequences

    fit = fit_hmm(
        step_sequences(canonical), n_states=n_pop, tau_s=params.tau_s,
        localization_sigma_um=sigma, n_starts=max(n_starts // 3, 4),
    )
    dwell = np.round(fit.dwell_times_s * 1000, 0).tolist()
    return {
        "slow_fraction_percent": float(fit.occupancy_percent[0]),
        "D_slow": float(fit.D[0]),
        "D_fast": float(fit.D[-1]),
        "note": f"switching-corrected; dwell {dwell} ms (model-derived), σ = {sigma * 1000:g} nm",
    }


def _spoton_row(canonical, n_pop, dZ_um, sigma, n_starts):
    from .spoton import fit_spoton

    fit = fit_spoton(
        canonical, n_states=n_pop, dZ_um=dZ_um, localization_sigma_um=sigma,
        lags=(1, 2, 3), n_starts=max(n_starts // 2, 8),
    )
    apparent = 100.0 * fit.apparent_fractions(fit.lags[0])[0]
    return {
        "slow_fraction_percent": float(fit.fractions_percent[0]),
        "D_slow": float(fit.D[0]),
        "D_fast": float(fit.D[-1]),
        "note": (
            f"defocalization-corrected (dZ = {dZ_um:g} µm); uncorrected at lag 1 "
            f"would read {apparent:.1f} %"
        ),
    }


def _saspt_row(canonical, cutoff):
    from .saspt_adapter import run_saspt

    result = run_saspt(canonical)
    modes = result.modes()
    if modes.empty:
        raise PanelError("saSPT found no occupancy peak")
    fast_D = float(modes["D_um2_per_s"].max())
    if cutoff is None:
        cutoff = float(np.sqrt(max(modes["D_um2_per_s"].min(), 1e-6) * fast_D))

    low, high = _SLOW_CUTOFF_SENSITIVITY
    span = (
        100.0 * result.occupancy_below(cutoff * low),
        100.0 * result.occupancy_below(cutoff * high),
    )
    return {
        "slow_fraction_percent": 100.0 * result.occupancy_below(cutoff),
        "D_slow": float(modes["D_um2_per_s"].min()),
        "D_fast": fast_D,
        "note": (
            f"{len(modes)} resolved peak(s); continuous profile, so the slow "
            f"fraction needs a boundary: at D < {cutoff:.3g} it is "
            f"{100.0 * result.occupancy_below(cutoff):.1f} %, ranging "
            f"{span[0]:.1f}–{span[1]:.1f} % from half to twice the cutoff"
        ),
    }
