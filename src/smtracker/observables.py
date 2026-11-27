"""Squared displacements and the empirical CDF, the SQD observable, once.

Everything the SQD/appD fits consume is computed here. Units and the
track-length filter are *not* redone: they happened in ``smtracker.canonical``.

Definition
----------
For lag ``n`` (in frames) and localizations at frames ``f`` and ``f + n``:

    r²  =  (x[f+n] − x[f])²  +  (y[f+n] − y[f])²      [µm²]

with the lag time ``τ_lag = n · τ``. All pairs within a track are used
(overlapping pairs included), matching the legacy tool.

Pairing: frames vs rows
-----------------------
The legacy MATLAB pairs by **row offset**, ``track(i)`` with ``track(i+n)``,
which silently treats a bridged gap as if it were a single frame step, inflating
that displacement. This module pairs by **frame difference** by default
(``pairing="frame"``): a pair counts only if the two localizations really are
``n`` frames apart. ``pairing="row"`` reproduces the legacy behaviour for
comparison.

For u-track data tracked with ``gapCloseParam.timeWindow = 1`` there is no gap
closing, so tracks are contiguous and the two are **identical**, which is the
case for the reference project. The distinction matters for gapped data.

Empirical CDF
-------------
``ecdf`` matches MATLAB's ``ecdf(x, 'Function', 'cdf')`` for uncensored data,
including its leading ``(min(x), 0)`` point, so an ECDF-least-squares fit sees
exactly the points the legacy fit saw.

Invariants
----------
- Input must be a canonical table (µm, filtered). Fails loud otherwise.
- ``lag_frames >= 1``.
- Fails loud if no pair survives: an empty observable is never a valid result.
- Grouping columns ride along, so fits can aggregate per condition/day/cell.

Lags 1-4 together (v2's "1-4 together", 17 Oct 2026)
----------------------------------------------------
``joint_lag_displacements`` stacks the displacements 1 to 4 frames apart into ONE observable for ONE fit, each
r² DIVIDED BY ITS LAG, so the whole set is fitted at tau = one frame interval: under the SQD model (no
localisation term, as v2's), r²_n ~ Exp(4 D n tau), so r²_n / n ~ Exp(4 D tau) -- the same D and fractions at every
lag, which is v2's assumption. ``fit_ecdf`` then builds ONE ECDF PER LAG and stacks their points, so the least
squares is v2's own (four per-lag residual blocks, unweighted: each lag counts by its number of points), not a fit
to one pooled ECDF (which differs: on the reference, three populations, by up to 3.5 points). The likelihood is
the joint one exactly (dividing by n changes it by a constant). Each row keeps its ``lag_frames``; ``.attrs
["joint_lags"]`` says the set is joint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .canonical import CANONICAL_COLUMNS, LOC_SIGMA_COLUMN, CanonicalError

__all__ = [
    "ObservableError",
    "SQD_COLUMNS",
    "LOC_SIGMA_STEP_COLUMNS",
    "WEIGHTINGS",
    "squared_displacements",
    "displacement_weights",
    "ecdf",
    "fit_ecdf",
    "joint_lag_displacements",
    "lag_labels",
    "displacements_at",
    "JOINT_LAGS",
    "JOINT",
]

#: The lags v2's "1-4 together" fits at once (`joint_lag_displacements`), and the word a Run's lag setting uses for
#: them (`comparison.JOINT_LAG`, the Spatial Map's `MAP_LAGS`).
JOINT_LAGS = (1, 2, 3, 4)
JOINT = "1-4"

#: How much each displacement counts. See ``displacement_weights``.
WEIGHTINGS = ("step", "track", "cell")

SQD_COLUMNS = [
    "protein", "condition", "day", "movie", "cell", "uid",
    "frame", "lag_frames", "lag_time_s", "sq_disp_um2",
]

#: Cell coordinates carried through from the step's START point WHEN PRESENT, so a
#: projected table can be labelled per step. Not part of `SQD_COLUMNS`: an unprojected
#: table has none of them and must still produce a valid observable.
_CELL_COORDS = ("L_um", "D_um", "l_norm", "d_norm")
#: Each step's two spots' own precision (um, per axis, as the tracker reported it: `canonical.LOC_SIGMA_COLUMN`),
#: carried WHEN the canonical table has it; like the cell coordinates, not part of `SQD_COLUMNS`. Both ends, because a
#: calibration acts on each spot (`precision.step_noise`), not on the step's sum.
LOC_SIGMA_STEP_COLUMNS = ("loc_sigma_start_um", "loc_sigma_end_um")

_GROUP_COLUMNS = ["protein", "condition", "day", "movie", "cell"]


class ObservableError(RuntimeError):
    """Raised when squared displacements cannot be computed."""


def squared_displacements(
    canonical: pd.DataFrame,
    *,
    lag_frames: int = 1,
    pairing: str = "frame",
) -> pd.DataFrame:
    """Squared displacements at a given lag.

    Parameters
    ----------
    canonical
        Canonical table from ``smtracker.canonical.to_canonical``. Its
        ``AcquisitionParams`` (carried in ``.attrs``) supplies τ.
    lag_frames
        Lag ``n`` in frames (1 = consecutive frames).
    pairing
        ``"frame"`` (default) pairs localizations exactly ``n`` frames apart;
        ``"row"`` pairs the ``i``-th and ``(i+n)``-th localization of a track,
        reproducing the legacy MATLAB behaviour across gaps.

    Returns
    -------
    pandas.DataFrame
        ``SQD_COLUMNS``; one row per displacement, ``frame`` being the *first*
        frame of the pair. ``.attrs["tau_s"]`` and ``.attrs["lag_time_s"]``
        carry the frame interval and the lag time.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise ObservableError(
            f"not a canonical table (missing {missing}); run canonical.to_canonical first"
        )
    if lag_frames < 1:
        raise ObservableError(f"lag_frames must be >= 1, got {lag_frames}")
    if pairing not in ("frame", "row"):
        raise ObservableError(f"pairing must be 'frame' or 'row', got {pairing!r}")
    if len(canonical) == 0:
        raise ObservableError("canonical table is empty")

    tau_s = _tau_from(canonical)
    frames = canonical.sort_values(["uid", "frame"], kind="stable")

    grouped = frames.groupby("uid", sort=False, observed=True)
    shifted = grouped[["frame", "x_um", "y_um"]].shift(-lag_frames)
    dx = shifted["x_um"] - frames["x_um"]
    dy = shifted["y_um"] - frames["y_um"]
    df = shifted["frame"] - frames["frame"]

    valid = shifted["frame"].notna()
    if pairing == "frame":
        # Only true n-frame steps; a bridged gap is not a single step.
        valid &= df == lag_frames

    out = pd.DataFrame(
        {
            **{c: frames[c].to_numpy() for c in _GROUP_COLUMNS},
            "uid": frames["uid"].to_numpy(),
            "frame": frames["frame"].to_numpy(),
            "lag_frames": lag_frames,
            "lag_time_s": lag_frames * tau_s,
            "sq_disp_um2": (dx**2 + dy**2).to_numpy(),
        }
    ).loc[valid.to_numpy()]

    if len(out) == 0:
        raise ObservableError(
            f"no localization pairs at lag {lag_frames} frames (pairing={pairing!r}). "
            "Tracks may be shorter than the lag, or entirely gapped."
        )
    if not np.isfinite(out["sq_disp_um2"]).all():
        raise ObservableError("non-finite squared displacements: check the input table")

    # CELL COORDINATES RIDE ALONG WHEN THEY ARE THERE, from the step's START point.
    #
    # Without this, `maps.labels_from_sqd` could NEVER succeed: it calls this function
    # and then demands `l_norm`/`d_norm` on the result, which the fixed `SQD_COLUMNS`
    # list structurally could not carry. Exported, documented, and raising on every
    # possible input - the same "built but never exercised" shape this project keeps
    # finding, and it had no test and no caller.
    #
    # THE START POINT, not the midpoint, because that is the convention
    # `hmm.decode_states` already uses and the two must not drift apart: a step assigned
    # to one end here and the other end there would put the same molecule in different
    # bins depending on which labeller drew the map. The choice is worth real contrast
    # (CLAUDE.md records midpoint against start at 2.92 vs 1.63 median on the SDA maps),
    # so it is a convention to state, never a detail. `frames[c]` IS the start row.
    carried = [c for c in _CELL_COORDS if c in frames.columns]
    for column in carried:
        out[column] = frames[column].to_numpy()[valid.to_numpy()]
    # EACH STEP'S TWO SPOTS' OWN PRECISION, where the tracker recorded it (u-track; 6 Nov 2026, the per-spot precision
    # fit), as reported, UNCALIBRATED (a lower bound: `canonical.LOC_SIGMA_COLUMN`); `precision.step_noise` turns them
    # into the step's localisation variance through a calibration. NaN where a spot has none.
    sigmas = []
    if LOC_SIGMA_COLUMN in frames.columns:
        start, end = LOC_SIGMA_STEP_COLUMNS
        out[start] = frames[LOC_SIGMA_COLUMN].to_numpy(dtype=float)[valid.to_numpy()]
        out[end] = grouped[LOC_SIGMA_COLUMN].shift(-lag_frames).to_numpy(dtype=float)[valid.to_numpy()]
        sigmas = [start, end]

    out = out.reset_index(drop=True)
    out.attrs["tau_s"] = tau_s
    out.attrs["lag_time_s"] = lag_frames * tau_s
    out.attrs["pairing"] = pairing
    out.attrs["step_position"] = "start"
    out.attrs["cell_coords_carried"] = list(carried)
    return out[SQD_COLUMNS + carried + sigmas]


#: The message every pooled-displacement model shares, because they share the reason.
_MIXED_TAU_REASON = (
    "displacements measured at different frame intervals come from different "
    "distributions even for identical D, because τ is inside the SQD, Spot-On and HMM "
    "likelihoods. Fit each τ stratum separately and merge the fractions or the D's, "
    "never the displacements. Per-track appD is unaffected. See AcquisitionParams.tau_strata()."
)


def _tau_from(canonical: pd.DataFrame) -> float:
    """The single τ for a pooled-displacement model. RAISES on a mixed-τ table.

    Checks the DATA, not only the params: a subset frame can lose ``.attrs`` while
    still holding rows from two acquisition rates, and that is exactly the case
    where a scalar τ would silently produce a wrong D for half of them.
    """
    if "tau_s" in canonical.columns:
        rates = sorted({round(float(v), 9) for v in canonical["tau_s"].dropna()})
        if len(rates) > 1:
            ms = ", ".join(f"{r * 1000:g} ms" for r in rates)
            raise ObservableError(
                f"these tracks span {len(rates)} frame intervals ({ms}), so there is "
                f"no single τ for a pooled model. {_MIXED_TAU_REASON}")
        if rates:
            return float(rates[0])

    params = canonical.attrs.get("params")
    if params is None:
        raise ObservableError(
            "canonical table carries no AcquisitionParams in .attrs['params'] and no "
            "'tau_s' column; τ is required and must not be guessed. Rebuild it with "
            "canonical.to_canonical (note: pandas drops .attrs through some "
            "operations: pass the table straight through, or re-attach params)."
        )
    try:
        return float(params.tau_s)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ObservableError(f"unexpected params object in .attrs: {params!r}") from exc
    except Exception as exc:                        # mixed-τ project → CanonicalError
        raise ObservableError(f"{exc} {_MIXED_TAU_REASON}") from exc


def joint_lag_displacements(canonical: pd.DataFrame, *, lags=JOINT_LAGS, pairing: str = "frame") -> pd.DataFrame:
    """The displacements at every lag in ``lags`` (v2's "1-4 together"), each r² divided by its lag so the set is
    fitted at one frame interval; one row per displacement and lag, ``lag_frames`` and ``lag_time_s`` the row's own.
    ``.attrs["lag_time_s"]`` is the frame interval (the fit's tau), ``.attrs["joint_lags"]`` the lags. See the
    module head for why this is v2's joint fit exactly."""
    lags = tuple(int(n) for n in lags)
    if len(lags) < 2 or len(set(lags)) != len(lags) or min(lags) < 1:
        raise ObservableError(f"a joint fit needs two or more distinct lags >= 1, got {lags}")
    parts = []
    for n in lags:
        part = squared_displacements(canonical, lag_frames=n, pairing=pairing)
        part["sq_disp_um2"] = part["sq_disp_um2"] / n
        parts.append(part)
    tau_s = float(parts[0].attrs["tau_s"])
    out = pd.concat(parts, ignore_index=True)
    out.attrs["tau_s"] = tau_s
    out.attrs["lag_time_s"] = tau_s
    out.attrs["joint_lags"] = lags
    return out


def displacements_at(canonical: pd.DataFrame, lag) -> pd.DataFrame:
    """The squared displacements a lag setting names: ``lag`` frames apart, or, for `JOINT`, lags 1-4 together
    (`joint_lag_displacements`). One place, for every caller that takes the setting as it is."""
    if lag == JOINT:
        return joint_lag_displacements(canonical)
    lag = int(lag)
    return squared_displacements(canonical) if lag == 1 else squared_displacements(canonical, lag_frames=lag)


def lag_labels(observable) -> "np.ndarray | None":
    """Each row's lag when ``observable`` is a joint set (`joint_lag_displacements`), else None."""
    if not isinstance(observable, pd.DataFrame) or not observable.attrs.get("joint_lags"):
        return None
    if "lag_frames" not in observable.columns:
        raise ObservableError("a joint-lag observable has no 'lag_frames' column")
    return observable["lag_frames"].to_numpy()


def fit_ecdf(values, weights=None, lags=None, max_points=None) -> tuple[np.ndarray, np.ndarray]:
    """The ECDF points a least-squares SQD fit is made to: ``ecdf`` of ``values``, or with ``lags`` (each row's lag,
    `lag_labels`) ONE ECDF PER LAG, their points stacked (v2's "1-4 together", `sqdFit_.m:66-70`). ``max_points``
    thins to about that many evenly spaced points (the interactive budget), shared among the lags by their counts,
    so each lag keeps the weight it has in v2's sum."""
    values = np.asarray(values, dtype=float).ravel()
    if lags is None:
        return _thin(*ecdf(values, weights), max_points)
    lags = np.asarray(lags).ravel()
    if lags.shape != values.shape:
        raise ObservableError(f"lags must match values: {lags.shape} vs {values.shape}")
    xs, ys = [], []
    for n in sorted(np.unique(lags)):
        keep = lags == n
        x, y = ecdf(values[keep], None if weights is None else np.asarray(weights, dtype=float)[keep])
        budget = None if not max_points else max(2, int(round(max_points * keep.sum() / len(values))))
        x, y = _thin(x, y, budget)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def _thin(x, y, max_points):
    """About ``max_points`` evenly spaced ECDF points (all when it has fewer, or no budget)."""
    if not max_points or len(x) <= max_points:
        return x, y
    idx = np.unique(np.linspace(0, len(x) - 1, max_points).round().astype(int))
    return x[idx], y[idx]


def ecdf(values, weights=None) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF, matching MATLAB's ``ecdf(x, 'Function', 'cdf')``.

    Returns ``(x, y)`` evaluated at the unique sorted values, prefixed with
    ``(min(x), 0)`` exactly as MATLAB does, so an ECDF-least-squares fit sees
    the same points (and the same number of residuals) as the legacy tool.

    ``y[i]`` is the fraction of observations ``<= x[i]``; the final value is 1.

    ``weights`` (optional) gives each observation a share other than ``1/n``,
    used to stop long tracks dominating simply by contributing more steps (see
    ``displacement_weights``). Weights are normalised internally.
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ObservableError("cannot build an ECDF from zero observations")
    if not np.isfinite(arr).all():
        raise ObservableError("ECDF input contains non-finite values")

    if weights is None:
        arr = np.sort(arr)
        unique, counts = np.unique(arr, return_counts=True)
        y = np.cumsum(counts) / arr.size
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape != arr.shape:
            raise ObservableError(
                f"weights must match values: {w.shape} vs {arr.shape}"
            )
        if not np.isfinite(w).all() or np.any(w < 0):
            raise ObservableError("weights must be finite and non-negative")
        total = w.sum()
        if total <= 0:
            raise ObservableError("weights sum to zero")
        order = np.argsort(arr, kind="stable")
        arr = arr[order]
        w = w[order] / total
        unique, first = np.unique(arr, return_index=True)
        y = np.cumsum(w)[np.append(first[1:] - 1, len(arr) - 1)]

    x = np.concatenate(([unique[0]], unique))
    y = np.concatenate(([0.0], y))
    return x, y


def displacement_weights(observable: pd.DataFrame, weighting: str) -> np.ndarray | None:
    """Per-displacement weights that undo length/size over-representation.

    Why this exists
    ---------------
    A track with ``N`` localizations contributes ``N − 1`` steps at lag 1, so
    **long tracks contribute more steps** to the pooled distribution. Track
    length is not neutral: fast molecules leave focus or exceed the tracker's
    search radius sooner, so long tracks are preferentially *slow*. Pooling
    steps therefore over-represents slow molecules and can bias the slow
    fraction upward. This is the same survivorship effect that flattens the MSD
    tail, acting here at a single lag.

    ``weighting``
        ``"step"``  : every displacement counts equally (legacy behaviour, the
                       estimator that reproduces the frozen reference).
        ``"track"``  (every *track* counts equally, regardless of length.
        ``"cell"``  ) every *cell* counts equally, regardless of how many
                       tracks or steps it yielded.

    Returns ``None`` for ``"step"`` (no weighting needed), else an array aligned
    with the observable's rows.
    """
    if weighting == "step":
        return None
    if weighting not in ("track", "cell"):
        raise ObservableError(
            f"weighting must be 'step', 'track' or 'cell', got {weighting!r}"
        )
    column = "uid" if weighting == "track" else "cell"
    if column not in observable.columns:
        raise ObservableError(f"observable has no '{column}' column to weight by")
    if observable[column].isna().any():
        raise ObservableError(
            f"'{column}' is unassigned for some displacements, so equal-{weighting} "
            "weighting is not defined. Assign it, or use weighting='step'."
        )
    counts = observable.groupby(column, observed=True)[column].transform("size")
    return (1.0 / counts.to_numpy(dtype=float))
