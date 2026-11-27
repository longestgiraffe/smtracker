"""MSD: mean squared displacement vs lag, and the early-portion fit.

Scope: read this before quoting a D from here
----------------------------------------------
An ensemble MSD averages over **all** molecules, so when several diffusive
states are present its slope is **not** any state's D: it is dominated by the
fastest population and biased by the slowest. MSD is kept as a *cross-check* and
as a route to the localization-error offset and the anomalous exponent, the
state fractions and their D's come from ``smtracker.sqd``. Do not build a claim
about populations on an MSD slope.

Definition
----------
Built directly on ``observables.squared_displacements``, so displacements are
computed in exactly one place:

    MSD(nτ) = mean over all pairs (t, t + nτ) of  r²      [µm²]

Per track first, then a weighted ensemble mean over tracks. For 2-D Brownian
motion with localization error σ:

    MSD(t) = 2·ndim·D·t + 2·ndim·σ²  =  4·D·t + 4·σ²

so the slope gives ``D = slope / 4`` and the intercept gives
``σ = sqrt(intercept / 4)``: a useful independent estimate of the localization
precision, which u-track does not record.

The legacy bug this module does NOT reproduce
---------------------------------------------
``getFitMSD_gui.m`` fits ``fit(msmsd(:,1), msmsd(:,2), 'm*x + b')`` over
``2:limit``, but inside a ``try/catch`` that silently falls back to ``2:end``,
i.e. the **whole** curve, whenever ``limit`` exceeds the available lags. Long
lags are averaged over very few, highly correlated pairs and are extremely
noisy; including them corrupts the slope. It also computes weights (``N``,
``sem``) and then fits **unweighted**.

Here: the fit uses only the first ``n_lags`` lags (default 4), the number of
lags used is always reported, asking for more lags than exist **raises** rather
than silently widening the window, and the fit is weighted by the number of
contributing pairs unless disabled.

Honest error bars
-----------------
Never the regression's own standard error: consecutive MSD points share
displacements and are strongly correlated, so that interval is meaningless (the
same flaw as the legacy ``nlparci`` fractions). Use ``bootstrap_msd`` over cells
/ movies / days.

Pairing follows ``observables``: by frame difference, so a bridged gap is not
mistaken for one step. ``pairing="row"`` reproduces the legacy behaviour.

Direction: the camera plane, or the CELL'S OWN AXES
---------------------------------------------------
v2's "Choose direction" (xy / x / y) read ``projTracks`` (``calcMSD_gui.m``), the
tracks projected onto each cell's centreline, so its x and y were ALONG and ACROSS
the cell, never the camera's axes. ``direction="along"`` / ``"across"`` reproduce
that from ``Project.projected`` (``L_um`` / ``D_um``); ``"xy"`` is the camera
plane. One axis is ONE dimension, ``MSD = 2·D·t + 2·σ²``, so the fit divides by
``2·ndim`` and the table carries ``ndim`` rather than a caller remembering it.
Across a rod about a micron wide the curve flattens within a few lags: that is
the cell wall, the same confinement that inflates the intercept (CLAUDE.md, the
σ-is-about-17-nm section), made visible on one axis.

v2's own per-track fit
----------------------
``per_track_fits`` is ``fitMSDTracks.m`` in its Linear mode: tracks of at least
``min_length`` localisations (v2's default 9), each track's own MSD weighted by
``1/std`` (``linWfit``), the last lag excluded by default. In v2's default "specific
τ" mode ``calcMSD_gui.m:102-104`` first CUTS every track's curve to its first
``min_length`` rows (lags 0 to 8), so the fit uses lags 1 to 7, not all of a long
track's lags; ``specific_tau=False`` is v2's "all τ" mode. v2's
"Localization error estimation" panel is the mean of ``sqrt(|b|/4)`` over the
tracks ticked in its table, and the table pre-ticks ``R² > 0.99`` with a positive
intercept (``fitMSDTracks.m``: ``check_``) -- so by default the ``abs()`` never
sees a negative intercept: the negatives are DROPPED by the tick rule instead,
which truncates the noise at zero and biases the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .observables import ObservableError, squared_displacements
from .sqd import BOOTSTRAP_UNITS, FEW_UNITS

__all__ = [
    "MSDError",
    "MSDFit",
    "msd_per_track",
    "ensemble_msd",
    "fit_msd",
    "bootstrap_msd",
    "per_track_fits",
    "legacy_ensemble_window",
    "DEFAULT_FIT_LAGS",
    "DEFAULT_MAX_LAG",
    "LEGACY_MIN_LENGTH",
    "LEGACY_FIT_LAGS",
    "LEGACY_TICK_R2",
    "MSD_DIRECTIONS",
]

#: Lags used for the linear fit. Only the early portion is reliable.
DEFAULT_FIT_LAGS = 4
#: Lags computed for the curve itself (the tail is plotted, not fitted).
DEFAULT_MAX_LAG = 10

#: v2's MSD tab defaults (`SMTracker.m:441-444, 508`): "specific tau" 9 -- only tracks
#: of at least 9 localisations enter the curve, each CUT to its first 9 rows, lags 0-8
#: (`calcMSD_gui.m:41, 102-104`) -- and the "Linear Fit" box 9, unweighted
#: (`getFitMSD_gui.m`), the per-track fit's minimum length following the same box.
#: The BOX says 9; the fit uses lags 1-8. See `legacy_ensemble_window`.
LEGACY_MIN_LENGTH = 9
LEGACY_FIT_LAGS = 9
#: v2's table pre-ticks a track for the localisation-error panel at R² above this AND a
#: positive intercept (`fitMSDTracks.m`, `check_`).
LEGACY_TICK_R2 = 0.99



def legacy_ensemble_window(min_length: int = LEGACY_MIN_LENGTH,
                           linear_fit: int = LEGACY_FIT_LAGS) -> dict:
    """The lags v2's ensemble "Linear Fit" really fits, in its default "specific τ" mode.

    ``calcMSD_gui.m:102-104`` cuts every track's curve to rows ``1:min_length``, lags 0 to
    ``min_length - 1``, so the pooled curve (``getMSD``) has ``min_length`` rows.
    ``getFitMSD_gui.m:8-14`` sets ``limit = linear_fit + 1`` and fits rows ``2:limit``,
    lags 1 to ``linear_fit``, inside a try; when that runs past the curve the catch fits
    rows ``2:end``, lags 1 to ``min_length - 1``. v2's own ``validate()``
    (``SMTracker.m:2410-2412``) clamps ``linear_fit`` to at most ``min_length``, so at
    ``linear_fit == min_length`` -- the default, 9 and 9 -- the try ALWAYS fails and the
    fit is lags 1 to 8. Returns ``n_lags`` (fitted), ``curve_lags`` (the curve's length
    after lag 0) and ``fell_back``.
    """
    if min_length < 2:
        raise MSDError(f"min_length must be >= 2, got {min_length}")
    linear_fit = min(int(linear_fit), int(min_length))       # validate(), SMTracker.m:2410
    curve_lags = int(min_length) - 1
    fell_back = linear_fit > curve_lags
    return {"n_lags": curve_lags if fell_back else linear_fit,
            "curve_lags": curve_lags, "fell_back": bool(fell_back),
            "linear_fit_box": int(linear_fit)}


#: ``"xy"`` the camera plane; ``"along"`` / ``"across"`` the cell's own axes (v2's x / y).
MSD_DIRECTIONS = ("xy", "along", "across")
_DIRECTION_COLUMN = {"along": "L_um", "across": "D_um"}

_NDIM = 2
_GROUP_COLUMNS = ["protein", "condition", "day", "movie", "cell"]


class MSDError(RuntimeError):
    """Raised when an MSD curve cannot be computed or fitted."""


@dataclass
class MSDFit:
    """A linear fit of the early MSD portion. ``D`` is apparent, not physical."""

    D_um2_per_s: float
    intercept_um2: float
    n_lags_used: int
    lag_times_s: np.ndarray
    weighted: bool
    r_squared: float
    alpha: float | None = None          # anomalous exponent, if fitted
    n_tracks: int = 0
    #: Dimensions the curve was built from: 2 in the camera plane, 1 along or across.
    ndim: int = _NDIM

    @property
    def localization_sigma_um(self) -> float | None:
        """σ from the intercept, ``sqrt(intercept / (2·ndim))``.

        ``None`` when the intercept is negative, which happens with motion blur
        or an over-corrected background and means the offset should not be read
        as a localization precision.
        """
        if self.intercept_um2 <= 0:
            return None
        return float(np.sqrt(self.intercept_um2 / (2 * self.ndim)))

    def summary(self) -> str:
        sigma = self.localization_sigma_um
        sigma_text = (
            f"σ ≈ {sigma * 1000:.0f} nm" if sigma is not None
            else f"intercept {self.intercept_um2:.2g} µm² ≤ 0 (not a σ)"
        )
        alpha_text = f", α = {self.alpha:.2f}" if self.alpha is not None else ""
        return (
            f"MSD fit over {self.n_lags_used} lag(s), "
            f"{'weighted' if self.weighted else 'unweighted'}: "
            f"D = {self.D_um2_per_s:.4g} µm²/s{alpha_text}, {sigma_text}, "
            f"R² = {self.r_squared:.4f}. Ensemble slope: not a single state's D."
        )


def msd_per_track(
    canonical: pd.DataFrame,
    *,
    max_lag: int = DEFAULT_MAX_LAG,
    pairing: str = "frame",
    direction: str = "xy",
) -> pd.DataFrame:
    """MSD per track and lag, from the shared squared-displacement observable.

    Returns one row per ``(uid, lag_frames)`` with ``msd_um2``, ``n_pairs`` and
    ``std_um2``, plus the grouping columns. Lags with no pairs are absent rather
    than zero-filled. ``direction`` (`MSD_DIRECTIONS`) picks the camera plane or one
    of the cell's own axes; ``.attrs["ndim"]`` says how many dimensions that is.
    """
    if max_lag < 1:
        raise MSDError(f"max_lag must be >= 1, got {max_lag}")
    source, ndim = _components(canonical, direction)

    frames = []
    for lag in range(1, max_lag + 1):
        try:
            displacements = squared_displacements(
                source, lag_frames=lag, pairing=pairing
            )
        except ObservableError:
            # No pair survives at this lag; the curve simply ends here.
            break
        grouped = displacements.groupby("uid", observed=True)
        table = grouped.agg(
            msd_um2=("sq_disp_um2", "mean"),
            std_um2=("sq_disp_um2", "std"),
            n_pairs=("sq_disp_um2", "size"),
            **{c: (c, "first") for c in _GROUP_COLUMNS},
        ).reset_index()
        table["lag_frames"] = lag
        table["lag_time_s"] = displacements.attrs["lag_time_s"]
        frames.append(table)

    if not frames:
        raise MSDError(
            "no MSD point could be computed: no track has two localizations at "
            "any lag. Check the input table."
        )
    out = pd.concat(frames, ignore_index=True)
    out.attrs["tau_s"] = out["lag_time_s"].min()
    out.attrs["pairing"] = pairing
    out.attrs["direction"] = direction
    out.attrs["ndim"] = ndim
    return out


def _components(canonical: pd.DataFrame, direction: str) -> tuple[pd.DataFrame, int]:
    """The table whose ``x_um``/``y_um`` the displacements are taken from, and its
    dimension. One axis of the cell is put in ``x_um`` with ``y_um`` zero, so the one
    squared-displacement function serves all three directions and pairs them alike."""
    if direction not in MSD_DIRECTIONS:
        raise MSDError(f"direction must be one of {MSD_DIRECTIONS}, got {direction!r}")
    if direction == "xy":
        return canonical, _NDIM
    column = _DIRECTION_COLUMN[direction]
    if column not in canonical.columns:
        raise MSDError(
            f"direction {direction!r} needs the cell's own coordinates ({column}); "
            "project the tracks onto their cells first (Project.projected)")
    values = canonical[column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise MSDError(f"{column} is not finite everywhere; drop rows without a "
                       "centreline before measuring along or across the cell")
    table = canonical.copy()
    table["x_um"] = values
    table["y_um"] = 0.0
    return table, 1


def ensemble_msd(per_track: pd.DataFrame, *, weighted: bool = True) -> pd.DataFrame:
    """Average the per-track MSD curves into one ensemble curve.

    Weighted by the number of contributing pairs (as the legacy computes but
    then ignores), so a lag averaged over many pairs counts for more. The
    returned ``sem`` is the spread **across tracks**, indicative only; honest
    intervals come from ``bootstrap_msd`` over cells or days.
    """
    required = {"uid", "lag_frames", "lag_time_s", "msd_um2", "n_pairs"}
    missing = required - set(per_track.columns)
    if missing:
        raise MSDError(f"not a per-track MSD table; missing {sorted(missing)}")
    if len(per_track) == 0:
        raise MSDError("per-track MSD table is empty")

    rows = []
    for lag, frame in per_track.groupby("lag_frames", observed=True, sort=True):
        values = frame["msd_um2"].to_numpy(dtype=float)
        weights = (
            frame["n_pairs"].to_numpy(dtype=float) if weighted
            else np.ones(len(frame), dtype=float)
        )
        total = weights.sum()
        if total <= 0:
            continue
        mean = float(np.average(values, weights=weights))
        variance = float(np.average((values - mean) ** 2, weights=weights))
        n_tracks = len(frame)
        rows.append(
            {
                "lag_frames": int(lag),
                "lag_time_s": float(frame["lag_time_s"].iloc[0]),
                "msd_um2": mean,
                "std_um2": float(np.sqrt(variance)),
                "sem_um2": float(np.sqrt(variance / n_tracks)) if n_tracks else np.nan,
                "n_tracks": n_tracks,
                "n_pairs": int(frame["n_pairs"].sum()),
            }
        )
    if not rows:
        raise MSDError("no lag had any weight; cannot build an ensemble MSD")
    out = pd.DataFrame(rows)
    out.attrs["weighted"] = weighted
    out.attrs["ndim"] = int(per_track.attrs.get("ndim", _NDIM))
    out.attrs["direction"] = per_track.attrs.get("direction", "xy")
    return out


def fit_msd(
    ensemble: pd.DataFrame,
    *,
    n_lags: int = DEFAULT_FIT_LAGS,
    weighted: bool = True,
    anomalous: bool = False,
    ndim: int | None = None,
) -> MSDFit:
    """Fit ``MSD = 2·ndim·D·t + 2·ndim·σ²`` over the first ``n_lags`` lags only.

    Parameters
    ----------
    ensemble
        Curve from ``ensemble_msd``.
    n_lags
        Number of lags to fit, from the shortest. Default 4. **Raises** if the
        curve is shorter: the legacy code silently widened to the whole curve
        instead, which is the bug this module exists to avoid.
    weighted
        Weight each lag by its pair count (default). The legacy fit was
        unweighted despite having the weights to hand.
    anomalous
        Also fit ``log MSD = α·log t + c`` to report the anomalous exponent α.
        α ≈ 1 is Brownian, α < 1 subdiffusive.
    ndim
        Dimensions of the curve: 2 in the camera plane, 1 along or across the cell.
        ``None`` reads the curve's own ``.attrs["ndim"]`` (2 when absent), so a
        one-axis curve is never divided as though it had two.

    Raises
    ------
    MSDError
        If fewer than ``n_lags`` lags exist, if fewer than 2 points remain, or
        if the fit is degenerate.
    """
    for column in ("lag_time_s", "msd_um2", "n_pairs"):
        if column not in ensemble.columns:
            raise MSDError(f"not an ensemble MSD table; missing '{column}'")
    if n_lags < 2:
        raise MSDError(f"n_lags must be >= 2 to fit a line, got {n_lags}")
    if ndim is None:
        ndim = int(ensemble.attrs.get("ndim", _NDIM))
    if ndim not in (1, 2):
        raise MSDError(f"ndim must be 1 or 2, got {ndim}")

    curve = ensemble.sort_values("lag_frames", kind="stable")
    if len(curve) < n_lags:
        raise MSDError(
            f"asked to fit {n_lags} lags but the curve has only {len(curve)}. "
            "Refusing to silently widen the fit window to the noisy tail "
            "(the legacy code did exactly that). Lower n_lags deliberately, or "
            "use longer tracks."
        )
    curve = curve.head(n_lags)

    t = curve["lag_time_s"].to_numpy(dtype=float)
    y = curve["msd_um2"].to_numpy(dtype=float)
    w = curve["n_pairs"].to_numpy(dtype=float) if weighted else np.ones(len(curve))
    if not np.isfinite(y).all() or not np.isfinite(t).all():
        raise MSDError("ensemble MSD contains non-finite values")
    if np.ptp(t) <= 0:
        raise MSDError("all fitted lags have the same lag time; cannot fit a slope")

    slope, intercept = _weighted_linear_fit(t, y, w)
    if slope <= 0:
        raise MSDError(
            f"fitted MSD slope is {slope:.4g} <= 0, so D would be non-positive. "
            "The curve is not diffusive over this window: inspect it rather "
            "than reporting a zero or negative D."
        )

    predicted = slope * t + intercept
    residual = np.sum(w * (y - predicted) ** 2)
    total = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    r_squared = float(1.0 - residual / total) if total > 0 else float("nan")

    alpha = None
    if anomalous:
        positive = (t > 0) & (y > 0)
        if positive.sum() < 2:
            raise MSDError("need at least two positive MSD points to fit α")
        alpha, _ = _weighted_linear_fit(
            np.log(t[positive]), np.log(y[positive]), w[positive]
        )
        alpha = float(alpha)

    return MSDFit(
        D_um2_per_s=float(slope / (2 * ndim)),
        intercept_um2=float(intercept),
        n_lags_used=int(n_lags),
        lag_times_s=t,
        weighted=weighted,
        r_squared=r_squared,
        alpha=alpha,
        n_tracks=int(curve["n_tracks"].max()) if "n_tracks" in curve else 0,
        ndim=int(ndim),
    )


def _weighted_linear_fit(x, y, w) -> tuple[float, float]:
    """Weighted least squares for ``y = m·x + b``. Returns ``(m, b)``."""
    w = np.asarray(w, dtype=float)
    if np.any(w < 0):
        raise MSDError("negative weights in the MSD fit")
    total = w.sum()
    if total <= 0:
        raise MSDError("all MSD fit weights are zero")
    mean_x = np.average(x, weights=w)
    mean_y = np.average(y, weights=w)
    variance = np.sum(w * (x - mean_x) ** 2)
    if variance <= 0:
        raise MSDError("no spread in lag times; cannot fit a slope")
    slope = np.sum(w * (x - mean_x) * (y - mean_y)) / variance
    return float(slope), float(mean_y - slope * mean_x)


def bootstrap_msd(
    canonical: pd.DataFrame,
    *,
    unit: str = "cell",
    n_boot: int = 200,
    n_lags: int = DEFAULT_FIT_LAGS,
    max_lag: int = DEFAULT_MAX_LAG,
    seed: int | None = 0,
    weighted: bool = True,
    pairing: str = "frame",
    direction: str = "xy",
    per_track: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Honest MSD error bars: resample independent units, recompute, refit.

    The regression's own standard error is not usable, consecutive MSD points
    share displacements and are strongly correlated. Units are resampled with
    replacement, exactly as in ``sqd.bootstrap_sqd``; ``.attrs`` flag a small
    unit count, where the percentile interval becomes quantized.

    EACH TRACK'S CURVE IS COMPUTED ONCE. A track's MSD depends on that track alone,
    so a resample is a weighted sum of per-UNIT totals, each unit counted as often as
    it was drawn -- the same ensemble the old route rebuilt from scratch per replicate
    (it re-keyed duplicated tracks and recomputed every displacement), with the same
    random draws, at a small fraction of the cost: 8.2 s for 20 replicates on the
    reference condition became well under a second for 200. A resample whose
    curve stops early ends there, as recomputing it would. ``per_track`` may be passed
    when the caller already holds it (it must come from the same table and settings).
    """
    if unit not in BOOTSTRAP_UNITS:
        raise MSDError(f"unit must be one of {BOOTSTRAP_UNITS}, got {unit!r}")
    if unit not in canonical.columns:
        raise MSDError(f"canonical table has no '{unit}' column")
    if canonical[unit].isna().any():
        raise MSDError(
            f"'{unit}' is unassigned for some tracks; bootstrapping over it would "
            "silently drop data."
        )
    if n_boot < 1:
        raise MSDError(f"n_boot must be >= 1, got {n_boot}")

    units = list(canonical.groupby(unit, observed=True).groups)
    if len(units) < 2:
        raise MSDError(
            f"only {len(units)} distinct {unit}(s); a bootstrap over one unit is "
            "meaningless."
        )
    if per_track is None:
        per_track = msd_per_track(canonical, max_lag=max_lag, pairing=pairing,
                                  direction=direction)
    ndim = int(per_track.attrs.get("ndim", _NDIM))
    per_track = per_track[per_track["lag_frames"] <= max_lag]

    # Per (unit, lag) totals: sum of w*msd, sum of w, pairs and tracks.
    lags = np.sort(per_track["lag_frames"].unique())
    lag_index = {int(lag): i for i, lag in enumerate(lags)}
    unit_index = {key: i for i, key in enumerate(units)}
    lag_time = (per_track.groupby("lag_frames")["lag_time_s"].first()
                .reindex(lags).to_numpy(dtype=float))
    u = per_track[unit].map(unit_index).to_numpy()
    if np.isnan(u.astype(float)).any():
        raise MSDError(f"the per-track table holds a {unit} the canonical table does not")
    u = u.astype(int)
    li = per_track["lag_frames"].map(lag_index).to_numpy(dtype=int)
    values = per_track["msd_um2"].to_numpy(dtype=float)
    pairs = per_track["n_pairs"].to_numpy(dtype=float)
    w = pairs if weighted else np.ones(len(per_track))
    shape = (len(units), len(lags))
    sum_wm, sum_w, sum_pairs, n_tracks = (np.zeros(shape) for _ in range(4))
    np.add.at(sum_wm, (u, li), w * values)
    np.add.at(sum_w, (u, li), w)
    np.add.at(sum_pairs, (u, li), pairs)
    np.add.at(n_tracks, (u, li), 1.0)

    rng = np.random.default_rng(seed)
    rows, failures, curves = [], 0, []
    for _ in range(n_boot):
        picked = rng.choice(units, size=len(units), replace=True)
        counts = np.bincount([unit_index[key] for key in picked], minlength=len(units))
        tracks = counts @ n_tracks
        present = tracks > 0
        # The curve ends at the first lag no drawn track reaches (msd_per_track's break).
        stop = int(np.argmin(present)) if not present.all() else len(lags)
        weight = (counts @ sum_w)[:stop]
        curve = pd.DataFrame({
            "lag_frames": lags[:stop],
            "lag_time_s": lag_time[:stop],
            "msd_um2": (counts @ sum_wm)[:stop] / np.where(weight > 0, weight, np.nan),
            "n_pairs": (counts @ sum_pairs)[:stop],
            "n_tracks": tracks[:stop],
        })
        try:
            fit = fit_msd(curve, n_lags=n_lags, weighted=weighted, ndim=ndim)
        except MSDError:
            failures += 1
            continue
        # THE RESAMPLE'S CURVE TOO (17 Oct 2026): each lag's ensemble MSD, NaN past where it stops -- what a
        # point's interval over cells is read from (`msd_view`'s bars under Uncertainty = Cells)
        full = np.full(len(lags), np.nan)
        full[:stop] = curve["msd_um2"].to_numpy(dtype=float)
        curves.append(full)
        rows.append(
            {
                "D_um2_per_s": fit.D_um2_per_s,
                "intercept_um2": fit.intercept_um2,
                "localization_sigma_um": fit.localization_sigma_um,
                "r_squared": fit.r_squared,
            }
        )

    if not rows:
        raise MSDError(
            f"every one of {n_boot} bootstrap replicates failed to fit the MSD."
        )
    out = pd.DataFrame(rows)
    out.attrs["curves"] = np.vstack(curves)
    out.attrs["curve_lag_s"] = lag_time.tolist()
    out.attrs["unit"] = unit
    out.attrs["n_units"] = len(units)
    out.attrs["n_requested"] = n_boot
    out.attrs["n_failed"] = failures
    out.attrs["few_units"] = len(units) < FEW_UNITS
    out.attrs["direction"] = direction
    out.attrs["caveat"] = (
        f"only {len(units)} {unit}(s): the percentile interval is coarse."
        if len(units) < FEW_UNITS else ""
    )
    return out


def per_track_fits(
    canonical: pd.DataFrame,
    *,
    min_length: int = LEGACY_MIN_LENGTH,
    exclude_last: bool = True,
    specific_tau: bool = True,
    pairing: str = "frame",
    direction: str = "xy",
) -> pd.DataFrame:
    """v2's per-track Linear fit, ``fitMSDTracks.m`` -> ``linWfit``, reproduced.

    Tracks of at least ``min_length`` localisations; each track's own MSD (lag 0
    excluded), weighted by ``1/std`` of that lag's squared displacements, the last lag
    dropped when ``exclude_last`` (v2's checkbox, on by default). ``specific_tau`` (v2's
    default, "specific τ" = ``min_length``) first cuts each curve to lags 1 to
    ``min_length - 1`` (``calcMSD_gui.m:102-104``, rows ``1:MinLength`` with lag 0 in
    row 1), so with 9 and the last lag dropped every track is fitted over lags 1 to 7;
    ``specific_tau=False`` is v2's "all τ" mode, every lag of the track. A lag resting on
    one pair has no spread: v2's ``std`` gives 0 and an infinite weight, which ``linWfit``
    replaces by ``1 / sum(1/w)`` over the finite ones of the (cut) curve -- reproduced,
    in v2's order: cut, replace, then drop the last lag.

    One row per track: ``uid``, ``n_localisations``, ``n_lags``, ``D_um2_per_s``,
    ``intercept_um2``, ``r_squared`` and ``ticked`` -- v2's table pre-ticks a track
    for the localisation-error panel at ``R² > LEGACY_TICK_R2`` with a POSITIVE
    intercept -- plus the grouping columns. ``D`` divides the slope by ``2·ndim``:
    v2 divided by 4 for its x and y as well (``fillMSDtable.m:50``), which halves a
    one-axis D; that is NOT reproduced.
    """
    from .canonical import without_attrs

    if min_length < 3:
        raise MSDError(f"min_length must be >= 3 to fit a line per track, got {min_length}")
    lengths = without_attrs(canonical).groupby("uid", observed=True).size()
    keep = lengths.index[lengths >= min_length]
    if len(keep) == 0:
        raise MSDError(f"no track reaches {min_length} localisations")
    subset = canonical[canonical["uid"].isin(keep)]
    # "specific tau": the curve stops at lag min_length - 1 (calcMSD_gui.m:102-104).
    longest = int(min_length) - 1 if specific_tau else int(lengths.max()) - 1
    curves = msd_per_track(subset, max_lag=longest, pairing=pairing, direction=direction)
    ndim = int(curves.attrs["ndim"])

    rows = []
    for uid, part in without_attrs(curves).groupby("uid", sort=False, observed=True):
        part = part.sort_values("lag_frames")
        t = part["lag_time_s"].to_numpy(dtype=float)
        y = part["msd_um2"].to_numpy(dtype=float)
        std = np.nan_to_num(part["std_um2"].to_numpy(dtype=float), nan=0.0)
        with np.errstate(divide="ignore"):
            w = 1.0 / std
        finite = np.isfinite(w)
        if not finite.all() and finite.any():
            w[~finite] = 1.0 / np.sum(1.0 / w[finite])
        keep_n = len(t) - (1 if exclude_last else 0)
        t, y, w = t[:keep_n], y[:keep_n], w[:keep_n]
        ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        t, y, w = t[ok], y[ok], w[ok]
        if len(t) < 2 or np.ptp(t) <= 0:
            continue
        slope, intercept = _weighted_linear_fit(t, y, w)
        predicted = slope * t + intercept
        total = np.sum(w * (y - np.average(y, weights=w)) ** 2)
        r2 = float(1.0 - np.sum(w * (y - predicted) ** 2) / total) if total > 0 else np.nan
        rows.append({
            "uid": uid,
            "n_localisations": int(lengths.loc[uid]),
            "n_lags": int(len(t)),
            "D_um2_per_s": float(slope / (2 * ndim)),
            "intercept_um2": float(intercept),
            "r_squared": r2,
            "ticked": bool(r2 > LEGACY_TICK_R2 and intercept > 0),
            **{c: part[c].iloc[0] for c in _GROUP_COLUMNS if c in part.columns},
        })
    if not rows:
        raise MSDError("no track produced a per-track fit")
    out = pd.DataFrame(rows)
    out.attrs.update({"min_length": int(min_length), "exclude_last": bool(exclude_last),
                      "specific_tau": bool(specific_tau),
                      "lags_fitted": (f"1-{int(min_length) - 1 - int(bool(exclude_last))}"
                                      if specific_tau else "all but the last"
                                      if exclude_last else "all"),
                      "direction": direction, "ndim": ndim,
                      "n_tracks_total": int(len(lengths)),
                      "n_tracks_long_enough": int(len(keep))})
    return out


# --------------------------------------------------------------------------- #
# Berglund motion-blur correction (for D_fast)
# --------------------------------------------------------------------------- #

#: Motion-blur coefficient for continuous illumination over the full exposure
#: (exposure time = frame time). Berglund (2010), R = 1/6.
BERGLUND_R_CONTINUOUS = 1.0 / 6.0


def berglund_correct_D(D_apparent, tau_s, *, R=BERGLUND_R_CONTINUOUS):
    """Correct an apparent single-lag D for motion blur (Berglund 2010).

    A camera integrates the molecule's position over the exposure, so a
    single-frame displacement is *smaller* than the instantaneous motion would
    give: the fast state is underestimated. For continuous illumination the
    measured single-lag mean-squared displacement relates to the true D by

        <r²>(τ)  =  2·d·D·τ·(1 - 2R)  +  4·σ²      (blur reduces the slope)

    so the blur-only relation (σ folded out) is
    ``D_apparent = D_true·(1 - 2R)`` and the correction is
    ``D_true = D_apparent / (1 - 2R)``. With R = 1/6 that is a factor 3/2 in the
    continuous-illumination limit; a shorter effective exposure (smaller R)
    corrects less.

    Not load-bearing for fractions: a fraction is set by the *split*, not the
    absolute D, and blur is a shared confound that cancels across conditions. It
    is offered for an honest D_fast. Validated against the simulator, whose blur
    knob (`ImagingSpec.n_subframes`) measures the bias directly.
    """
    import numpy as np

    D_apparent = np.asarray(D_apparent, dtype=float)
    if not 0.0 <= R < 0.5:
        raise MSDError(f"blur coefficient R must be in [0, 0.5), got {R}")
    if tau_s <= 0:
        raise MSDError(f"tau_s must be > 0, got {tau_s}")
    return D_apparent / (1.0 - 2.0 * R)


def berglund_correct_apparent_D(D_apparent, tau_s, *, sigma_um,
                                R=BERGLUND_R_CONTINUOUS):
    """Combined single-lag D correction (item C): static localization error THEN blur.

    A single-lag apparent D carries two shared, condition-independent biases (both
    quantified factorially in ``scripts/confinement_factorial.py``):

    - the static localization error adds ``σ²/τ`` to the apparent D (the fit models
      no error offset), and
    - motion blur multiplies the true D by ``(1 - 2R)``, the Berglund slope factor.

    Removing them in that order:

        ``D_true = (D_apparent - σ²/τ) / (1 - 2R)``

    With R = 1/6 (continuous illumination) this is ``(3/2)·(D_app - σ²/τ)``, the
    exact expression item C specifies. σ is a per-axis localization SD in µm.

    Blur is a SHARED confound that cancels in a between-condition fraction, so this
    is offered for an honest D *magnitude*, never for the fractions. The correction
    can drive a low apparent D at or below the measurement floor, the caller must
    render those as BOUNDS, not points (the value is no longer resolvable).
    """
    D_apparent = np.asarray(D_apparent, dtype=float)
    if not 0.0 <= R < 0.5:
        raise MSDError(f"blur coefficient R must be in [0, 0.5), got {R}")
    if tau_s <= 0:
        raise MSDError(f"tau_s must be > 0, got {tau_s}")
    if sigma_um < 0:
        raise MSDError(f"sigma_um must be ≥ 0, got {sigma_um}")
    return (D_apparent - sigma_um ** 2 / tau_s) / (1.0 - 2.0 * R)
