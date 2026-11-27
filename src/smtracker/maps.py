"""Spatial maps: where in the cell each diffusive state actually is.

Method-agnostic by design. A map needs two things: a **position** in cell
coordinates (``smtracker.cellcoords``) and a **label** per localization or step.
Where the label comes from is the caller's choice, and the choice matters:

``labels_from_appd``  (the legacy speed map, kept for continuity)
    One D per track, binned. This is what the old tool drew. Its weakness is
    structural: a molecule that binds and unbinds mid-track gets a single
    averaged D (a muddy middle value that belongs to neither state) so the map
    shows *where slow molecules are on average*, blurred by switching.

``labels_from_hmm``  (**the flagship**)
    A state per *step*, from posterior decoding. Because the HMM models the
    transitions, it can say "this molecule was bound **here**, then free
    **there**", so the map localises individual **binding events** rather than
    switching-blurred averages. This is the map that answers "where does it
    bind: poles, midcell, nucleoid?", which a per-track-D map structurally
    cannot.

``labels_from_sqd``
    Posterior state per step from the SQD mixture, switching-blind like appD,
    but per-step rather than per-track, so a useful middle ground.

Normalised coordinates make pooling legitimate
----------------------------------------------
Maps are drawn in ``(l_norm, d_norm)``: position along the cell in ``[0, 1]``
and across it in ``[-1, 1]``. Because both are normalised by each cell's own
length and local width, cells of different sizes can be summed into one map
without smearing, which is the whole reason the cell-coordinate step exists.

Honest limits
-------------
- A map of a few hundred localizations is a rumour. ``spatial_map`` reports the
  count per bin so sparse bins are visible, and ``occupancy_profile`` gives the
  1-D version, which needs far less data to be meaningful.
- Cells are symmetric under both flips (which pole is ``l = 0``, which side is
  ``d > 0`` are arbitrary), so an apparent asymmetry is almost always noise
  unless the cells were oriented by an independent marker. ``fold_map`` applies
  the symmetry explicitly.
- Density is *per unit normalised area*, not per µm²; comparing across cell
  sizes needs care.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "MapError",
    "labels_from_hmm",
    "labels_from_sqd",
    "labels_from_appd",
    "labels_from_confinement",
    "labels_from_clusters",
    "spatial_map",
    "occupancy_profile",
    "fold_map",
    "state_enrichment",
]

_COORDS = ("l_norm", "d_norm")


class MapError(RuntimeError):
    """Raised when a spatial map cannot be built."""


def labels_from_hmm(projected: pd.DataFrame, fit, *, min_posterior: float = 0.0):
    """Per-STEP state labels by posterior decoding, the flagship map input.

    ``min_posterior`` drops ambiguous steps; raising it trades coverage for
    confidence, and the fraction kept is reported.
    """
    from .hmm import decode_states

    decoded = decode_states(projected, fit)
    missing = [c for c in _COORDS if c not in decoded.columns]
    if missing:
        raise MapError(
            f"decoded steps lack {missing}; project the canonical table with "
            "cellcoords.project_tracks before decoding, so positions travel with "
            "the states."
        )
    kept = decoded[decoded["posterior"] >= min_posterior].copy()
    if kept.empty:
        raise MapError(f"no step reached a posterior of {min_posterior}")
    # Cell geometry must survive labelling: the renderer needs it to state a bin
    # size in µm, and losing it here would only surface as a failure much later.
    kept.attrs.update(projected.attrs)
    kept.attrs["label_source"] = "hmm"
    kept.attrs["fraction_kept"] = len(kept) / len(decoded)
    kept.attrs["n_states"] = fit.n_states
    return kept


def per_spot_steps(steps: pd.DataFrame, fit) -> tuple:
    """``(steps, noise)`` under a PER-SPOT fit (`sqd.SQDFit.per_spot_precision`; 6 Nov 2026): the steps whose spots
    carry a precision (the rest left out, as the fit left them; above `precision.MAX_MISSING_SHARE` refused) and each
    one's localisation noise through the fit's OWN calibration (``precision_calibration``), so every map labels its
    steps with the noise the fit was made with. ``(steps, None)`` for any other fit."""
    if not getattr(fit, "per_spot_precision", False):
        return steps, None
    calibration = getattr(fit, "precision_calibration", None)
    if calibration is None:
        raise MapError("a per-spot fit made from bare step noise carries no calibration: the map cannot work out its "
                       "own steps' noise (fit with precision=...)")
    from .precision import Calibration, PrecisionError, step_noise, usable_steps

    try:
        ok = usable_steps(steps)
        kept = steps.loc[ok] if not ok.all() else steps
        return kept, step_noise(kept, Calibration(c_nm=float(calibration[0]), p=float(calibration[1]),
                                                  source="the fit's"))
    except PrecisionError as exc:
        raise MapError(str(exc)) from exc


def labels_from_sqd(projected: pd.DataFrame, fit, *, lag_frames=1):
    """Per-step posterior state from an SQD mixture (switching-blind, per-step). ``lag_frames`` a number of frames, or
    "1-4" for lags 1-4 together (`observables.displacements_at`: each jump's r^2 divided by its lag, so the joint
    fit's one tau reads every lag's jumps)."""
    from .observables import displacements_at

    steps = displacements_at(projected, lag_frames)
    missing = [c for c in _COORDS if c not in steps.columns]
    if missing:
        raise MapError(f"observable lacks {missing}; project the table first")

    from .sqd import posteriors

    # `sqd.posteriors`, the one formula (the immobile population of exact-zero jumps included); under a per-spot fit
    # each step with its own noise (`per_spot_steps`)
    steps, noise = per_spot_steps(steps, fit)
    values = steps["sq_disp_um2"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        posterior = posteriors(values, fit.D, fit.fractions, fit.tau_lag_s, step_noise=noise)
    if not np.all(np.isfinite(posterior)):
        raise MapError("degenerate SQD responsibilities; check the fit")

    out = steps.copy()
    out["state"] = np.argmax(posterior, axis=1) + 1
    out["posterior"] = posterior.max(axis=1)
    out.attrs.update(projected.attrs)
    out.attrs["label_source"] = "sqd"
    out.attrs["n_states"] = len(fit.D)
    return out


def labels_from_appd(projected: pd.DataFrame, *, n_bins: int = 3, n_steps: int = 4):
    """Per-TRACK D bins, the legacy speed map, kept for continuity.

    Structurally switching-blurred: one D per track means a molecule that binds
    mid-track lands in a middle bin belonging to neither state. Use
    ``labels_from_hmm`` when the question is *where binding happens*.
    """
    from .appd import apparent_D

    table = apparent_D(projected, n_steps=n_steps, mode="split")
    if table.empty:
        raise MapError("no sub-track yielded an apparent D")

    per_track = table.groupby("uid")["appd_um2_per_s"].mean()
    edges = np.quantile(per_track, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-12
    bins = pd.Series(
        np.digitize(per_track, edges[1:-1], right=True) + 1, index=per_track.index
    )

    out = projected.copy()
    out["state"] = out["uid"].map(bins)
    out["posterior"] = np.nan            # a bin is not a probability
    out = out[out["state"].notna()]
    if out.empty:
        raise MapError("no localization could be labelled")
    out["state"] = out["state"].astype(int)
    out.attrs["label_source"] = "appd"
    out.attrs["n_states"] = n_bins
    out.attrs["bin_edges"] = edges
    return out


def labels_from_confinement(
    projected: pd.DataFrame, *, radius_um: float, min_steps: int = 3, gap_mode: str = "legacy"
):
    """Per-localization confinement labels, state 1 = in a dwell zone, 2 = free.

    A geometric detector, not a model: it asks whether the molecule stayed
    inside a circle, so it needs no diffusive-state fit. That independence is
    the point: it is a cross-check on ``labels_from_hmm``, arriving at "where
    does it stop" from a different direction. Its two parameters are choices,
    not measurements: run ``confinement.radius_sensitivity`` before letting a
    map built this way carry a claim.
    """
    from .confinement import label_confinement

    labelled = label_confinement(
        projected, radius_um=radius_um, min_steps=min_steps, gap_mode=gap_mode
    )
    missing = [c for c in _COORDS if c not in labelled.columns]
    if missing:
        raise MapError(f"table lacks {missing}; project with cellcoords first")

    out = labelled.copy()
    out["state"] = np.where(out["confined"], 1, 2)
    out["posterior"] = np.nan            # a geometric flag is not a probability
    out.attrs.update(labelled.attrs)
    out.attrs["label_source"] = "confinement"
    out.attrs["n_states"] = 2
    return out


def labels_from_clusters(projected: pd.DataFrame, report):
    """Per-TRACK cluster labels, the legacy per-cluster heatmap.

    Switching-blurred in the same structural way as ``labels_from_appd``: one
    label per track. And a cluster is a *curve shape*, not a diffusive state,
    only draw this once ``clustering.select_k`` has shown the structure exceeds
    the free-Brownian null, or the map shows where k-means split noise.
    """
    out = projected.copy()
    out["state"] = out["uid"].map(report.labels)
    out["posterior"] = np.nan            # a cluster index is not a probability
    out = out[out["state"].notna()]
    if out.empty:
        raise MapError("no localization belongs to a clustered track")
    out["state"] = out["state"].astype(int)
    out.attrs.update(projected.attrs)
    out.attrs["label_source"] = "clusters"
    out.attrs["n_states"] = report.k
    return out


def _axis_range(axis: str, labelled: pd.DataFrame) -> tuple[float, float]:
    """Support of a coordinate, honouring whether the map has been folded.

    Folding halves the support (``l`` becomes 0..0.5 pole-to-midcell, ``d``
    becomes 0..1 axis-to-membrane). Binning over the unfolded range afterwards
    would leave half the bins empty and inflate every enrichment value, so the
    flag is respected rather than assumed.
    """
    folded = bool(labelled.attrs.get("folded", False))
    if axis == "l_norm":
        return (0.0, 0.5) if folded else (0.0, 1.0)
    return (0.0, 1.0) if folded else (-1.0, 1.0)


def spatial_map(
    labelled: pd.DataFrame,
    *,
    state: int | None = None,
    bins: tuple[int, int] = (20, 10),
    normalize: bool = True,
) -> pd.DataFrame:
    """2-D density in normalised cell coordinates, optionally for one state.

    Returns a long table (``l_bin``, ``d_bin``, ``l_center``, ``d_center``,
    ``count``, ``density``) rather than a matrix, so it plots directly and the
    per-bin ``count`` stays visible, a density with ``count = 3`` is a rumour.
    """
    for column in _COORDS:
        if column not in labelled.columns:
            raise MapError(f"missing '{column}'; project with cellcoords first")
    l_low, l_high = _axis_range("l_norm", labelled)
    d_low, d_high = _axis_range("d_norm", labelled)
    frame = labelled if state is None else labelled[labelled["state"] == state]
    if frame.empty:
        raise MapError(f"no localizations for state {state}")

    n_l, n_d = bins
    if n_l < 1 or n_d < 1:
        raise MapError(f"bins must be positive, got {bins}")

    l_edges = np.linspace(l_low, l_high, n_l + 1)
    d_edges = np.linspace(d_low, d_high, n_d + 1)
    counts, _, _ = np.histogram2d(
        np.clip(frame["l_norm"].to_numpy(), l_low, l_high),
        np.clip(frame["d_norm"].to_numpy(), d_low, d_high),
        bins=[l_edges, d_edges],
    )

    l_centers = (l_edges[:-1] + l_edges[1:]) / 2
    d_centers = (d_edges[:-1] + d_edges[1:]) / 2
    rows = []
    total = counts.sum()
    for i, lc in enumerate(l_centers):
        for j, dc in enumerate(d_centers):
            rows.append(
                {
                    "l_bin": i,
                    "d_bin": j,
                    "l_center": float(lc),
                    "d_center": float(dc),
                    "count": int(counts[i, j]),
                    "density": float(counts[i, j] / total) if normalize and total else float(counts[i, j]),
                }
            )
    out = pd.DataFrame(rows)
    out.attrs["state"] = state
    out.attrs["n_localizations"] = int(total)
    out.attrs["label_source"] = labelled.attrs.get("label_source")
    if total < 500:
        out.attrs["warning"] = (
            f"only {int(total)} localizations across {n_l * n_d} bins: too sparse "
            "for a 2-D map; use occupancy_profile instead."
        )
    return out


def occupancy_profile(
    labelled: pd.DataFrame, *, axis: str = "l_norm", bins: int = 20
) -> pd.DataFrame:
    """1-D profile along or across the cell, per state.

    Needs far less data than a 2-D map, so this is usually the honest figure.
    ``axis="l_norm"`` is pole-to-pole; ``"d_norm"`` is axis-to-membrane.
    """
    if axis not in _COORDS:
        raise MapError(f"axis must be one of {_COORDS}, got {axis!r}")
    if axis not in labelled.columns:
        raise MapError(f"missing '{axis}'; project with cellcoords first")
    if "state" not in labelled.columns:
        raise MapError("no 'state' column; label the table first")

    low, high = _axis_range(axis, labelled)
    edges = np.linspace(low, high, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    rows = []
    for state, frame in labelled.groupby("state", observed=True):
        values = np.clip(frame[axis].to_numpy(), low, high)
        counts, _ = np.histogram(values, bins=edges)
        total = counts.sum()
        for center, count in zip(centers, counts):
            rows.append(
                {
                    "state": int(state),
                    axis: float(center),
                    "count": int(count),
                    "fraction": float(count / total) if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def fold_map(labelled: pd.DataFrame) -> pd.DataFrame:
    """Apply the cell's symmetries: fold ``l`` about midcell and ``d`` about 0.

    Which pole is ``l = 0`` and which side is ``d > 0`` are arbitrary unless the
    cells were oriented by an independent marker, so an unfolded map's apparent
    asymmetry is almost always noise. Folding doubles the counts per bin and
    removes a class of spurious asymmetry claims.
    """
    for column in _COORDS:
        if column not in labelled.columns:
            raise MapError(f"missing '{column}'")
    out = labelled.copy()
    out["l_norm"] = 0.5 - np.abs(out["l_norm"] - 0.5)   # 0 = pole, 0.5 = midcell
    out["d_norm"] = np.abs(out["d_norm"])               # 0 = axis, 1 = membrane
    out.attrs.update(labelled.attrs)
    out.attrs["folded"] = True
    return out


def state_enrichment(
    labelled: pd.DataFrame, *, axis: str = "l_norm", bins: int = 10
) -> pd.DataFrame:
    """Is a state enriched anywhere, relative to where molecules are at all?

    A state can look polar simply because *every* molecule is polar. This
    divides each state's profile by the profile of all localizations, so a value
    above 1 means genuine enrichment beyond the overall distribution. That is
    the quantity a "binding happens at the poles" claim needs.
    """
    profile = occupancy_profile(labelled, axis=axis, bins=bins)
    overall = (
        profile.groupby(axis, observed=True)["count"].sum().rename("total").reset_index()
    )
    merged = profile.merge(overall, on=axis, how="left")
    # Sum the per-bin totals ONCE, not once per state: `merged` repeats each
    # bin for every state, so summing there would divide the expected share by
    # the number of states and inflate every enrichment value.
    grand_total = float(overall["total"].sum())
    if grand_total <= 0:
        raise MapError("no localizations to compute enrichment from")

    expected_share = merged["total"] / grand_total
    state_totals = merged.groupby("state", observed=True)["count"].transform("sum")
    with np.errstate(divide="ignore", invalid="ignore"):
        observed_share = merged["count"] / state_totals
        merged["enrichment"] = np.where(
            expected_share > 0, observed_share / expected_share, np.nan
        )
    return merged[["state", axis, "count", "total", "enrichment"]]


def state_map_from_projected(projected, *, n_states: int = 2, state=None,
                             bins=(20, 10), normalize: bool = True,
                             localization_sigma_um: float = 0.0,
                             n_starts: int = 12):
    """The catalogue target for the flagship map: a projected table in, the grid out.

    `spatial_map` takes an already-LABELLED table, which is the output of fitting an HMM
    and decoding it. Nothing ran those steps for it, so the entry resolved, passed every
    signature check, and raised the moment anyone clicked it. This runs the chain.

    The labels come from the HMM, not from the SQD fit, and the difference is the entry's
    whole claim: per-STEP posterior decoding localises the individual bound segments,
    where a per-track average is switching-blurred and puts a binding molecule at a muddy
    middle value. Measured on the reference data, the HMM map has about 2.7x the radial
    dynamic range of the per-track appD map.

    `localization_sigma_um` defaults to zero on purpose. A sigma-corrected HMM is not
    robust to the value - the slow fraction rises by tens of points between 0 or 30 nm and
    76 nm, where the slow state is entirely explained by localisation noise - and
    sigma is unmeasured in this project. Zero keeps the D values APPARENT and comparable
    with the SQD fit; raising it is a sensitivity check, not a correction.
    """
    from .hmm import decode_states, fit_hmm, step_sequences

    tau = projected.attrs.get("tau_s")
    if tau is None:
        params = projected.attrs.get("params")
        if params is None:
            raise MapError(
                "no frame interval: the table carries neither 'tau_s' nor "
                "AcquisitionParams in .attrs, and every rate and D depends on it")
        tau = float(params.frame_interval_ms) / 1000.0

    fit = fit_hmm(step_sequences(projected), n_states=n_states, tau_s=float(tau),
                  localization_sigma_um=localization_sigma_um, n_starts=n_starts)
    labelled = decode_states(projected, fit)
    grid = spatial_map(labelled, state=state, bins=bins, normalize=normalize)
    grid.attrs["hmm_D"] = list(map(float, fit.D))
    grid.attrs["hmm_occupancy"] = list(map(float, fit.occupancy))
    grid.attrs["localization_sigma_um"] = float(localization_sigma_um)
    grid.attrs["step_position"] = "start"
    return grid
