"""Spatial map rendering: the estimate and the display, kept apart.

Governing principle
-------------------
A map has exactly **two** numbers that set its real resolution, and both are
printed on the figure:

1. **bin size**, in physical µm in the normalized-cell frame;
2. **bandwidth h**, in physical µm, tied to the localization precision σ.

Rendered pixel density and colormap are cosmetic and must never imply
resolution finer than ``h``. Smoothness here comes from a *stated kernel*,
never from a spline through sparse bins.

Kept from the legacy ``plotHeatmap.m``
--------------------------------------
The model-cell coordinate frame (``l_norm``/``d_norm`` already are it) and the
two-path density-vs-value structure.

Five deliberate breaks from it
------------------------------
1. **No cubic upsample.** ``interp2(hist3(20x20), 5, 'cubic')`` turned ~40 real
   bins into ~1000, and cubic splines overshoot, producing densities below zero
   and above anything in the data. Here the count grid is Gaussian-smoothed with
   ``h = max(σ, bin half-width)``, and ``h`` is reported. Render at any pixel
   density you like: the effective resolution is the kernel, and the kernel is
   declared.
2. **Folding SUMS counts** (≡ mean density), never ``max``. Folding pools the
   symmetry-equivalent positions to cut noise, which is a pooling operation:
   summing quadruples the effective N per bin, which is the entire point.
   ``max`` picks the noisiest of the four corners, biases every pixel upward,
   and destroys comparability between folded and unfolded maps. (Folding the
   coordinates and then binning is identical to folding the count grid, given
   symmetric bins: that is how it is done here.)
3. **The boundary is enforced in data space, not painted over.** The legacy
   filled a white polygon *on top of* the interpolated density, so the spill
   past the cell was hidden rather than absent and the visible edge was not
   where the data ended. Here the kernel is reflected at the domain boundary,
   so no density crosses it and "hidden" and "absent" are the same thing.

   Note on the model-cell outline: because ``d_norm`` is normalized by each
   cell's **local** half-width, the cell in map coordinates *is* the rectangle
   ``l ∈ [0,1] × d ∈ [-1,1]``: the taper is already divided out. Clipping to a
   capsule outline here would delete real polar localizations, so the legacy
   cosmetic clip is retired rather than ported. Draw a capsule if you like the
   look; do not let it remove data.
4. **Minimum-count mask: undefined, not zero.** Bins below ``n_min`` (counted
   post-fold, pre-smooth) are set to NaN, are excluded from normalization, and
   are not fed to the kernel: the smoothing renormalizes by the surviving
   weight so neighbours are not diluted by a hole. ``n_min`` is stricter for
   value maps: a value bin is an average of noisy per-point quantities, so a
   one-point D bin is pure noise, whereas a one-count density bin is merely
   sparse.
5. **Deterministic, shared colour scale.** The legacy's ratcheting
   ``heatmapCaxis`` made the scale depend on which panel you drew first.
   ``shared_color_limits`` computes the scale over the whole set *before*
   rendering, and the limits are printed.

Two paths, one geometry
-----------------------
Density and value maps share the grid, the fold, the boundary handling, the
min-count mask and the reported ``(bin size, h, n_min)``. They differ only in
the reduction:

- **density**, summed counts → Gaussian-smoothed → normalized by total
  detections;
- **value**, count-weighted mean of a per-point quantity → Gaussian-smoothed
  with count as the weight → stricter ``n_min``.

Prefer enrichment for any claim
-------------------------------
Raw density is dominated by the projection path length through the cell centre.
``enrichment_grid`` divides by the matched spatial null (where localizations are
at all), which removes most of that: the same logic as the confinement
observed/chance/excess. Heatmap for orientation, enrichment for claims.

Invariants
----------
- ``bandwidth_um`` is either given or derived from σ; σ is never invented.
- Physical scales come from the projected table's cell geometry, or are given
  explicitly. Fails loud rather than assuming a cell size.
- Every grid carries a ``caption()`` with everything a reader needs to judge it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .maps import MapError, _axis_range

__all__ = [
    "RenderError",
    "MapGrid",
    "map_grid",
    "enrichment_grid",
    "shared_color_limits",
    "robust_color_limits",
    "DEFAULT_N_MIN",
]

#: Minimum bin counts. A value bin averages noisy per-point quantities, so it
#: needs far more support than a density bin, which is merely sparse.
DEFAULT_N_MIN = {"density": 5, "value": 20}


class RenderError(RuntimeError):
    """Raised when a map cannot be rendered honestly."""


@dataclass(frozen=True)
class MapGrid:
    """A rendered map plus everything needed to judge it.

    ``values`` is ``(n_d, n_l)`` with NaN where the bin is undefined, ready for
    ``imshow(..., extent=grid.extent, origin="lower")``.
    """

    values: np.ndarray
    counts: np.ndarray
    l_edges: np.ndarray
    d_edges: np.ndarray
    kind: str
    quantity: str | None
    bin_um: tuple[float, float]
    bandwidth_um: float
    sigma_um: float | None
    n_min: int
    folded: bool
    pooling: str
    n_localizations: int
    n_bins_masked: int
    color_limits: tuple[float, float]
    shared_scale: bool = False
    null: str | None = None
    label_source: str | None = None

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (
            float(self.l_edges[0]),
            float(self.l_edges[-1]),
            float(self.d_edges[0]),
            float(self.d_edges[-1]),
        )

    def with_color_limits(self, limits, *, shared: bool = True) -> "MapGrid":
        low, high = (float(limits[0]), float(limits[1]))
        if not high > low:
            raise RenderError(f"colour limits must increase, got {limits}")
        return MapGrid(**{**self.__dict__, "color_limits": (low, high),
                          "shared_scale": bool(shared)})

    def caption(self) -> str:
        """The block that must appear on the figure."""
        sigma = "unset" if self.sigma_um is None else f"{1000 * self.sigma_um:.0f} nm"
        fold = (f"folded ({self.pooling})" if self.folded else "unfolded")
        scale = "shared" if self.shared_scale else "per-panel"
        parts = [
            f"bin {1000 * self.bin_um[0]:.0f}x{1000 * self.bin_um[1]:.0f} nm",
            f"h {1000 * self.bandwidth_um:.0f} nm (sigma {sigma})",
            f"n_min {self.n_min} ({self.n_bins_masked} bins masked)",
            fold,
            f"scale [{self.color_limits[0]:.3g}, {self.color_limits[1]:.3g}] {scale}",
            f"N {self.n_localizations}",
        ]
        if self.null:
            parts.append(f"null: {self.null}")
        return " | ".join(parts)


# --------------------------------------------------------------------------- #
# Physical scale of the normalized axes
# --------------------------------------------------------------------------- #


def _physical_scale(
    labelled: pd.DataFrame,
    cell_length_um: float | None,
    cell_half_width_um: float | None,
) -> tuple[float, float]:
    """µm per unit of ``l_norm`` and of ``d_norm``.

    ``l_norm`` spans one cell length; ``d_norm = 1`` is one **local half**-width.
    Both are needed to state a bin size and a bandwidth in µm, and they differ
    by roughly 6x on real cells, so the kernel is genuinely anisotropic in
    normalized coordinates, and pretending otherwise would smooth ~6x too hard
    along the cell.
    """
    if cell_length_um is not None and cell_half_width_um is not None:
        if cell_length_um <= 0 or cell_half_width_um <= 0:
            raise RenderError("cell dimensions must be > 0")
        return float(cell_length_um), float(cell_half_width_um)

    geometry = labelled.attrs.get("geometry")
    if not geometry:
        raise RenderError(
            "no cell geometry on the table and no cell_length_um / "
            "cell_half_width_um given: a bin size in µm cannot be invented. "
            "Pass the projected table from cellcoords.project_tracks, or give "
            "the dimensions explicitly."
        )
    lengths = np.array([g.length_um for g in geometry.values()], dtype=float)
    widths = np.array([g.max_width_um for g in geometry.values()], dtype=float)
    return float(lengths.mean()), float(widths.mean() / 2.0)


def _resolve_bandwidth(
    bandwidth_um: float | None,
    sigma_um: float | None,
    bin_um: tuple[float, float],
) -> float:
    if bandwidth_um is not None:
        if bandwidth_um <= 0:
            raise RenderError(f"bandwidth_um must be > 0, got {bandwidth_um}")
        return float(bandwidth_um)
    if sigma_um is None:
        raise RenderError(
            "give either bandwidth_um or sigma_um: the smoothing scale follows the localisation "
            "precision and is never guessed (σ is itself uncertain, 30-76 nm)"
        )
    if sigma_um <= 0:
        raise RenderError(f"sigma_um must be > 0, got {sigma_um}")
    return float(max(sigma_um, max(bin_um) / 2.0))


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #


def _smooth(weighted: np.ndarray, valid: np.ndarray, sigma_bins) -> np.ndarray:
    """Gaussian-smooth, excluding invalid bins and renormalizing for them.

    ``mode="reflect"`` is the boundary condition: the kernel does not leak out
    of the cell, so no density is created outside it that a cosmetic mask would
    then have to hide.
    """
    numerator = gaussian_filter(np.where(valid, weighted, 0.0), sigma_bins, mode="reflect")
    denominator = gaussian_filter(valid.astype(float), sigma_bins, mode="reflect")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denominator > 0, numerator / denominator, np.nan)
    return out


def map_grid(
    labelled: pd.DataFrame,
    *,
    quantity: str | None = None,
    state: int | None = None,
    bins: tuple[int, int] = (30, 15),
    bandwidth_um: float | None = None,
    sigma_um: float | None = None,
    n_min: int | None = None,
    cell_length_um: float | None = None,
    cell_half_width_um: float | None = None,
) -> MapGrid:
    """Render a density map (``quantity=None``) or a value map.

    ``quantity`` names a per-localization column (an apparent D, a posterior, a
    speed); the map is then the **count-weighted mean** of it per bin, which is
    why it needs a stricter ``n_min`` than a density map.
    """
    for column in ("l_norm", "d_norm"):
        if column not in labelled.columns:
            raise MapError(f"missing '{column}'; project with cellcoords first")
    frame = labelled if state is None else labelled[labelled["state"] == state]
    if frame.empty:
        raise RenderError(f"no localizations for state {state}")
    if quantity is not None and quantity not in frame.columns:
        raise RenderError(f"no column {quantity!r} to average")

    kind = "density" if quantity is None else "value"
    if n_min is None:
        n_min = DEFAULT_N_MIN[kind]
    if n_min < 1:
        raise RenderError(f"n_min must be >= 1, got {n_min}")
    n_l, n_d = bins
    if n_l < 1 or n_d < 1:
        raise RenderError(f"bins must be positive, got {bins}")

    folded = bool(labelled.attrs.get("folded", False))
    l_low, l_high = _axis_range("l_norm", labelled)
    d_low, d_high = _axis_range("d_norm", labelled)
    l_edges = np.linspace(l_low, l_high, n_l + 1)
    d_edges = np.linspace(d_low, d_high, n_d + 1)

    length_um, half_width_um = _physical_scale(labelled, cell_length_um, cell_half_width_um)
    bin_um = (
        float((l_high - l_low) / n_l * length_um),
        float((d_high - d_low) / n_d * half_width_um),
    )
    bandwidth = _resolve_bandwidth(bandwidth_um, sigma_um, bin_um)
    # The kernel is isotropic in µm, so it is anisotropic in bins.
    sigma_bins = (bandwidth / bin_um[1], bandwidth / bin_um[0])   # (d, l) = row, col

    l_values = np.clip(frame["l_norm"].to_numpy(), l_low, l_high)
    d_values = np.clip(frame["d_norm"].to_numpy(), d_low, d_high)
    counts, _, _ = np.histogram2d(d_values, l_values, bins=[d_edges, l_edges])

    valid = counts >= n_min
    if not valid.any():
        raise RenderError(
            f"every bin holds fewer than n_min = {n_min} localizations "
            f"({int(counts.sum())} in {counts.size} bins): use fewer bins, fold "
            "the map, or drop to a 1-D profile"
        )

    if kind == "density":
        smoothed = _smooth(counts, valid, sigma_bins)
        total = np.nansum(np.where(valid, counts, 0.0))
        values = smoothed / total
    else:
        weights = np.zeros_like(counts)
        np.add.at(
            weights,
            (
                np.clip(np.digitize(d_values, d_edges) - 1, 0, n_d - 1),
                np.clip(np.digitize(l_values, l_edges) - 1, 0, n_l - 1),
            ),
            frame[quantity].to_numpy(dtype=float),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            per_bin_mean = np.where(counts > 0, weights / counts, 0.0)
        # Count-weighted: a bin with more support pulls the smoothed mean more.
        numerator = _smooth(per_bin_mean * counts, valid, sigma_bins)
        denominator = _smooth(counts, valid, sigma_bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(denominator > 0, numerator / denominator, np.nan)

    values = np.where(valid, values, np.nan)
    finite = values[np.isfinite(values)]
    limits = (float(np.nanmin(finite)), float(np.nanmax(finite)))
    if limits[1] <= limits[0]:
        limits = (limits[0], limits[0] + 1e-12)

    return MapGrid(
        values=values,
        counts=counts,
        l_edges=l_edges,
        d_edges=d_edges,
        kind=kind,
        quantity=quantity,
        bin_um=bin_um,
        bandwidth_um=bandwidth,
        sigma_um=sigma_um,
        n_min=int(n_min),
        folded=folded,
        pooling="sum" if folded else "none",
        n_localizations=int(len(frame)),
        n_bins_masked=int((~valid).sum()),
        color_limits=limits,
        label_source=labelled.attrs.get("label_source"),
    )


def enrichment_grid(
    labelled: pd.DataFrame,
    *,
    state: int,
    bins: tuple[int, int] = (30, 15),
    bandwidth_um: float | None = None,
    sigma_um: float | None = None,
    n_min: int | None = None,
    cell_length_um: float | None = None,
    cell_half_width_um: float | None = None,
) -> MapGrid:
    """One state's density divided by the density of *all* localizations.

    The matched spatial null is the overall distribution: it carries the same
    projection path-length bias, the same cell shapes and the same sampling, so
    dividing by it leaves the part that is about this state. 1.0 means no
    preference, and the colour scale is symmetric about it.
    """
    common = dict(
        bins=bins,
        bandwidth_um=bandwidth_um,
        sigma_um=sigma_um,
        n_min=n_min,
        cell_length_um=cell_length_um,
        cell_half_width_um=cell_half_width_um,
    )
    if "state" not in labelled.columns:
        raise RenderError("no 'state' column; label the table first")
    this = map_grid(labelled, state=state, **common)
    overall = map_grid(labelled, **common)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(overall.values > 0, this.values / overall.values, np.nan)

    finite = ratio[np.isfinite(ratio)]
    if finite.size == 0:
        raise RenderError("no bin survived in both the state map and the null")
    return MapGrid(
        **{
            **this.__dict__,
            "values": ratio,
            "kind": "enrichment",
            "quantity": f"state {state}",
            "color_limits": _ratio_limits(finite),
            "null": "all localizations (same cells, same projection)",
        }
    )


def _ratio_limits(values: np.ndarray) -> tuple[float, float]:
    """Limits symmetric about 1 **multiplicatively**: ``(1/r, r)``.

    Enrichment is a ratio, and "twice as likely" and "half as likely" are the
    same size of departure from 1. A linearly symmetric scale ``(1 - s, 1 + s)``
    gets this wrong twice over: it treats 2.0 as a bigger deviation than 0.5,
    and once any bin exceeds 2 it puts the lower half of the colour range below
    zero, where no ratio can ever live: half the colours wasted on impossible
    values. This is the only deliberate departure from the written spec's
    "symmetric around enrichment = 1", and it is the same symmetry, taken in the
    space the quantity actually lives in.
    """
    positive = values[values > 0]
    if positive.size == 0:
        raise RenderError("no positive enrichment value to scale")
    r = float(max(positive.max(), 1.0 / positive.min()))
    return (1.0 / r, r)


def shared_color_limits(grids, *, symmetric_about: float | None = None):
    """Colour limits over a whole set, computed BEFORE anything is rendered.

    The legacy scale ratcheted upward as panels were drawn, so it depended on
    draw order. Compute once, apply to all, print the limits.

    ``symmetric_about=1.0`` gives the multiplicative symmetry described in
    ``_ratio_limits``; any other value is treated as a linear centre.
    """
    grids = list(grids)
    if not grids:
        raise RenderError("no grids to scale")
    finite = np.concatenate([g.values[np.isfinite(g.values)].ravel() for g in grids])
    if finite.size == 0:
        raise RenderError("no finite bin across the set")
    if symmetric_about == 1.0:
        return _ratio_limits(finite)
    if symmetric_about is not None:
        spread = float(np.max(np.abs(finite - symmetric_about)))
        return (symmetric_about - spread, symmetric_about + spread)
    return (float(finite.min()), float(finite.max()))


def robust_color_limits(grids, *, symmetric_about=None, lo_pct=2.0, hi_pct=98.0):
    """Colour limits CLIPPED to the ``[lo_pct, hi_pct]`` percentiles of unmasked
    bins, so a single low-count spike cannot define the top of the scale.

    The same "compute once, apply to all, print it" discipline as
    ``shared_color_limits``, but percentile-robust: min–max lets one 2–3-count bin
    blow out the whole panel and crush the real signal into black. For an
    enrichment set (``symmetric_about=1.0``) the robust ratio ``r`` is
    ``max(p_hi, 1/p_lo)`` over the clipped values, kept multiplicatively symmetric.
    """
    grids = list(grids)
    if not grids:
        raise RenderError("no grids to scale")
    finite = np.concatenate([g.values[np.isfinite(g.values)].ravel() for g in grids])
    if finite.size == 0:
        raise RenderError("no finite bin across the set")
    lo = float(np.percentile(finite, lo_pct))
    hi = float(np.percentile(finite, hi_pct))
    if symmetric_about == 1.0:
        positive_hi = max(hi, 1e-6)
        positive_lo = lo if lo > 0 else float(np.min(finite[finite > 0], initial=positive_hi))
        r = float(max(positive_hi, 1.0 / max(positive_lo, 1e-6)))
        return (1.0 / r, r)
    if symmetric_about is not None:
        spread = float(max(abs(hi - symmetric_about), abs(lo - symmetric_about)))
        return (symmetric_about - spread, symmetric_about + spread)
    if not hi > lo:                                        # a flat panel
        hi = lo + 1e-9
    return (lo, hi)
