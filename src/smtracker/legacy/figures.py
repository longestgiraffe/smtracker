"""The legacy figures for items 2, 3 and 4. Drawing only; nothing here computes.

Why this is one module. Every legacy figure must carry the same four things, and putting
the drawing in one place is what stops one of them being forgotten on one figure:

1. a title marked ``legacy``, through ``legacy_title``;
2. its own limitation line, from ``limitations.limitation_line``, filled with numbers
   measured on the group being drawn;
3. **both counts, never their difference** -- N tracks and N localisations are printed
   side by side because the legacy printed one number that was neither;
4. the vocabulary gate, ``assert_neutral(fig)``, which reads all three title artists.

Reproducing the old METHOD does not license reproducing old LABELS, so a legacy figure is
held to the same vocabulary rule as every other figure in this project.

The colour scale is computed over the whole panel set before any panel is drawn. The
legacy raises ``params.heatmapCaxis`` when a panel exceeds it and never lowers it, and it
does not redraw the panels already on screen (``plotHeatmap.m:65``), so the scale a panel
carries depends on the order the panels were clicked. ``panel_limits`` takes every grid
first and returns one pair; the applied limits are printed on every panel either way.

Figures go to ``legacy_out/figures/`` and nowhere else, and the name of each carries the
word ``legacy``. Both are enforced by ``legacy_path`` rather than by convention.
"""
from __future__ import annotations

import numpy as np

from .guard import assert_neutral, legacy_path, legacy_title
from .limitations import limitation_line

__all__ = ["FigureError", "panel_limits", "draw_speedmap", "draw_sda_maps",
           "draw_sla_classification", "CAPTION_SEPARATOR"]

CAPTION_SEPARATOR = chr(10) * 2

#: One warm ramp, used for every legacy map so two of them can be read against each other.
_CMAP = "magma_r"
_INK = "#0b0b0b"
_NOTE = "#52514e"
_ACCENT = "#D47B59"


class FigureError(RuntimeError):
    """Raised when a legacy figure cannot be drawn honestly."""


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def panel_limits(grids, *, percentile: float = 99.0) -> tuple:
    """One colour range for a whole set of panels, computed BEFORE any is drawn.

    The legacy scale grows panel by panel and never shrinks, so two panels in one figure
    can sit on different scales and a reader cannot compare them by eye. Taking every grid
    first is the fix, and it has to happen before drawing rather than during it.

    A percentile rather than the maximum, because one bin holding a single large value
    would otherwise set the scale for everything.
    """
    values = []
    for grid in grids:
        arr = np.asarray(grid, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)
    if not values:
        raise FigureError("every panel is empty; there is no scale to compute")
    stacked = np.concatenate(values)
    hi = float(np.percentile(stacked, percentile))
    lo = float(stacked.min())
    if not np.isfinite(hi) or hi <= lo:
        hi = float(stacked.max())
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi


def _counts_line(n_tracks, n_localisations, extra=""):
    """BOTH counts, named, never their difference.

    ``plotSpeedmap.m:182`` prints ``size(normTracks,1)`` after one padding row per track
    has been appended, so its single number is localisations PLUS tracks and is neither
    quantity. Printing both named is the fix.
    """
    line = f"N tracks {n_tracks}   N localisations {n_localisations}"
    return f"{line}   {extra}" if extra else line


def _footer(fig, lines, *, top=0.30, step=0.048):
    for i, text in enumerate(lines):
        if text:
            fig.text(0.02, top - step * i, str(text), fontsize=4.6, color=_NOTE,
                     wrap=True)


def _caption(name, blocks, root=None):
    path = legacy_path("figures", name, root=root)
    path.write_text(CAPTION_SEPARATOR.join(str(b) for b in blocks) + chr(10),
                    encoding="utf-8")
    return path


# ------------------------------------------------------------------- item 2, speed map
def draw_speedmap(census: dict, fits, *, enzyme: str, condition: str, clim=None,
                  trim=None, root=None):
    """The apparent-D speed map: the per-track D distribution and the track census.

    ``clim`` is the ``--speedmap-clim`` pair. The legacy calls ``caxis(ax,'auto')`` after
    every draw (``plotSpeedmap.m:184``), so each panel is scaled to its own maximum and two
    panels can never be compared; passing a pair applies one scale to both. The default
    stays automatic so the default figure looks like the legacy figure, and the applied
    limits are printed either way.
    """
    plt = _mpl()
    d = np.asarray(fits["appd_um2_per_s"], dtype=float)
    d = d[np.isfinite(d) & (d > 0)]
    if not d.size:
        raise FigureError(f"{enzyme}/{condition}: no track produced an apparent D")

    lo, hi = (float(clim[0]), float(clim[1])) if clim is not None else (d.min(), d.max())
    scale_source = "supplied, one scale for every panel" if clim is not None else (
        "automatic, this panel's own range: the legacy behaviour, and two panels drawn "
        "this way cannot be compared")

    fig = plt.figure(figsize=(120 / 25.4, 100 / 25.4))
    ax = fig.add_axes((0.13, 0.44, 0.55, 0.48))
    bins = np.logspace(np.log10(max(d.min(), 1e-4)), np.log10(d.max()), 40)
    ax.hist(d, bins=bins, color=_ACCENT, edgecolor=_INK, linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("apparent D per track (um^2/s)", fontsize=7)
    ax.set_ylabel("tracks", fontsize=7)
    ax.set_title(legacy_title(f"appD speed map, {enzyme} {condition}"), loc="left",
                 fontsize=8)
    ax.tick_params(labelsize=6)
    ax.axvline(lo, color=_INK, lw=0.6, ls=":")
    ax.axvline(hi, color=_INK, lw=0.6, ls=":")

    bar = fig.add_axes((0.75, 0.44, 0.20, 0.48))
    kept, total = census["n_tracks_out"], census["n_tracks_in"]
    bar.bar([0, 1], [total, kept], color=[_NOTE, _ACCENT], edgecolor=_INK, linewidth=0.3)
    bar.set_xticks([0, 1])
    bar.set_xticklabels(["all", f">= {census['legacy_min_length']}"], fontsize=6)
    bar.set_ylabel("tracks", fontsize=7)
    bar.set_title("track census", loc="left", fontsize=7)
    bar.tick_params(labelsize=6)

    _footer(fig, [
        _counts_line(census["n_tracks_in"], census["localisations_in"],
                     f"drawn from {kept} tracks, {census['percent_kept']:.1f} per cent"),
        f"colour limits {lo:.4g} to {hi:.4g} um^2/s, {scale_source}",
        (f"contour trim: {trim['tracks_partially_trimmed']} tracks shortened, "
         f"{trim['localisations_removed_from_surviving_tracks']} localisations removed, "
         f"{trim['tracks_that_became_empty']} tracks emptied. {trim['direction']}"
         if trim else ""),
        limitation_line("speedmap", min_length=census["legacy_min_length"],
                        kept=kept, total=total,
                        percent_kept=census["percent_kept"]),
    ])
    assert_neutral(fig)

    stem = f"legacy_speedmap_{enzyme.lower()}_{condition.lower()}"
    _caption(f"{stem}_caption.txt", [
        legacy_title(f"appD speed map, {enzyme} {condition}"),
        _counts_line(census["n_tracks_in"], census["localisations_in"]),
        census["which_binds"],
        census["two_effects"],
        limitation_line("speedmap", min_length=census["legacy_min_length"], kept=kept,
                        total=total, percent_kept=census["percent_kept"]),
    ], root=root)
    path = legacy_path("figures", f"{stem}.png", root=root)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


# -------------------------------------------------------------------- item 3, SDA maps
def draw_sda_maps(location: dict, value: dict, *, enzyme: str, condition: str,
                  bins: dict, provenance: dict, contrast: dict, clim=None,
                  fold: dict | None = None, root=None):
    """The two SDA maps side by side: a count per bin, and the per-jump apparent D.

    Nothing is interpolated. The legacy up-scales a 20-node grid to 609 by ``interp2``
    (``plotHeatmap.m:17``), over a short-axis bin that is already below one camera pixel,
    so the added detail is empty. The bin size in nanometres is printed on the figure, per
    axis, because the verdict differs per axis.
    """
    plt = _mpl()
    counts = np.asarray(location["counts"], dtype=float)
    mean = np.asarray(value["mean"], dtype=float)
    if not np.isfinite(mean).any():
        raise FigureError(f"{enzyme}/{condition}: every bin of the value map is empty")

    lo, hi = (float(clim[0]), float(clim[1])) if clim is not None else panel_limits([mean])
    extent = [location["x_edges"][0], location["x_edges"][-1],
              location["y_edges"][0], location["y_edges"][-1]]

    fig = plt.figure(figsize=(150 / 25.4, 92 / 25.4))
    ax1 = fig.add_axes((0.07, 0.50, 0.38, 0.40))
    im1 = ax1.imshow(counts.T, origin="lower", extent=extent, aspect="auto",
                     cmap=_CMAP, interpolation="nearest")
    ax1.set_title(legacy_title(f"SDA location, {enzyme} {condition}"), loc="left",
                  fontsize=8)
    fig.colorbar(im1, ax=ax1, fraction=0.04).ax.tick_params(labelsize=5)

    ax2 = fig.add_axes((0.56, 0.50, 0.38, 0.40))
    im2 = ax2.imshow(np.ma.masked_invalid(mean).T, origin="lower", extent=extent,
                     aspect="auto", cmap=_CMAP, vmin=lo, vmax=hi,
                     interpolation="nearest")
    ax2.set_title(legacy_title(f"SDA apparent D, {enzyme} {condition}"), loc="left",
                  fontsize=8)
    fig.colorbar(im2, ax=ax2, fraction=0.04).ax.tick_params(labelsize=5)

    for ax in (ax1, ax2):
        ax.set_xlabel("position along the cell (normalised)", fontsize=6.5)
        ax.set_ylabel("position across the cell", fontsize=6.5)
        ax.tick_params(labelsize=5.5)

    long_px = bins["long_axis_data_bin_nm"] / bins["pixel_nm"]
    short_px = bins["short_axis_data_bin_nm"] / bins["pixel_nm"]
    _footer(fig, [
        _counts_line(provenance["n_tracks"], provenance["n_localisations"],
                     f"N jumps {provenance['n_jumps']}"),
        (f"bin: long axis {bins['long_axis_data_bin_nm']:.0f} nm ({long_px:.2f} camera "
         f"pixels), short axis {bins['short_axis_data_bin_nm']:.0f} nm "
         f"({short_px:.2f} pixels). Nothing is interpolated; the legacy up-scales by "
         f"{bins['upscale_factor_per_axis']} per axis"),
        (f"bins holding one jump {value['n_bins_holding_one_point']} of "
         f"{value['n_bins_occupied']} occupied; {value['sparse_note']}"),
        (f"the maximum fold sits {fold['median_ratio_max_over_mean']:.3f} times above a "
         f"pooling fold at the median bin" if fold else ""),
        f"colour limits {lo:.4g} to {hi:.4g} um^2/s",
        limitation_line("sda", midpoint_contrast=contrast["midpoint_contrast"],
                        start_contrast=contrast["start_contrast"]),
    ], top=0.36, step=0.052)
    assert_neutral(fig)

    stem = f"legacy_sda_{enzyme.lower()}_{condition.lower()}"
    _caption(f"{stem}_caption.txt", [
        legacy_title(f"SDA maps, {enzyme} {condition}"),
        provenance["convention_note"],
        provenance["legacy_interpolation"],
        limitation_line("sda", midpoint_contrast=contrast["midpoint_contrast"],
                        start_contrast=contrast["start_contrast"]),
    ], root=root)
    path = legacy_path("figures", f"{stem}.png", root=root)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------- item 4, SLA classifier
def draw_sla_classification(msd_matrix, verdict: dict, counts: dict, *, enzyme: str,
                            condition: str, missed: int, tried: int, population: str,
                            divisor: dict | None = None, root=None):
    """The pooled MSD curve, both fits, and the legacy title.

    ``population`` is required and has no default, because the verdict is a property of
    the curve that was pooled and the title does not say which one that was. The legacy
    classifies the confined segments only (``confinedMSDMotion.m:11`` reads
    ``npTracks_segments``); running the identical test on every track of a group is a
    different question with an identical-looking answer. A figure that cannot name its
    own input is the figure this argument exists to prevent.

    The title is the legacy's own, ``F-test: <label> (p value: <p>)``. Two error bars are
    drawn where the per-lag count is available: the one the legacy draws, scaled by
    ``numel`` of the whole matrix, and the one its own legend claims, scaled by the per-lag
    count in column 4. Both are labelled; neither is presented as the other.
    """
    plt = _mpl()
    m = np.asarray(msd_matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 3:
        raise FigureError("expected the getMSD matrix [dT, M, STD, N]")
    t, msd, std = m[:, 0], m[:, 1], m[:, 2]

    fig = plt.figure(figsize=(120 / 25.4, 100 / 25.4))
    ax = fig.add_axes((0.15, 0.44, 0.80, 0.46))
    if divisor is not None and divisor.get("intended_N_lives_in_column_4"):
        ax.errorbar(t, msd, yerr=std / np.sqrt(divisor["legacy_N"]), fmt="o",
                    ms=2.5, lw=0.7, color=_NOTE,
                    label=f"legacy bar, divided by numel = {divisor['legacy_N']}")
        ax.errorbar(t, msd, yerr=std / np.sqrt(np.maximum(m[:, 3], 1)), fmt="o",
                    ms=2.5, lw=0.7, color=_INK,
                    label="per-lag count, from column 4")
    else:
        ax.plot(t, msd, "o", ms=2.5, color=_INK, label="pooled curve")

    fine = np.linspace(t.min(), t.max(), 200)
    ax.plot(fine, 4.0 * verdict["D_um2_per_s"] * fine, color=_ACCENT, lw=1.0,
            label=f"straight line, D {verdict['D_um2_per_s']:.4g}")
    ax.set_xlabel("lag (s)", fontsize=7)
    ax.set_ylabel("MSD (um^2)", fontsize=7)
    ax.set_title(legacy_title(verdict["title"]), loc="left", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.legend(frameon=False, fontsize=5.5)

    _footer(fig, [
        f"curve pooled over: {population}",
        _counts_line(counts["n_tracks"], counts["n_localisations"],
                     f"N confined segments {counts['n_confined_segments']}"),
        (f"F statistic {verdict['f_statistic']:.4g} on {verdict['n_used_in_test']} data "
         f"against {verdict['n_lags']} lags; {verdict['correlated_lags_note']}"),
        (f"exponent {verdict['alpha']:.3f}, threshold p = {verdict['p_threshold']}"),
        counts["gap_allowance"],
        limitation_line("sla_classification", missed=missed, tried=tried),
    ], top=0.34, step=0.050)
    assert_neutral(fig)

    stem = f"legacy_sla_{enzyme.lower()}_{condition.lower()}"
    _caption(f"{stem}_caption.txt", [
        legacy_title(verdict["title"]),
        f"curve pooled over: {population}",
        verdict["caveat"],
        counts["gap_allowance"],
        limitation_line("sla_classification", missed=missed, tried=tried),
    ], root=root)
    path = legacy_path("figures", f"{stem}.png", root=root)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path
