"""Matplotlib rendering of an inspection: shared by the script and the widget.

Why this is in ``src`` when the other plotting lives in ``scripts/``
--------------------------------------------------------------------
The napari dock widget and ``scripts/figure_track_inspector.py`` must draw the
*same* figure. Leaving the drawing in the script and copying it into the widget
would give two divergent renderings of one analysis, the exact duplication the
one-source-of-truth rule forbids. So the drawing lives here, and both callers
use it.

It still computes nothing: every number comes from ``smtracker.inspector``. And
it imports no Qt: the widget supplies a canvas, this fills a ``Figure``.

matplotlib is an optional extra (``pip install smtracker[figures]``); the
analysis spine never imports this module.
"""

from __future__ import annotations

import numpy as np

__all__ = ["STATE_COLORS", "STATE_NAMES", "FLOOR_COLORS", "draw_inspection"]

# The state-colour grammar is defined ONCE in smtracker.figstyle (Okabe-Ito,
# colourblind-safe) so slow/fast/third read the same in every view.
from .figstyle import STATE_COLORS, STATE_NAMES  # noqa: E402,F401
#: One colour per σ in the floor range. The band is a RANGE because σ is not
#: measured, see ``inspector.SIGMA_RANGE_NM``.
FLOOR_COLORS = {30.0: "#f6ad55", 76.0: "#e53e3e"}


def draw_inspection(inspection, figure, *, outline_um=None):
    """Draw ``inspection`` onto ``figure``. Returns the figure.

    Layout follows the question the view exists to answer: the step-size trace
    with the localization-floor band is the load-bearing panel, because "is this
    slow call real binding or jitter below the resolution limit?" is what a
    suspicious track actually raises.
    """
    figure.clear()
    grid = figure.add_gridspec(
        2, 2, width_ratios=[1.0, 1.35], height_ratios=[1.0, 1.0],
        hspace=0.38, wspace=0.28,
    )
    ax_traj = figure.add_subplot(grid[:, 0])
    ax_step = figure.add_subplot(grid[0, 1])
    ax_d = figure.add_subplot(grid[1, 1], sharex=ax_step)

    states = (
        inspection.states if inspection.states is not None
        else np.full(inspection.n_steps, -1)
    )

    # --- trajectory on the cell, coloured by state ------------------------- #
    if outline_um is not None:
        closed = np.vstack([outline_um, outline_um[:1]])
        ax_traj.fill(*outline_um.T, color="#edf2f7", zorder=0)
        ax_traj.plot(*closed.T, color="#2d3748", lw=1.2, zorder=1)
    for i in range(inspection.n_steps):
        ax_traj.plot(
            inspection.x_um[i:i + 2], inspection.y_um[i:i + 2], "-",
            color=STATE_COLORS.get(int(states[i]), "#999999"), lw=2.0, zorder=2,
        )
    ax_traj.plot(inspection.x_um[0], inspection.y_um[0], "o", color="k", ms=5, zorder=3)
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("x [µm]")
    ax_traj.set_ylabel("y [µm]")
    ax_traj.set_title("trajectory, coloured by HMM state", fontsize=9, loc="left")

    import matplotlib.lines as mlines

    handles = [
        mlines.Line2D([], [], color=STATE_COLORS.get(s, "#999999"), lw=2,
                      label=STATE_NAMES.get(s, f"state {s}"))
        for s in sorted({int(v) for v in states})
    ]
    if handles:
        ax_traj.legend(handles=handles, fontsize=7, frameon=False, loc="best")

    # --- step trace with the floor band (the load-bearing panel) ----------- #
    frames = inspection.step_frames
    for sigma_um, band in sorted(inspection.sigma_bands_um.items()):
        nm = 1000 * sigma_um
        color = FLOOR_COLORS.get(round(nm), "#e53e3e")
        ax_step.axhspan(0, band, color=color, alpha=0.16, zorder=0)
        ax_step.axhline(band, color=color, lw=1.0, ls="--", zorder=1)
        ax_step.text(
            frames[0], band, f"  σ = {nm:.0f} nm floor (q={inspection.quantile:.2f})",
            fontsize=7, va="bottom", color=color,
        )
    ax_step.plot(frames, inspection.steps_um, "-", color="#4a5568", lw=0.8, zorder=2)
    for i in range(inspection.n_steps):
        ax_step.plot(frames[i], inspection.steps_um[i], "o", ms=5, zorder=3,
                     color=STATE_COLORS.get(int(states[i]), "#999999"))
    for i in np.flatnonzero(~inspection.step_is_contiguous):
        ax_step.axvline(frames[i], color="k", ls=":", lw=1.0)
    ax_step.set_ylabel("step size [µm]")
    ax_step.set_ylim(bottom=0)
    ax_step.set_title(
        "step size vs frame: below the band is indistinguishable from an "
        "immobile molecule", fontsize=9, loc="left",
    )

    # --- instantaneous D, same limit expressed as a D ---------------------- #
    ax_d.plot(frames, inspection.instantaneous_D, "-", color="#4a5568", lw=0.8)
    for i in range(inspection.n_steps):
        ax_d.plot(frames[i], inspection.instantaneous_D[i], "o", ms=4,
                  color=STATE_COLORS.get(int(states[i]), "#999999"))
    if inspection.appd_um2_per_s is not None:
        ax_d.axhline(
            inspection.appd_um2_per_s, color="#805ad5", lw=1.4,
            label=f"appD, one D per track = {inspection.appd_um2_per_s:.3g}",
        )
    if inspection.localization_floor_D is not None:
        ax_d.axhline(
            inspection.localization_floor_D, color="#e53e3e", ls="--", lw=1.0,
            label=f"localization floor D = {inspection.localization_floor_D:.3g}",
        )
    ax_d.set_yscale("log")
    ax_d.set_xlabel("frame")
    ax_d.set_ylabel("instantaneous D [µm²/s]")
    ax_d.legend(fontsize=7, frameon=False)
    ax_d.set_title("per-step D: for looking, never for quoting", fontsize=9, loc="left")

    figure.suptitle(inspection.uid, fontsize=10)
    figure.text(0.01, 0.015, inspection.disagreement(), fontsize=7.0, color="#2d3748")
    figure.text(0.01, -0.005, "floor: " + inspection.floor_verdict(), fontsize=7.0,
                color="#742a2a")
    return figure


# --------------------------------------------------------------------------- #
# QC dashboard
# --------------------------------------------------------------------------- #

#: Flagged units are drawn in this colour. Flagging is for inspection only,
#: nothing is ever removed, so the colour must read as "look here", not "gone".
FLAG_COLOR = "#c53030"
OK_COLOR = "#2b6cb0"


def _caption(ax, text, *, y=-0.32, color="#444444"):
    import textwrap

    ax.text(0.0, y, textwrap.fill(text, 78), transform=ax.transAxes, fontsize=6.5,
            va="top", color=color, family="monospace")


def draw_qc_dashboard(figure, *, lengths, density, counts, units, in_cell=None):
    """Five panels: the filter's cost, density over time, N, per-unit spread, and
    how much of each MOVIE survived the cell filter.

    Every panel prints its own N and any threshold it applied, the same rule the
    spatial maps follow. Computation lives in ``smtracker.qc``; this only draws.

    THE FIFTH PANEL WAS ADDED BECAUSE THE OTHER FOUR CANNOT SHOW AN ABSENCE. All four
    are drawn from the canonical table, which is POST-cell-filter, so a movie the filter
    emptied has no rows and appears nowhere -- `series009` in the reference data sits at
    376 tracks found and 0 kept, and the dashboard was silent about it. ``in_cell`` is
    `qc.in_cell_share`; passing None draws the panel as unavailable and says why, since a
    table loaded without those counts is UNMEASURED rather than clean.

    IT IS OPTIONAL FOR THAT REASON ALONE, not to keep an old caller working. A default of
    "no panel" would let the absence go unremarked, which is the failure the panel exists
    to fix, so the None case still draws and still explains itself.
    """
    figure.clear()
    grid = figure.add_gridspec(3, 2, hspace=0.95, wspace=0.28,
                               height_ratios=(1.0, 1.0, 0.85))
    ax_len = figure.add_subplot(grid[0, 0])
    ax_den = figure.add_subplot(grid[0, 1])
    ax_n = figure.add_subplot(grid[1, 0])
    ax_unit = figure.add_subplot(grid[1, 1])
    ax_cell = figure.add_subplot(grid[2, :])

    # --- 1. track-length distribution and the cost of the cut -------------- #
    centres = (lengths.edges[:-1] + lengths.edges[1:]) / 2
    kept = centres > lengths.min_steps
    ax_len.bar(centres[~kept], lengths.counts[~kept], width=1.0, color="#a0aec0",
               label="removed by the filter")
    ax_len.bar(centres[kept], lengths.counts[kept], width=1.0, color=OK_COLOR,
               label="kept")
    ax_len.axvline(lengths.min_steps + 0.5, color=FLAG_COLOR, lw=1.4, ls="--")
    ax_len.set_xlabel("steps per track")
    ax_len.set_ylabel("tracks")
    ax_len.set_yscale("log")
    ax_len.legend(fontsize=7, frameon=False)
    ax_len.set_title("track-length distribution", fontsize=9, loc="left")
    _caption(ax_len, lengths.caption())

    # --- 2. localizations per frame ---------------------------------------- #
    for movie, part in density.groupby("movie", observed=True):
        ordered = part.sort_values("frame")
        ax_den.plot(ordered["frame"], ordered["n_localizations"], lw=0.7,
                    label=str(movie))
    threshold = density.attrs["threshold"]
    if np.isfinite(threshold):
        ax_den.axhline(threshold, color=FLAG_COLOR, lw=1.2, ls="--")
        ambiguous = density[density["ambiguous"]]
        if len(ambiguous):
            ax_den.plot(ambiguous["frame"], ambiguous["n_localizations"], "o",
                        color=FLAG_COLOR, ms=3, label="ambiguous linking")
    ax_den.set_xlabel("frame")
    ax_den.set_ylabel("localizations")
    if density["movie"].nunique() <= 6:
        ax_den.legend(fontsize=6, frameon=False, ncol=2)
    ax_den.set_title("localizations per frame", fontsize=9, loc="left")
    _caption(
        ax_den,
        f"N = {density.attrs['n_frames']} frames | "
        f"{density.attrs['n_ambiguous']} flagged | criterion: "
        f"{density.attrs['criterion']} | nothing is cropped",
    )

    # --- 3. N per group ----------------------------------------------------- #
    ax_n.axis("off")
    keys = list(counts.attrs.get("by", ()))
    columns = keys + ["n_tracks", "n_localizations", "n_cells"]
    shown = counts[columns].head(12)
    table = ax_n.table(
        cellText=[[f"{v}" for v in row] for row in shown.to_numpy()],
        colLabels=columns, loc="upper center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    ax_n.set_title("N per group", fontsize=9, loc="left")
    _caption(
        ax_n,
        f"grouped by {keys} | {counts.attrs['n_tracks']} tracks total"
        + (f" | showing 12 of {len(counts)} groups" if len(counts) > 12 else "")
        + " | the independent unit is the CELL or the DAY, never the displacement",
        y=-0.05,
    )

    # --- 4. per-unit readout with outliers flagged ------------------------- #
    target = units.attrs["flagged_on"]
    unit = units.attrs["unit"]
    values = units[target].to_numpy()
    positions = np.arange(len(units))
    colors = [FLAG_COLOR if f else OK_COLOR for f in units["flagged"]]
    ax_unit.bar(positions, values, color=colors)
    if units.attrs.get("scale", 0) > 0:
        ax_unit.axhline(units.attrs["median"], color="#2d3748", lw=1.0)
        ax_unit.axhspan(units.attrs["low"], units.attrs["high"], color="#2d3748",
                        alpha=0.08)
    ax_unit.set_xticks(positions)
    ax_unit.set_xticklabels(units[unit].astype(str), rotation=90, fontsize=5)
    ax_unit.set_ylabel(target)
    ax_unit.set_title(f"per-{unit} {target}, outliers flagged", fontsize=9, loc="left")
    _caption(ax_unit, units.attrs["caption"] + " | flagged for inspection, never dropped")

    # --- 5. tracks in cells, per movie -------------------------------------- #
    ax_cell.set_title("tracks inside a cell outline, per movie", fontsize=9, loc="left")
    if in_cell is None or not len(in_cell):
        ax_cell.axis("off")
        ax_cell.text(0.5, 0.5,
                     "not measured: this table carries no per-movie cell-filter counts.\n"
                     "Re-load the project with use_canonical_cache=False.",
                     ha="center", va="center", fontsize=8, color=FLAG_COLOR)
        return figure

    positions = np.arange(len(in_cell))
    inside = in_cell["tracks_in_cells"].to_numpy(dtype=float)
    outside = in_cell["tracks_outside"].to_numpy(dtype=float)
    # STACKED, SO THE WHOLE BAR IS WHAT WAS FOUND. A share alone hides that one movie
    # contributed 40 tracks and another 400, and the two want different responses.
    ax_cell.bar(positions, inside, color=OK_COLOR, label="inside a cell")
    ax_cell.bar(positions, outside, bottom=inside, color="#cbd5e0",
                label="outside every outline")
    flagged = in_cell["flagged"].to_numpy(dtype=bool)
    if flagged.any():
        ax_cell.bar(positions[flagged], (inside + outside)[flagged], facecolor="none",
                    edgecolor=FLAG_COLOR, lw=1.4, label="unlike its condition")
    ax_cell.set_xticks(positions)
    ax_cell.set_xticklabels(in_cell["movie"].astype(str), rotation=90, fontsize=5)
    ax_cell.set_ylabel("tracks")
    ax_cell.legend(fontsize=6, frameon=False, ncol=3)

    twin = ax_cell.twinx()
    twin.plot(positions, 100.0 * in_cell["in_cell_share"].to_numpy(dtype=float), "o-",
              color="#2d3748", lw=0.9, ms=2.5)
    twin.set_ylabel("% inside", fontsize=7)
    twin.set_ylim(0, 105)
    twin.tick_params(labelsize=6)

    empty = in_cell[in_cell["tracks_in_cells"] == 0]
    said = in_cell.attrs.get("note", "")
    if len(empty):
        said += (" CONTRIBUTING NOTHING: "
                 + ", ".join(str(m) for m in empty["movie"]) + ".")
    _caption(ax_cell, said + f" | {in_cell.attrs.get('n_flagged', 0)} movie(s) named | "
             "flagged for inspection, never dropped")
    return figure


# --------------------------------------------------------------------------- #
# SuperPlot (Lord et al. 2020): the pseudoreplication-honest fraction figure
# --------------------------------------------------------------------------- #

#: A stable colour per replicate day. Kept small; cycles for more days.
_DAY_COLORS = ["#3182ce", "#e53e3e", "#38a169", "#805ad5", "#dd6b20", "#319795"]
#: Neutral colour when there is NO real replicate-day structure to encode.
_SINGLE_SESSION_COLOR = "#4a5568"


def _day_color(day, days):
    return _DAY_COLORS[list(days).index(day) % len(_DAY_COLORS)]


def _has_real_days(days) -> bool:
    """≥2 distinct replicate days worth encoding by colour/axis. A single session
    must NOT be drawn per day: the figure would assert a replicate structure the
    data lacks (mirrors ``compare_plots.has_real_days`` and the aggregate guard)."""
    return len({d for d in days if d is not None}) >= 2


def draw_superplot_distribution(figure, points, *, value="appd_um2_per_s",
                                unit_col="uid", day_col="day"):
    """Per-TRACK apparent-D as a log-axis strip, coloured by replicate day.

    This is the *distribution* SuperPlot: one point per track (never per step),
    so the slow/fast bumps are visible without the pseudoreplication of pooling
    steps. It is explicitly the switching-degraded appD view (a molecule that
    binds mid-track lands at an averaged D) and is labelled as such.
    """
    figure.clear()
    ax = figure.add_subplot(111)
    days = sorted(points[day_col].unique())
    by_day = _has_real_days(days)
    rng = _np_for_jitter()
    if by_day:
        for day in days:
            sub = points[points[day_col] == day]
            vals = sub[value].to_numpy(dtype=float)
            vals = vals[vals > 0]
            x = list(days).index(day) + 1 + rng.uniform(-0.25, 0.25, len(vals))
            ax.scatter(x, vals, s=8, alpha=0.4, color=_day_color(day, days),
                       edgecolors="none")
        ax.set_xticks(range(1, len(days) + 1))
        ax.set_xticklabels([f"replicate {d}" for d in days])
        unit_phrase = "by replicate"
    else:
        # single session: one neutral column, no per-day axis or colour
        vals = points[value].to_numpy(dtype=float)
        vals = vals[vals > 0]
        x = 1 + rng.uniform(-0.25, 0.25, len(vals))
        ax.scatter(x, vals, s=8, alpha=0.4, color=_SINGLE_SESSION_COLOR,
                   edgecolors="none")
        ax.set_xticks([1])
        ax.set_xticklabels(["all tracks\n(single session)"])
        unit_phrase = "(single session)"
    ax.set_yscale("log")
    ax.set_ylabel(f"per-track {value} (log)")
    ax.set_title(
        f"SuperPlot: per-track apparent D {unit_phrase}\n"
        "(switching-degraded appD view; one point per track, never per step)",
        fontsize=9, loc="left")
    return figure


def draw_superplot_fraction(figure, per_cell, *, frac="slow_fraction",
                            day_col="day", ci=None, overall=None):
    """Per-CELL slow fraction, coloured by day, with per-day means + overall CI.

    The pseudoreplication-honest headline figure: each cell is one point, cells
    are coloured by their replicate day, each day's mean is drawn as a large
    marker, and the overall bootstrap CI (over the correct unit) is shaded. The
    whole point is *colour-by-replicate + summarise-on-replicates*, the eye sees
    both the cell spread and the day structure, and the CI is the honest one.
    """
    figure.clear()
    ax = figure.add_subplot(111)
    days = sorted(per_cell[day_col].unique())
    by_day = _has_real_days(days)
    rng = _np_for_jitter()

    if ci is not None and overall is not None:
        ax.axhspan(ci[0], ci[1], color="#a0aec0", alpha=0.25, zorder=0)
        ax.axhline(overall, color="#2d3748", lw=1.2, zorder=1,
                   label=f"overall {overall:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

    if by_day:
        for day in days:
            sub = per_cell[per_cell[day_col] == day]
            vals = sub[frac].to_numpy(dtype=float)
            colour = _day_color(day, days)
            x = list(days).index(day) + 1 + rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(x, vals, s=20, alpha=0.5, color=colour, edgecolors="none")
            ax.plot(list(days).index(day) + 1, vals.mean(), "D", ms=11,
                    color=colour, markeredgecolor="k", zorder=3)
        ax.set_xticks(range(1, len(days) + 1))
        ax.set_xticklabels([f"replicate {d}\n(mean)" for d in days])
        unit_phrase = "by replicate"
        mean_phrase = "diamonds = per-replicate means; "
    else:
        # single session: one neutral cell column, NO per-day means (there is one
        # "day", so a per-day mean would be a fabricated replicate). The overall
        # cell-bootstrap CI stays: that is the honest single-session interval.
        vals = per_cell[frac].to_numpy(dtype=float)
        x = 1 + rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(x, vals, s=20, alpha=0.5, color=_SINGLE_SESSION_COLOR,
                   edgecolors="none")
        ax.set_xticks([1])
        ax.set_xticklabels([f"cells (n={len(per_cell)})\n(single session)"])
        unit_phrase = "(single session)"
        mean_phrase = ""
    ax.set_ylabel(f"per-cell {frac}")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"SuperPlot: per-cell slow fraction {unit_phrase}\n"
        f"({mean_phrase}band = overall bootstrap CI)",
        fontsize=9, loc="left")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    return figure


def _np_for_jitter():
    import numpy as np

    return np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# Uncertainty-first plotting, never a bare mean
# --------------------------------------------------------------------------- #

#: Below this many units a region is "thin data" and is drawn faded, so a
#: summary from too few points reads as tentative rather than authoritative.
THIN_DATA_UNITS = 5


def draw_uncertainty_summary(ax, groups, *, positions=None, thin_below=THIN_DATA_UNITS,
                             point_color=None, band_color="#4a5568"):
    """Draw a summary the honest way: per-unit points + a bootstrap band, faded
    where the data is thin. Never a bare mean.

    ``groups`` is ``{label: values}``: the per-unit values behind each summary
    (per-cell fractions, per-day means, …). For each: the individual points are
    scattered (so the spread is visible), the mean is a heavy marker, and a
    2.5–97.5 percentile band is shaded. A group with fewer than ``thin_below``
    units is drawn at reduced alpha, because a mean of three points is not a
    mean to trust. Returns the per-group summary stats.

    This is the shared helper the fraction plots, the SuperPlot and the map
    panels use so *every* summary shows its spread by default.
    """
    import numpy as np

    labels = list(groups)
    positions = positions if positions is not None else range(len(labels))
    rng = np.random.default_rng(0)
    stats = {}
    for pos, label in zip(positions, labels):
        vals = np.asarray(groups[label], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        thin = vals.size < thin_below
        alpha = 0.25 if thin else 0.6
        lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals.size > 1
                  else (vals[0], vals[0]))
        colour = point_color or "#2b6cb0"
        # the spread, as a shaded band around the mean
        ax.plot([pos, pos], [lo, hi], color=band_color, lw=6,
                alpha=alpha * 0.4, solid_capstyle="round", zorder=1)
        # the per-unit points
        x = pos + rng.uniform(-0.08, 0.08, vals.size)
        ax.scatter(x, vals, s=16, color=colour, alpha=alpha, edgecolors="none",
                   zorder=2)
        # the mean, heavy
        ax.plot(pos, vals.mean(), "D", ms=9, color=colour, markeredgecolor="k",
                zorder=3)
        if thin:
            ax.annotate("thin", (pos, hi), fontsize=6, color="#a0aec0",
                        ha="center", va="bottom")
        stats[label] = {"mean": float(vals.mean()), "ci": (float(lo), float(hi)),
                        "n": int(vals.size), "thin": bool(thin)}
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    return stats


def draw_replicate_fit(figure, payload):
    """One condition's per-replicate SIMULTANEOUS SQD fit as a BUBBLE plot, the
    SAME visual grammar as the main SQD view, so the inspector never switches
    language: x = replicate, y = D (log), bubble area ∝ fraction, colour =
    population. Pure; every number comes from ``comparison.condition_replicate_fit``.
    NEUTRAL: populations by index + numeric D, no interpretive state word.

    Under condition/global scope every population's bubbles sit at the SAME D (a
    horizontal row), so a shared D is obvious and only the SIZES (fractions) change
    across replicates. Under replicate scope the bubbles step in D per replicate.
    Below: a small text table.
    """
    from .figstyle import STATE_COLORS

    def _state_colour(k):
        return STATE_COLORS.get(k, STATE_COLORS[-1])

    figure.clear()
    grid = figure.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.35)
    ax = figure.add_subplot(grid[0, 0])

    cond = payload["condition"]
    shared_D = payload["shared_D"]
    thr = payload["threshold_um2_s"]
    reps = payload["replicates"]
    scope = payload.get("scope", "condition")
    n_pop = payload["n_pop"]
    xs = np.arange(len(reps))

    seen = {}
    for x, r in zip(xs, reps):
        D = r["D"] if (scope == "replicate" or shared_D is None) else shared_D
        for k, (Dk, frac) in enumerate(zip(D, r["fractions"]), start=1):
            ax.scatter(x, Dk, s=120 + 3200 * frac, color=_state_colour(k),
                       alpha=0.55, edgecolors="k", linewidths=0.6, zorder=3)
            ax.annotate(f"{frac*100:.0f}%", (x, Dk), ha="center", va="center",
                        fontsize=7, zorder=4)
            seen.setdefault(k, Dk)

    ax.axhline(thr, color="#2d3748", lw=1.0, ls=":", zorder=1)
    ax.annotate(f"D = {thr:g} µm²/s", (len(reps) - 0.5, thr), fontsize=6.5,
                va="bottom", ha="right", color="#2d3748")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"rep {r['replicate']}" for r in reps])
    ax.set_ylabel("D (µm²/s, log)")
    ax.set_xlim(-0.6, len(reps) - 0.4)
    title_scope = ("per-replicate D" if (scope == "replicate" or shared_D is None)
                   else "shared D: rows align")
    ax.set_title(f"populations per replicate ({title_scope}); area ∝ fraction",
                 fontsize=8, loc="left")

    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], marker="o", ls="", color=_state_colour(k),
                             markeredgecolor="k",
                             label=f"pop {k}: D = {seen[k]:.3g}")
               for k in sorted(seen)]
    if handles:
        ax.legend(handles=handles, fontsize=6, frameon=False, loc="best")

    # the small table below: per-replicate fractions for every population.
    hdr = f"{'replicate':>10} {'cells':>6} {'tracks':>7}  " + \
        " ".join(f"{'f'+str(k):>7}" for k in range(1, n_pop + 1))
    lines = [hdr]
    for r in reps:
        fr = " ".join(f"{f*100:>6.1f}%" for f in r["fractions"])
        lines.append(f"{str(r['replicate']):>10} {r['n_cells']:>6d} "
                     f"{r['n_tracks']:>7d}  {fr}")
    from .workspace_view import scope_words

    if scope == "replicate" or shared_D is None:
        dnote = f"per-replicate D ({scope_words('replicate')})"
    else:
        dstr = ", ".join(f"{d:.3g}" for d in shared_D)
        group = payload.get("group") or {}
        # Force same D across in its one spelling (`workspace_view.scope_words`, 1 Nov 2026)
        where = (f"{scope_words('group', group.get('factor') or '')}: {group.get('name', 'its group')}"
                 if scope == "group" else scope_words(scope))
        dnote = f"shared D = [{dstr}] µm²/s ({where})"
    estimand = ("molecule fraction" if payload.get("weighting") == "track"
                else "displacement/time fraction")
    header = f"{dnote}   (n_pop = {n_pop}, {estimand})"
    figure.text(0.01, 0.20, header, fontsize=7.5, family="monospace", color="#2d3748")
    figure.text(0.01, 0.02, "\n".join(lines), fontsize=7, family="monospace",
                color="#4a5568", va="bottom")

    subtitle = ("per-replicate D" if (scope == "replicate" or shared_D is None)
                else "shared D, per-replicate fractions")
    figure.suptitle(f"{cond}: simultaneous replicate fit ({subtitle})", fontsize=9)
    return ax


# =====================================================================================
# THE BUBBLE PLOT AND THE CROSSED ERROR BARS (SMTracker.m:6361 and :5019)
# =====================================================================================
#
# Both appear on three v2 tabs (GMM, SQD, APPD) and show the same two numbers per
# population: a FRACTION and a DIFFUSION COEFFICIENT. The bubble plot puts D on the y
# axis and encodes the fraction as bubble area; the crossed error bars put the fraction on
# the x axis and D on the y, with an interval in both directions.
#
# THREE THINGS ARE FIXED, each confirmed in the source.
#
# **1. The labels asserted an interpretation.** ``crossedErrorbar`` hard-codes
# ``diffusionStr = {'Static','Mobile','Fast Mobile'}`` and assigns them BY SORT ORDER, so
# the population with the smallest D is called "Static" whatever its D turned out to be.
# Populations here are numbered by ascending D and named D1, D2, D3. The reader is told
# what was measured, not what to conclude.
#
# **2. Negative fractions were silently clamped.** ``bubbleplot`` sets any fraction at or
# below zero to ``1e-5`` and writes a console line. A negative fraction is not a small
# fraction -- it means the fit was unconstrained and returned something impossible, and
# drawing it as a dot hides that. Here it RAISES.
#
# **3. Bubble area could not be read back.** v2 draws area as ``20 * fraction`` with no
# size key, so a reader can compare two bubbles but cannot recover a number from one. A
# size legend is drawn, and the fraction is still printed beside each bubble as v2 did.
#
# THE INTERVALS ARE THE CALLER'S. Both functions take explicit low/high bounds rather than
# computing anything, because where the interval came from decides what it means: v2's
# ``nlparci`` bounds are 90 to 131 times too narrow on correlated points, and a bootstrap
# over cells or days is the honest alternative. The source is printed on the figure.


def _population_labels(n_pop):
    """D1, D2, D3 by ascending D. Never Static / Mobile / Fast Mobile."""
    return ["D$_{%d}$" % (i + 1) for i in range(n_pop)]


def draw_bubble_plot(ax, table, *, interval_source, fraction_col="fraction",
                     d_col="D_um2_per_s", d_lo_col=None, d_hi_col=None,
                     condition_col="condition", area_per_percent=20.0):
    """v2's bubble plot: D on y, fraction as bubble AREA, one colour per condition.

    ``table`` is one row per (condition, population) and must already be sorted so that
    populations run by ascending D within each condition.
    """
    import numpy as np

    frac = np.asarray(table[fraction_col], dtype=float)
    if np.any(~np.isfinite(frac)) or np.any(frac <= 0):
        bad = table.loc[(frac <= 0) | ~np.isfinite(frac)]
        raise ValueError(
            "%d population(s) carry a fraction at or below zero:\n%s\n"
            "v2 clamps these to 1e-5 and writes a console line. A negative fraction is "
            "not a small fraction -- it means the fit was unconstrained and returned "
            "something impossible, and drawing it as a dot hides that. Constrain the fit "
            "or drop the population explicitly."
            % (len(bad), bad.to_string(index=False)))

    d = np.asarray(table[d_col], dtype=float)
    conditions = list(dict.fromkeys(table[condition_col]))
    colours = _DAY_COLORS * (1 + len(conditions) // len(_DAY_COLORS))
    colour_of = {c: colours[i] for i, c in enumerate(conditions)}

    x = np.arange(len(table), dtype=float)
    ax.scatter(x, d, s=area_per_percent * frac * 100.0,
               c=[colour_of[c] for c in table[condition_col]],
               edgecolors="#0b0b0b", linewidths=0.5, zorder=3)
    if d_lo_col and d_hi_col:
        lo = np.asarray(table[d_lo_col], dtype=float)
        hi = np.asarray(table[d_hi_col], dtype=float)
        ax.errorbar(x, d, yerr=[d - lo, hi - d], fmt="none", ecolor="#0b0b0b",
                    elinewidth=0.8, capsize=2, zorder=2)
    for xi, di, fi in zip(x, d, frac):
        ax.annotate("%.1f %%" % (100.0 * fi), (xi, di), textcoords="offset points",
                    xytext=(9, 0), va="center", fontsize=6)

    n_pop = len(table) // max(len(conditions), 1)
    labels = _population_labels(n_pop)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[i % n_pop] for i in range(len(table))], fontsize=7)
    ax.set_ylabel("D (um$^2$/s)", fontsize=8)
    ax.tick_params(labelsize=7)
    # WHICH CONDITION each group belongs to. Without this the axis reads D1 D2 D1 D2
    # and the reader has to infer the grouping from the colours.
    for ci, cond in enumerate(conditions):
        centre = np.mean(x[ci * n_pop:(ci + 1) * n_pop])
        ax.annotate(str(cond), (centre, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(0, -20), textcoords="offset points", ha="center",
                    fontsize=7, color=colour_of[cond], fontweight="bold")
    ax.margins(x=0.18)

    # THE SIZE KEY v2 NEVER DREW, so a bubble can be read back to a number.
    for pct in (10, 30, 60):
        ax.scatter([], [], s=area_per_percent * pct, c="#bbbbbb",
                   edgecolors="#0b0b0b", linewidths=0.5, label="area = %d %%" % pct)
    # BELOW the axes, horizontally. loc="best" put the key on top of the largest bubble;
    # moving it to the right then collided with the per-bubble percentage labels, which
    # extend rightward by design. Below is the only side nothing else uses.
    # No legend TITLE: with ncol=3 matplotlib lays the title among the entries rather
    # than above them, so the key is spelled into each label instead.
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), columnspacing=2.6, handletextpad=1.3)
    ax.text(0.0, -0.52, "interval: %s" % interval_source, transform=ax.transAxes,
            fontsize=5.5, color="#52514e")
    return ax


def draw_crossed_errorbars(ax, table, *, interval_source, fraction_col="fraction",
                           frac_lo_col=None, frac_hi_col=None, d_col="D_um2_per_s",
                           d_lo_col=None, d_hi_col=None, condition_col="condition"):
    """v2's crossed error bars: fraction on x, D on y, an interval in both directions.

    One marker shape and colour per condition, as v2 does. Populations are labelled by
    ascending D, NOT by an asserted mobility class.
    """
    import numpy as np

    frac = np.asarray(table[fraction_col], dtype=float) * 100.0
    d = np.asarray(table[d_col], dtype=float)
    conditions = list(dict.fromkeys(table[condition_col]))
    colours = _DAY_COLORS * (1 + len(conditions) // len(_DAY_COLORS))
    shapes = ["o", "s", "D", "^", "v", ">", "<", "P", "X", "*"]
    n_pop = len(table) // max(len(conditions), 1)
    labels = _population_labels(n_pop)

    for i, row in enumerate(table.itertuples(index=False)):
        cond = getattr(row, condition_col)
        ci = conditions.index(cond)
        xerr = None
        if frac_lo_col and frac_hi_col:
            lo = getattr(row, frac_lo_col) * 100.0
            hi = getattr(row, frac_hi_col) * 100.0
            xerr = [[frac[i] - lo], [hi - frac[i]]]
        yerr = None
        if d_lo_col and d_hi_col:
            yerr = [[d[i] - getattr(row, d_lo_col)], [getattr(row, d_hi_col) - d[i]]]
        ax.errorbar([frac[i]], [d[i]], xerr=xerr, yerr=yerr,
                    fmt=shapes[ci % len(shapes)], color=colours[ci],
                    markerfacecolor=colours[ci], markeredgecolor="#0b0b0b",
                    markersize=5, elinewidth=0.9, capsize=2,
                    label="%s %s" % (cond, labels[i % n_pop]))
    ax.set_xlabel("fraction (%)", fontsize=8)
    ax.set_ylabel("D (um$^2$/s)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    ax.legend(frameon=False, fontsize=5.5, ncol=2)
    ax.text(0.0, -0.22, "interval: %s" % interval_source, transform=ax.transAxes,
            fontsize=5.5, color="#52514e")
    return ax


def draw_appd_histogram(ax, appd, *, n_bins=40, fit=None, n_steps=None,
                        label="apparent D per track"):
    """v2's apparent-diffusion histogram, with the fitted mixture over it if given.

    ``fit`` is a ``smtracker.appd_gamma.GammaFit``. Drawing the components as well as the
    sum is what makes a mixture readable: v2 draws them, and without them a reader cannot
    see which population carries which part of the curve.
    """
    import numpy as np

    x = np.asarray(appd, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if not len(x):
        raise ValueError("no positive apparent-D values to draw")
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    ax.hist(x, bins=edges, color="#c6c6c6", edgecolor="#0b0b0b", linewidth=0.3,
            density=True, label="%s (n = %d)" % (label, len(x)))
    ax.set_xscale("log")

    if fit is not None:
        from .appd_gamma import gamma_logpdf, mixture_logpdf

        shape = n_steps if n_steps is not None else fit.n_steps
        grid = np.logspace(np.log10(x.min()), np.log10(x.max()), 400)
        for k, (dk, fk) in enumerate(zip(fit.D, fit.fraction)):
            if dk == 0:
                # the immobile population of exact-zero tracks (`appd_gamma`'s module head): no density on this axis
                ax.plot([], [], lw=1.0, ls="--", color=_DAY_COLORS[k % len(_DAY_COLORS)],
                        label="D$_{%d}$ = 0 (exact-zero tracks), %.1f %%" % (k + 1, 100 * fk))
                continue
            ax.plot(grid, fk * np.exp(gamma_logpdf(grid, dk, shape)),
                    lw=1.0, ls="--", color=_DAY_COLORS[k % len(_DAY_COLORS)],
                    label="D$_{%d}$ = %.4g, %.1f %%" % (k + 1, dk, 100 * fk))
        ax.plot(grid, np.exp(mixture_logpdf(grid, fit.D, fit.fraction, shape,
                                            immobile=bool(getattr(fit, "immobile", False)))),
                lw=1.4, color="#0b0b0b", label="mixture")
    # EVERY FIGURE STATES ITS OWN N. This one predates that convention and had no
    # caption at all, so a reader could not tell 50 tracks from 5000.
    _n = int(np.sum(np.isfinite(np.asarray(appd, dtype=float))))
    ax.set_title(f"N {_n} per-track apparent D values", loc="left", fontsize=5.5)
    ax.set_xlabel("apparent D (um$^2$/s)", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.tick_params(labelsize=7)
    # Upper right, INSIDE: on a log x axis the mixture peak sits left of centre and the
    # upper right is empty. Headroom is added so the legend cannot clip the peak, which
    # it did at the default limit.
    top = ax.get_ylim()[1]
    ax.set_ylim(top=top * 1.38)
    ax.legend(frameon=False, fontsize=5.5, loc="upper right")
    return ax


# =====================================================================================
# ALL TRACKS ON ONE CELL (``@SMT/allTracksonCell.m``) -- the STATIC viewer
# =====================================================================================
#
# v2's fast track view: every track of a condition projected into one normalised cell and
# drawn as a still image. No movie, no Qt, no OpenGL -- which is why it opens instantly
# where the napari viewer has a window to build.
#
# **THE CELL LENGTH WAS HARD-CODED, AND THE LINE THAT READ THE REAL ONE IS COMMENTED OUT
# DIRECTLY ABOVE IT.** ``allTracksonCell.m:18-19``:
#
#     % avgCellLength = obj.cellstats.cdata{3,i_condition+1};
#     avgCellLength = 3;
#
# So every condition is drawn at 3 um whatever its cells measure. On a lab project
# the conditions' cell lengths differ by about 12 per cent,
# so the drawn aspect is wrong per condition AND identical across conditions, which hides
# a real difference in exactly the picture meant to show differences. The commented-out
# line proves the intent. Here the length is measured and PRINTED, and passing one
# explicitly is allowed but never silent.
#
# THE NaN-SEPARATED POLYLINE IS KEPT, and it is the reason this is fast: v2 joins every
# track into a single vector separated by ``[NaN NaN]`` and draws ONE line. Drawing each
# track as its own artist is the obvious alternative and is far slower for no gain.


def draw_all_tracks_on_cell(ax, canonical, *, cell_length_um=None, cell_width_um=None,
                            l_col="l_norm", d_col="d_norm", colour="#2b6cb0",
                            linewidth=0.35, alpha=0.55, max_tracks=None, seed=0):
    """Every track of a group in one normalised cell. ``allTracksonCell`` reproduced.

    ``cell_length_um`` and ``cell_width_um`` default to the MEASURED median of the cells
    that contributed, never to a literal. Pass them to override, and the figure says which
    happened.

    ``max_tracks`` subsamples for interactivity and, when it bites, says so on the figure
    rather than quietly drawing a subset.
    """
    import numpy as np

    for column in ("uid", l_col, d_col):
        if column not in canonical.columns:
            raise ValueError(
                f"missing column {column!r}: this view needs cell-frame coordinates, "
                f"which cellcoords.project_tracks supplies")
    frame = canonical[canonical[l_col].notna() & canonical[d_col].notna()]
    if frame.empty:
        raise ValueError("no localisation carries cell-frame coordinates")

    measured = True
    if cell_length_um is None or cell_width_um is None:
        geometry = canonical.attrs.get("geometry") or {}
        lengths = [g.length_um for g in geometry.values()
                   if getattr(g, "length_um", None)]
        widths = [g.max_width_um for g in geometry.values()
                  if getattr(g, "max_width_um", None)]
        if not lengths or not widths:
            raise ValueError(
                "no cell geometry on the table, so the cell size cannot be measured. "
                "v2 hard-codes 3 um here with the line that read the real value "
                "commented out above it; a drawn cell size is a measurement of these "
                "cells and a default printed beside real tracks reads as one. Pass "
                "cell_length_um and cell_width_um explicitly, or project the table "
                "through cellcoords.project_tracks first.")
        cell_length_um = float(np.median(lengths))
        cell_width_um = float(np.median(widths))
    else:
        measured = False

    uids = frame["uid"].unique()
    n_total = len(uids)
    subsampled = False
    if max_tracks is not None and n_total > max_tracks:
        rng = np.random.default_rng(seed)
        uids = rng.choice(uids, size=int(max_tracks), replace=False)
        frame = frame[frame["uid"].isin(set(uids))]
        subsampled = True

    # ONE polyline, tracks separated by NaN. v2 does this and it is why the view is fast:
    # a separate artist per track is the obvious alternative and is far slower for no gain.
    ordered = frame.sort_values(["uid", "frame"], kind="stable")
    x = ordered[l_col].to_numpy(dtype=float) * cell_length_um
    y = ordered[d_col].to_numpy(dtype=float) * cell_width_um / 2.0
    boundaries = np.flatnonzero(ordered["uid"].to_numpy()[1:]
                                != ordered["uid"].to_numpy()[:-1]) + 1
    xs = np.insert(x.astype(float), boundaries, np.nan)
    ys = np.insert(y.astype(float), boundaries, np.nan)
    ax.plot(xs, ys, "-", color=colour, lw=linewidth, alpha=alpha, solid_capstyle="round")

    R = cell_width_um / 2.0
    Lc = max(cell_length_um - 2 * R, 0.0)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 120)
    ring_x = np.concatenate([Lc / 2 + R * np.cos(theta), -Lc / 2 - R * np.cos(theta),
                             [Lc / 2 + R * np.cos(theta[0])]]) + cell_length_um / 2.0
    ring_y = np.concatenate([R * np.sin(theta), -R * np.sin(theta),
                             [R * np.sin(theta[0])]])
    ax.plot(ring_x, ring_y, "k-", lw=1.2)

    margin = 0.15
    ax.set_xlim(-margin, cell_length_um + margin)
    ax.set_ylim(-R - margin, R + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("[um]", fontsize=7)
    ax.set_ylabel("[um]", fontsize=7)
    ax.tick_params(labelsize=6)

    # SATURATION IS A PROPERTY OF THE VIEW, AND IT MUST BE SAID. Above one localisation
    # per camera pixel of cell area the ink cannot separate one track from another, and
    # any texture in it is the drawing rather than the data. That threshold is a
    # resolution statement and not a taste one: the detector cannot place two
    # localisations inside one pixel apart either.
    #
    # MEASURED on a lab project: pooled localisations in a 2.75 x 0.90 um
    # cell come to **236.4 localisations per camera pixel**, so the pooled view is saturated by
    # more than two orders of magnitude. Subsampling barely helps -- 150 tracks still
    # gives 5.76, and clearing 1.0 needs about **25 tracks**, which is roughly one cell's
    # worth. So this view shows the ENVELOPE the molecules occupy, not the tracks, at
    # every track count anyone would pool. It is still the right picture for "where does
    # anything go at all"; it is the wrong picture for reading individual paths, and
    # ``draw_cell_explorer`` is the one for that. v2 drew this and said nothing.
    pixel_um = None
    params = canonical.attrs.get("params")
    if params is not None and getattr(params, "pixel_size_nm", None):
        pixel_um = float(params.pixel_size_nm) / 1000.0
    density = None
    if pixel_um:
        area_um2 = (cell_length_um - cell_width_um) * cell_width_um +             np.pi * (cell_width_um / 2.0) ** 2
        density = len(frame) / max(area_um2 / pixel_um ** 2, 1e-9)

    source = "median of the contributing cells" if measured else "supplied"
    note = ("N: %d tracks, %d localisations   cell %.2f x %.2f um (%s)"
            % (len(frame["uid"].unique()), len(frame), cell_length_um,
               cell_width_um, source))
    if subsampled:
        note += "   SUBSAMPLED from %d tracks" % n_total
    if density is not None:
        note += "   %.1f localisations per camera pixel" % density
        if density > 1.0:
            note += " -- SATURATED, texture here is ink not data"
    ax.set_title(note, loc="left", fontsize=6)
    ax.saturation_localisations_per_pixel = density
    return ax


# =====================================================================================
# ONE CELL, ONE TRACK (``@SMT/SMTracker_explorer.m``) -- the static single-track view
# =====================================================================================
#
# THE SAME QUANTITY GETS TWO ANSWERS IN TWO FILES. ``SMTracker_explorer.m:34`` reads the
# real cell length off the cell (``length*pixel_size/1000``); ``allTracksonCell.m:19``
# hard-codes ``3``. Whichever is right, they cannot both be.
#
# AND THE WIDTH IS NEVER SCALED AT ALL. ``SMT.m:42`` documents the contour as "a model
# unity cell, with width 1 and length 1", and ``SMTracker_explorer.m:35`` draws
#
#     plot(ax, modelCellContour(:,1)*cell_length, modelCellContour(:,2), ...)
#
# -- x multiplied by the real length, y left at unity. The TRACKS are in real um
# (``projTracks(:,2:3)*pixel_size/1000``), so a real-um molecule is drawn against a
# fixed 1.0 um-wide outline. It passes the eye because B. subtilis is about a micron
# across; measured on this project the cells run 0.90 to 1.25 um wide, so at the wide end
# a molecule genuinely at the membrane is drawn about 0.13 um OUTSIDE the cell it is
# inside. Here both axes are scaled and the drawn size is printed.
#
# The time colouring is kept exactly: ``AdvancedColormap('autumn', n)`` over the track's
# own localisations, one segment per step, with the start marked.


def draw_cell_explorer(ax, canonical, *, cell, uid=None, cell_length_um=None,
                       cell_width_um=None, l_col="l_norm", d_col="d_norm",
                       cmap="autumn", context_colour="0.62", linewidth=1.8):
    """One cell's tracks, one of them coloured by time. ``SMTracker_explorer`` reproduced.

    Every track assigned to ``cell`` is drawn in grey; ``uid`` is drawn over them coloured
    along its own length, earliest to latest, with its first localisation marked. Pass
    ``uid=None`` to draw the context alone.
    """
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from matplotlib.collections import LineCollection

    frame = canonical[(canonical["cell"] == cell)
                      & canonical[l_col].notna() & canonical[d_col].notna()]
    if frame.empty:
        raise ValueError(f"cell {cell!r} carries no projected localisation")

    if cell_length_um is None or cell_width_um is None:
        geometry = (canonical.attrs.get("geometry") or {}).get(cell)
        if geometry is None:
            raise ValueError(
                f"no geometry for cell {cell!r}, so its drawn size would be a default. "
                f"v2 draws every cell 1.0 um wide here whatever it measures; pass "
                f"cell_length_um and cell_width_um to choose that deliberately.")
        cell_length_um = float(geometry.length_um)
        cell_width_um = float(geometry.max_width_um)

    R = cell_width_um / 2.0
    ordered = frame.sort_values(["uid", "frame"], kind="stable")
    x = ordered[l_col].to_numpy(dtype=float) * cell_length_um
    y = ordered[d_col].to_numpy(dtype=float) * R
    ids = ordered["uid"].to_numpy()
    cuts = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    ax.plot(np.insert(x, cuts, np.nan), np.insert(y, cuts, np.nan), "-",
            color=context_colour, lw=0.6, zorder=1)

    n_steps = 0
    if uid is not None:
        one = ordered[ordered["uid"] == uid]
        if one.empty:
            raise ValueError(f"track {uid!r} is not in cell {cell!r}")
        tx = one[l_col].to_numpy(dtype=float) * cell_length_um
        ty = one[d_col].to_numpy(dtype=float) * R
        points = np.column_stack([tx, ty]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        n_steps = len(segments)
        if n_steps:
            lc = LineCollection(segments, cmap=cmap,
                                norm=Normalize(0, max(n_steps - 1, 1)), lw=linewidth,
                                zorder=3)
            lc.set_array(np.arange(n_steps))
            ax.add_collection(lc)
        ax.plot(tx[0], ty[0], "o", ms=5, mfc=colormaps[cmap](0.0), mec="#1a365d",
                mew=1.0, zorder=4)

    Lc = max(cell_length_um - 2 * R, 0.0)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 120)
    ring_x = np.concatenate([Lc / 2 + R * np.cos(theta), -Lc / 2 - R * np.cos(theta),
                             [Lc / 2 + R * np.cos(theta[0])]]) + cell_length_um / 2.0
    ring_y = np.concatenate([R * np.sin(theta), -R * np.sin(theta),
                             [R * np.sin(theta[0])]])
    ax.plot(ring_x, ring_y, "k-", lw=1.0, zorder=2)

    margin = 0.15
    ax.set_xlim(-margin, cell_length_um + margin)
    ax.set_ylim(-R - margin, R + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("[um]", fontsize=7)
    ax.set_ylabel("[um]", fontsize=7)
    ax.tick_params(labelsize=6)
    title = ("cell %s: %d tracks   cell %.2f x %.2f um (measured)"
             % (cell, frame["uid"].nunique(), cell_length_um, cell_width_um))
    if uid is not None:
        title += "   track %s: %d steps, colour = time" % (uid, n_steps)
    ax.set_title(title, loc="left", fontsize=6)
    return ax


# =====================================================================================
# THE SQD FIT (`smtracker.sqd`) - the fraction readout, drawn
# =====================================================================================
#
# A FIT NEEDS ITS DATA TO BE JUDGED. Every other drawing here takes ``(ax, result)``, but
# a model curve with no measured points beside it cannot show whether it fits: the
# residual is the whole reason to look. So this one takes the squared displacements as
# well, as an optional keyword, and the panel supplies them when the drawing asks for
# them. Called without them it still draws the model and says the data is missing rather
# than quietly showing a curve that looks like a fit to nothing.
#
# WHAT IS AND IS NOT LEGIBLE HERE. The FRACTIONS are the readout and are printed. The D's
# are printed as bounds: D1 sits at or below the localisation floor under either sigma
# estimate (30 or 76 nm), so it is not a physical coefficient, and the caption says so
# rather than leaving a reader to take the number at face value.


def draw_sqd_fit(ax, fit, *, data=None, n_points=400, show_components=True,
                 data_color="#2b6cb0", fit_color="#c53030"):
    """The squared-displacement CDF: the measured ECDF and the fitted mixture.

    ``fit`` is an :class:`smtracker.sqd.SQDFit`. ``data`` is the squared displacements in
    um^2 the fit was made on; without it the model is drawn alone and labelled as such.
    """
    import numpy as np

    D = np.asarray(fit.D, dtype=float)
    fractions = np.asarray(fit.fractions, dtype=float)
    tau = float(fit.tau_lag_s)
    sigma = float(getattr(fit, "localization_sigma_um", 0.0) or 0.0)
    order = np.argsort(D)
    D, fractions = D[order], fractions[order]

    if data is not None and len(np.asarray(data)):
        x = np.sort(np.asarray(data, dtype=float))
        x = x[np.isfinite(x)]
        ecdf = np.arange(1, len(x) + 1) / len(x)
        # One polyline, not one marker per point: at 42 587 points the markers are the
        # cost and the picture is the same.
        ax.plot(x, ecdf, "-", color=data_color, lw=1.4, label="measured (ECDF)")
        grid = np.linspace(0.0, float(np.quantile(x, 0.999)), n_points)
        n_shown = len(x)
    else:
        grid = np.linspace(0.0, 4 * float(D.max()) * tau * 5, n_points)
        n_shown = 0

    msd = 4.0 * D * tau + 4.0 * sigma ** 2
    model = 1.0 - np.sum(fractions[None, :] * np.exp(-grid[:, None] / msd[None, :]),
                         axis=1)
    ax.plot(grid, model, "--", color=fit_color, lw=1.6,
            label=f"{len(D)}-population fit")

    if show_components and len(D) > 1:
        for k in range(len(D)):
            component = fractions[k] * (1.0 - np.exp(-grid / msd[k]))
            ax.plot(grid, component, ":", lw=1.0, alpha=0.75,
                    label=f"pop {k + 1}: {100 * fractions[k]:.1f} %, "
                          f"D {D[k]:.3f} um2/s")

    ax.set_xlabel("squared displacement [um$^2$]", fontsize=8)
    ax.set_ylabel("cumulative probability", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower right", frameon=False)

    # The caption states the unit, the population and the limit, in the same place as the
    # numbers. The fractions are the readout; D1 is a bound, not a coefficient.
    # WRAPPED, because an overflowing caption collides with the panel beside it and what
    # the reader then loses is the limit, not the decoration.
    import textwrap

    if n_shown:
        head = (f"N {n_shown} steps | weighting {fit.weighting} | "
                f"seeds {fit.starts_strategy}")
    else:
        head = (f"MODEL ONLY, no data supplied | fit N {fit.n_observations} | "
                f"weighting {fit.weighting}")
    limit = (f"fractions are the readout. D1 {D[0]:.3f} is at or below the localisation "
             f"floor (0.011 at 30 nm, 0.072 at 76 nm): a BOUND, not a coefficient")
    caption = textwrap.fill(head, 62) + "\n" + textwrap.fill(limit, 62)
    ax.set_title(caption, loc="left", fontsize=5.0, linespacing=1.35)
    return ax


# =====================================================================================
# STATE-COUNT SELECTION (`sqd.select_n_pop`)
# =====================================================================================
#
# BIC RECOMMENDS, THE USER DECIDES, and the figure has to show both or it becomes an
# assertion. The recommendation is marked, any override is marked separately WITH its
# written reason, and a gap below 2 is drawn as inconclusive rather than resolved.


DELTA_BIC_INCONCLUSIVE = 2.0


def draw_n_pop_selection(ax, selection, colour="#4a5568", best_colour="#2f855a",
                         chosen_colour="#c53030"):
    """Delta BIC per state count. ``selection`` is an ``NPopSelection``."""
    import numpy as np

    table = selection.table()
    n_pop = table["n_pop"].to_numpy()
    delta = table["delta_bic"].to_numpy(dtype=float)
    colours = [best_colour if r else colour for r in table["recommended"]]
    if "chosen" in table.columns:
        colours = [chosen_colour if c else col
                   for c, col in zip(table["chosen"], colours)]
    ax.bar(n_pop, delta, color=colours, width=0.55)
    # SYMLOG WITH THE THRESHOLD AT THE DECISION BOUNDARY. On a linear axis a delta of
    # 2246 for one population flattens the 0-against-13 comparison that actually decides
    # the count, and that comparison is the whole reason to draw this. symlog is linear
    # below 2 (exactly the span called inconclusive) and logarithmic above it, so both
    # scales are readable and the axis changes character where the decision does.
    ax.set_yscale("symlog", linthresh=DELTA_BIC_INCONCLUSIVE)
    ax.axhline(DELTA_BIC_INCONCLUSIVE, ls=":", lw=1.0, color="#718096")
    ax.text(n_pop.min() - 0.5, DELTA_BIC_INCONCLUSIVE * 1.15, "inconclusive below 2",
            fontsize=5.5, va="bottom", ha="left", color="#718096")
    for x, d, row in zip(n_pop, delta, table.itertuples()):
        ax.text(x, max(d, DELTA_BIC_INCONCLUSIVE * 1.05),
                f" {getattr(row, 'evidence_against', '')}", fontsize=5,
                rotation=90, va="bottom", ha="center")
    # THE RECOMMENDED BAR HAS ZERO HEIGHT, so its colour never shows and the one thing
    # the reader is looking for is the one thing invisible. Mark it on the axis.
    best = n_pop[np.asarray(table["recommended"], dtype=bool)]
    ax.plot(best, np.zeros_like(best, dtype=float), "^", ms=7, color=best_colour,
            clip_on=False, zorder=5)
    ax.set_xticks(n_pop)
    ax.set_xlim(n_pop.min() - 0.6, n_pop.max() + 0.6)
    ax.set_xlabel("populations", fontsize=8)
    ax.set_ylabel("$\\Delta$BIC from the best", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylim(0, delta.max() * 8)

    # `.recommended` is the FIT, not the count: printing it dumps the whole repr into the
    # caption, which is what the first version did.
    recommended_n = int(table.loc[table["recommended"], "n_pop"].iloc[0])
    head = f"BIC recommends {recommended_n}"
    chosen = getattr(selection, "chosen_n_pop", None)
    if chosen is not None and chosen != recommended_n:
        head += (f"; OVERRIDDEN to {chosen} - "
                 f"{getattr(selection, 'choice_reason', 'no reason recorded')}")
    limit = ("the recommendation is a statistical preference, not a biological verdict: "
             "a state whose D sits at the localisation floor may be measurement "
             "heterogeneity")
    ax.set_title(_wrap2(head, limit), loc="left", fontsize=5.0, linespacing=1.35)
    return ax


def _wrap2(head, limit, width=62):
    """Two wrapped caption lines. An overflowing caption collides with the next panel."""
    import textwrap
    return textwrap.fill(head, width) + "\n" + textwrap.fill(limit, width)


# =====================================================================================
# CLUSTERING AGAINST ITS NULL (`clustering.select_k`)
# =====================================================================================
#
# THE NULL IS THE WHOLE POINT AND MUST BE IN THE PICTURE. k-means always returns clusters
# and Calinski-Harabasz never says "none", so an observed CH drawn alone says nothing. v2
# drew exactly that. Here observed and chance are side by side and the EXCESS is what is
# read; a negative excess is a real result and is drawn as one rather than hidden.


def draw_k_selection(ax, selection, observed_colour="#2b6cb0", chance_colour="#a0aec0"):
    """Observed against chance Calinski-Harabasz per k. ``selection`` is a ``KSelection``."""
    import numpy as np

    table = selection.table
    k = table["k"].to_numpy(dtype=float)
    observed = table["observed"].to_numpy(dtype=float)
    chance = table["chance"].to_numpy(dtype=float)
    sd = table["chance_sd"].to_numpy(dtype=float)
    excess = table["excess"].to_numpy(dtype=float)

    width = 0.36
    ax.bar(k - width / 2, observed, width, color=observed_colour, label="observed")
    ax.bar(k + width / 2, chance, width, color=chance_colour, yerr=sd, capsize=2,
           error_kw={"lw": 0.8}, label="chance (free Brownian)")
    for xi, e in zip(k, excess):
        ax.text(xi, max(observed.max(), chance.max()) * 1.02,
                f"{e:+.0f}", fontsize=6, ha="center",
                color="#2f855a" if e > 0 else "#c53030")
    ax.set_xticks(k)
    ax.set_xlabel("clusters (k)", fontsize=8)
    ax.set_ylabel("Calinski-Harabasz", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="upper left")

    beats = excess > 0
    head = (f"feature {selection.feature} | {selection.n_tracks} tracks | "
            f"null D {selection.D_um2_s} um2/s | {int(beats.sum())} of {len(k)} k beat "
            f"the null")
    limit = ("a cluster is a CURVE SHAPE, not a diffusive state, and never the fraction "
             "readout. The excess is the number; the observed value alone means nothing "
             "because k-means always returns clusters")
    ax.set_title(_wrap2(head, limit), loc="left", fontsize=5.0, linespacing=1.35)
    return ax


# =====================================================================================
# CONFINEMENT AGAINST ITS CHANCE RATE (`confinement.confinement_null`)
# =====================================================================================
#
# A random walk revisits its own neighbourhood, so the detector reports confinement on
# purely Brownian tracks: at 150 nm and D = 0.2, 79 per cent of simulated FREE
# localisations are flagged. The RAW fraction rises monotonically with the radius and
# means nothing on its own. So observed, chance and excess are drawn together and the
# excess is labelled as the readout.


def draw_confinement_null(ax, null, observed_colour="#2b6cb0", chance_colour="#a0aec0"):
    """Observed against chance for each confinement quantity. ``null`` is a
    ``ConfinementNull``."""
    import numpy as np

    keys = [k for k in null.observed if k in null.chance]
    y = np.arange(len(keys), dtype=float)
    observed = np.array([null.observed[k] for k in keys], dtype=float)
    chance = np.array([null.chance[k] for k in keys], dtype=float)
    sd = np.array([null.chance_sd.get(k, 0.0) for k in keys], dtype=float)

    height = 0.36
    ax.barh(y + height / 2, observed, height, color=observed_colour, label="observed")
    ax.barh(y - height / 2, chance, height, color=chance_colour, xerr=sd, capsize=2,
            error_kw={"lw": 0.8}, label="chance (free Brownian)")
    for yi, o, c in zip(y, observed, chance):
        ax.text(max(o, c) + 0.02, yi, f"excess {o - c:+.3f}", fontsize=6, va="center",
                color="#2f855a" if o - c > 0 else "#c53030")
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace("_", " ") for k in keys], fontsize=6)
    ax.set_xlabel("fraction", fontsize=8)
    ax.set_xlim(0, 1.35)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="lower right")

    head = (f"radius {null.radius_um * 1000:.0f} nm | min steps {null.min_steps} | "
            f"null D {null.D_um2_s} um2/s | {null.n_replicates} replicates")
    limit = ("the EXCESS is the readout. The raw fraction rises monotonically with the "
             "radius and means nothing alone: at 150 nm and D = 0.2, 79 per cent of "
             "simulated FREE localisations are flagged confined")
    ax.set_title(_wrap2(head, limit), loc="left", fontsize=5.0, linespacing=1.35)
    return ax


# =====================================================================================
# THE HMM STATES (`hmm.fit_hmm`)
# =====================================================================================
#
# LEAD WITH RATES, NOT DWELL TIMES. The directly measured quantity is the per-frame
# transition probability; dwell times are model extrapolations that here EXCEED the
# observation window, so most are censored. And occupancy is a step-weighted TIME
# AVERAGE, not the stationary distribution - it sits between the initial distribution and
# the stationary one in 10 of 10 groups because a 5-step track never lets the chain mix.
# All three are drawn so none can be read as the others.


def draw_hmm_states(ax, fit, colours=("#2b6cb0", "#c53030", "#2f855a", "#805ad5")):
    """Per-state D and occupancy, with the switching rates. ``fit`` is an ``HMMFit``."""
    import numpy as np

    D = np.asarray(fit.D, dtype=float)
    occupancy = np.asarray(fit.occupancy, dtype=float)
    order = np.argsort(D)
    D, occupancy = D[order], occupancy[order]
    x = np.arange(len(D), dtype=float)

    bars = ax.bar(x, 100 * occupancy, width=0.55,
                  color=[colours[i % len(colours)] for i in range(len(D))])
    for xi, bar, d, o in zip(x, bars, D, occupancy):
        ax.text(xi, 100 * o + 1.5, f"D {d:.3f}\n{100 * o:.1f} %", fontsize=6,
                ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([f"state {i + 1}" for i in range(len(D))], fontsize=7)
    ax.set_ylabel("occupancy [%]", fontsize=8)
    ax.set_ylim(0, 118)
    ax.tick_params(labelsize=7)

    # The rates, from the transition matrix, per second.
    T = np.asarray(fit.transition_matrix, dtype=float)[np.ix_(order, order)]
    tau = float(fit.tau_s)
    rates = []
    for i in range(len(D)):
        for j in range(len(D)):
            if i != j:
                rates.append(f"{i + 1}->{j + 1} {T[i, j] / tau:.1f}/s")
    # EVERY RATE, NOT THE FIRST FOUR. The first version wrote ``rates[:4]``, which drops
    # two of the six rates of a 3-state fit and says nothing, the silent cap this whole
    # project refuses everywhere else. A 3-state chain has 6 rates and a 4-state one 12;
    # the caption wraps, which is the correct way to fit them rather than dropping them.
    head = (f"{fit.n_steps} steps in {fit.n_sequences} tracks | "
            f"{fit.n_converged_starts}/{fit.n_starts} starts converged | "
            f"{len(rates)} rates: " + " | ".join(rates))
    limit = ("rates are measured; occupancy is a step-weighted TIME AVERAGE and not the "
             "stationary distribution; dwell times are model extrapolations that exceed "
             "the observation window here, so most are censored")
    ax.set_title(_wrap2(head, limit), loc="left", fontsize=5.0, linespacing=1.35)
    return ax


# =====================================================================================
# THE MSD CURVE AND ITS FIT (`smtracker.msd`)
# =====================================================================================
#
# THE WHOLE POINT OF THIS FIGURE IS WHICH LAGS WERE FITTED. v2 fitted the full curve,
# which is worth 51 per cent in D on this data, because the tail flattens: it is a
# shrinking, slow-biased subset as fast molecules leave focus. So the fitted lags are
# drawn solid and filled, the unfitted tail is drawn open and greyed, and the caption
# says how many of how many were used. A reader must not have to take that on trust.
#
# AND MSD IS A CROSS-CHECK, NEVER THE FRACTION READOUT. An ensemble MSD of a mixture is
# dominated by the fast state, measured here D_msd = 0.30 between SQD's 0.055 and 0.85,
# so a single D from this curve is not a population.


def draw_msd_fit(ax, fit, *, data=None, fitted_colour="#2b6cb0",
                 unfitted_colour="#a0aec0", line_colour="#c53030"):
    """Ensemble MSD with the early-lag fit. ``data`` is the ensemble table."""
    import numpy as np

    n_fitted = int(fit.n_lags_used)
    if data is not None and len(data):
        lag = np.asarray(data["lag_time_s"], dtype=float)
        msd = np.asarray(data["msd_um2"], dtype=float)
        sem = np.asarray(data["sem_um2"], dtype=float)
        used = np.arange(len(lag)) < n_fitted
        ax.errorbar(lag[used], msd[used], yerr=sem[used], fmt="o", ms=4,
                    color=fitted_colour, capsize=2, lw=0.9,
                    label=f"fitted ({n_fitted} lags)")
        if (~used).any():
            ax.errorbar(lag[~used], msd[~used], yerr=sem[~used], fmt="o", ms=4,
                        mfc="none", color=unfitted_colour, capsize=2, lw=0.9,
                        label="NOT fitted (tail)")
        grid = np.linspace(0, lag.max(), 100)
        n_total = len(lag)
    else:
        grid = np.linspace(0, float(np.max(fit.lag_times_s)), 100)
        n_total = n_fitted

    model = 4.0 * fit.D_um2_per_s * grid + fit.intercept_um2
    ax.plot(grid, model, "--", color=line_colour, lw=1.5,
            label=f"4Dt + b, D {fit.D_um2_per_s:.3f} um2/s")
    ax.set_xlabel("lag time [s]", fontsize=8)
    ax.set_ylabel("MSD [um$^2$]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="upper left")

    sigma_nm = (np.sqrt(max(fit.intercept_um2, 0.0) / 4.0) * 1000.0
                if fit.intercept_um2 > 0 else 0.0)
    head = (f"{fit.n_tracks} tracks | fitted {n_fitted} of {n_total} lags | "
            f"R2 {fit.r_squared:.3f} | intercept implies sigma {sigma_nm:.0f} nm")
    limit = ("MSD is a CROSS-CHECK, never the fraction readout: an ensemble MSD of a "
             "mixture is dominated by the fast state. The tail is excluded because it is "
             "a shrinking, slow-biased subset, and fitting it is worth 51 per cent in D")
    ax.set_title(_wrap2(head, limit), loc="left", fontsize=5.0, linespacing=1.35)
    return ax


# =====================================================================================
# THE DISPLACEMENT MIXTURE (`smtracker.jd`)
# =====================================================================================
#
# The model is a ZERO-MEAN NORMAL MIXTURE on SIGNED 1-D displacements with
# D = sigma^2/(2*tau), which is what `getDispl_gui` always produced. Signed, so the
# histogram is symmetric about zero and a fit that is not would be visible immediately.


def draw_jd_fit(ax, result, *, data=None, n_bins=60, data_colour="#2b6cb0",
                fit_colour="#c53030"):
    """Signed-displacement histogram with the fitted mixture. ``result`` is fit_jd's dict."""
    import numpy as np

    D = np.asarray(result["D_um2_s"], dtype=float)
    fractions = np.asarray(result["fraction"], dtype=float)
    tau = float(result["tau_s"])
    order = np.argsort(D)
    D, fractions = D[order], fractions[order]
    sigma = np.sqrt(2.0 * D * tau)

    if data is not None and len(np.asarray(data)):
        x = np.asarray(data, dtype=float)
        x = x[np.isfinite(x)]
        ax.hist(x, bins=n_bins, density=True, color=data_colour, alpha=0.55,
                label=f"measured ({len(x)} steps)")
        span = float(np.quantile(np.abs(x), 0.999))
        n_shown = len(x)
    else:
        span = 4.0 * float(sigma.max())
        n_shown = 0

    grid = np.linspace(-span, span, 400)
    total = np.zeros_like(grid)
    for k in range(len(D)):
        component = (fractions[k] / (sigma[k] * np.sqrt(2 * np.pi))
                     * np.exp(-0.5 * (grid / sigma[k]) ** 2))
        total += component
        ax.plot(grid, component, ":", lw=1.0, alpha=0.8,
                label=f"pop {k + 1}: {100 * fractions[k]:.1f} %, D {D[k]:.3f} um2/s")
    ax.plot(grid, total, "--", color=fit_colour, lw=1.5,
            label=f"{len(D)}-population fit")

    ax.set_xlabel("signed displacement [um]", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False)

    head = (f"N {n_shown if n_shown else result['n_displacements']} signed displacements "
            f"| tau {tau * 1000:.0f} ms | zero-mean normal mixture, D = sigma^2/(2 tau)")
    if n_shown == 0:
        head += " | MODEL ONLY, no data supplied"
    limit = ("SIGNED displacements, so the histogram is symmetric about zero by "
             "construction. The fractions are the readout; the smallest D sits near the "
             "localisation floor and is a bound")
    ax.set_title(_wrap2(head, limit), loc="left", fontsize=5.0, linespacing=1.35)
    return ax


# =====================================================================================
# THE GAMMA MIXTURE ON PER-TRACK appD (`smtracker.appd_gamma`)
# =====================================================================================
#
# Shape = n_steps and mean = D, so the width of each component is SET BY THE TRACK LENGTH
# rather than fitted. Passing the median track length instead of the exact value widens
# every component without moving any fitted D, which is invisible in the fitted numbers
# and visible here.


def draw_gamma_fit(ax, fit, *, data=None, n_bins=60, data_colour="#2b6cb0",
                   fit_colour="#c53030"):
    """Per-track appD histogram with the fitted gamma mixture."""
    import numpy as np
    from scipy import special

    # STRAIGHT OFF THE FIT. The first version read n_steps from `fit.table()`, which has
    # only population / D / fraction, so it came back None, the model block was skipped,
    # and the figure drew the histogram ALONE with "gamma shape = None steps" in the
    # caption. It looked like a successful render and the smoke test passed it, because
    # a title existed. A figure that silently omits the fit is the defect this project
    # refuses everywhere else, so a missing shape now RAISES.
    D = np.asarray(fit.D, dtype=float)
    fractions = np.asarray(fit.fraction, dtype=float)
    n_steps = getattr(fit, "n_steps", None)
    if not n_steps:
        raise ValueError(
            "the fit carries no n_steps, so the gamma components cannot be drawn. "
            "Drawing the histogram alone would look like a fit that happens to be "
            "invisible")
    n_steps = int(n_steps)
    order = np.argsort(D)
    D, fractions = D[order], fractions[order]

    if data is not None and len(np.asarray(data)):
        x = np.asarray(data, dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
        ax.hist(x, bins=n_bins, density=True, color=data_colour, alpha=0.55,
                label=f"measured ({len(x)} tracks)")
        grid = np.linspace(1e-6, float(np.quantile(x, 0.995)), 400)
        n_shown = len(x)
    else:
        grid = np.linspace(1e-6, 4 * float(D.max()), 400)
        n_shown = 0

    total = np.zeros_like(grid)
    for k in range(len(D)):
        scale = D[k] / n_steps
        log_pdf = (special.xlogy(n_steps - 1, grid) - grid / scale
                   - special.gammaln(n_steps) - n_steps * np.log(scale))
        component = fractions[k] * np.exp(log_pdf)
        total += component
        ax.plot(grid, component, ":", lw=1.0, alpha=0.8,
                label=f"pop {k + 1}: {100 * fractions[k]:.1f} %, D {D[k]:.3f} um2/s")
    ax.plot(grid, total, "--", color=fit_colour, lw=1.5,
            label=f"{len(D)}-population gamma")

    ax.set_xlabel("apparent D [um$^2$/s]", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False)

    head = (f"N {n_shown} per-track appD | gamma shape = {n_steps} steps"
            + ("" if n_shown else " | MODEL ONLY, no data supplied"))
    limit = ("the component WIDTH is set by the track length, not fitted, so a wrong "
             "n_steps widens every component without moving any D. appD averages over "
             "within-track switching and COMPRESSES a condition difference")
    ax.set_title(_wrap2(head, limit), loc="left", fontsize=5.0, linespacing=1.35)
    return ax


# =====================================================================================
# SPOT-ON (`smtracker.spoton`)
# =====================================================================================
#
# dZ IS UNMEASURED AND MOVES THE ANSWER BY TENS OF POINTS: the corrected slow fraction differs
# widely between a corrected dZ and no correction at all. So the dZ used
# is printed at the same size as the fractions, and the per-lag z-corrections are drawn
# because they are what the number rests on.


def draw_spoton_fit(ax, fit, colours=("#2b6cb0", "#c53030", "#2f855a", "#805ad5")):
    """Per-state D and fraction, with the defocalisation correction that produced them."""
    import numpy as np

    D = np.asarray(fit.D, dtype=float)
    fractions = np.asarray(fit.fractions, dtype=float)
    order = np.argsort(D)
    D, fractions = D[order], fractions[order]
    x = np.arange(len(D), dtype=float)

    ax.bar(x, 100 * fractions, width=0.55,
           color=[colours[i % len(colours)] for i in range(len(D))])
    for xi, d, f in zip(x, D, fractions):
        ax.text(xi, 100 * f + 1.5, f"D {d:.3f}\n{100 * f:.1f} %", fontsize=6,
                ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([f"state {i + 1}" for i in range(len(D))], fontsize=7)
    ax.set_ylabel("fraction [%]", fontsize=8)
    ax.set_ylim(0, 118)
    ax.tick_params(labelsize=7)

    corrections = np.asarray(getattr(fit, "z_corrections", []), dtype=float)
    head = (f"{fit.n_jumps} jumps | lags {list(fit.lags)} | dZ {fit.dZ_um} um | "
            f"{fit.n_converged_starts}/{fit.n_starts} starts converged")
    if corrections.size:
        head += (f" | z-correction {corrections.min():.3f} to {corrections.max():.3f}")
    limit = ("dZ IS UNMEASURED and the data cannot supply it: the corrected slow fraction "
             "swings by tens of points between a corrected dZ and none. Run dz_sensitivity before quoting any corrected fraction")
    ax.set_title(_wrap2(head, limit), loc="left", fontsize=5.0, linespacing=1.35)
    return ax


# =====================================================================================
# THE CONFINEMENT CLASSES (`confinement.confinement_events`)
# =====================================================================================
#
# free / mixed / confined per track, as v2 printed them. THE CHANCE RATE IS NOT IN THIS
# FIGURE and the caption says so: this is the observed split, and a split without its null
# is the thing `draw_confinement_null` exists to supply. Drawing them together would let a
# reader take the raw fraction as a result, which is exactly what v2 invited.


def draw_confinement_classes(ax, events, colours=("#a0aec0", "#4a5568", "#2b6cb0")):
    """Track-class split and the per-track confined fraction."""
    import numpy as np

    classes = ["free", "mixed", "confined"]
    counts = [int((events["track_class"] == c).sum()) for c in classes]
    total = max(sum(counts), 1)
    bars = ax.bar(np.arange(3), counts, width=0.55, color=list(colours))
    for x, (c, n) in enumerate(zip(classes, counts)):
        ax.text(x, n, f"{n}\n{100 * n / total:.1f} %", fontsize=6, ha="center",
                va="bottom")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(classes, fontsize=7)
    ax.set_ylabel("tracks", fontsize=8)
    ax.set_ylim(0, max(counts) * 1.25)
    ax.tick_params(labelsize=7)

    attrs = getattr(events, "attrs", {})
    head = (f"{total} tracks | radius {attrs.get('radius_um', float('nan')) * 1000:.0f} nm "
            f"| min steps {attrs.get('min_steps', '?')} | confined segment fraction "
            f"{attrs.get('confined_segment_fraction', float('nan')):.3f}")
    limit = ("THIS FIGURE HAS NO CHANCE RATE IN IT. A random walk revisits its own "
             "neighbourhood, so the detector flags free Brownian tracks; run "
             "confinement_null before reading any of these numbers as a result")
    ax.set_title(_wrap2(head, limit), loc="left", fontsize=5.0, linespacing=1.35)
    return ax


# =====================================================================================
# TRANSITION COUNTS (`confinement.transitions`)
# =====================================================================================
#
# THE TWO STRICTNESS SETTINGS ARE THE POINT. The legacy guards an interior gap with three
# steps and guards the head and tail with nothing, and that unguarded branch supplies
# about HALF of all counted transitions (median ratio 2.07 across twelve groups). Both
# totals are drawn and NEITHER is called the difference.


def draw_transitions(ax, table, colour="#2b6cb0", other_colour="#a0aec0"):
    """Per-track transition counts, with both strictness totals."""
    import numpy as np

    counts = np.asarray(table["n_transitions"], dtype=float)
    edges = np.arange(-0.5, max(counts.max(), 3) + 1.5)
    ax.hist(counts, bins=edges, color=colour, alpha=0.8)
    ax.set_xlabel("transitions per track", fontsize=8)
    ax.set_ylabel("tracks", fontsize=8)
    ax.tick_params(labelsize=7)

    attrs = getattr(table, "attrs", {})
    this = attrs.get("n_transitions", int(counts.sum()))
    other = attrs.get("n_transitions_other_strictness")
    head = (f"{len(counts)} tracks | {this} transitions at strict="
            f"{attrs.get('strict', '?')} | {other} under the other strictness")
    limit = ("the legacy guards an interior gap with three steps and the head and tail "
             "with nothing, and that unguarded branch supplies about half of all counted "
             "transitions. Both totals are reported and neither is the difference")
    ax.set_title(_wrap2(head, limit), loc="left", fontsize=5.0, linespacing=1.35)
    return ax


# =====================================================================================
# AXIAL ORDER AGAINST ITS CHANCE RATE (`directionality.axial_null`)
# =====================================================================================
#
# A ROD-SHAPED CELL PRODUCES AXIAL ANISOTROPY WITH NO BIOLOGY IN IT, and the chance rate
# GROWS WITH STEP SIZE: +0.009 at D = 0.055 against +0.055 at D = 0.85 in a 3.0 x 0.9 um
# cell. So a low-D and a high-D population cannot be compared to each other, only each to
# its own chance rate, and the figure has to make that impossible to miss.


def draw_axial_order(ax, result, observed_colour="#2b6cb0", chance_colour="#a0aec0"):
    """Observed axial order against the chance rate for the same geometry and D."""
    import numpy as np

    observed = float(result["observed"])
    chance = float(result["chance"])
    sd = float(result.get("chance_sd", 0.0) or 0.0)
    ax.bar([0], [observed], width=0.5, color=observed_colour, label="observed")
    ax.bar([1], [chance], width=0.5, color=chance_colour, yerr=[sd], capsize=3,
           error_kw={"lw": 0.9}, label="chance (free, same box and D)")
    ax.axhline(0.0, color="#4a5568", lw=0.8)
    ax.text(0.5, max(observed, chance) * 1.08 + 0.002,
            f"excess {observed - chance:+.4f}", fontsize=6.5, ha="center",
            color="#2f855a" if observed > chance else "#c53030")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["observed", "chance"], fontsize=7)
    ax.set_ylabel("mean cos 2theta  (+1 = along the cell)", fontsize=7.5)
    ax.tick_params(labelsize=7)

    head = (f"D {result['D_um2_per_s']} um2/s | cell {result['cell_length_um']} x "
            f"{result['cell_width_um']} um (aspect {result['aspect_ratio']:.2f}) | "
            f"step sigma {result['sigma_step_um'] * 1000:.0f} nm | "
            f"{result['n_steps_observed']} observed against "
            f"{result['n_steps_null']} simulated steps")
    limit = ("THE CHANCE RATE GROWS WITH STEP SIZE, so a faster population picks up more "
             "axial order from the geometry alone. Read each population against ITS OWN "
             "chance rate, never against another population")
    ax.set_title(_wrap2(head, limit), loc="left", fontsize=5.0, linespacing=1.35)
    return ax


# =====================================================================================
# WHERE SWITCHING HAPPENS (`smtracker.switching`)
# =====================================================================================
#
# TWO PROFILES, NEVER ONE. Capture and release are different questions -- where binding
# begins and where it ends -- and pooling them answers neither. Each is drawn against its
# OWN denominator: capture per FAST step, release per SLOW step, because capture can only
# happen from fast and release only from slow. A per-total-step version of this figure
# came out with the opposite sign, so the denominator is printed on the axis label.
#
# THE GRADIENT IS NOT A NULL-TESTED QUANTITY BY ITSELF. The caption says whether a null
# was supplied and refuses to imply one that was not run.


def draw_switching_profiles(ax, profiles, *, null=None, colours=("#2b6cb0", "#c53030"),
                            null_colour="#a0aec0"):
    """Capture and release rate against distance from the axis.

    ``profiles`` is ``{event: radial_profile table}``. ``null`` is the same mapping from a
    spatially-uniform simulation, or ``None`` -- in which case the caption says so rather
    than letting a bare gradient read as a tested result.
    """
    import numpy as np

    for colour, (event, table) in zip(colours, sorted(profiles.items())):
        shell = np.asarray(table["shell"], dtype=float)
        rate = np.asarray(table["rate"], dtype=float)
        n = np.asarray(table["n_eligible"], dtype=float)
        # Binomial standard error on the rate: the outer shell is always the thinnest and
        # a bare line there invites reading noise as a trend.
        err = np.sqrt(np.clip(rate * (1 - rate), 0, None) / np.maximum(n, 1))
        ax.errorbar(shell, rate, yerr=err, fmt="o-", color=colour, lw=1.6, ms=5,
                    capsize=2, label=f"{event} ({table.attrs['gradient']:+.4f})")
        if null is not None and event in null:
            nt = null[event]
            ax.plot(np.asarray(nt["shell"], dtype=float),
                    np.asarray(nt["rate"], dtype=float), "s--", color=null_colour,
                    lw=1.2, ms=4, label=f"{event}, uniform null")

    ax.set_xticks(sorted({int(s) for t in profiles.values() for s in t["shell"]}))
    ax.set_xlabel("shell   0 = axis, last = membrane", fontsize=7.5)
    ax.set_ylabel("events per step OF THE ELIGIBLE STATE", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False)

    counts = " | ".join(f"{e} n={int(t['n_events'].sum())}"
                        for e, t in sorted(profiles.items()))
    head = counts + " | error bars are binomial on the rate"
    if null is None:
        limit = ("NO NULL SUPPLIED, so a gradient here is not a tested result: a rod-shaped "
                 "cell and a state-dependent decoder can both produce one. Simulate "
                 "uniform switching and decode it with the same fit before reading it")
    else:
        limit = ("the null is spatially uniform switching decoded by the SAME fit, so it "
                 "measures decoder geometry. It does NOT model the cell filter, whose "
                 "loss is D-selective at the boundary and must be tested by re-running")
    ax.set_title(_wrap2(head, limit), loc="left", fontsize=5.0, linespacing=1.35)
    return ax


def draw_jump_density_map(ax, result, *, population=None):
    """One population's jump-density map inside the cell outline.

    One axes, many populations: the panel gives each condition a single axes, so this
    draws ONE population and names it rather than silently picking. The default is the
    SLOWEST, because that is the one the spatial question is usually about; `population`
    takes an index into the fitted D order.

    The caption states the assignment, the weight behind the map and the fold, because
    each changes what the picture means: soft maps blend the populations, hard maps can
    omit one entirely, and a folded map is four-fold symmetric by construction.
    """
    import numpy as np
    import matplotlib.patheffects as pe
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch
    from scipy.ndimage import gaussian_filter

    from .spatialsqd import capsule_outline, local_half_width

    cfg = result["config"]
    labels = list(result["labels"])
    D = np.asarray(result["D"], dtype=float)
    index = int(np.argmin(D)) if population is None else int(population)
    label = labels[index]
    grid = gaussian_filter(np.asarray(result["grids"][label], dtype=float),
                           sigma=cfg.smooth_bins)

    cap_x, cap_y = capsule_outline(cfg)
    half = cfg.cell_width_um / 2.0
    xc, yc = cfg.centres()
    XX, YY = np.meshgrid(xc, yc)
    inside = np.abs(YY) <= local_half_width(XX, cfg)
    vmax = float(np.nanpercentile(grid[inside], 99)) if inside.any() else 1.0

    image = ax.imshow(grid, origin="lower", cmap="jet", vmin=0.0, vmax=max(vmax, 1e-12),
                      extent=[0.0, cfg.cell_length_um, -half, half],
                      interpolation="bilinear", aspect="equal")
    clip = PathPatch(MplPath(np.column_stack([cap_x, cap_y])), transform=ax.transData,
                     facecolor="none", lw=0)
    ax.add_patch(clip)
    image.set_clip_path(clip)
    ax.plot(cap_x, cap_y, color="0.15", lw=1.2,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    ax.set_xlim(-0.05, cfg.cell_length_um + 0.05)
    ax.set_ylim(-half - 0.05, half + 0.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    weight = result["weights"][label]
    # ONE LINE, because the panel puts the condition on the CENTRED title of the same
    # axes. A comment upstream says the two cannot collide since one is left and one is
    # centred; drawn, a two-line left caption runs straight through the centred one.
    caption = (f"{label.split(' (')[0]} · D {D[index]:.3g} · "
               f"{weight:,.0f} {result['weight_unit'].replace('weighted', 'wt.')} · "
               f"{result['assignment']} · "
               f"{'folded' if result['mirrored'] else 'unfolded'}")
    if not result["hard_assignment_possible"]:
        caption += " · hard cannot separate these"
    ax.set_title(caption, loc="left", fontsize=5.5)
    return ax


def draw_consistency_panel(ax, result):
    """Every estimator's slow fraction side by side, with the ones that cannot be
    compared marked rather than dropped.

    Agreement across methods with different assumptions is the evidence, so the figure
    is built to make agreement and disagreement equally visible: one row per method,
    ordered as the table gives them, and a shaded band spanning the methods that DO
    agree. A method sitting outside that band is the finding.

    Three things are drawn that a bare bar chart would lose:

    - **MSD has no fraction at all** and is shown as a row with no marker and the reason
      written in. An absent bar reads as zero; a named absence does not. The ensemble
      MSD of a mixture is dominated by the fast state, so it never had a fraction to
      report - it is on the panel because its D lands BETWEEN the two states, which is
      itself the diagnosis.
    - **A method whose D_slow is at or below the localisation floor** gets a hollow
      marker and a floor tag, because its slow state is a bound and not a measurement.
      That is the panel's "refusing false precision" job, and it must survive being
      looked at rather than living in a column.
    - **A method that FAILED** keeps its row and prints its error. This module's own
      table does that; throwing it away at the drawing stage would undo it.

    The agreement band is the inter-quartile range of the methods that reported a
    fraction, NOT a mean with error bars: these are different estimators of one
    quantity, not repeated measurements of it, so their spread is not an uncertainty
    and must not be drawn as one.
    """
    import numpy as np

    rows = [r for _, r in result.iterrows()]
    y = np.arange(len(rows))[::-1]

    quoted = [float(r["slow_fraction_percent"]) for r in rows
              if r["ok"] and r["slow_fraction_percent"] == r["slow_fraction_percent"]]
    if quoted:
        lo, hi = np.percentile(quoted, [25, 75])
        ax.axvspan(lo, hi, color="0.90", zorder=0)
        ax.axvline(float(np.median(quoted)), color="0.55", lw=1.0, ls="--", zorder=1)

    labels = []
    for pos, r in zip(y, rows):
        method = str(r["method"])
        value = r["slow_fraction_percent"]
        has_value = bool(r["ok"]) and value == value and value is not None
        floored = "FLOOR" in str(r.get("limits", "")).upper().split("|")[0]
        if not r["ok"]:
            ax.text(0.02, pos, f"failed: {str(r['note'])[:60]}", va="center",
                    fontsize=6.5, color="#a1362a",
                    transform=ax.get_yaxis_transform())
        elif not has_value:
            ax.text(0.02, pos, "no fraction: a mixture average, not a state",
                    va="center", fontsize=6.5, color="0.35",
                    transform=ax.get_yaxis_transform())
        else:
            ax.plot([float(value)], [pos], marker="o", markersize=8,
                    markerfacecolor="white" if floored else "#3c6e9f",
                    markeredgecolor="#3c6e9f", markeredgewidth=1.6, zorder=3)
        switching = str(r["switching_aware"]).strip().lower() in ("yes", "true")
        labels.append(f"{method}{'  (switching-aware)' if switching else ''}")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("slow fraction (%)", fontsize=9)
    ax.set_xlim(0, max(60.0, (max(quoted) + 10.0) if quoted else 60.0))
    ax.grid(axis="x", color="0.92", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    n_ok = sum(1 for r in rows if r["ok"])
    spread = (f"{max(quoted) - min(quoted):.1f} pp apart" if len(quoted) > 1
              else "one method")
    first = (f"cross-method consistency · {n_ok}/{len(rows)} ran · {len(quoted)} report "
             f"a fraction, {spread}\nband = their interquartile range: a spread across "
             f"estimators, NOT an uncertainty")

    # THE FLOOR VERDICT IS A FUNCTION OF AN UNMEASURED NUMBER, so the figure names the
    # sigma it used. On this data every method's slow D sits below the floor at the
    # PSF-derived sigma and ABOVE it at 30 nm, so a panel that drew the hollow markers
    # without saying which sigma produced them would be presenting a choice as a result.
    n_floored = sum(1 for r in rows
                    if "FLOOR" in str(r.get("limits", "")).upper().split("|")[0])
    params = getattr(result, "attrs", {}).get("params")
    floor = getattr(params, "localization_floor_D", None) if params is not None else None
    if n_floored:
        if floor is not None:
            sigma_nm = getattr(params, "localization_precision_nm", None)
            sigma = (f"σ {sigma_nm:.0f} nm" if sigma_nm else "σ from the PSF, unmeasured")
            first += (f"\nhollow = a bound: {n_floored} of {len(quoted)} slow D below "
                      f"the {floor:.3f} µm²/s floor ({sigma})")
        else:
            first += (f"\nhollow = a bound: {n_floored} of {len(quoted)} slow D at or "
                      f"below the localisation floor")
    ax.set_title(first, loc="left", fontsize=6.0, linespacing=1.5)
    return ax


def draw_crop_curves(figure, curves, *, crop=None, proposal=None):
    """The two curves the crop is read from, with the floor and the chosen range.

    Two axes, one figure, because the decision is made by reading them TOGETHER: the
    frame mean says how far the field has bleached, the spot density says whether what
    is left is still single molecules. Either alone can be misread.

    The floor is drawn as a floor. It is the earliest start the images defend, not a
    recommendation (the three rules behind it disagree sixfold on real data) so it is
    a line with a label saying so, never a shaded "suggested region". A proposal, when
    one has been asked for, is drawn separately and names its rule.
    """
    import numpy as np

    figure.clear()
    # THREE PANELS: the two full-length curves, and the KEPT RANGE ON ITS OWN SCALE.
    # The legacy macro drew that third curve for a reason, the whole-acquisition plot
    # is dominated by the first few hundred frames, so a kept range that is still
    # bleaching looks perfectly flat next to the spike. Reading the crop alone is how an
    # analyst decides the field has reached single molecules.
    top, bottom, zoom = figure.subplots(3, 1, height_ratios=(1.0, 1.0, 1.1))

    # NOTHING IS "KEPT" WHEN NOTHING IS CROPPED. A range covering every frame painted the
    # entire curve orange, which reads as "all of this was selected" when it means "no
    # crop has been set": the experimenter's question was literally why series006 was
    # completely orange. Those are opposite states and must not share a colour.
    whole = crop is None or (crop[0] <= 1 and crop[1] >= int(curves.n_frames))

    # ONE QUANTITY PER PANEL, IN TWO COLOURS: the WHOLE acquisition in grey-blue and the
    # KEPT range over it in orange. The first version drew the frame mean in blue and
    # the spot density in orange, which reads as two curves to compare rather than one
    # curve and the part of it being kept: the experimenter's first question was
    # literally "what's the blue and orange curve?".
    #
    # WHAT THIS MAKES ANSWERABLE is the question the crop is actually for: is the KEPT
    # part flat, or still falling? An orange segment that still slopes means the field
    # is still bleaching down and the start is too early.
    def draw(axis, frame, value, colour_full="#7f9bb5", colour_kept="#d9791f"):
        """Returns the line artists it made, in order.

        Returned rather than looked up. `axvline` appends to `axis.lines` too, so
        indexing `lines[:2]` afterwards picked up the dashed FLOOR as the second curve
        and handed the legend three handles for two labels: matplotlib warned, and the
        legend would have named the wrong artist.
        """
        frame = np.asarray(frame, dtype=float)
        value = np.asarray(value, dtype=float)
        made = list(axis.plot(frame, value, lw=0.8, color=colour_full, zorder=1))
        if not whole:
            keep = (frame >= crop[0]) & (frame <= crop[1])
            if keep.any():
                made += list(axis.plot(frame[keep], value[keep], lw=1.3,
                                       color=colour_kept, zorder=3))
        axis.tick_params(labelsize=7)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        return made

    curve_lines = draw(top, curves.bleaching["frame"], curves.bleaching["mean_counts"])
    top.set_ylabel("frame mean", fontsize=8)
    draw(bottom, curves.density["frame"], curves.density["n_spots"])
    bottom.set_ylabel("spots / frame", fontsize=8)
    bottom.tick_params(labelbottom=False)

    # --- the kept range, on its own scale --------------------------------------
    # IT ALWAYS SHOWS A RANGE, NEVER THE WHOLE MOVIE AGAIN. With no crop set the first
    # version drew every frame here, which is panel one a second time, and the spike
    # then dominates this axis too, so the panel that exists to reveal a slow decline
    # showed nothing but the bleach-down. With no crop it previews what the FLOOR would
    # give, which is the question actually being asked at that moment.
    frames = np.asarray(curves.bleaching["frame"], dtype=float)
    means = np.asarray(curves.bleaching["mean_counts"], dtype=float)
    floor_start = None
    if curves.floor is not None and curves.floor.earliest_defensible is not None:
        floor_start = int(curves.floor.earliest_defensible)

    if not whole:
        lo, hi, colour, prefix = crop[0], crop[1], "#d9791f", ""
    elif floor_start is not None:
        lo, hi, colour, prefix = (floor_start, int(curves.n_frames), "#7f9bb5",
                                  f"preview at the floor {floor_start}: ")
    else:
        lo, hi, colour, prefix = 1, int(curves.n_frames), "#7f9bb5", ""

    inside = (frames >= lo) & (frames <= hi)
    zoom.plot(frames[inside], means[inside], lw=0.8, color=colour)
    zoom.set_ylabel("kept range" if not whole else "preview", fontsize=8)
    zoom.set_xlabel("frame", fontsize=8)
    zoom.tick_params(labelsize=7)
    for side in ("top", "right"):
        zoom.spines[side].set_visible(False)

    verdict = ""
    try:
        from .movieprep import crop_quality

        quality = crop_quality(curves, lo, hi)
    except Exception:                                  # noqa: BLE001 - cosmetic
        quality = None
    if quality is not None:
        verdict = (f"{prefix}{quality['verdict']}; "
                   f"{quality['mean_spots_per_frame']:.1f} spots/frame (mean)")
        zoom.set_title(verdict, loc="left", fontsize=7.5, color="#555")

    # THE TITLE CARRIES THE NUMBERS, so the legend does not have to. Four lines of
    # legend text is four lines nobody reads.
    kept = "NO CROP SET: all frames" if whole else f"keeping {crop[0]}–{crop[1]}"
    top.set_title(f"{curves.movie}    {kept}", loc="left", fontsize=9,
                  fontweight="bold")

    floor = curves.floor
    for axis in (top, bottom):
        if floor is not None and floor.earliest_defensible is not None:
            axis.axvline(floor.earliest_defensible, color="#a1362a", ls="--", lw=1.0,
                         zorder=2)
        if proposal is not None:
            for edge in (proposal["start"], proposal["end"]):
                axis.axvline(edge, color="#2f8f5b", lw=1.0, ls=":", zorder=2)

    # TWO ENTRIES AT MOST. The curve lines are handles 0 and 1 by construction above;
    # the dashed floor is named in the third only when there is one.
    handles = list(curve_lines)
    labels = ["whole acquisition", "kept"][:len(handles)]
    if floor is not None and floor.earliest_defensible is not None:
        from matplotlib.lines import Line2D

        handles.append(Line2D([], [], color="#a1362a", ls="--", lw=1.0))
        labels.append(f"no earlier than {floor.earliest_defensible}")
    top.legend(handles, labels, fontsize=6.5, loc="upper right", framealpha=0.9)

    # THE EDGES ARE DRAWN AS HANDLES so the panel can make them draggable. They are
    # returned rather than hunted for afterwards: searching an axis for "the line at
    # x = start" breaks the moment anything else draws a vertical line, and this figure
    # already has two (the floor, and the proposal).
    edges = []
    if crop is not None:
        for index, edge in enumerate((crop[0], crop[1])):
            for axis in (top, bottom):
                line = axis.axvline(edge, color="#d9791f", lw=1.6, alpha=0.9, zorder=5)
                line.set_picker(True)
                edges.append((index, line))
    figure.crop_edges = edges
    return top, bottom


def draw_bleaching_pair(figure, full, cropped, *, movie: str, crop=None,
                        frame_interval_s: float | None = None):
    """The macro's two bleaching plots, from the CSVs this project writes.

    The picture is back, and the numbers stay. The Fiji macro saved each curve as a TIFF
    **picture of a plot**, which cannot be rescaled, re-read or checked against anything;
    this project writes the curve as CSV for exactly that reason. But the CSV was meant
    to be ADDED, not to replace the image: nobody opens a spreadsheet to see whether a
    movie bleached down, and the experimenter noticed the picture had gone.

    So both ship: the CSV is the record, this is the view, and the view is drawn from
    the record so the two cannot disagree.

    ``frame_interval_s``, when given, adds a seconds axis: the macro's own x axis is
    integration time rather than frame number.
    """
    figure.clear()
    top, bottom = figure.subplots(2, 1)

    # THE KEPT PART IS ORANGE HERE TOO. Two vertical lines mark where the crop is; they
    # do not show WHICH SIDE was kept, and the exported picture then disagreed with the
    # panel the crop was chosen in, same data, same colours available, different
    # meaning. One convention: blue is the whole acquisition, orange is what is kept.
    import numpy as np

    frames = np.asarray(full["frame"], dtype=float)
    values = np.asarray(full["mean_counts"], dtype=float)
    top.plot(frames, values, lw=0.8, color="#7f9bb5")
    if crop is not None:
        keep = (frames >= crop[0]) & (frames <= crop[1])
        if keep.any():
            top.plot(frames[keep], values[keep], lw=1.2, color="#d9791f")
    top.set_ylabel("frame mean", fontsize=8)
    top.set_title(f"{movie}: whole acquisition", loc="left", fontsize=9,
                  fontweight="bold")

    bottom.plot(cropped["frame"], cropped["mean_counts"], lw=0.8, color="#d9791f")
    bottom.set_ylabel("frame mean", fontsize=8)
    bottom.set_xlabel("frame", fontsize=8)
    kept = f": kept {crop[0]}–{crop[1]}" if crop is not None else ""
    bottom.set_title(f"cropped range{kept}", loc="left", fontsize=9)

    if frame_interval_s:
        seconds = bottom.secondary_xaxis(
            "top", functions=(lambda f: f * frame_interval_s,
                              lambda t: t / frame_interval_s))
        seconds.set_xlabel("integration time (s)", fontsize=7)
        seconds.tick_params(labelsize=6)

    for axis in (top, bottom):
        axis.tick_params(labelsize=7)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    return top, bottom


#: How each SERIES is distinguished, in that order. The encoding is deliberately
#: REDUNDANT -- a shade AND an outline style -- because this project has already
#: recorded that a colour-only encoding does not survive greyscale reproduction, and a
#: reader should not have to trust the print pipeline to tell two fits apart.
BUBBLE_SERIES_STYLES = (
    {"fill": 1.00, "linestyle": "solid", "linewidth": 0.9},
    {"fill": 0.00, "linestyle": "dashed", "linewidth": 2.0},
    {"fill": 0.55, "linestyle": "dotted", "linewidth": 2.0},
    {"fill": 0.25, "linestyle": "dashdot", "linewidth": 2.0},
)


def _lighten(colour, amount):
    """Blend towards white. ``amount`` 1.0 keeps the colour, 0.0 gives white."""
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(colour)
    return (1.0 - amount * (1.0 - r), 1.0 - amount * (1.0 - g),
            1.0 - amount * (1.0 - b))


def draw_bubble_panel(ax, table, *, condition_col="condition", series_col="series",
                      d_col="D_um2_per_s", fraction_col="fraction",
                      sd_col="fraction_sd", colour="#4878a8",
                      min_radius_points=6.0, max_radius_points=34.0,
                      annotate_population=1, log_y=True, title=None):
    """The bubble panel: D on a log axis, fraction as bubble AREA, one x per condition.

    Several **series** -- ``(fit_mode, n_pop)`` combinations from
    ``compare.bubble_series`` -- are drawn TOGETHER, offset within each condition, so a
    reader can see how much the arrangement is worth. Pass a one-series table and it
    draws a single set, which is the ordinary case.

    AREA CARRIES THE VALUE, so the radius goes as its square root. Scaling the radius
    with the fraction instead would exaggerate every difference by squaring it, which is
    the classic bubble-chart defect and is why this is computed rather than left to a
    default.

    THE STATE COUNT IS ON THE FIGURE, always. A fraction from a 2-population fit and one
    from a 3-population fit are different quantities; ``compare.compare_fractions``
    refuses to tabulate them together, and drawing them side by side is only defensible
    while the reader can see which is which. The series label carries it, so it cannot
    be cropped away from the number.

    ``annotate_population`` picks which population's fraction is printed under the axis
    (1 = the slowest, the usual readout). ``None`` prints none.
    """
    import numpy as np

    conditions = list(dict.fromkeys(table[condition_col]))
    series = list(dict.fromkeys(table[series_col]))
    styles = [BUBBLE_SERIES_STYLES[i % len(BUBBLE_SERIES_STYLES)]
              for i in range(len(series))]

    span = 0.0 if len(series) == 1 else 0.34
    offsets = (np.zeros(1) if len(series) == 1
               else np.linspace(-span, span, len(series)))

    def radius(fraction):
        return min_radius_points + (max_radius_points - min_radius_points) * np.sqrt(
            max(float(fraction), 0.0))

    if log_y:
        ax.set_yscale("log")
    ax.grid(axis="y", color="#eeeeee", lw=0.8)
    ax.set_axisbelow(True)

    for si, (name, style) in enumerate(zip(series, styles)):
        part = table[table[series_col] == name]
        face = _lighten(colour, style["fill"]) if style["fill"] > 0 else "white"
        edge = "#222222" if style["fill"] >= 1.0 else colour
        for ci, condition in enumerate(conditions):
            block = part[part[condition_col] == condition]
            if block.empty:
                continue
            x = ci + float(offsets[si])
            for _, row in block.iterrows():
                d = float(row[d_col])
                ax.scatter([x], [d], s=radius(row[fraction_col]) ** 2,
                           facecolor=face, edgecolor=edge,
                           linewidth=style["linewidth"], linestyle=style["linestyle"],
                           zorder=3)
                # The BAR marks the fitted D itself. A bubble has extent and a D does
                # not, so without it the eye reads the bubble's edge as an interval.
                ax.plot([x - 0.085, x + 0.085], [d, d],
                        color=edge if style["fill"] < 1.0 else "#222222",
                        lw=1.4, zorder=4, solid_capstyle="butt")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([str(c) for c in conditions], rotation=30, ha="right",
                       fontsize=9)
    ax.set_xlim(-0.6, len(conditions) - 0.4)
    ax.set_ylabel("Diffusion coefficient (µm²/s)", fontsize=10)
    if title:
        ax.set_title(str(title), fontsize=13, fontweight="bold")

    if annotate_population is not None:
        wanted = table[table["population"] == int(annotate_population)]
        for si, name in enumerate(series):
            face_colour = colour if si else "#222222"
            for ci, condition in enumerate(conditions):
                block = wanted[(wanted[series_col] == name)
                               & (wanted[condition_col] == condition)]
                if block.empty:
                    continue
                row = block.iloc[0]
                text = "%.0f%%" % (100.0 * float(row[fraction_col]))
                sd = float(row[sd_col]) if sd_col in block.columns else float("nan")
                if sd == sd:                       # not NaN
                    text += "\n± %.0f" % (100.0 * sd)
                ax.annotate(text, (ci + float(offsets[si]), 0.02),
                            xycoords=("data", "axes fraction"), ha="center",
                            fontsize=9, color=face_colour,
                            fontweight="bold" if si else "normal")

    return ax


def bubble_panel_legend(series, colour="#4878a8"):
    """Legend handles for :func:`draw_bubble_panel`, so a figure can place them itself.

    Returned rather than drawn: a multi-panel figure wants ONE legend, and a function
    that draws its own would give it as many as there are panels.
    """
    import matplotlib.pyplot as plt

    handles = []
    for i, name in enumerate(series):
        style = BUBBLE_SERIES_STYLES[i % len(BUBBLE_SERIES_STYLES)]
        handles.append(plt.Line2D(
            [], [], marker="o", ls="none", ms=11,
            markerfacecolor=(_lighten(colour, style["fill"]) if style["fill"] > 0
                             else "white"),
            markeredgecolor="#222222" if style["fill"] >= 1.0 else colour,
            markeredgewidth=style["linewidth"], label=str(name)))
    return handles
