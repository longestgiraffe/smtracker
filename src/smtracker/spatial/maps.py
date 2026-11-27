"""Mirrored full-cell maps + map_bins + diagnostics for ``smtracker.spatial``.

Pure figures: no ``plt.show``, no globals; every plot function accepts and returns
``(fig, ax)`` and computes nothing that the table builders do not already provide.

Three defects of the legacy appD speed map are fixed here and pinned by tests:
1. the map counts the wrong unit (one long immobile molecule fills one bin) so the
   default ``weight`` is ``'track'`` (each track total weight 1), not localisation;
2. a max-scaled colourmap wastes its range on a heavy tail, colour limits come from
   a percentile (``clim_pct``, default 99), with over-range bins flagged;
3. cell length is a covariate (``l_abs`` normalises per cell so the cap sits at a
   different ``l_abs`` in every cell) so ``length_window`` selects which cells enter
   and the axial profile must be read on BOTH ``l_abs`` and ``d_pole_um``.

Figure text is NEUTRAL: populations are named by their D range; never *bound*,
*free*, *slow*, *fast*, *mobile*, *immobile*, *binding*, *diffusing*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "MapError",
    "WEIGHTS",
    "FORBIDDEN_WORDS",
    "loc_weights",
    "select_length_window",
    "compute_map_bins",
    "contributor_sums",
    "effective_contributors",
    "compute_enrichment_bins",
    "map_diagnostics",
    "plot_cell_map",
    "plot_enrichment_map",
    "assert_figure_neutral",
    "shell_additive_pp",
    "radial_profile",
    "RADIAL_BANDS",
    "ENRICHMENT_FORMULA",
    "MIN_INDEPENDENT_VISITS",
    "shell_summary",
]

#: The statistic, as algebra, so no caller has to infer it from a name.
ENRICHMENT_FORMULA = (
    "enrichment = [Sw(pop in bin)/Sw(all in bin)] / [Sw(pop total)/Sw(all total)], "
    "w = 1/n_loc_track (unit = track). A ratio of shares, not area-normalised. It "
    "inflates when the global share is small, so it is not comparable across "
    "conditions with different baselines; use the percentage-point statistic."
)

#: Below this many independent visits (distinct tracks contributing to the region,
#: not localisations, which are correlated within a track) a region is not mappable
#: and ``shell_summary`` flags it instead of letting a number stand.
#:
#: PROVENANCE: unknown. Traced (ON3d) with ``git log -S`` on the literal: introduced in
#: 256af33 (2026-07-31), whose message justifies only WHAT is counted, "distinct
#: tracks (not localisations)", and never the value 50. No doc, comment or test cites
#: a basis for it, and no power calculation or simulation stands behind it. It is a
#: round number chosen by hand. Recorded as unsourced rather than given an invented
#: justification; if it ever becomes load-bearing, derive it or sweep it.
MIN_INDEPENDENT_VISITS = 50


class MapError(ValueError):
    """Raised when a map cannot be built (empty selection, bad weight/mode)."""


#: No interpretive vocabulary on any rendered figure text: populations are named by
#: their D range. ``assert_figure_neutral`` enforces it across every text artist.
FORBIDDEN_WORDS = (
    "bound", "free", "slow", "fast", "mobile", "immobile", "binding", "diffusing",
)


def assert_figure_neutral(fig) -> None:
    """Raise ``AssertionError`` if any text artist in ``fig`` (titles, axis + tick
    labels, legend entries + title, annotations, colourbar labels, figure texts)
    contains a :data:`FORBIDDEN_WORDS` substring. Deliberately strict: a word inside
    a longer token still trips it, which is the safe direction."""
    import matplotlib

    offenders = []

    def _scan(artist):
        if isinstance(artist, matplotlib.text.Text):
            content = artist.get_text().lower()
            for w in FORBIDDEN_WORDS:
                if w in content:
                    offenders.append((w, artist.get_text()))

    for ax in fig.axes:                                   # colourbar axes included
        # ax.title is ONLY the centred title; loc="left"/"right" are separate
        # artists absent from ax.texts. Read all three (public accessor).
        arts = [ax.xaxis.label, ax.yaxis.label]
        titles = [ax.get_title(loc=loc) for loc in ("left", "center", "right")]
        arts += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        arts += list(ax.texts)
        leg = ax.get_legend()
        if leg is not None:
            arts += list(leg.get_texts())
            if leg.get_title() is not None:
                arts.append(leg.get_title())
        for a in arts:
            _scan(a)
        for title in titles:
            for w in FORBIDDEN_WORDS:
                if w in title.lower():
                    offenders.append((w, title))
    for a in fig.texts:
        _scan(a)
    if offenders:
        raise AssertionError("figure prints interpretive word(s): "
                             + "; ".join(f"{w!r} in {t!r}" for w, t in offenders))


WEIGHTS = ("localisation", "track", "cell", "distinct_track")
_GROUP_KEYS = ("dataset", "enzyme", "condition", "replicate", "session")


def loc_weights(locs: pd.DataFrame, weight: str) -> np.ndarray:
    """Per-localisation weight under ``weight`` (see :data:`WEIGHTS`).

    ``'track'`` (default of the map) gives each track total weight 1, i.e.
    ``1/n_loc_track`` per localisation, so one long immobile molecule can no longer
    dominate a bin. ``'cell'`` gives each cell total weight 1 spread over its tracks.
    ``'localisation'`` is the legacy behaviour, kept only for comparison.
    """
    if weight not in WEIGHTS:
        raise MapError(f"weight must be one of {WEIGHTS}, got {weight!r}")
    n_loc_track = locs["n_loc_track"].to_numpy(dtype=float)
    if weight == "localisation":
        return np.ones(len(locs))
    if weight in ("track", "distinct_track"):
        return 1.0 / n_loc_track
    # cell: each cell total 1 → 1/(n_tracks_in_cell · n_loc_track)
    n_tracks_in_cell = locs.groupby("cell_id")["track_id"].transform("nunique")
    return 1.0 / (n_tracks_in_cell.to_numpy(dtype=float) * n_loc_track)


def select_length_window(locs, cells, length_window="iqr"):
    """Return the ``cell_id`` set kept by ``length_window`` and the number excluded.

    ``'iqr'`` keeps cells whose ``length_um`` is within the interquartile range;
    ``None`` keeps all; an explicit ``(lo, hi)`` tuple in µm overrides. Cell length
    is a covariate, not noise: a few long cells pull the pooled map toward their
    geometry and blur the pole region across ``l_abs`` bins.
    """
    lengths = cells.set_index("cell_id")["length_um"]
    if length_window is None:
        keep = set(lengths.index)
    elif length_window == "iqr":
        lo, hi = lengths.quantile([0.25, 0.75])
        keep = set(lengths[(lengths >= lo) & (lengths <= hi)].index)
    else:
        lo, hi = length_window
        keep = set(lengths[(lengths >= lo) & (lengths <= hi)].index)
    present = set(locs["cell_id"].unique())
    n_excluded = len(present - keep)
    return keep, n_excluded


def _grid(cells_used, bin_um):
    """(u_edges, v_edges, median_L, median_W, iqr_L), edges SYMMETRIC through 0 so
    the four-fold fold/reflect is exact."""
    L = float(np.median(cells_used["length_um"]))
    W = float(np.median(cells_used["width_um"]))
    q1, q3 = np.quantile(cells_used["length_um"], [0.25, 0.75])
    nu = int(np.ceil((L / 2.0) / bin_um))
    nv = int(np.ceil((W / 2.0) / bin_um))
    u_edges = np.arange(-nu, nu + 1) * bin_um
    v_edges = np.arange(-nv, nv + 1) * bin_um
    return u_edges, v_edges, L, W, (float(q1), float(q3))


def contributor_sums(bin_id, contributor, weight=None, n_bins: int | None = None):
    """Per bin, ``(S1, S2) = (Σ c_i, Σ c_i²)`` with ``c_i`` the summed weight of contributor
    ``i`` in that bin -- THE one place a bin's contributions are grouped (handover
    §4a.2, rule 5). Both spatial systems call it: ``compute_map_bins`` for its
    ``design_effect`` (S2/S1) and n_eff columns, and ``spatialsqd``/``spatialhmm``/
    ``spatialappd`` for the GUI's per-bin n_eff.

    ``bin_id`` is a non-negative integer per row, ``contributor`` any hashable label
    per row (a track, or a cell); ``weight`` defaults to 1 per row and may be
    fractional (a soft posterior). A contributor is grouped WITHIN a bin, so the same
    track in two bins counts in both. Returns two float arrays of length ``n_bins``
    (default ``max(bin_id) + 1``); a bin with no rows has ``S1 = S2 = 0``.
    """
    bin_id = np.asarray(bin_id, dtype=np.int64)
    codes, _ = pd.factorize(pd.Series(contributor, copy=False), sort=False)
    if (codes < 0).any():
        raise MapError("a contributor label is missing (NaN); give every row one")
    w = np.ones(len(bin_id)) if weight is None else np.asarray(weight, dtype=float)
    if not (len(bin_id) == len(codes) == len(w)):
        raise MapError("bin_id, contributor and weight must have one entry per row")
    n = int(n_bins) if n_bins is not None else (int(bin_id.max()) + 1 if len(bin_id) else 0)
    if len(bin_id) == 0:
        return np.zeros(n), np.zeros(n)
    span = int(codes.max()) + 1
    pair, inverse = np.unique(bin_id * span + codes, return_inverse=True)
    c = np.bincount(inverse, weights=w)                 # one entry per (bin, contributor)
    owner = pair // span
    s1 = np.bincount(owner, weights=c, minlength=n)
    s2 = np.bincount(owner, weights=c * c, minlength=n)
    return s1, s2


def effective_contributors(s1, s2) -> np.ndarray:
    """``n_eff = S1² / S2``: how many EQUAL contributors would give the bin this
    concentration (the participation ratio, or inverse Simpson index). 20 tracks of 2
    give 20.00; one track of 30 plus 10 of 1 gives 1.76. An empty bin is 0 -- it has
    no contributor -- never NaN, so a cutoff reads it as failing, like any bin with
    too few."""
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    out = np.zeros(np.broadcast(s1, s2).shape)
    np.divide(s1 * s1, s2, out=out, where=s2 > 0)
    return out


def compute_map_bins(locs: pd.DataFrame, cells: pd.DataFrame, *, bin_um: float = 0.04,
                     weight: str = "track", population: str | None = None,
                     length_window="iqr") -> pd.DataFrame:
    """The mirrored histogram behind the map: the picture AND the numbers.

    Each localisation is placed at ``(u, v)`` and its three reflections, so the grid
    is four-fold symmetric (the symmetry is IMPOSED). Per bin: raw ``count``,
    ``weighted_count`` (under ``weight``), the realised ``design_effect`` =
    ``Σ n_i² / Σ n_i`` over tracks (its precision cost: a big bin can be worth a
    few independent units), ``n_eff_track`` / ``n_eff_cell`` (= ``count / design_effect``
    by track, and the same by cell; see ``effective_contributors``),
    ``top_track_share`` (the busiest single track's share),
    ``n_cells``/``n_tracks`` and ``source``. Returns one row per non-empty bin.
    """
    keep_cells, _ = select_length_window(locs, cells, length_window)
    sub = locs[locs["cell_id"].isin(keep_cells)].copy()
    if population is not None:
        sub = sub[sub["population"].astype(str) == str(population)]
    if sub.empty:
        raise MapError("no localisation in the selected cells / population")
    cells_used = cells[cells["cell_id"].isin(sub["cell_id"].unique())]
    u_edges, v_edges, *_ = _grid(cells_used, bin_um)
    w = loc_weights(sub, weight)

    # fold to |u|,|v| then bin: each real loc contributes ONCE to its folded bin;
    # the four signed quadrants are reflections of that folded histogram.
    fu, fv = np.abs(sub["u_um"].to_numpy()), np.abs(sub["v_um"].to_numpy())
    fu_edges = u_edges[u_edges >= -1e-12]              # [0, bin, …] (edges pass 0)
    fv_edges = v_edges[v_edges >= -1e-12]
    n_fv = len(fv_edges) - 1
    iu = np.clip(np.digitize(fu, fu_edges[1:-1]), 0, len(fu_edges) - 2)
    iv = np.clip(np.digitize(fv, fv_edges[1:-1]), 0, n_fv - 1)

    bin_id = iu * n_fv + iv
    df = pd.DataFrame({"bin": bin_id, "w": w, "tid": sub["track_id"].to_numpy(),
                       "cell": sub["cell_id"].to_numpy()})
    per_bin = df.groupby("bin").agg(
        count=("w", "size"), weighted_count=("w", "sum"),
        n_tracks=("tid", "nunique"), n_cells=("cell", "nunique"))
    bt = df.groupby(["bin", "tid"]).size().rename("ni").reset_index()
    de = bt.groupby("bin")["ni"].agg(top_track_share=lambda n: float(n.max() / n.sum()))
    per_bin = per_bin.join(de).reset_index()
    # design_effect and n_eff through the ONE shared grouping (rule 5), by track and,
    # a second time, by cell -- the two fail independently (handover §4a.2).
    bins = per_bin["bin"].to_numpy()
    n_all = int(bin_id.max()) + 1
    t1, t2 = contributor_sums(bin_id, df["tid"].to_numpy(), n_bins=n_all)
    c1, c2 = contributor_sums(bin_id, df["cell"].to_numpy(), n_bins=n_all)
    per_bin["design_effect"] = t2[bins] / t1[bins]
    per_bin["n_eff_track"] = effective_contributors(t1, t2)[bins]
    per_bin["n_eff_cell"] = effective_contributors(c1, c2)[bins]
    per_bin["distinct_track"] = per_bin["n_tracks"]
    per_bin["iu"] = per_bin["bin"].to_numpy() // n_fv    # recover from the bin id
    per_bin["iv"] = per_bin["bin"].to_numpy() % n_fv
    fu_ctr = 0.5 * (fu_edges[:-1] + fu_edges[1:])
    fv_ctr = 0.5 * (fv_edges[:-1] + fv_edges[1:])
    per_bin["u_folded"] = fu_ctr[per_bin["iu"].to_numpy()]
    per_bin["v_folded"] = fv_ctr[per_bin["iv"].to_numpy()]

    # reflect the folded quadrant into all four (the symmetric panel + map_bins)
    out = []
    for su in (1, -1):
        for sv in (1, -1):
            q = per_bin.copy()
            q["u_bin_centre"] = su * q["u_folded"]
            q["v_bin_centre"] = sv * q["v_folded"]
            out.append(q)
    full = pd.concat(out, ignore_index=True)
    full["population"] = "all" if population is None else str(population)
    full["source"] = sub["source"].iloc[0] if "source" in sub else "observed"
    for k in _GROUP_KEYS:
        if k in sub.columns:
            full[k] = sub[k].iloc[0]
    full.attrs["u_edges"] = u_edges
    full.attrs["v_edges"] = v_edges
    full.attrs["median_L"] = float(np.median(cells_used["length_um"]))
    full.attrs["median_W"] = float(np.median(cells_used["width_um"]))
    full.attrs["n_cells"] = int(sub["cell_id"].nunique())
    full.attrs["n_tracks"] = int(sub["track_id"].nunique())
    return full[["u_bin_centre", "v_bin_centre", "count", "weighted_count",
                 "design_effect", "n_eff_track", "n_eff_cell", "top_track_share",
                 "distinct_track", "n_cells",
                 "n_tracks", "population", "source"]
                + [k for k in _GROUP_KEYS if k in full.columns]]


def map_diagnostics(locs: pd.DataFrame, cells: pd.DataFrame, *,
                    bin_um: float = 0.04) -> pd.DataFrame:
    """Per group × population: is one molecule making the spot? Answers the
    reviewer's first question before it is asked.

    Columns: ``top_track_busiest_bin`` (busiest bin's top-track share), the top
    1/5/10 % of cells' share of localisations, the Gini across cells, the median
    ``design_effect`` over the busiest 5 % of bins, and the max-to-99th-percentile
    count ratio.
    """
    rows = []
    pops = [None] + sorted(locs["population"].dropna().astype(str).unique().tolist())
    # AQ9: this is the ONE function that groups on the identity keys, so it is the one
    # place a wrong-but-consistent `enzyme` label silently MERGES two proteins. An unset
    # label is exactly that state, so refuse rather than group into it.
    if "enzyme" in locs.columns:
        vals = set(locs["enzyme"].astype(str).unique())
        if vals & {"UNSPECIFIED", "nan", "None", ""}:
            raise MapError(
                "map_diagnostics groups on `enzyme`, and this table carries an UNSET "
                f"enzyme label ({sorted(vals)}). Grouping on an unset identity merges "
                "whatever was concatenated. Build the table through "
                "smtracker.spatial.pipeline.spatial_localisations, which requires the "
                "label, or set it explicitly before calling.")
    grouping = [k for k in _GROUP_KEYS if k in locs.columns]
    for keys, gloc in (locs.groupby(grouping) if grouping else [((), locs)]):
        for pop in pops:
            try:
                mb = compute_map_bins(gloc, cells, bin_um=bin_um, weight="localisation",
                                      population=pop, length_window=None)
            except MapError:
                continue
            counts = mb["count"].to_numpy()
            busiest = mb.sort_values("count", ascending=False)
            top5 = busiest.head(max(1, int(np.ceil(0.05 * len(busiest)))))
            per_cell = (gloc if pop is None else
                        gloc[gloc["population"].astype(str) == pop]
                        ).groupby("cell_id").size().sort_values(ascending=False)
            share = per_cell.to_numpy() / per_cell.sum()
            row = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
            row.update({
                "population": "all" if pop is None else pop,
                "top_track_busiest_bin": float(busiest["top_track_share"].iloc[0]),
                "top1pct_cell_share": float(share[:max(1, len(share) // 100)].sum()),
                "top5pct_cell_share": float(share[:max(1, len(share) // 20)].sum()),
                "top10pct_cell_share": float(share[:max(1, len(share) // 10)].sum()),
                "gini_cells": _gini(per_cell.to_numpy()),
                "median_design_effect_busiest5pct": float(top5["design_effect"].median()),
                "max_over_p99_count": float(counts.max() / np.percentile(counts, 99)),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def _gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _spherocylinder_outline(L, W, n=200):
    from . import geometry as g

    u = np.linspace(-L / 2, L / 2, n)
    h = g.half_width(u, L, W)
    uu = np.concatenate([u, u[::-1]])
    vv = np.concatenate([h, -h[::-1]])
    return uu, vv


def plot_cell_map(locs: pd.DataFrame, cells: pd.DataFrame, *,
                  mode: str = "density", population: str | None = None,
                  group: str | None = None, bin_um: float = 0.04,
                  weight: str = "track", length_window="iqr",
                  clim_pct: float = 99.0, ax=None):
    """Mirrored full-cell map (``density`` or ``tracks``). Returns ``(fig, ax)``.

    The four-fold mirroring is IMPOSED and stated on the figure. Colour limits are
    the ``clim_pct`` percentile of non-empty bins (over-range bins flagged), never
    the maximum of a heavy-tailed count. The outline is a spherocylinder at the
    MEDIAN length/width of the contributing cells; ``n_cells`` and the length IQR
    are drawn because pooling cells of different lengths blurs the axial axis.
    """
    from matplotlib.figure import Figure

    if mode not in ("density", "tracks"):
        raise MapError(f"mode must be 'density' or 'tracks', got {mode!r}")
    if ax is None:
        fig = Figure(figsize=(6.0, 3.0))
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    keep_cells, n_excl = select_length_window(locs, cells, length_window)
    sub = locs[locs["cell_id"].isin(keep_cells)].copy()
    if population is not None:
        sub = sub[sub["population"].astype(str) == str(population)]
    if sub.empty:
        raise MapError("no localisation to map after selection")
    cells_used = cells[cells["cell_id"].isin(sub["cell_id"].unique())]
    u_edges, v_edges, L, W, iqr_L = _grid(cells_used, bin_um)
    src = sub["source"].iloc[0] if "source" in sub else "observed"

    over_note = ""
    if mode == "density":
        u = sub["u_um"].to_numpy()
        v = sub["v_um"].to_numpy()
        w = loc_weights(sub, weight)
        uu = np.concatenate([u, -u, u, -u])
        vv = np.concatenate([v, v, -v, -v])
        ww = np.concatenate([w, w, w, w])
        H, _, _ = np.histogram2d(uu, vv, bins=[u_edges, v_edges], weights=ww)
        nonzero = H[H > 0]
        vmax = np.percentile(nonzero, clim_pct) if nonzero.size else 1.0
        n_over = int((H > vmax).sum())
        over_note = f"{n_over} bin(s) > p{clim_pct:g}"
        disp = np.ma.masked_where(H.T == 0, H.T)
        im = ax.pcolormesh(u_edges, v_edges, np.clip(disp, None, vmax),
                           cmap="viridis", vmin=0, vmax=vmax)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label(f"{weight}-weighted count (clip p{clim_pct:g})")
    else:                                              # tracks
        labels = sorted(sub["population"].dropna().astype(str).unique())
        palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
        for i, lab in enumerate(labels):
            col = palette[i % len(palette)]
            for _, tr in sub[sub["population"].astype(str) == lab].groupby("track_id"):
                tr = tr.sort_values("frame")
                u, v = tr["u_um"].to_numpy(), tr["v_um"].to_numpy()
                for su in (1, -1):
                    for sv in (1, -1):
                        ax.plot(su * u, sv * v, "-", lw=0.5, color=col, alpha=0.5)
            ax.plot([], [], "-", color=col, label=lab)   # legend proxy (D range)
        ax.legend(fontsize=6, frameon=False, loc="upper right", title="D µm²/s")

    ox, oy = _spherocylinder_outline(L, W)
    ax.plot(ox, oy, "-", color="#2d3748", lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("u (µm): long axis")
    ax.set_ylabel("v (µm)")
    n_tracks = int(sub["track_id"].nunique())
    pop_txt = "all populations" if population is None else f"population {population}"
    ax.set_title(f"{mode} · {pop_txt} · source={src}", fontsize=8, loc="left")
    note = (f"mirrored (fourfold symmetry imposed) · n_cells={sub['cell_id'].nunique()} "
            f"(len IQR {iqr_L[0]:.2f}–{iqr_L[1]:.2f} µm, {n_excl} excluded) · "
            f"n_tracks={n_tracks} · weight={weight}")
    if over_note:
        note += f" · {over_note}"
    fig.text(0.01, 0.005, note, fontsize=5.5, color="#4a5568")
    return fig, ax


def _fold_index(sub, u_edges, v_edges):
    """(iu, iv, n_fv, folded edges) for |u|, |v| on the symmetric grid."""
    fu_edges = u_edges[u_edges >= -1e-12]
    fv_edges = v_edges[v_edges >= -1e-12]
    n_fv = len(fv_edges) - 1
    iu = np.clip(np.digitize(np.abs(sub["u_um"].to_numpy()), fu_edges[1:-1]),
                 0, len(fu_edges) - 2)
    iv = np.clip(np.digitize(np.abs(sub["v_um"].to_numpy()), fv_edges[1:-1]),
                 0, n_fv - 1)
    return iu, iv, n_fv, fu_edges, fv_edges


def compute_enrichment_bins(locs: pd.DataFrame, cells: pd.DataFrame, population: str, *,
                            reference: str = "pooled", bin_um: float = 0.04,
                            weight: str = "track", length_window="iqr") -> pd.DataFrame:
    """Per-bin enrichment of ``population`` vs a baseline, on the mirrored grid.

    THE FORMULA (pinned, because a name is not a definition)::

        enrichment(bin) = [ Σw(pop ∈ bin) / Σw(all ∈ bin) ]
                          ---------------------------------
                          [ Σw(pop total) / Σw(all total)  ]

    with ``w = 1/n_loc_track`` under the default ``weight='track'`` (unit of
    analysis = TRACK, spread over its localisations). So it is a **ratio of
    SHARES** (the population's share of this bin ÷ its share overall) NOT a
    density or occupancy enrichment, and NOT normalised by bin area or volume.

    CONSEQUENCE, and the reason ``shell_summary`` exists: a ratio of shares
    INFLATES where the global share is small. Two conditions with the SAME
    additive shift (e.g. +7 pp more low-D in the shell) return very different
    ratios if their baselines differ (5 % → 2.4x, 19 % → 1.4x). Never compare the
    ratio across conditions with unequal baselines; use the additive
    (percentage-point) statistic for that, and carry the denominator in any label.

    ``reference='pooled'``: ``(pop weighted_count / pooled weighted_count) / (pop
    global weighted share)``: cancels detection AND geometry together (default).
    ``reference='proj_weight'``: ``(pop weighted_count / summed proj_weight) / (pop
    total / total proj_weight)``. NEVER normalised by bin area. Enrichment is 1 where
    the population has no spatial preference. Returns the ``map_bins`` rows (four
    quadrant reflections) with ``pooled_count``/``pooled_weighted_count``/
    ``summed_proj_weight`` so the picture and the numbers are the same object.
    """
    if reference not in ("pooled", "proj_weight"):
        raise MapError(f"reference must be 'pooled' or 'proj_weight', got {reference!r}")
    keep, _ = select_length_window(locs, cells, length_window)
    sel = locs[locs["cell_id"].isin(keep)].copy()
    if sel.empty:
        raise MapError("no localisation in the selected cells")
    cells_used = cells[cells["cell_id"].isin(sel["cell_id"].unique())]
    u_edges, v_edges, medL, medW, _ = _grid(cells_used, bin_um)
    iu, iv, n_fv, fu_edges, fv_edges = _fold_index(sel, u_edges, v_edges)
    sel = sel.assign(_bin=iu * n_fv + iv, _w=loc_weights(sel, weight),
                     _pw=sel["proj_weight"].to_numpy())
    is_pop = sel["population"].astype(str).to_numpy() == str(population)

    pooled = sel.groupby("_bin").agg(
        pooled_count=("_w", "size"), pooled_weighted_count=("_w", "sum"),
        summed_proj_weight=("_pw", "sum"), n_cells=("cell_id", "nunique"))
    popd = sel[is_pop]
    pop = popd.groupby("_bin").agg(count=("_w", "size"),
                                   weighted_count=("_w", "sum"),
                                   n_tracks=("track_id", "nunique"))
    bt = popd.groupby(["_bin", "track_id"]).size().rename("ni").reset_index()
    de = bt.groupby("_bin")["ni"].agg(
        design_effect=lambda n: float((n ** 2).sum() / n.sum()),
        top_track_share=lambda n: float(n.max() / n.sum()))

    tbl = pooled.join(pop, how="left").join(de, how="left")
    tbl[["count", "weighted_count", "n_tracks"]] = tbl[
        ["count", "weighted_count", "n_tracks"]].fillna(0.0)
    pop_share = popd["_w"].sum() / sel["_w"].sum() if sel["_w"].sum() else np.nan
    if reference == "pooled":
        tbl["enrichment"] = (tbl["weighted_count"] / tbl["pooled_weighted_count"]
                             ) / pop_share
    else:
        total_ratio = (popd["_w"].sum() / sel["_pw"].sum()
                       if sel["_pw"].sum() else np.nan)
        tbl["enrichment"] = (tbl["weighted_count"] / tbl["summed_proj_weight"]
                             ) / total_ratio
    tbl = tbl.reset_index()
    tbl["iu"] = tbl["_bin"].to_numpy() // n_fv
    tbl["iv"] = tbl["_bin"].to_numpy() % n_fv
    fu_ctr = 0.5 * (fu_edges[:-1] + fu_edges[1:])
    fv_ctr = 0.5 * (fv_edges[:-1] + fv_edges[1:])
    tbl["u_folded"] = fu_ctr[tbl["iu"].to_numpy()]
    tbl["v_folded"] = fv_ctr[tbl["iv"].to_numpy()]

    out = []
    for su in (1, -1):
        for sv in (1, -1):
            q = tbl.copy()
            q["u_bin_centre"] = su * q["u_folded"]
            q["v_bin_centre"] = sv * q["v_folded"]
            out.append(q)
    full = pd.concat(out, ignore_index=True)
    full["population"] = str(population)
    full["source"] = sel["source"].iloc[0] if "source" in sel else "observed"
    full["reference"] = reference
    for k in _GROUP_KEYS:
        if k in sel.columns:
            full[k] = sel[k].iloc[0]
    full.attrs["u_edges"], full.attrs["v_edges"] = u_edges, v_edges
    full.attrs["median_L"], full.attrs["median_W"] = medL, medW
    full.attrs["n_cells"] = int(sel["cell_id"].nunique())
    full.attrs["n_tracks_pop"] = int(popd["track_id"].nunique())
    cols = ["u_bin_centre", "v_bin_centre", "count", "weighted_count", "pooled_count",
            "pooled_weighted_count", "summed_proj_weight", "enrichment",
            "design_effect", "top_track_share", "n_cells", "n_tracks", "population",
            "source", "reference"] + [k for k in _GROUP_KEYS if k in full.columns]
    return full[cols]


def plot_enrichment_map(locs: pd.DataFrame, cells: pd.DataFrame, population: str, *,
                        reference: str = "pooled", min_pooled: int = 20,
                        bin_um: float = 0.04, weight: str = "track",
                        length_window="iqr", ax=None):
    """Mirrored enrichment map of ``population`` (diverging, centred on 1.0, symmetric
    limits). Returns ``(fig, ax)``. Bins with pooled count < ``min_pooled`` are
    greyed; if > half the bins are below threshold the WHOLE panel is greyed with the
    reason drawn on it (this fires on the low-fraction panels and should)."""
    from matplotlib.figure import Figure

    if ax is None:
        fig = Figure(figsize=(6.0, 3.0))
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    mb = compute_enrichment_bins(locs, cells, population, reference=reference,
                                 bin_um=bin_um, weight=weight,
                                 length_window=length_window)
    u_edges, v_edges = mb.attrs["u_edges"], mb.attrs["v_edges"]
    nu, nv = len(u_edges) - 1, len(v_edges) - 1
    grid = np.full((nu, nv), np.nan)
    iu = np.digitize(mb["u_bin_centre"].to_numpy(), u_edges) - 1
    iv = np.digitize(mb["v_bin_centre"].to_numpy(), v_edges) - 1
    valid = mb["pooled_count"].to_numpy() >= min_pooled
    ok = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
    enr = mb["enrichment"].to_numpy()
    sel = ok & valid
    grid[iu[sel], iv[sel]] = enr[sel]

    n_valid = int(valid.sum())
    n_total = int(len(mb))
    L, W = mb.attrs["median_L"], mb.attrs["median_W"]
    if n_valid < 0.5 * n_total:
        ax.text(0.5, 0.5, f"panel greyed: {n_valid}/{n_total} bins ≥ {min_pooled} "
                "pooled\n(too sparse to read)", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#4a5568")
        ax.set_xticks([]); ax.set_yticks([])
    else:
        span = np.nanmax(np.abs(grid - 1.0))
        span = 0.5 if not np.isfinite(span) or span == 0 else span
        im = ax.pcolormesh(u_edges, v_edges, np.ma.masked_invalid(grid.T),
                           cmap="RdBu_r", vmin=1 - span, vmax=1 + span)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("enrichment (1 = no preference)")
    ox, oy = _spherocylinder_outline(L, W)
    ax.plot(ox, oy, "-", color="#2d3748", lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("u (µm): long axis")
    ax.set_ylabel("v (µm)")
    ax.set_title(f"enrichment · population {population} · ref={reference} · "
                 f"source={mb['source'].iloc[0]}", fontsize=8, loc="left")
    fig.text(0.01, 0.005, f"mirrored (fourfold symmetry imposed) · "
             f"n_cells={mb.attrs['n_cells']} · n_tracks={mb.attrs['n_tracks_pop']} · "
             f"grey < {min_pooled} pooled", fontsize=5.5, color="#4a5568")
    return fig, ax


def _shell_point_estimate(w, is_pop, in_shell):
    """``(additive_pp, ratio)`` for one weighted selection. The arithmetic, once."""
    if w.sum() <= 0 or w[in_shell].sum() <= 0:
        return np.nan, np.nan
    s_all = w[is_pop].sum() / w.sum()
    s_shell = w[in_shell & is_pop].sum() / w[in_shell].sum()
    return 100.0 * (s_shell - s_all), (s_shell / s_all if s_all > 0 else np.nan)


#: Default radial bands: six equal bins of ``r_norm`` from the axis to the boundary.
#: Six because that is what every script in this project converged on independently;
#: it is a DISPLAY choice, not a resolution claim.
RADIAL_BANDS = tuple(np.linspace(0.0, 1.0, 7))


def radial_profile(locs: pd.DataFrame, *, bands=RADIAL_BANDS, coord: str = "r_norm",
                   weight: str = "track", population: str | None = None,
                   normalise: bool = True) -> np.ndarray:
    """Share of weight per radial band: the profile six scripts each reimplemented.

    One source of truth for the radial band histogram. Every caller had its own
    four-line copy of ``digitize -> sum -> divide``, which is fine until two of them
    clip the top band differently or normalise over a different denominator, at which
    point two figures in the same document disagree for an invisible reason.

    ``weight`` follows :func:`loc_weights`; the default ``'track'`` (w = 1/n_loc_track)
    makes the unit the TRACK, so one long track cannot dominate a band. ``population``
    restricts to one D-range label before profiling. ``normalise=False`` returns the
    raw weighted totals instead of shares.

    Returns an array of ``len(bands) - 1`` values. Bins are closed on the right at the
    outer edge, so a localisation exactly at ``r_norm = 1`` lands in the last band
    rather than falling off the end.
    """
    b = np.asarray(bands, dtype=float)
    d = locs if population is None else locs[
        locs["population"].astype(str) == str(population)]
    if not len(d):
        return np.full(len(b) - 1, np.nan)
    w = loc_weights(d, weight)
    idx = np.clip(np.digitize(d[coord].to_numpy(), b[1:-1]), 0, len(b) - 2)
    total = np.array([w[idx == k].sum() for k in range(len(b) - 1)])
    if not normalise:
        return total
    return total / total.sum() if total.sum() > 0 else total


def shell_additive_pp(locs: pd.DataFrame, population: str, *, shell: float = 0.75,
                      coord: str = "r_norm", weight: str = "track") -> float:
    """The additive shell statistic in percentage points, point estimate only.

    One source of truth for the primary endpoint's arithmetic: :func:`shell_summary`
    calls the same kernel, so a boundary sweep and a bootstrapped single boundary can
    never drift apart. Extracted because the AJ3 sweep had grown its own copy,
    a third implementation of the same formula, in a script.

    Returns the point estimate WITHOUT an interval and WITHOUT the
    ``MIN_INDEPENDENT_VISITS`` guard, because a sweep needs the whole curve including
    its thin end. Anything quoted as a result must come from :func:`shell_summary`,
    which carries the interval and refuses to emit a number below the visit floor.

    ``population`` is a per-track apparent-D range LABEL, not an HMM state occupancy;
    name it wherever this number is printed. Unit is the TRACK under the default
    ``weight="track"``, where ``w = 1/n_loc_track`` makes the weighted share collapse
    to a plain track-count fraction.
    """
    is_pop = locs["population"].astype(str).to_numpy() == str(population)
    w = loc_weights(locs, weight)
    in_shell = locs[coord].to_numpy() > shell
    return float(_shell_point_estimate(w, is_pop, in_shell)[0])


def shell_summary(locs: pd.DataFrame, population: str, *, shell: float = 0.75,
                  coord: str = "r_norm", weight: str = "track", n_boot: int = 500,
                  seed: int = 0, min_visits: int = MIN_INDEPENDENT_VISITS,
                  allow_insufficient: bool = False) -> dict:
    """PRIMARY shell statistic on the ADDITIVE scale, with a cell bootstrap interval.

    Returns ``additive_pp`` (percentage-point excess of the population's share inside
    the shell over its share overall) as the primary quantity, and ``ratio`` (the
    module's :data:`ENRICHMENT_FORMULA`) as a clearly-labelled SECONDARY that carries
    its denominator (``share_all``) in the result.

    Why additive is primary: the ratio is a ratio of SHARES, so it inflates where the
    baseline share is small. Two conditions with an IDENTICAL additive shift return
    very different ratios if their baselines differ, so a ratio difference across
    conditions is not evidence of a difference in spatial preference. The additive
    statistic does not have that failure mode.

    ``n_independent_visits`` counts distinct tracks of this population in the shell,
    not localisations (correlated within a track) and NOT all tracks visiting the
    shell. The population's own track count is the binding constraint: on real
    data many tracks can visit the shell while only a small fraction carry the low-D label,
    and it is those the estimate rests on. Below ``min_visits`` the result is
    flagged ``insufficient=True``; a caller must refuse to map rather than let the
    value stand. ``n_visits_all`` is reported alongside for context.

    All quantities are computed on the ``population`` LABEL supplied (a per-track
    apparent-D range label), name that population wherever these numbers are
    printed; it is not an HMM state occupancy.
    """
    d = locs.copy()
    is_pop = d["population"].astype(str).to_numpy() == str(population)
    w = loc_weights(d, weight)
    in_shell = d[coord].to_numpy() > shell
    cells = d["cell_id"].to_numpy()

    def _stat(idx):
        return _shell_point_estimate(w[idx], is_pop[idx], in_shell[idx])

    add, ratio = _stat(np.arange(len(d)))
    # R6: a flagged condition must not EMIT a number. Compute the flag first and
    # blank the estimates unless the caller explicitly opts in, otherwise an
    # insufficient panel silently supplies a value that then gets quoted.
    _in_shell_pop = in_shell & is_pop
    _visits = int(pd.unique(d.loc[_in_shell_pop, "track_id"]).size)
    if _visits < min_visits and not allow_insufficient:
        return {
            "population": str(population),
            "additive_pp": float("nan"), "additive_ci": (float("nan"), float("nan")),
            "ratio": float("nan"), "ratio_ci": (float("nan"), float("nan")),
            "share_all": float(w[is_pop].sum() / w.sum()) if w.sum() else float("nan"),
            "n_independent_visits": _visits,
            "n_visits_all": int(pd.unique(d.loc[in_shell, "track_id"]).size),
            "insufficient": True,
            "unit": "track" if weight in ("track", "distinct_track") else weight,
            "formula": ENRICHMENT_FORMULA,
            "reason": (f"only {_visits} independent visits of {population!r} in the "
                       f"shell (< {min_visits}); refusing to emit an enrichment "
                       "number. Pass allow_insufficient=True to override knowingly."),
        }
    rng = np.random.default_rng(seed)
    uniq = np.unique(cells)
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(cells == c) for c in pick])
        boot.append(_stat(idx))
    # n_boot=0 is a legitimate "point estimate, no interval" request (a boundary sweep
    # wants the curve, not 500 resamples per point). Indexing an empty array raised
    # IndexError instead, a latent crash on a documented-looking argument.
    if n_boot <= 0:
        add_ci = ratio_ci = (float("nan"), float("nan"))
    else:
        b = np.asarray(boot, dtype=float)
        add_ci = tuple(np.nanpercentile(b[:, 0], [2.5, 97.5]))
        ratio_ci = tuple(np.nanpercentile(b[:, 1], [2.5, 97.5]))
    # the BINDING constraint is this population's own tracks in the shell, not every
    # track that passes through it (see the docstring)
    visits = int(pd.unique(d.loc[in_shell & is_pop, "track_id"]).size)
    visits_all = int(pd.unique(d.loc[in_shell, "track_id"]).size)
    share_all = float(w[is_pop].sum() / w.sum()) if w.sum() else np.nan
    return {
        "population": str(population),
        "additive_pp": float(add), "additive_ci": add_ci,       # PRIMARY
        "ratio": float(ratio), "ratio_ci": ratio_ci,            # secondary
        "share_all": share_all,                                 # the ratio's denominator
        "n_independent_visits": visits,                         # population's own tracks
        "n_visits_all": visits_all,                             # context only
        "insufficient": bool(visits < min_visits),
        "unit": "track" if weight in ("track", "distinct_track") else weight,
        "formula": ENRICHMENT_FORMULA,
    }
