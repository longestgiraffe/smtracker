"""The legacy window's Clustering tab numbers: SMTracker 2.0's k-means on MSD curves (`SMTracker.m:553-621`,
`SMT_GUI/Help_fcn/clusteringMSD.m`, `@SMT/calcMSD_clustering.m`; manual Fig. 9, p. 15).

2.0's method, kept: every track of at least "Specific tau" localisations gets its time-averaged MSD at the lags 1 to
tau - 1 (all pairs of the whole track, projected onto its cell's axes: `projTracks`); k-means on those vectors
(squared Euclidean, 20 replicates, an emptied cluster refilled) for 1 to "Max k clusters", the count chosen by the
Calinski-Harabasz index (`evalclusters`, the largest); each cluster's tracks pooled into one weighted MSD curve
(`getMSD`), the anomalous F-test and its fit (`ftestAnomalous`, `diffusionClassification`, ported in
`smtracker.legacy.sla`); one heat map per cluster (the positions of its tracks on the model cell of length 1 and
width 1, `plotHeatmap.m`: a 20 by 20 histogram, cubic up-scaling by five refinements, divided by the rows 2.0 counted).

Everything is reused: the k-means and the index are `smtracker.clustering`'s, the curve pooling and the F-test
`smtracker.legacy.sla`'s. What is new here is only the glue 2.0 had in `clusteringMSD.m` and the heat map's
arithmetic.

KNOWN DEVIATIONS from 2.0 (each on purpose, said in the tab's Console):
- 2.0's per-cluster fit crashed on a cluster of ONE track (its MSD has no spread, so the fit weights are NaN). Here
  that cluster is listed with the words "not classified" and the reason, never dropped and never given a number.
- k-means starts are `smtracker.clustering.kmeans_fit`'s (k-means++, a fixed seed) where MATLAB's were random.

INTERFACES
- `clustering_run(project, condition, *, max_clusters, specific_tau)` -> the run payload (see its docstring).
- `cluster_table_rows(runs)` -> 2.0's table, rows of two strings, one block per condition.
- `heat_map(points, n_rows)` -> the cubic-interpolated surface 2.0 draws; `MODEL_CELL_CONTOUR`, `inside_model_cell`.
- `MAX_CLUSTER_CHOICES`: 2.0's "Max k clusters" popup entries.
"""
from __future__ import annotations

import math

import numpy as np

from .legacy_view import LegacyViewError

__all__ = ["MAX_CLUSTER_CHOICES", "DEFAULT_MAX_CLUSTERS", "DEFAULT_SPECIFIC_TAU", "MODEL_CELL_CONTOUR",
           "HEAT_GRID_NODES", "HEAT_REFINEMENTS", "HEAT_X", "HEAT_Y", "clustering_run", "cluster_table_rows",
           "cluster_title", "heat_map", "inside_model_cell", "pooled_curve"]

#: 2.0's "Max k clusters" popup (`SMTracker.m:565`): 2 to 6, the fifth entry (6) selected.
MAX_CLUSTER_CHOICES = (2, 3, 4, 5, 6)
DEFAULT_MAX_CLUSTERS = 6
#: "Specific tau" (`SMTracker.m:569`): the track length from which the MSD is computed; 5 localisations, 4 lags.
DEFAULT_SPECIFIC_TAU = 5
#: `plotHeatmap.m:15-17`: a 20 by 20 histogram on `linspace(-0.1, 1.1, 20)` by `linspace(-0.6, 0.6, 20)`, then
#: `interp2(v, 5, 'cubic')`.
HEAT_GRID_NODES = 20
HEAT_REFINEMENTS = 5
HEAT_X = (-0.1, 1.1)
HEAT_Y = (-0.6, 0.6)
#: `typeOfMotion` of `clusteringMSD.m:15`, by `anomalousWfit`'s code (1 Brownian, 2 super, 3 sub).
MOTION_WORDS = ("Brownian", "Super-diffusive", "Sub-diffusive")
#: `anomalousWfit.m:7`: an exponent within 0.1 of one counts as Brownian.
LINEAR_TOLERANCE = 0.1

#: 2.0's model cell (`SMT_GUI/Help_fcn/config/modelCell.mat`, `modelCellContour`): a cell of length 1 and width 1,
#: its long axis along x from 0 to 1, its short axis along y from -0.5 to 0.5, 96 vertices.
MODEL_CELL_CONTOUR = np.array([
    (0.00000, 0.00000),
    (0.00330, -0.09972),
    (0.01085, -0.19453),
    (0.02255, -0.28123),
    (0.03821, -0.35623),
    (0.05726, -0.41575),
    (0.07877, -0.45802),
    (0.10179, -0.48377),
    (0.12547, -0.49623),
    (0.14925, -0.50000),
    (0.17311, -0.50000),
    (0.19689, -0.50000),
    (0.22075, -0.50000),
    (0.24462, -0.50000),
    (0.26840, -0.50000),
    (0.29226, -0.50000),
    (0.31613, -0.50000),
    (0.33991, -0.50000),
    (0.36377, -0.50000),
    (0.38764, -0.50000),
    (0.41142, -0.50000),
    (0.43528, -0.50000),
    (0.45906, -0.50000),
    (0.48292, -0.50000),
    (0.50679, -0.50000),
    (0.53057, -0.50000),
    (0.55443, -0.50000),
    (0.57830, -0.50000),
    (0.60208, -0.50000),
    (0.62594, -0.50000),
    (0.64981, -0.50000),
    (0.67358, -0.50000),
    (0.69745, -0.50000),
    (0.72132, -0.50000),
    (0.74509, -0.50000),
    (0.76896, -0.50000),
    (0.79283, -0.50000),
    (0.81660, -0.50000),
    (0.84047, -0.50000),
    (0.86425, -0.49877),
    (0.88802, -0.49123),
    (0.91142, -0.47264),
    (0.93377, -0.43906),
    (0.95434, -0.38858),
    (0.97547, -0.32179),
    (0.98623, -0.24075),
    (0.99538, -0.14792),
    (1.00000, -0.04981),
    (1.00000, 0.04981),
    (0.99538, 0.14792),
    (0.98623, 0.24075),
    (0.97547, 0.32179),
    (0.95434, 0.38858),
    (0.93377, 0.43906),
    (0.91142, 0.47264),
    (0.88802, 0.49123),
    (0.86425, 0.49877),
    (0.84047, 0.50000),
    (0.81660, 0.50000),
    (0.79283, 0.50000),
    (0.76896, 0.50000),
    (0.74509, 0.50000),
    (0.72132, 0.50000),
    (0.69745, 0.50000),
    (0.67358, 0.50000),
    (0.64981, 0.50000),
    (0.62594, 0.50000),
    (0.60208, 0.50000),
    (0.57830, 0.50000),
    (0.55443, 0.50000),
    (0.53057, 0.50000),
    (0.50679, 0.50000),
    (0.48292, 0.50000),
    (0.45906, 0.50000),
    (0.43528, 0.50000),
    (0.41142, 0.50000),
    (0.38764, 0.50000),
    (0.36377, 0.50000),
    (0.33991, 0.50000),
    (0.31613, 0.50000),
    (0.29226, 0.50000),
    (0.26840, 0.50000),
    (0.24462, 0.50000),
    (0.22075, 0.50000),
    (0.19689, 0.50000),
    (0.17311, 0.50000),
    (0.14925, 0.50000),
    (0.12547, 0.49623),
    (0.10179, 0.48377),
    (0.07877, 0.45802),
    (0.05726, 0.41575),
    (0.03821, 0.35623),
    (0.02255, 0.28123),
    (0.01085, 0.19453),
    (0.00330, 0.09972),
    (0.00000, 0.00000)
])


def inside_model_cell(xy) -> np.ndarray:
    """`inCellContour.m`: which ``(n, 2)`` points lie inside the model cell (`inpolygon`: the boundary counts)."""
    from matplotlib.path import Path

    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if xy.size == 0:
        return np.zeros(0, dtype=bool)
    finite = np.isfinite(xy).all(axis=1)
    out = np.zeros(len(xy), dtype=bool)
    path = Path(MODEL_CELL_CONTOUR)
    out[finite] = path.contains_points(xy[finite], radius=1e-9) | path.contains_points(xy[finite], radius=-1e-9)
    return out


def _keys_weights(n_nodes: int, refinements: int) -> np.ndarray:
    """The matrix that takes ``n_nodes`` samples to ``2**refinements * (n_nodes - 1) + 1`` by cubic convolution
    (Keys, a = -0.5), the kernel of MATLAB's ``interp2(v, k, 'cubic')``; ends replicated."""
    n_out = 2 ** refinements * (n_nodes - 1) + 1
    u = np.linspace(0.0, n_nodes - 1.0, n_out)
    weights = np.zeros((n_out, n_nodes))
    a = -0.5
    for row, position in enumerate(u):
        base = min(int(math.floor(position)), n_nodes - 2)
        s = position - base
        for offset in (-1, 0, 1, 2):
            d = abs(s - offset)
            if d <= 1:
                w = (a + 2) * d ** 3 - (a + 3) * d ** 2 + 1
            elif d < 2:
                w = a * d ** 3 - 5 * a * d ** 2 + 8 * a * d - 4 * a
            else:
                w = 0.0
            weights[row, min(max(base + offset, 0), n_nodes - 1)] += w
    return weights


def heat_map(points, n_rows: int) -> dict:
    """`plotHeatmap.m`'s location surface for ``points`` (``(n, 2)``, positions in cell lengths and widths):
    ``hist3`` on 20 nodes (a point past the last node falls in the last bin, one before the first is dropped),
    cubic up-scaling five times, divided by ``n_rows``, the row count of the matrix 2.0 handed over (its points and
    one NaN separator per track, `size(nTracks, 1)`). Returns ``surface`` (rows x, columns y: 609 by 609),
    ``counts`` (20 by 20), ``extent`` ``(x0, x1, y0, y1)`` and ``maximum``."""
    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    xy = xy[np.isfinite(xy).all(axis=1)]
    nodes_x = np.linspace(*HEAT_X, HEAT_GRID_NODES)
    nodes_y = np.linspace(*HEAT_Y, HEAT_GRID_NODES)
    ix = np.searchsorted(nodes_x, xy[:, 0], side="right") - 1
    iy = np.searchsorted(nodes_y, xy[:, 1], side="right") - 1
    keep = (ix >= 0) & (iy >= 0)
    counts = np.zeros((HEAT_GRID_NODES, HEAT_GRID_NODES))
    np.add.at(counts, (ix[keep], iy[keep]), 1.0)
    weights = _keys_weights(HEAT_GRID_NODES, HEAT_REFINEMENTS)
    surface = weights @ counts @ weights.T / max(int(n_rows), 1)
    return {"surface": surface, "counts": counts, "extent": (*HEAT_X, *HEAT_Y), "maximum": float(surface.max())}


def pooled_curve(means, pairs, delays) -> np.ndarray:
    """`getMSD.m`'s pooled curve ``[delay, mean, std, N]`` of many tracks' MSD values at the same delays.

    ``means`` and ``pairs``: ``(tracks, lags)`` (each track's mean MSD at a lag and the number of pairs behind it);
    ``delays``: the lags in seconds. Mean weighted by the pairs; std ``sqrt(sw / (sw^2 - ssw) * sum n (m - M)^2)``;
    ``N = sw^2 / ssw`` (`smtracker.legacy.sla.pooled_msd` is the one-curve-at-a-time form of the same arithmetic;
    one track alone has no spread, NaN, as in 2.0)."""
    m = np.asarray(means, dtype=float)
    n = np.asarray(pairs, dtype=float)
    sw = n.sum(axis=0)
    ssw = (n ** 2).sum(axis=0)
    mean = (n * m).sum(axis=0) / sw
    swv = (n * (m - mean) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        std = np.sqrt(sw / (sw ** 2 - ssw) * swv)
    return np.column_stack([np.asarray(delays, dtype=float), mean, std, sw ** 2 / ssw])


def _classify(curve: np.ndarray) -> dict:
    """`ftestAnomalous` on a pooled curve: the F-test (`legacy.sla.classify_motion`), then the coefficients 2.0 kept:
    the anomalous ones where the test rejected a straight line, the straight line's otherwise, and the motion word
    from `anomalousWfit`'s own tolerance (an exponent within 0.1 of one is called Brownian)."""
    from .legacy import sla

    verdict = sla.classify_motion(curve)
    usable = np.isfinite(curve[:, :3]).all(axis=1)
    t, msd, w = curve[usable, 0], curve[usable, 1], curve[usable, 2]
    if verdict["p_value"] < sla.LEGACY_P_THRESHOLD:
        ano = sla.fit_anomalous_weighted(t, msd, w)
        alpha, D = float(ano["alpha"]), float(ano["D_alpha"])
        code = 1 if abs(alpha - 1.0) < LINEAR_TOLERANCE else (2 if alpha > 1.0 else 3)
    else:
        alpha, D, code = 1.0, float(verdict["D_um2_per_s"]), 1
    return {"alpha": alpha, "D": D, "p_value": float(verdict["p_value"]), "motion": MOTION_WORDS[code - 1],
            "anomalous_chosen": verdict["chosen_model"] == "anomalous"}


def _num2str(value: float, digits: int) -> str:
    return "NaN" if not math.isfinite(value) else f"{value:.{digits}g}"


def cluster_title(cluster: dict) -> str:
    """The heat map's second title line (`clusteringMSD.m:151`): ``D_{alpha}:D - motion (p-value: p)``, in mathtext."""
    if not cluster.get("classified", True):
        return "not classified"
    return (f"D$_{{{_num2str(cluster['alpha'], 3)}}}$:{_num2str(cluster['D'], 3)} - {cluster['motion']} "
            f"(p-value: {_num2str(cluster['p_value'], 2)})")


def _table_line(cluster: dict) -> str:
    """One row of 2.0's cluster table (`clusteringMSD.m:116-121`)."""
    if not cluster["classified"]:
        return f" - not classified: {cluster['reason']} ({_num2str(100 * cluster['fraction'], 3)}%)"
    return (f" - alpha:{_num2str(cluster['alpha'], 3)} D_{{{_num2str(cluster['alpha'], 3)}}} :"
            f"{_num2str(cluster['D'], 3)}( {_num2str(100 * cluster['fraction'], 3)}%) - {cluster['motion']} "
            f"(p-value: {_num2str(cluster['p_value'], 3)})")


def cluster_table_rows(runs) -> list:
    """2.0's "Characteristics of cluster" table (`fillClusterTable`): per condition run so far a row ``Condition``,
    a row ``#Clusters`` and one ``Cluster i`` row per cluster, biggest first; rows of two strings."""
    rows = []
    for run in runs:
        rows.append(["Condition", run["condition"]])
        rows.append(["#Clusters", str(run["n_clusters"])])
        for i, cluster in enumerate(run["clusters"], start=1):
            rows.append([f"Cluster {i}", _table_line(cluster)])
    return rows


def _cell_extent(table) -> tuple:
    """Per row, its cell's length and largest width in um (2.0's ``normCell``: the cell's length and the spread of its
    outline across the axis), from the projection's geometry."""
    geometry = table.attrs.get("geometry") or {}
    keys = table["cell"].astype(str).to_numpy()
    length = np.array([geometry[k].length_um if k in geometry else np.nan for k in keys])
    width = np.array([geometry[k].max_width_um if k in geometry else np.nan for k in keys])
    return length, width


def clustering_run(project, condition: str, *, max_clusters: int = DEFAULT_MAX_CLUSTERS,
                   specific_tau: int = DEFAULT_SPECIFIC_TAU, n_restarts: int = 20, seed: int = 0) -> dict:
    """2.0's Clustering tab's ``Run!`` on one condition.

    Returns ``condition``, ``tau_s``, ``n_lags``, ``specific_tau``, ``n_tracks`` (the tracks clustered), ``ch`` (the
    Calinski-Harabasz index at each k tried, k = 1 absent: undefined), ``n_clusters`` (the k chosen) and
    ``clusters``: one dict per cluster, the biggest first, each with ``fraction``, ``n_tracks``, ``classified``
    (False with a ``reason``), ``alpha``, ``D``, ``p_value``, ``motion``, ``curve`` (the pooled
    ``[delay, mean, std, N]``) and ``heat`` (`heat_map` plus ``n_inside`` tracks, ``percent`` of all clustered
    tracks, ``overlay`` the inside points with a NaN row between tracks)."""
    from .clustering import ClusteringError, calinski_harabasz, kmeans_labels
    from .legacy.sla import SLAError
    from .msd import MSDError, msd_per_track

    if int(max_clusters) not in MAX_CLUSTER_CHOICES:
        raise LegacyViewError(f"Max clusters must be one of {MAX_CLUSTER_CHOICES}, got {max_clusters!r}")
    if int(specific_tau) < 3:
        raise LegacyViewError(f"Specific tau {specific_tau}: the MSD needs at least 2 lags, so 3 localisations")
    n_lags = int(specific_tau) - 1
    try:
        table = project.projected(condition).copy()
    except Exception as exc:                              # noqa: BLE001 - said by name, never swallowed
        raise LegacyViewError(f"{condition}: no cell coordinates for the clustering ({exc})") from exc
    # 2.0's tracks here are the PROJECTED ones: distances along and across the cell's own axes
    table["x_um"] = table["L_um"].to_numpy(dtype=float)
    table["y_um"] = table["D_um"].to_numpy(dtype=float)
    try:
        per_track = msd_per_track(table, max_lag=n_lags)
    except MSDError as exc:
        raise LegacyViewError(f"{condition}: {exc}") from exc
    tau = float(per_track.attrs["tau_s"])
    mean_wide = per_track.pivot(index="uid", columns="lag_frames", values="msd_um2")
    pairs_wide = per_track.pivot(index="uid", columns="lag_frames", values="n_pairs")
    lags = list(range(1, n_lags + 1))
    if not set(lags) <= set(mean_wide.columns):
        raise LegacyViewError(f"{condition}: no track has {specific_tau} localisations; Specific tau is too long")
    mean_wide = mean_wide[lags].dropna()           # tracks with every lag only (a ragged matrix clusters length)
    pairs_wide = pairs_wide.loc[mean_wide.index, lags]
    uids = list(mean_wide.index)
    X = mean_wide.to_numpy(dtype=float)
    if len(X) < 2 * int(max_clusters):
        raise LegacyViewError(f"{condition}: only {len(X)} track(s) of at least {specific_tau} localisations; "
                              f"too few to cluster up to {max_clusters} groups")
    ch, labels_by_k = {}, {}
    try:
        for k in range(1, int(max_clusters) + 1):
            labels_by_k[k] = kmeans_labels(X, k, n_restarts=n_restarts, seed=seed)
            if k >= 2:
                ch[k] = calinski_harabasz(X, labels_by_k[k])
    except ClusteringError as exc:
        raise LegacyViewError(f"{condition}: {exc}") from exc
    k_best = max(ch, key=lambda k: (ch[k] if math.isfinite(ch[k]) else -math.inf))
    labels = labels_by_k[k_best]
    n_clusters = int(len(np.unique(labels)))

    # each track's points on the model cell (x / cell length, y / cell width), inside the model contour
    uid_rows = table["uid"].to_numpy()
    length, width = _cell_extent(table)
    nx, ny = table["L_um"].to_numpy() / length, table["D_um"].to_numpy() / width
    inside_all = inside_model_cell(np.column_stack([nx, ny]))
    order = {u: i for i, u in enumerate(uids)}
    track_of_row = np.array([order.get(u, -1) for u in uid_rows])
    delays = [lag * tau for lag in lags]
    n_all = len(uids)
    pairs = pairs_wide.to_numpy(dtype=float)
    clusters = []
    for c in range(n_clusters):
        members = np.flatnonzero(labels == c)
        curve = pooled_curve(X[members], pairs[members], delays)
        cluster = {"fraction": len(members) / n_all, "n_tracks": int(len(members)), "curve": curve}
        try:
            cluster.update(_classify(curve), classified=True)
        except (SLAError, ValueError, FloatingPointError, RuntimeError) as exc:
            cluster.update(classified=False, reason=f"{type(exc).__name__}: {exc}"[:160])
        rows = np.flatnonzero(np.isin(track_of_row, members) & inside_all)
        tracks_in = track_of_row[rows]
        n_inside = int(len(np.unique(tracks_in)))
        points = np.column_stack([nx[rows], ny[rows]])
        # `normTracks` carries one NaN row per track after its points: that count is the divisor 2.0 used
        heat = heat_map(points, n_rows=len(points) + n_inside)
        breaks = np.flatnonzero(np.diff(tracks_in) != 0) + 1 if len(tracks_in) else np.array([], dtype=int)
        overlay = np.insert(points, breaks, np.nan, axis=0) if len(points) else points
        heat.update(n_inside=n_inside, percent=100.0 * n_inside / n_all, overlay=overlay)
        cluster["heat"] = heat
        clusters.append(cluster)
    clusters.sort(key=lambda d: -d["fraction"])
    return {"condition": condition, "tau_s": tau, "n_lags": n_lags, "specific_tau": int(specific_tau),
            "n_tracks": n_all, "ch": ch, "n_clusters": n_clusters, "max_clusters": int(max_clusters),
            "clusters": clusters}
