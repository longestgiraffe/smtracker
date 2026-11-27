"""Track clustering by MSD shape, and the null that decides whether it is real.

The legacy method (`clusteringMSD.m`)
-------------------------------------
Take each track's MSD curve over the first ``n_lags`` lags, run k-means on those
vectors for k = 1..K, pick k by the Calinski-Harabasz index, then fit each
cluster's ensemble MSD and F-test it for anomalous diffusion. Reproduced here.

Why this needs a null more than anything else in the codebase
-------------------------------------------------------------
**k-means always returns clusters, and Calinski-Harabasz never says "none".**
Neither can express "this is one population". Worse, the input makes false
structure almost inevitable: a per-track MSD estimated from a handful of steps
is *enormously* noisy, for a single Brownian population the per-track MSD is
broadly spread with a long tail, so partitioning it produces well-separated,
convincing-looking groups made entirely of estimation noise. Split a unimodal
cloud in three and the three means genuinely differ; that is arithmetic, not
biology.

So the question is never "what does k-means find?" but "does it find *more* than
it finds on data with no clusters in it?". ``select_k`` runs the identical
pipeline on free Brownian tracks matched to the real track-length distribution
(``smtracker.simulate``) and reports the real Calinski-Harabasz index beside the
null's: the same observed / chance / excess discipline as
``smtracker.confinement``. If the real index sits inside the null's spread, the
honest report is *one population with a broad per-track spread*, whatever
k-means returned.

**This null has no free parameter.** A free Brownian MSD scales exactly with D,
so changing the null D merely rescales the cloud (raw features) or shifts it
(log features), and Calinski-Harabasz is invariant to both, verified to machine
precision in the tests. Unlike the confinement radius, σ, or Spot-On's dZ, there
is nothing here to tune into a preferred answer. Only ``sigma_um > 0`` breaks
it, by adding a constant 4σ² that does not scale with D; ``KSelection`` reports
which case it is.

The feature scale decides the verdict: measured, not assumed
-------------------------------------------------------------
On the reference data the two scales disagree completely at k = 2:

- ``feature="raw"`` (the legacy): observed 2568 vs chance 2703 ± 83 → **no
  excess**. The raw scale is dominated by the extreme upper tail, so the index
  is measuring tail shape, not population structure.
- ``feature="log"``: observed 3390 vs chance 2020 ± 88 → **+1370**, and k = 2
  wins over 3 and 4.

The legacy would have reported a large-looking index of 2568 with nothing to
compare it against. ``"raw"`` remains the default for legacy fidelity, exactly
as ``weighting="step"`` does in ``sqd``, but on this data it is the log scale
that carries the signal, and the difference is a *measured* claim per dataset,
not an assumed one. Report which scale was used.

Survivorship rides along
------------------------
Long tracks are preferentially slow (see CLAUDE.md), and a track's MSD vector
length is its track length, so any clustering of MSD curves partly clusters
*track length*. The null inherits the real length distribution, so this bias is
present on both sides and does not inflate the excess, but a cluster is still
never automatically a molecular state.

What a cluster is NOT
--------------------
Not a diffusive state. `smtracker.hmm` and `smtracker.sqd` estimate states with
a likelihood and a model; this partitions curves by shape. Use it as a
cross-check and an exploratory view, never as the fraction readout.

Invariants
----------
- ``n_lags >= 2`` and every clustered track must have all ``n_lags`` lags: a
  ragged feature matrix would cluster on *missingness*, i.e. on track length.
- k-means uses multiple restarts and the best inertia; a single restart is a
  coin flip on the initialization.
- Fails loud on too few tracks, an empty cluster, or a degenerate feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

from .canonical import CANONICAL_COLUMNS, AcquisitionParams
from .msd import ensemble_msd, fit_msd, msd_per_track
from .simulate import matched_brownian_null

__all__ = [
    "ClusteringError",
    "FEATURES",
    "msd_features",
    "kmeans_labels",
    "calinski_harabasz",
    "kmeans_fit",
    "assign_to_centroids",
    "pool_tables",
    "SharedClusters",
    "shared_clusters",
    "KSelection",
    "select_k",
    "cluster_tracks",
    "ClusterReport",
    "describe_clusters",
]

#: Feature scalings. ``"raw"`` is the legacy; ``"log"`` damps the noisy tail.
FEATURES = ("raw", "log")


#: Below this, the chance SD is too poorly determined to quote. 20 draws give an
#: SD with roughly 16 per cent relative uncertainty; 5 give roughly 35 per cent.
MIN_REPLICATES_FOR_A_QUOTABLE_SD = 20

#: How many chance SDs the observed index must exceed. A CONVENTION (the
#: two-sided 5 per cent landmark for a normal deviate, 1.96 rounded to 2), not a
#: derivation, and not a p value. See KSelection.exceeds_null.
EXCEEDS_NULL_SD = 2.0


class ClusteringError(RuntimeError):
    """Raised when a clustering cannot be computed or trusted."""


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


def msd_features(
    canonical: pd.DataFrame, *, n_lags: int = 4, feature: str = "raw"
) -> tuple[np.ndarray, list[str]]:
    """Per-track MSD vectors, one row per track: ``(matrix, uids)``.

    Only tracks with **every** lag from 1 to ``n_lags`` are kept. A ragged matrix
    filled with zeros or NaNs would make the clustering partly a clustering of
    track length, which is exactly the survivorship confound this analysis is
    most exposed to.
    """
    if feature not in FEATURES:
        raise ClusteringError(f"feature must be one of {FEATURES}, got {feature!r}")
    if n_lags < 2:
        raise ClusteringError(f"n_lags must be >= 2 to have a shape, got {n_lags}")
    missing = [c for c in CANONICAL_COLUMNS if c not in canonical.columns]
    if missing:
        raise ClusteringError(f"not a canonical table (missing {missing})")

    per_track = msd_per_track(canonical, max_lag=n_lags)
    wide = per_track.pivot(index="uid", columns="lag_frames", values="msd_um2")
    wanted = list(range(1, n_lags + 1))
    if not set(wanted) <= set(wide.columns):
        raise ClusteringError(
            f"no track reaches lag {n_lags}; the longest available is "
            f"{int(max(wide.columns))}"
        )
    wide = wide[wanted].dropna()
    if len(wide) < 2:
        raise ClusteringError(
            f"only {len(wide)} track(s) have all {n_lags} lags: nothing to cluster. "
            "Use fewer lags."
        )
    matrix = wide.to_numpy(dtype=float)
    if feature == "log":
        floor = matrix[matrix > 0].min() if np.any(matrix > 0) else 1e-12
        matrix = np.log(np.maximum(matrix, floor))
    return matrix, list(wide.index)


# --------------------------------------------------------------------------- #
# k-means and its index
# --------------------------------------------------------------------------- #


def kmeans_fit(
    features: np.ndarray, k: int, *, n_restarts: int = 20, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """k-means ``(centroids, labels)``, best of ``n_restarts`` by within-cluster inertia.

    A single run depends on its random initialization, so the legacy's
    ``'Replicates', 20`` is kept: one restart would make the answer a coin flip.
    The centroids are returned because the shared-centroid mode (`shared_clusters`)
    assigns OTHER tracks to them.
    """
    features = np.atleast_2d(np.asarray(features, dtype=float))
    if k < 1:
        raise ClusteringError(f"k must be >= 1, got {k}")
    if k > len(features):
        raise ClusteringError(f"k = {k} exceeds the {len(features)} tracks available")
    if k == 1:
        return features.mean(axis=0, keepdims=True), np.zeros(len(features), dtype=int)

    best, best_inertia = None, np.inf
    for restart in range(n_restarts):
        centroids, labels = kmeans2(
            features, k, minit="++", seed=seed + restart, missing="warn"
        )
        if len(np.unique(labels)) < k:
            continue                     # an emptied cluster: this restart failed
        inertia = float(((features - centroids[labels]) ** 2).sum())
        if inertia < best_inertia:
            best, best_inertia = (centroids, labels), inertia

    if best is None:
        raise ClusteringError(
            f"every one of {n_restarts} restarts collapsed a cluster at k = {k}; "
            "the data does not support that many groups"
        )
    return best[0], best[1].astype(int)


def kmeans_labels(
    features: np.ndarray, k: int, *, n_restarts: int = 20, seed: int = 0
) -> np.ndarray:
    """k-means labels, best of ``n_restarts`` by within-cluster inertia (`kmeans_fit`)."""
    return kmeans_fit(features, k, n_restarts=n_restarts, seed=seed)[1]


def assign_to_centroids(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Each row of ``features`` to its nearest centroid (0-based)."""
    features = np.atleast_2d(np.asarray(features, dtype=float))
    centroids = np.atleast_2d(np.asarray(centroids, dtype=float))
    distance = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return distance.argmin(axis=1).astype(int)


def calinski_harabasz(features: np.ndarray, labels: np.ndarray) -> float:
    """Between-cluster over within-cluster dispersion, per degree of freedom.

    The legacy's selection criterion. Note what it cannot do: it is undefined at
    k = 1 and increases whenever a cloud is split into genuinely-different means,
    so it will happily rank k = 3 above k = 2 on a single Gaussian blob. It ranks
    partitions; it never says "do not partition".
    """
    features = np.atleast_2d(np.asarray(features, dtype=float))
    labels = np.asarray(labels)
    n = len(features)
    unique = np.unique(labels)
    k = len(unique)
    if k < 2 or k >= n:
        return float("nan")

    overall = features.mean(axis=0)
    between = 0.0
    within = 0.0
    for value in unique:
        members = features[labels == value]
        centroid = members.mean(axis=0)
        between += len(members) * float(((centroid - overall) ** 2).sum())
        within += float(((members - centroid) ** 2).sum())
    if within <= 0:
        return float("inf")
    return (between / (k - 1)) / (within / (n - k))


# --------------------------------------------------------------------------- #
# Choosing k, against the null
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KSelection:
    """Calinski-Harabasz for the data and for structureless Brownian tracks.

    ``verdict`` is the only line that should ever leave this analysis unquoted:
    it says whether the data's cluster structure exceeds what k-means invents on
    data with no clusters in it.
    """

    table: pd.DataFrame
    D_um2_s: float
    n_replicates: int
    n_tracks: int
    n_lags: int
    feature: str
    sigma_um: float = 0.0
    #: What the chance draws were: "free" (one free Brownian population), or see `clustering_null`.
    null_kind: str = "free"

    @property
    def criterion(self) -> str:
        """What ``best_k`` maximises: ``"z"``, the excess in chance SDs, when every k has a quotable SD (two or more
        draws), else ``"excess"``, the raw excess in index units (said by `report`). Calinski-Harabasz units are not
        comparable across k: the chance value itself rises and falls with k, and the raw excess favours small k
        (the review of 7 Nov 2026). The excess over the chance SD asks how far beyond chance, in chance's own spread."""
        z = self.table["z"] if "z" in self.table.columns else None
        if z is not None and z.notna().all() and len(z):
            return "z"
        return "excess"

    @property
    def best_k(self) -> int:
        """The k with the largest STANDARDISED excess (excess / chance SD, `criterion`), not the largest index and not
        the largest raw excess."""
        column = "z" if self.criterion == "z" else "excess"
        finite = self.table.dropna(subset=[column])
        if finite.empty:
            raise ClusteringError("no k had a defined index")
        return int(finite.loc[finite[column].idxmax(), "k"])

    @property
    def exceeds_null(self) -> bool:
        """Does the best k beat the chance value by more than ``EXCEEDS_NULL_SD``?

        WHERE THE 2 comes from, because a threshold with no stated provenance is asked
        about once and defended forever. It is a CONVENTION, not a derivation: 2 SD is
        the familiar two-sided 5 per cent landmark for a normal deviate (1.96, rounded).
        Three things it is NOT:

        * it is not a p value. The chance draws are not assumed normal and no tail
          probability is computed;
        * it is not corrected for testing several k. Every k is compared to its own
          chance value, and the largest excess is then selected, so the selection itself
          is not accounted for;
        * it is not stable when ``n_replicates`` is small. The SD is estimated from that
          many draws and carries roughly 35 per cent relative uncertainty at 5 draws
          against 16 per cent at 20, so the threshold moves with the draw count. See
          ``MIN_REPLICATES_FOR_A_QUOTABLE_SD``.

        Report the observed value, the chance value, the SD and the draw count. This
        boolean is a convenience, not the result.
        """
        row = self.table.set_index("k").loc[self.best_k]
        return bool(row["observed"]
                    > row["chance"] + EXCEEDS_NULL_SD * row["chance_sd"])

    @property
    def null_is_D_invariant(self) -> bool:
        """Does the verdict depend on the null D at all?

        With no localization noise it does **not**, on either feature scale, and
        for two different reasons. A free Brownian track's MSD scales exactly
        with D, so the null cloud is *rescaled* on the raw scale and merely
        *shifted* on the log scale, and Calinski-Harabasz is invariant to both.
        Verified numerically to machine precision in the tests.

        This is worth stating plainly: unlike the confinement radius, σ, or
        Spot-On's dZ, **this null has no free parameter to argue about**. Only a
        non-zero ``sigma_um`` breaks it, by adding a constant 4σ² to every MSD
        that does not scale with D.
        """
        return self.null_kind == "free" and self.sigma_um == 0.0

    @property
    def verdict(self) -> str:
        if self.null_kind == "sqd":
            if not self.exceeds_null:
                return (
                    "NO cluster structure beyond the SQD mixture: tracks drawn from this fit's own populations, "
                    "with the same lengths and noise, cluster as well as the data. The curves hold nothing the "
                    "populations the SQD fit found do not already explain."
                )
            return (
                f"cluster structure at k = {self.best_k} exceeds the SQD-mixture null: the curves hold shape the "
                "fitted populations, track lengths and noise do not produce (switching, confinement, "
                "or survivorship, which the null does not carry). It is a curve shape, not a diffusive state."
            )
        if not self.exceeds_null:
            return (
                "NO cluster structure beyond chance: k-means on matched free "
                "Brownian tracks scores as well as it does on the data. Report "
                "one population with a broad per-track spread."
            )
        return (
            f"cluster structure at k = {self.best_k} exceeds the free-Brownian "
            "null: more than one diffusion coefficient in the curves, which says nothing yet about "
            "whether it is more than the SQD populations; a cluster is a curve shape, not a diffusive state."
        )

    def report(self) -> str:
        lines = [
            f"Calinski-Harabasz vs "
            f"{'the fitted-SQD-mixture null' if self.null_kind == 'sqd' else 'a free-Brownian null'} "
            f"(D = {self.D_um2_s:.3g} µm²/s, {self.n_replicates} replicates, "
            f"{self.n_tracks} tracks, {self.n_lags} lags, feature={self.feature})",
            *(["  (this null is D-invariant to machine precision: a Brownian "
               "MSD scales exactly with D, and the index is invariant to "
               "rescaling and to translation, so the verdict has no free "
               "parameter)"] if self.null_is_D_invariant else []),
            f"{'k':>3}{'observed':>12}{'chance':>10}{'± sd':>9}{'excess':>11}{'z':>8}",
        ]
        for _, row in self.table.iterrows():
            lines.append(
                f"{int(row['k']):>3}{row['observed']:>12.1f}{row['chance']:>10.1f}"
                f"{row['chance_sd']:>9.1f}{row['excess']:>+11.1f}{row.get('z', float('nan')):>+8.1f}"
            )
        lines.append(f"k chosen on the {'excess in chance SDs (z)' if self.criterion == 'z' else 'raw excess'}")
        lines.append(f"VERDICT: {self.verdict}")
        return "\n".join(lines)


def select_k(
    canonical: pd.DataFrame,
    *,
    D_um2_s: float,
    k_values=(2, 3, 4),
    n_lags: int = 4,
    feature: str = "raw",
    sigma_um: float = 0.0,
    n_replicates: int = 5,
    n_restarts: int = 20,
    seed: int = 0,
    null_tracks=None,
    null_kind: str = "free",
) -> KSelection:
    """Score every k on the data and on matched structureless tracks.

    ``D_um2_s`` has no default: the null is "one Brownian population", and which
    D that is must come from the analysis, not from this function's imagination
    the same rule as ``confinement_null`` and ``spoton.dz_sensitivity``.

    ``null_tracks``: ``callable(seed) -> canonical table`` making one set of chance tracks; None is the free Brownian
    null matched to the data's track lengths. ``null_kind`` only labels it.
    """
    if n_replicates < 1:
        raise ClusteringError(f"n_replicates must be >= 1, got {n_replicates}")
    # A DEFAULT THAT PRODUCES AN UNUSABLE NUMBER IS THE SAME DEFECT CLASS AS A GUARD
    # THAT CANNOT FIRE. The chance value's standard deviation is estimated from
    # n_replicates draws; below about 20 that SD is itself so poorly determined that the
    # plus-or-minus cannot be quoted, and `exceeds_null` compares against 2 SD. The
    # default stays low because the simulation is the expensive part, so it WARNS rather
    # than silently returning a figure nobody should print.
    if n_replicates < MIN_REPLICATES_FOR_A_QUOTABLE_SD:
        warnings.warn(
            f"n_replicates={n_replicates}: the chance SD is estimated from "
            f"{n_replicates} draws and is too poorly determined to quote as a plus or "
            f"minus, and `exceeds_null` tests against 2 of these SDs. Use at least "
            f"{MIN_REPLICATES_FOR_A_QUOTABLE_SD} before reporting the interval.",
            RuntimeWarning, stacklevel=2)
    params = canonical.attrs.get("params")
    if not isinstance(params, AcquisitionParams):
        raise ClusteringError("canonical table carries no AcquisitionParams")

    features, uids = msd_features(canonical, n_lags=n_lags, feature=feature)
    observed = {
        k: calinski_harabasz(
            features, kmeans_labels(features, k, n_restarts=n_restarts, seed=seed)
        )
        for k in k_values
    }

    draws: dict[int, list[float]] = {k: [] for k in k_values}
    for replicate in range(n_replicates):
        if null_tracks is None:
            drawn = matched_brownian_null(
                canonical, D_um2_s=D_um2_s, sigma_um=sigma_um, seed=seed + replicate
            )
        else:
            drawn = null_tracks(seed + replicate)
        null_features, _ = msd_features(drawn, n_lags=n_lags, feature=feature)
        for k in k_values:
            draws[k].append(
                calinski_harabasz(
                    null_features,
                    kmeans_labels(
                        null_features, k, n_restarts=n_restarts, seed=seed + replicate
                    ),
                )
            )

    rows = []
    for k in k_values:
        values = np.asarray(draws[k], dtype=float)
        chance = float(np.mean(values))
        chance_sd = float(np.std(values, ddof=1)) if n_replicates > 1 else float("nan")
        rows.append(
            {
                "k": int(k),
                "observed": float(observed[k]),
                "chance": chance,
                "chance_sd": chance_sd,
                "excess": float(observed[k] - chance),
                # the excess in chance SDs: comparable across k, unlike the raw excess in index units
                "z": (float(observed[k] - chance) / chance_sd
                      if math.isfinite(chance_sd) and chance_sd > 0 else float("nan")),
            }
        )
    return KSelection(
        table=pd.DataFrame(rows),
        D_um2_s=float(D_um2_s),
        n_replicates=int(n_replicates),
        n_tracks=len(uids),
        n_lags=int(n_lags),
        feature=feature,
        sigma_um=float(sigma_um),
        null_kind=null_kind,
    )


# --------------------------------------------------------------------------- #
# The clustering itself
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClusterReport:
    """One clustering: per-cluster fractions and ensemble MSD fits."""

    labels: pd.Series                # index uid -> cluster (1-based)
    clusters: pd.DataFrame           # per-cluster fraction, D, alpha, R², motion
    k: int
    n_lags: int
    feature: str
    n_tracks: int
    caveat: str = field(default="")

    def summary(self) -> str:
        lines = [f"{self.k} clusters over {self.n_tracks} tracks "
                 f"({self.n_lags} lags, feature={self.feature})"]
        for _, row in self.clusters.iterrows():
            lines.append(
                f"  cluster {int(row['cluster'])}: {100 * row['fraction']:5.1f} % "
                f"of tracks, D = {row['D_um2_per_s']:.4g} µm²/s, "
                f"alpha = {row['alpha']:.2f}, R² = {row['r_squared']:.3f} "
                f"[{row['motion']}]"
            )
        if self.caveat:
            lines.append(f"CAVEAT: {self.caveat}")
        return "\n".join(lines)


def _motion_label(alpha: float) -> str:
    if alpha > 1.1:
        return "super-diffusive"
    if alpha < 0.9:
        return "sub-diffusive"
    return "Brownian"


def describe_clusters(canonical: pd.DataFrame, assignment: pd.Series, *, k: int, n_lags: int) -> pd.DataFrame:
    """Per cluster (1..k): its tracks, share of the clustered tracks, and the anomalous ensemble-MSD fit's D, alpha,
    R² and motion label. ``assignment``: uid -> cluster (1-based)."""
    rows = []
    for cluster in range(1, k + 1):
        members = assignment.index[assignment == cluster]
        subset = canonical[canonical["uid"].isin(members)]
        per_track = msd_per_track(subset, max_lag=n_lags)
        fit = fit_msd(ensemble_msd(per_track), n_lags=n_lags, anomalous=True)
        rows.append(
            {
                "cluster": cluster,
                "n_tracks": len(members),
                "fraction": len(members) / len(assignment),
                "D_um2_per_s": fit.D_um2_per_s,
                "alpha": fit.alpha if fit.alpha is not None else np.nan,
                "r_squared": fit.r_squared,
                "motion": _motion_label(fit.alpha if fit.alpha is not None else 1.0),
            }
        )
    return pd.DataFrame(rows).sort_values("fraction", ascending=False).reset_index(drop=True)


def cluster_tracks(
    canonical: pd.DataFrame,
    *,
    k: int,
    n_lags: int = 4,
    feature: str = "raw",
    n_restarts: int = 20,
    seed: int = 0,
) -> ClusterReport:
    """Partition tracks by MSD shape and describe each cluster.

    Per-cluster ``alpha`` comes from an anomalous MSD fit. Read it with care:
    CLAUDE.md records that α < 1 on an ensemble curve here is produced by
    population mixing and the survivorship tail-sag, not by anomalous motion,
    and clustering *by MSD magnitude* selects on exactly those, so a cluster's
    α is doubly confounded. It is reported because the legacy reported it, and
    labelled accordingly.
    """
    features, uids = msd_features(canonical, n_lags=n_lags, feature=feature)
    labels = kmeans_labels(features, k, n_restarts=n_restarts, seed=seed)
    assignment = pd.Series(labels + 1, index=uids, name="cluster")
    clusters = describe_clusters(canonical, assignment, k=k, n_lags=n_lags)

    smallest = clusters["n_tracks"].min()
    caveat = ""
    if smallest < 30:
        caveat = (
            f"the smallest cluster holds {int(smallest)} tracks: too few for its "
            "MSD fit or its spatial map to mean much"
        )
    return ClusterReport(
        labels=assignment,
        clusters=clusters,
        k=int(k),
        n_lags=int(n_lags),
        feature=feature,
        n_tracks=len(assignment),
        caveat=caveat,
    )


# --------------------------------------------------------------------------- #
# One set of clusters for every condition (shared centroids)
# --------------------------------------------------------------------------- #

#: Joins a condition and a uid in a pooled table. A unit separator no uid carries.
POOL_SEP = " | "


def pool_tables(tables: dict) -> pd.DataFrame:
    """Several conditions' canonical tables as ONE table, every uid made unique (``condition`` + ``POOL_SEP`` +
    uid) so two conditions that number their tracks alike never collide (RULE 0: nothing is joined on a shared
    label). The pooled table carries the first table's ``AcquisitionParams``."""
    if not tables:
        raise ClusteringError("no conditions to pool")
    parts = []
    first = None
    for name, table in tables.items():
        if first is None:
            first = table
        part = table.copy()
        part["uid"] = [f"{name}{POOL_SEP}{u}" for u in part["uid"]]
        parts.append(part)
    pooled = pd.concat(parts, ignore_index=True)
    pooled.attrs["params"] = first.attrs.get("params")
    return pooled


@dataclass(frozen=True)
class SharedClusters:
    """Centroids fitted ONCE on the pooled tracks of every condition, and each condition's tracks assigned to them.

    Cluster 1 is the slowest (the centroid with the smallest mean feature), so the number means the same in every
    condition: that is the point, a cluster label from per-condition k-means is not comparable across conditions.
    ``assignments[condition]`` is a Series uid -> cluster (1-based) over that condition's clustered tracks.
    """

    k: int
    n_lags: int
    feature: str
    centroids: np.ndarray                    # (k, n_lags), slowest first
    assignments: dict
    n_pooled: int

    def shares(self, condition: str) -> np.ndarray:
        """The condition's share of its clustered tracks in each cluster (tracks, not steps)."""
        labels = self.assignments[condition]
        return np.array([(labels == c).mean() for c in range(1, self.k + 1)], dtype=float)


def shared_clusters(tables: dict, *, k: int, n_lags: int = 4, feature: str = "raw",
                    n_restarts: int = 20, seed: int = 0) -> SharedClusters:
    """Fit ``k`` centroids on the pooled tracks of ``tables`` ({condition: canonical table}), then assign each
    condition's tracks to the nearest. Every condition must have tracks reaching all ``n_lags`` lags."""
    feats, uids = {}, {}
    for name, table in tables.items():
        feats[name], uids[name] = msd_features(table, n_lags=n_lags, feature=feature)
    pooled = np.vstack(list(feats.values()))
    centroids, _labels = kmeans_fit(pooled, k, n_restarts=n_restarts, seed=seed)
    order = np.argsort(centroids.mean(axis=1), kind="stable")
    centroids = centroids[order]
    assignments = {}
    for name in tables:
        labels = assign_to_centroids(feats[name], centroids) + 1
        assignments[name] = pd.Series(labels, index=uids[name], name="cluster")
    return SharedClusters(k=int(k), n_lags=int(n_lags), feature=feature, centroids=centroids,
                          assignments=assignments, n_pooled=int(len(pooled)))
