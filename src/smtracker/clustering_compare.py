"""The Clustering section's comparison ACROSS conditions, headless (7 Nov 2026).

WHY. The per-condition mode (`clustering_view.compute_clustering_view`) clusters each condition alone, with its own k, so
"cluster 1" of one condition is not "cluster 1" of another: a label is a rank by D inside one condition, and a share
cannot be set beside another condition's. The question the experiment asks is "does the share of slow-curve tracks
move with the condition?", which needs ONE set of clusters.

THE SHARED MODE. The k-means centroids are fitted ONCE on the pooled tracks of every condition
(`clustering.shared_clusters`), k chosen on the pool against the same null (`clustering.select_k`) or set by hand, and
every condition's tracks are assigned to the nearest centroid. Clusters are numbered slowest first, the same in every
condition. Each condition's share per cluster is then given PER REPLICATE.

INVARIANTS.
- A share is a fraction of TRACKS (each clustered track counts once), never of steps: it is not an SQD fraction
  (`clustering_view.TRACKS_NOT_STEPS`).
- RULE 0. The replicate (the ``day`` column of a condition, a within-condition ordinal) is the independent unit. Two
  conditions are two independent samples compared with Welch (`uncertainty.welch_test`, the brackets' test), never
  paired by replicate number; pooled tracks are never the unit of a test. Dropping the pairing narrows every interval
  (through the degrees of freedom, not the data), the direction that flatters the effect: the comparison says so.
- The pooled clustering's verdict ("beyond chance") is on the POOLED tracks of every replicate and condition. It says the
  curves hold structure; it does not say a condition differs. Only the replicate-level comparison does that.
- A condition none of whose tracks reach all the lags clustered is named and refused, not dropped silently.

Plain data (picklable) for `ui.workspace.clustering_section`; the comparison is a function of the payload.
"""
from __future__ import annotations

import math

from .clustering_view import (CLUSTER_FEATURE_LABELS, K_VALUES, N_REPLICATES, TRACKS_NOT_STEPS, cluster_inks, headline,
                              section_config)

__all__ = ["compute_shared_clustering", "shared_card_payload", "shares_payload", "compare_shares", "replicate_shares",
           "RULE0_SENTENCE", "SHARES_TEST_NAME", "comparison_table", "all_pairs_table"]

SHARES_TEST_NAME = ("Welch's t-test (two-sided, unequal variances) on each replicate's share of tracks in the cluster, "
                    "two independent samples")
RULE0_SENTENCE = ("The two conditions are two independent samples: replicates are never matched by number. Dropping "
                  "the pairing narrows the interval through its degrees of freedom, which favours the effect.")


def _replicate_name(day) -> str:
    try:
        return f"replicate {int(day)}"
    except (TypeError, ValueError):
        return f"replicate {day}"


def replicate_shares(table, labels, k: int) -> list:
    """One condition's clusters per replicate: ``[{"replicate", "n_tracks", "shares"}]``, the share of that
    replicate's clustered tracks in each cluster 1..k. ``labels``: uid -> cluster (1-based). A table without a ``day``
    column is one replicate."""
    from .canonical import without_attrs

    plain = without_attrs(table)
    if "day" not in plain.columns:
        return [{"replicate": "replicate 1", "n_tracks": int(len(labels)),
                 "shares": [float((labels == c).mean()) for c in range(1, k + 1)]}]
    day_of = plain.drop_duplicates("uid").set_index("uid")["day"].reindex(labels.index)
    out = []
    for day in sorted(day_of.dropna().unique()):
        mine = labels[(day_of == day).to_numpy()]
        if len(mine):
            out.append({"replicate": _replicate_name(day), "n_tracks": int(len(mine)),
                        "shares": [float((mine == c).mean()) for c in range(1, k + 1)]})
    return out


def _spread(replicates: list, k: int) -> tuple:
    """``(mean, sd)`` of the replicate shares per cluster; sd None below two replicates (no spread exists)."""
    import numpy as np

    shares = np.array([r["shares"] for r in replicates], dtype=float)
    mean = [float(v) for v in shares.mean(axis=0)]
    if len(replicates) < 2:
        return mean, None
    return mean, [float(v) for v in shares.std(axis=0, ddof=1)]


def compute_shared_clustering(project, conditions, *, n_lags: int = 4, feature: str = "raw", k=None,
                              k_values=K_VALUES, n_replicates: int = N_REPLICATES, seed: int = 0, null: str = "free",
                              sigma_nm=None, sqd_pop=None, precision=None) -> dict:
    """Shared centroids over ``conditions``: k chosen on the pool (``k`` None) or set by hand, every condition's
    tracks assigned to them, the shares per replicate. An expected refusal comes back as
    ``{"mode": "shared", "settings", "error"}``. ``null``, ``sigma_nm``, ``sqd_pop``: the chance the index is scored
    against (`clustering_null`)."""
    import warnings

    from .canonical import without_attrs
    from .clustering import (POOL_SEP, ClusteringError, describe_clusters, msd_features, pool_tables, select_k,
                             shared_clusters)
    from .clustering_null import null_for
    from .clustering_view import _cluster_curves, _pooled_null_D
    from .msd import MSDError

    conditions = list(conditions)
    settings = section_config(n_lags, feature, "shared", k, null, sigma_nm, sqd_pop,
                              per_spot=any(v is not None for v in (precision or {}).values()) or null != "sqd")
    try:
        if len(conditions) < 2:
            raise ClusteringError("shared clusters compare conditions: load at least two")
        tables = {c: project.condition(c) for c in conditions}
        counts = {}
        for name, table in tables.items():
            try:
                counts[name] = len(msd_features(table, n_lags=n_lags, feature=feature)[1])
            except ClusteringError as exc:
                raise ClusteringError(f"{name}: {exc}") from exc
        pooled = pool_tables(tables)
        n_total = {c: int(without_attrs(t)["uid"].nunique()) for c, t in tables.items()}
        d_null = _pooled_null_D(pooled, n_lags)
        selection = None
        if k is None:
            ks = tuple(v for v in k_values if v < sum(counts.values()))
            if not ks:
                raise ClusteringError("too few tracks to cluster")
            null_tracks, null_info = null_for(null, tables, sigma_nm=sigma_nm, sqd_pop=sqd_pop, precision=precision)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                selection = select_k(pooled, D_um2_s=d_null, k_values=ks, n_lags=n_lags, feature=feature,
                                     n_replicates=n_replicates, null_tracks=null_tracks, null_kind=null_info["kind"],
                                     seed=seed)
            best = selection.best_k
        else:
            best = int(k)
            if best >= sum(counts.values()):
                raise ClusteringError(f"k = {best} exceeds the {sum(counts.values())} tracks available")
            null_info = {"kind": "none"}
        shared = shared_clusters(tables, k=best, n_lags=n_lags, feature=feature, seed=seed)
    except (ClusteringError, MSDError, ValueError) as exc:
        return {"mode": "shared", "condition": "", "settings": settings, "error": f"{type(exc).__name__}: {exc}"}

    pooled_labels = {}
    for name, labels in shared.assignments.items():
        for uid, c in labels.items():
            pooled_labels[f"{name}{POOL_SEP}{uid}"] = int(c)
    import pandas as pd

    pooled_assignment = pd.Series(pooled_labels, name="cluster")
    described = describe_clusters(pooled, pooled_assignment, k=best, n_lags=n_lags).sort_values(
        "cluster").reset_index(drop=True)
    rows = []
    for c in described.itertuples():
        alpha = float(c.alpha)
        rows.append({"cluster": int(c.cluster), "n_tracks": int(c.n_tracks), "fraction": float(c.fraction),
                     "D": float(c.D_um2_per_s), "alpha": alpha if math.isfinite(alpha) else None,
                     "r2": float(c.r_squared), "motion": str(c.motion)})
    curves = _cluster_curves(pooled, pooled_assignment, best, n_lags)

    per_condition = {}
    for name in conditions:
        reps = replicate_shares(tables[name], shared.assignments[name], best)
        mean, sd = _spread(reps, best)
        per_condition[name] = {
            "n_tracks": n_total[name], "n_clustered": counts[name],
            "pooled_shares": [float(v) for v in shared.shares(name)],
            "replicates": reps, "mean": mean, "sd": sd, "n": len(reps)}
    sel = None
    if selection is not None:
        sel = [{"k": int(r.k), "observed": float(r.observed), "chance": float(r.chance),
                "chance_sd": float(r.chance_sd), "excess": float(r.excess),
                "z": None if not math.isfinite(float(r.z)) else float(r.z)} for r in selection.table.itertuples()]
    smallest = min(r["n_tracks"] for r in rows)
    caveat = (f"the smallest cluster holds {smallest} tracks: too few for its MSD fit to mean much"
              if smallest < 30 else "")
    return {
        "mode": "shared", "condition": "", "settings": settings, "conditions": conditions,
        "n_tracks": sum(n_total.values()), "n_clustered": int(sum(counts.values())), "d_null": d_null,
        "n_replicates": int(n_replicates), "null": null_info, "criterion": None if selection is None else selection.criterion,
        "selection": sel, "best_k": int(best),
        "k_fixed": k is not None,
        "exceeds_null": None if selection is None else bool(selection.exceeds_null),
        "verdict": None if selection is None else selection.verdict,
        "null_d_invariant": None if selection is None else bool(selection.null_is_D_invariant),
        "clusters": rows, "curves": curves, "per_condition": per_condition, "caveat": caveat,
    }


def shared_card_payload(result: dict) -> dict:
    """The shared card as drawn: the verdict and its lines, the selection rows, the pooled clusters' table and one
    row per condition with its share per cluster (mean ± SD over replicates)."""
    from .clustering_null import NULL_QUESTIONS, null_words

    if "error" in result:
        return {"error": result["error"]}
    s = result["settings"]
    k = result["best_k"]
    feature = CLUSTER_FEATURE_LABELS[s["feature"]]
    if result["selection"] is None:
        head = (f"k = {k}, set by hand: no test of whether the curves hold structure beyond chance was made. The "
                f"shares below are comparable across conditions, because every condition is assigned to the same "
                f"clusters.")
        beyond = None
    else:
        head = (("Beyond chance: " if result["exceeds_null"] else "NOT beyond chance: ") + result["verdict"])
        beyond = bool(result["exceeds_null"])
    lines = [
        head,
        f"{result['n_clustered']:,} of {result['n_tracks']:,} tracks reach all {s['n_lags']} lags and are clustered, "
        f"all conditions pooled to find {k} clusters (feature: {feature}); each condition's tracks are then assigned "
        f"to those clusters, numbered from the slowest, the same in every condition.",
    ]
    if result["selection"] is not None:
        lines.append("k chosen on the excess in chance SDs; "
                     + null_words(result["null"], result["n_replicates"], result["null_d_invariant"])
                     + " Question asked of the data: " + NULL_QUESTIONS[result["null"]["kind"]])
    if result.get("caveat"):
        lines.append(result["caveat"][0].upper() + result["caveat"][1:] + ".")
    lines.append(TRACKS_NOT_STEPS)
    lines.append("The verdict is on the pooled tracks of all replicates and conditions. Whether conditions differ "
                 "is judged on the replicates' shares only.")
    cluster_rows = [[str(c["cluster"]), f"{c['n_tracks']:,}", f"{100 * c['fraction']:.1f} %", f"{c['D']:.3g}",
                     "" if c["alpha"] is None else f"{c['alpha']:.2f}", f"{c['r2']:.3f}"]
                    for c in result["clusters"]]
    selection_rows = []
    if result["selection"]:
        selection_rows = [[str(r["k"]), f"{r['observed']:.1f}", f"{r['chance']:.1f} ± {r['chance_sd']:.1f}",
                           f"{r['excess']:+.1f}", "" if r["z"] is None else f"{r['z']:+.1f}"]
                          for r in result["selection"]]
    share_columns = ["Condition", "Replicates", "Tracks"] + [f"Cluster {c['cluster']}" for c in result["clusters"]]
    share_rows = []
    for name in result["conditions"]:
        e = result["per_condition"][name]
        cells = []
        for i in range(k):
            if e["sd"] is None:
                cells.append(f"{100 * e['mean'][i]:.1f} %")
            else:
                cells.append(f"{100 * e['mean'][i]:.1f} ± {100 * e['sd'][i]:.1f} %")
        share_rows.append([name, str(e["n"]), f"{e['n_clustered']:,}", *cells])
    if any(result["per_condition"][n]["n"] < 2 for n in result["conditions"]):
        lines.append("A condition with one replicate has no spread and cannot be tested.")
    return {"k": k, "exceeds_null": beyond, "lines": lines, "headline": headline(result),
            "selection_columns": ["k", "Index", "Chance ± SD", "Excess", "Excess / SD"],
            "selection_rows": selection_rows,
            "columns": ["Cluster", "Tracks", "Share of pooled tracks", "D (µm²/s)", "alpha", "R²"],
            "rows": cluster_rows, "curves": result["curves"],
            "share_columns": share_columns, "share_rows": share_rows}


def shares_payload(result: dict, labels: dict | None = None, colours: dict | None = None, dark: bool = False) -> dict:
    """The shares figure's data: per condition, each cluster's replicate shares (the dots), their mean and SD, in the
    conditions' own colours; clusters in the palette's inks (`clustering_view.cluster_inks`, fitted to the page). Both the screen's chart and the figure export read it."""
    if "error" in result:
        raise ValueError(result["error"])
    labels, colours = labels or {}, colours or {}
    columns = []
    for name in result["conditions"]:
        e = result["per_condition"][name]
        columns.append({"key": name, "name": labels.get(name, name), "colour": colours.get(name),
                        "n": e["n"], "n_tracks": e["n_clustered"], "mean": e["mean"], "sd": e["sd"],
                        "replicates": [{"replicate": r["replicate"], "n_tracks": r["n_tracks"], "shares": r["shares"]}
                                       for r in e["replicates"]]})
    return {"k": result["best_k"], "columns": columns, "settings": result["settings"],
            "inks": cluster_inks(result["best_k"], dark),
            "caption": (f"Each condition's share of tracks in each of {result['best_k']} shared clusters (numbered "
                        f"from the slowest). Dots: one replicate each; bar: mean over replicates; whiskers: SD. "
                        + TRACKS_NOT_STEPS)}


def compare_shares(result: dict, pairs, cluster=1) -> list:
    """Welch on the replicate shares of ``cluster`` (a whole number, or ``"all"`` for every cluster), one
    `significance.Comparison` per distinct pair and cluster in the order picked, Holm-adjusted over every test made
    (a test that could not be made is not part of the family). Opt-in: nothing calls it unless the reader asks.

    TWO INDEPENDENT SAMPLES (RULE 0): the replicates of one condition are never matched to the other's by number. A pair
    with fewer than two replicates on a side comes back untested, its ``reason`` saying why. Raises ValueError on a
    condition not in the result or a cluster outside 1..k. For a tested row the bracket's text is
    `significance.bracket_text(row.p_holm, label, thresholds)`."""
    from .significance import Comparison, _floor
    from .uncertainty import holm, welch_test

    if "error" in result or result.get("mode") != "shared":
        raise ValueError("there is no shared clustering to compare")
    k = result["best_k"]
    if cluster == "all":
        clusters = list(range(1, k + 1))
    else:
        if isinstance(cluster, bool) or int(cluster) != cluster or not 1 <= int(cluster) <= k:
            raise ValueError(f"the cluster compared is a whole number from 1 to {k}, or all, got {cluster!r}")
        clusters = [int(cluster)]
    per = result["per_condition"]
    distinct, seen = [], set()
    for a, b in pairs:
        a, b = str(a), str(b)
        for name in (a, b):
            if name not in per:
                raise ValueError(f"{name!r} is not a condition of this clustering")
        if a == b:
            raise ValueError("a condition is not compared with itself")
        if frozenset((a, b)) not in seen:
            seen.add(frozenset((a, b)))
            distinct.append((a, b))
    rows = []
    for c in clusters:
        for a, b in distinct:
            ra, rb = per[a]["replicates"], per[b]["replicates"]
            row = Comparison(a=a, b=b, population=c, days_a=[r["replicate"] for r in ra],
                             days_b=[r["replicate"] for r in rb], values_a=[r["shares"][c - 1] for r in ra],
                             values_b=[r["shares"][c - 1] for r in rb], kind="welch")
            row.floor = _floor(len(ra), len(rb))
            row.test = welch_test(row.values_a, row.values_b)
            row.reason = row.test.reason
            rows.append(row)
    adjusted = holm([r.test.p for r in rows])
    family = sum(1 for p in adjusted if math.isfinite(p))
    for row, p in zip(rows, adjusted):
        row.p_holm, row.family = p, family
    return rows


def comparison_table(result: dict, pairs, cluster="all", *, label: str = "stars", thresholds=None,
                     names: dict | None = None) -> dict:
    """`compare_shares` as a table the card draws: ``{"columns", "rows", "lines"}``. A row: the pair and cluster, the
    difference in percentage points (b minus a) with its 95 % interval, the Holm-adjusted p as the bracket prints it
    (``label``: stars, the exact p, or both). ``lines`` holds the test's name, the family and the Rule 0 sentence."""
    from .significance import DEFAULT_STARS, bracket_text

    thresholds = DEFAULT_STARS if thresholds is None else thresholds
    names = names or {}
    found = compare_shares(result, pairs, cluster)
    rows = []
    for r in found:
        shown = f"{names.get(r.a, r.a)} vs {names.get(r.b, r.b)}"
        if not r.tested:
            rows.append([shown, str(r.population), "", "", r.reason or "no test"])
            continue
        t = r.test
        rows.append([shown, str(r.population), f"{100 * t.difference:+.1f}",
                     f"[{100 * t.ci_low:+.1f}, {100 * t.ci_high:+.1f}]",
                     bracket_text(r.p_holm, label, thresholds)])
    lines = [SHARES_TEST_NAME + f"; Holm over the {found[0].family if found else 0} tests made.", RULE0_SENTENCE]
    return {"columns": ["Pair", "Cluster", "Difference (pp)", "95 % interval", "Holm-adjusted"], "rows": rows,
            "lines": lines}


def all_pairs_table(result: dict, names: dict | None = None) -> dict:
    """`comparison_table` over EVERY pair of conditions and every cluster, the card's opt-in "Test conditions"."""
    from itertools import combinations

    return comparison_table(result, list(combinations(result["conditions"], 2)), "all", names=names)
