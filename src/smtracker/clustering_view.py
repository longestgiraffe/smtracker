"""The workspace's Clustering section, headless (v2's Clustering tab, `clusteringMSD.m`; added 17 Oct 2026, "Clustering
tab (k-means on per-track MSD curves; classic browser only today)").

Per condition: every track's MSD curve over the first ``n_lags`` lags, k-means for each k asked (best of 20
restarts), Calinski-Harabasz for each -- and the SAME pipeline on free Brownian tracks matched to the condition's
track lengths (`clustering.select_k`), because k-means always returns clusters and the index never says "none". The
k shown is the one with the largest EXCESS over that chance value, never the largest index; the verdict says whether
it beats chance by more than two of the chance draws' SDs (a convention, not a p value: `clustering.KSelection`).

Then that k's clusters (`clustering.cluster_tracks`): each one's share of the tracks, its ensemble MSD and early-lag
fit (D, alpha). The clusters are ordered by D (slowest first) and numbered in that order. Each cluster's share is also
given PER REPLICATE DAY (RULE 0: days are independent): the share of each day's clustered tracks in it, and the mean
+- SD over days -- the only spread shown. A cluster is a curve shape, not a diffusive state (`clustering` module
head): the section is an exploratory cross-check, never the fraction readout.

The null's D is the condition's own ensemble-MSD D; with no localisation error in the null the verdict does not
depend on it (`KSelection.null_is_D_invariant`), which the payload states.

Plain data (picklable, one pool job per condition) for `ui.workspace.clustering_section`.
"""
from __future__ import annotations

import math

__all__ = ["config_args", "wants_per_spot",
           "CLUSTER_FEATURES", "CLUSTER_INKS", "CLUSTER_LAGS", "K_VALUES", "N_REPLICATES", "compute_clustering_view",
           "MODES", "MODE_LABELS", "SHARED_K_CHOICES", "K_MAX_CHOICES", "DRAW_CHOICES", "compute_kwargs",
           "cluster_inks", "run_problem", "STANDARD_CONFIG", "describe_config", "headline", "TRACKS_NOT_STEPS", "READING_NOTES", "section_config"]

#: The feature scale (`clustering.FEATURES`): "raw" is v2's; "log" damps the noisy upper tail. Raw first: legacy
#: fidelity, as the SQD weighting keeps v2's step; the scale used is always said.
CLUSTER_FEATURES = ("raw", "log")
CLUSTER_FEATURE_LABELS = {"raw": "MSD", "log": "log MSD"}
#: Lags in each track's curve; a track needs all of them (`clustering.msd_features`), so more lags keep fewer tracks.
CLUSTER_LAGS = (3, 4, 6, 8)
K_VALUES = (2, 3, 4)
#: One ink per cluster, slowest first (a cluster is a curve shape, so no condition colour: every card has the same
#: clusters 1..k by D). The screen's chart and the exported figure read this one list.
CLUSTER_INKS = ("#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#b7791f")
#: Chance draws: 20, the fewest whose SD can be quoted (`clustering.MIN_REPLICATES_FOR_A_QUOTABLE_SD`).
N_REPLICATES = 20

READING_NOTES = (
    "k-means groups each track's MSD curve (first lags) for each number of clusters k, and the Calinski-Harabasz "
    "index scores each grouping. k-means finds clusters in any data, so the same analysis is run on chance tracks "
    "that have none, and the data are judged by how far they go beyond them.\n\n"
    "Chance is the SQD mixture (the default) or one free population. The SQD mixture draws tracks from this "
    "condition's fitted SQD populations with its track lengths and localisation error; beating it means clustering "
    "finds something those populations do not explain. One free population draws tracks of a single D; beating it "
    "shows only that the curves are not those of one D.\n\n"
    "The k shown has the largest excess in chance SDs (excess over the chance value, divided by the chance draws' "
    "SD). The verdict says whether it exceeds two SDs; this is a convention and not a p value.\n\n"
    "A cluster is a curve shape and not a diffusive state. Long tracks are slower on average, so clustering "
    "curves partly clusters track length. Read the clusters beside the SQD fractions.\n\n"
    "A cluster's share is a fraction of tracks and not of steps, given per replicate as mean ± SD over replicates. "
    "Shared clusters put every condition on the same clusters, so their shares can be compared across conditions; "
    "clusters found in each condition alone cannot be compared.")


#: The two ways to cluster. "condition": each condition alone, its own k and its own clusters (cluster 1 of one
#: condition is NOT cluster 1 of another). "shared": the centroids fitted once on the pooled tracks of every condition
#: and each condition's tracks assigned to them, so a cluster means the same thing in every condition and the
#: conditions can be compared (`clustering_compare.compute_shared_clustering`).
MODES = ("condition", "shared")
#: The k a reader may fix in shared mode (the data chooses among `K_VALUES` otherwise).
SHARED_K_CHOICES = (2, 3, 4, 5)
#: The largest k the data may choose (k runs from 2 to it), and the chance draws (20 is the fewest whose SD can be quoted).
K_MAX_CHOICES = (3, 4, 5)
DRAW_CHOICES = (20, 50, 100)
MODE_LABELS = {"condition": "Each condition alone", "shared": "Shared clusters"}
#: Said beside every share: what the denominator is.
TRACKS_NOT_STEPS = ("A share is a fraction of tracks (each clustered track counts once), not of steps, so it is not "
                    "an SQD fraction and the two need not agree.")


def section_config(n_lags: int = 4, feature: str = "raw", mode: str = "condition", k=None, null: str = "free",
                   sigma_nm=None, sqd_pop=None, k_max: int = 4, draws: int = 20, *, per_spot: bool = True) -> dict:
    """The committed Run settings. RAISES on a value the section cannot offer. ``k`` None: chosen by the data (the
    largest excess over chance in chance SDs); a whole number from 2: set by hand (shared mode only). ``null``: the chance
    the index is scored against (`clustering_null.NULLS`); ``sigma_nm`` the localisation error of the SQD-mixture null
    (None: the acquisition's stated one); ``sqd_pop`` its populations (None: Best by BIC, to 3). The keys ``mode``,
    ``k``, ``null``, ``sigma_nm`` and ``sqd_pop`` are present only when they depart from the standard, so a setting made
    before them reads, and is keyed, the same."""
    from .clustering_null import NULLS

    if int(n_lags) not in CLUSTER_LAGS:
        raise ValueError(f"n_lags must be one of {CLUSTER_LAGS}, got {n_lags!r}")
    if feature not in CLUSTER_FEATURES:
        raise ValueError(f"feature must be one of {CLUSTER_FEATURES}, got {feature!r}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if null not in NULLS:
        raise ValueError(f"null must be one of {NULLS}, got {null!r}")
    if k is not None:
        if isinstance(k, bool) or int(k) != k or int(k) < 2:
            raise ValueError(f"k is a whole number from 2 (or left to the data), got {k!r}")
        if mode != "shared":
            raise ValueError("a fixed k is offered with shared clusters only")
    if sigma_nm is not None and (isinstance(sigma_nm, bool) or not 0 <= float(sigma_nm) <= 500):
        raise ValueError(f"sigma_nm is a localisation error from 0 to 500 nm (or left to the acquisition), got {sigma_nm!r}")
    if sqd_pop is not None and (isinstance(sqd_pop, bool) or int(sqd_pop) != sqd_pop or not 1 <= int(sqd_pop) <= 3):
        raise ValueError(f"sqd_pop is 1, 2 or 3 populations (or Best), got {sqd_pop!r}")
    if int(k_max) not in K_MAX_CHOICES:
        raise ValueError(f"k_max must be one of {K_MAX_CHOICES}, got {k_max!r}")
    if int(draws) not in DRAW_CHOICES:
        raise ValueError(f"draws must be one of {DRAW_CHOICES}, got {draws!r}")
    out = {"n_lags": int(n_lags), "feature": feature}
    if mode != "condition":
        out["mode"] = mode
        out["k"] = None if k is None else int(k)
    if null != "free":
        out["null"] = null
    if sigma_nm is not None:
        out["sigma_nm"] = float(sigma_nm)
    if sqd_pop is not None:
        out["sqd_pop"] = int(sqd_pop)
    if int(k_max) != K_VALUES[-1]:
        out["k_max"] = int(k_max)
    if int(draws) != N_REPLICATES:
        out["draws"] = int(draws)
    # PER SPOT (7 Nov 2026): the SQD-mixture null's standard where the tracks carry the tracker's precision; a typed
    # sigma replaces it, so the key is kept only when it is turned off with no sigma typed
    if not per_spot and sigma_nm is None:
        out["per_spot"] = False
    return out


def config_args(config: dict) -> dict:
    """A saved Run's settings as `section_config`'s arguments (a key missing is the standard)."""
    return {"n_lags": config.get("n_lags", 4), "feature": config.get("feature", "raw"),
            "mode": config.get("mode", "condition"), "k": config.get("k"), "null": config.get("null", "free"),
            "sigma_nm": config.get("sigma_nm"), "sqd_pop": config.get("sqd_pop"),
            "k_max": config.get("k_max", K_VALUES[-1]), "draws": config.get("draws", N_REPLICATES),
            "per_spot": config.get("per_spot", True)}


def wants_per_spot(config: dict) -> bool:
    """Whether a Run asks for the per-spot SQD-mixture null: the SQD mixture chosen, no sigma typed, per spot not
    turned off. The window then resolves the project's calibration for each condition that carries precision
    (`workspace_view.run_precision`), as every per-spot Run does."""
    c = section_config(**config_args(config))
    return c.get("null") == "sqd" and c.get("sigma_nm") is None and c.get("per_spot", True)


def compute_kwargs(config: dict) -> dict:
    """A committed Run's settings (`section_config`) as the arguments of `compute_clustering_view` /
    `compute_shared_clustering` (the mode is the caller's choice of function). Validated again, so a stale saved
    setting is refused by name here, not deep in a worker."""
    c = section_config(**config_args(config))
    out = {"n_lags": c["n_lags"], "feature": c["feature"], "null": c.get("null", "free"),
           "sigma_nm": c.get("sigma_nm"), "sqd_pop": c.get("sqd_pop"),
           "k_values": tuple(range(2, c.get("k_max", K_VALUES[-1]) + 1)),
           "n_replicates": c.get("draws", N_REPLICATES)}
    if c.get("mode") == "shared":
        out["k"] = c.get("k")
    return out


def run_problem(config: dict, n_conditions: int) -> str:
    """Why a Run cannot start with ``config`` over ``n_conditions``, or "" (the dialog's hint, and its Run button)."""
    try:
        compute_kwargs(config)
    except ValueError as exc:
        return str(exc)
    if n_conditions < 1:
        return "No conditions loaded."
    if config.get("mode") == "shared" and n_conditions < 2:
        return "Shared clusters compare conditions: load at least two."
    return ""


def cluster_inks(k: int, dark: bool = False) -> list:
    """One ink per cluster, slowest first: the first ``k`` colours of the project's palette
    (`hcl.default_palette`-sized, `condition_colours.default_palette`), fitted to the page (light or dark) so they
    hold their contrast on either. The screen's chart and the exported figure read this one rule; `CLUSTER_INKS` is
    the light-page list for code that has no theme."""
    from .condition_colours import default_palette, palette_slots

    k = max(1, int(k))
    return list(palette_slots(default_palette(k), dark=bool(dark))[:k])


#: The app's standard Run: the SQD mixture is the chance worth asking about (the library's own default stays the free
#: population, as the SQD weighting keeps v2's step).
STANDARD_CONFIG = section_config(null="sqd")


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def describe_config(config: dict) -> str:
    """The Run's settings in one line, for the page under the Run button."""
    from .clustering_null import NULL_LABELS

    c = section_config(**config_args(config))
    null = c.get("null", "free")
    parts = [f"{c['n_lags']} lags", CLUSTER_FEATURE_LABELS[c["feature"]], _lower_first(MODE_LABELS[c.get("mode", "condition")]),
             (f"k = {c['k']} set by hand" if c.get("k") else f"k from 2 to {c.get('k_max', K_VALUES[-1])}, chosen by the data"),
             f"chance: {_lower_first(NULL_LABELS[null])}"]
    if null == "sqd":
        sigma = (f"{c['sigma_nm']:g} nm" if c.get("sigma_nm") is not None
                 else "the acquisition's" if c.get("per_spot") is False else "per spot where the tracks carry it")
        pops = "Best (BIC)" if c.get("sqd_pop") is None else f"{c['sqd_pop']}"
        parts.append(f"localisation error {sigma}, SQD populations {pops}")
    if not c.get("k"):
        parts.append(f"{c.get('draws', N_REPLICATES)} chance draws")
    return "Run settings: " + " · ".join(parts) + ". Press Run to change them."


def headline(result: dict) -> str:
    """The first thing a card says: whether the clusters beat chance, and which chance (a result with no test, k set by
    hand, says so)."""
    k = result["best_k"]
    if result.get("exceeds_null") is None:
        return f"k = {k}, set by hand: not tested against chance"
    chance = "the SQD mixture" if result["null"]["kind"] == "sqd" else "one free population"
    if result["exceeds_null"]:
        return f"Cluster structure beyond {chance} at k = {k}"
    return f"No cluster structure beyond {chance}"


def _day_name(day) -> str:
    try:
        return f"replicate {int(day)}"
    except (TypeError, ValueError):
        return f"replicate {day}"


def _cluster_curves(table, labels, k: int, n_lags: int) -> list:
    """Each cluster's ensemble MSD over the lags clustered: ``[[(lag time s, MSD um2), ...], ...]`` for clusters
    1..k. ``labels``: uid -> cluster (1-based)."""
    from .canonical import without_attrs
    from .msd import ensemble_msd, msd_per_track

    plain = without_attrs(table)
    curves = []
    for new in range(1, k + 1):
        members = set(labels.index[labels == new])
        part = table[plain["uid"].isin(members).to_numpy()]
        ens = without_attrs(ensemble_msd(msd_per_track(part, max_lag=n_lags))).sort_values("lag_frames")
        curves.append([(float(r.lag_time_s), float(r.msd_um2)) for r in ens.itertuples()])
    return curves


def _pooled_null_D(table, n_lags: int) -> float:
    """The D the free Brownian null is given: the table's ensemble-MSD D (the verdict does not depend on it without
    localisation noise), 0.3 where that cannot be fitted."""
    from .msd import MSDError, ensemble_msd, fit_msd, msd_per_track

    try:
        d = float(fit_msd(ensemble_msd(msd_per_track(table, max_lag=n_lags)), n_lags=n_lags).D_um2_per_s)
    except MSDError:
        d = 0.3
    return d if math.isfinite(d) and d > 0 else 0.3


def compute_clustering_view(project, condition: str, *, n_lags: int = 4, feature: str = "raw",
                            k_values=K_VALUES, n_replicates: int = N_REPLICATES, null: str = "free", sigma_nm=None,
                            sqd_pop=None, precision=None) -> dict:
    """One condition's clustering: the k selection against chance, that k's clusters, their curves and their shares
    per replicate day. An expected refusal comes back as ``{"condition", "settings", "error"}``."""
    import warnings

    import numpy as np

    from .canonical import without_attrs
    from .clustering import ClusteringError, cluster_tracks, msd_features, select_k
    from .msd import MSDError

    from .clustering_null import null_for

    # per spot is the SQD mixture's alone: recorded as off only where that null ran without it
    settings = section_config(n_lags, feature, "condition", None, null, sigma_nm, sqd_pop,
                              per_spot=precision is not None or null != "sqd")
    try:
        table = project.condition(condition)
        n_total = int(table["uid"].nunique())
        d_null = _pooled_null_D(table, n_lags)
        _features, used = msd_features(table, n_lags=n_lags, feature=feature)
        ks = tuple(k for k in k_values if k < len(used))
        if not ks:
            raise ClusteringError(f"only {len(used)} tracks have all {n_lags} lags: too few to cluster")
        null_tracks, null_info = null_for(null, {condition: table}, sigma_nm=sigma_nm, sqd_pop=sqd_pop,
                                          precision=None if precision is None else {condition: precision})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            selection = select_k(table, D_um2_s=d_null, k_values=ks, n_lags=n_lags, feature=feature,
                                 n_replicates=n_replicates, null_tracks=null_tracks, null_kind=null_info["kind"])
        best = selection.best_k
        report = cluster_tracks(table, k=best, n_lags=n_lags, feature=feature)
    except (ClusteringError, MSDError, ValueError) as exc:
        return {"condition": condition, "settings": settings, "error": f"{type(exc).__name__}: {exc}"}

    # the clusters by D, slowest first, renumbered in that order
    clusters = report.clusters.sort_values("D_um2_per_s").reset_index(drop=True)
    renumber = {int(old): new for new, old in enumerate(clusters["cluster"], start=1)}
    labels = report.labels.map(renumber)
    rows = []
    for new, (_i, c) in enumerate(clusters.iterrows(), start=1):
        rows.append({"cluster": new, "n_tracks": int(c["n_tracks"]), "fraction": float(c["fraction"]),
                     "D": float(c["D_um2_per_s"]),
                     "alpha": None if not math.isfinite(float(c["alpha"])) else float(c["alpha"]),
                     "r2": float(c["r_squared"]), "motion": str(c["motion"])})
    plain = without_attrs(table)
    curves = _cluster_curves(table, labels, best, n_lags)
    # each cluster's share per replicate day
    days = []
    if "day" in plain.columns:
        day_of = plain.drop_duplicates("uid").set_index("uid")["day"]
        on = day_of.reindex(labels.index)
        for day in sorted(on.dropna().unique()):
            mine = labels[on == day]
            if len(mine):
                days.append({"day": _day_name(day), "n_tracks": int(len(mine)),
                             "shares": [float((mine == c).mean()) for c in range(1, best + 1)]})
    spread = None
    if len(days) >= 2:
        shares = np.array([d["shares"] for d in days])
        spread = {"mean": [float(v) for v in shares.mean(axis=0)],
                  "sd": [float(v) for v in shares.std(axis=0, ddof=1)], "n_days": len(days)}
    sel = [{"k": int(r.k), "observed": float(r.observed), "chance": float(r.chance),
            "chance_sd": float(r.chance_sd), "excess": float(r.excess),
            "z": None if not math.isfinite(float(r.z)) else float(r.z)} for r in selection.table.itertuples()]
    return {
        "condition": condition, "settings": settings,
        "n_tracks": n_total, "n_clustered": int(len(used)), "d_null": d_null,
        "null_d_invariant": bool(selection.null_is_D_invariant), "n_replicates": int(n_replicates),
        "null": null_info, "criterion": selection.criterion,
        "selection": sel, "best_k": int(best), "exceeds_null": bool(selection.exceeds_null),
        "verdict": selection.verdict, "clusters": rows, "curves": curves, "days": days, "day_spread": spread,
        "caveat": report.caveat,
    }


def card_payload(result: dict) -> dict:
    """One condition's card as drawn: the selection rows, the clusters' table, the curves and the words."""
    if "error" in result:
        return {"error": result["error"]}
    from .clustering_null import NULL_QUESTIONS, null_words

    s = result["settings"]
    k = result["best_k"]
    spread = result.get("day_spread")
    columns = ["Cluster", "Tracks", "Share of all tracks", "D (µm²/s)", "alpha", "R²"]
    if spread:
        columns.append(f"Per replicate, mean ± SD (n = {spread['n_days']})")
    rows = []
    for i, c in enumerate(result["clusters"]):
        row = [str(c["cluster"]), f"{c['n_tracks']:,}", f"{100 * c['fraction']:.1f} %", f"{c['D']:.3g}",
               "" if c["alpha"] is None else f"{c['alpha']:.2f}", f"{c['r2']:.3f}"]
        if spread:
            row.append(f"{100 * spread['mean'][i]:.1f} ± {100 * spread['sd'][i]:.1f} %")
        rows.append(row)
    selection_rows = [[str(r["k"]), f"{r['observed']:.1f}", f"{r['chance']:.1f} ± {r['chance_sd']:.1f}",
                       f"{r['excess']:+.1f}", "" if r.get("z") is None else f"{r['z']:+.1f}"]
                      for r in result["selection"]]
    feature = CLUSTER_FEATURE_LABELS[s["feature"]]
    lines = [
        ("Beyond chance: " if result["exceeds_null"] else "NOT beyond chance: ") + result["verdict"],
        f"{result['n_clustered']:,} of {result['n_tracks']:,} tracks reach all {s['n_lags']} lags and are clustered "
        f"(feature: {feature}); k chosen on the excess in chance SDs; "
        + null_words(result["null"], result["n_replicates"], result["null_d_invariant"]),
        "Question asked of the data: " + NULL_QUESTIONS[result["null"]["kind"]],
    ]
    if result.get("caveat"):
        lines.append((result["caveat"][0].upper() + result["caveat"][1:]).replace("msd fit", "MSD fit") + ".")
    lines.append(TRACKS_NOT_STEPS + " The verdict above is on the pooled tracks of all replicates.")
    if not spread:
        lines.append("One replicate: no spread between replicates.")
    return {"k": k, "exceeds_null": result["exceeds_null"], "headline": headline(result),
            "selection_columns": ["k", "Index", "Chance ± SD", "Excess", "Excess / SD"], "selection_rows": selection_rows,
            "columns": columns, "rows": rows, "curves": result["curves"], "lines": lines}
