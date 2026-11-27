"""One analysis section's result for one condition, kept between Runs and restarts (3 Nov 2026).

Asked: "spatial maps dont get cached? it seems slow? am i running different parameters?" and "same for sqd jump
distances. takes 5 + min even though same settings as yesterday". Only the SQD fits were kept (the workspace's
`WarmCache`, through `comparison_controller`); every other section computed every condition from scratch on every
Run, the same settings included, and again after a restart. Each section's per-condition job now goes through
`cached_job`: the same store, the same kind of key (`provenance.capture(...).content_key()`).

THE KEY. A job reads one condition of the project, so its key holds what that condition's result can depend on:
- every row of the condition's table in EVERY column (`rows`): coordinates, frames, cells, and the replicate labels
  too -- `provenance.hash_frame`'s default skips labels (the trap in CLAUDE.md), and relabelled replicates move every
  per-replicate result; an exclusion removes rows;
- the outlines of the condition's movies (`outlines`): the projection onto each cell, and every map, read them;
- the acquisition parameters (`params`: pixel size, frame interval, sigma). The table's other attrs are bookkeeping
  for the whole project (counts per condition) and stay out, or opening a different set of conditions would miss;
- the section, the job function, every setting the job is handed (a hidden track included), and the section's
  version below.
THE CODE IS NOT IN THE KEY, as for the SQD fits (`COMPARISON_CACHE_VERSION`): a change to what a section computes
RAISES ITS VERSION in `SECTION_CACHE_VERSIONS`, or results made before are served unchanged.

Every job function cached here is deterministic (its random draws are seeded), so a kept result is the result a
recompute gives. A job that raises stores nothing. The store is the workspace's own, per user, never on a data drive.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import asdict, is_dataclass

__all__ = ["SECTION_CACHE_VERSIONS", "condition_inputs", "cached_job", "cached_group_job"]

#: One version per section: RAISE IT when what the section computes for a condition changes, so every result kept
#: before is made again rather than served (the code is not in the key).
SECTION_CACHE_VERSIONS = {
    # Spatial Map: `spatial_map_view.compute_spatial_map` (a Run, and a redraw without a track); 2 since 3 Nov 2026,
    # when every SQD split also draws All jumps and a Best map keeps its BIC of every count
    # 3 since 3 Nov 2026 (later): the apparent-D and HMM maps draw their All column, HMM maps take 1-3 states
    # 4 since 4 Nov 2026: an HMM map at hard assignment names the states no step won (`never_most_likely`)
    # 5 the same day: an HMM map's steps are decoded under the fit's own initial distribution (`hmm.decode_states`,
    # tester C2), so its state shares are the fit's occupancy
    # 5 also, on another branch: a map at a count set by hand keeps Best's BIC beside it (`bic_forced`, `forced_count_text`)
    # 6 the same day: both 5s merged (a cache written by either branch is made again)
    # 7 the same evening: an SQD map's check beside a count set by hand is Best's own (maximum likelihood), no longer
    # its least-squares fits' (`_forced_selection`); HMM and apparent-D maps say the same as at 6
    # 8 the same evening (raised from main's 7): exact-zero jumps are an immobile population, counted
    # (`sqd.zero_jumps`), and the map says so
    # 9 since 4 Nov 2026: a fitted map may take a section's D (`map_fit_source`, the job's ``section_fit``, in the
    # key with the rest of its settings), and its payload says which D it used
    # 10 the same evening (merged on top of 9): an apparent-D map's exact-zero sub-tracks are an immobile population,
    # counted (`appd_gamma.zero_tracks`), cut at 0 and said; the grid says every condition's exact-zero note; and
    # the rule's floor (a 1 % floor, withdrawn the same day), SQD and apparent D alike (one raise for both on this branch)
    # 11 the same day: the rule has no floor, a handful of exact zeros is the immobile population too
    # 12 since 10 Nov 2026: an apparent-D map fitted Per spot uses the gamma's effective shape
    # (`appd_gamma.shape_and_scale`: consecutive steps share a localisation)
    # 13 since 22 Nov 2026: the HMM map's Beyond chance links its free pairs through the tracker's radius
    # (`hmm_angles.linking_for`) and walks a fast state in finer sub-steps (`hmm_angles._substeps_for`)
    # 14 the same day: its cells as wide as the fastest state's molecules fill (`hmm_angles.measure_inset`), its camera
    # open as long as the camera files say (`hmm_angles.exposure_for`), a state past the linking limit left without a
    # chance; 15 on 23 Nov 2026: an SDA map carries its x and y graphs (`spatialsda.sda_profiles`, payload["profiles"])
    "spatial": 15,
    "msd": 1,           # MSD: `msd_view.msd_condition`
    # SQD's Jump distances: `jumps_view.compute_jump_view` (the quick pass and the full one); 2 since 4 Nov 2026,
    # exact-zero jumps an immobile population (`sqd.zero_jumps`), the table's note and the ECDF read at its step
    # 3 the same evening (raised from main's 2): the exact-zero rule's floor (a 1 % floor, withdrawn the same day)
    # 4 the same day: no floor, a handful of exact zeros is the immobile population too
    "jumps": 4,
    # SQD's Signed jumps: `jd_view.compute_jd_view`; 2 since 4 Nov 2026, its SQD column under the exact-zero rule
    # 3 the same evening (raised from main's 2): the exact-zero rule's floor (a 1 % floor, withdrawn the same day)
    # 4 the same day: no floor, a handful of exact zeros is the immobile population too
    "jd": 4,
    # Apparent diffusion: `appd_view.appd_condition`; 3 since 3 Nov 2026, its result holding every model's design
    # (`appd_fits`) and its fit quality (the ECDF residuals and gap, the held-out replicates or cells: `fit_quality`)
    # 4 since 4 Nov 2026: no cell fractions under Don't force anywhere (no D shared by the cells; SQD's rule)
    # 5 the same day: Chunks withheld on a condition with replicates (`comparison.chunks_withheld`)
    # 6 the same day (raised from main's 5): its SQD-based cell shares and switching prediction under the exact-zero
    # rule (`sqd.zero_jumps`)
    # 7 the same evening (raised from main's 6): exact-zero tracks are an immobile population, counted, in every fit
    # of the section (`appd_gamma.zero_tracks`; they were dropped without a word), the histogram and its fits over the
    # tracks that move, the fit quality scored as the fits were made; and the rule's floor (a 1 % floor, withdrawn the same day,
    # 1 %), for its SQD cell shares and switching check too (one raise for both on this branch)
    # 8 the same day: no floor, a handful of exact zero tracks / jumps is the immobile population too
    # 9 since 10 Nov 2026: Localisation error Per spot fits the gamma at its effective shape
    # (`appd_gamma.shape_and_scale`: consecutive steps share a localisation, so the shape is under the step count;
    # Best (BIC) counted one bound population as two), and Not modelled says the cause where it shows
    "appd": 9,
    # HMM: `hmm_view.compute_hmm_view`; 2 since 3 Nov 2026, Force same D across (replicates held at the condition's D)
    # 3 since 4 Nov 2026: a chunk spread says chunks (its card lines are kept with the result); 4 the same day: the
    # lines carry SQD's fit warning (`hmm_view.fit_flags`)
    # 5 the same day: a theoretical sigma says so, a corrected D is never "apparent" (`hmm_view.d_word`)
    # 6 the same day: Chunks withheld on a condition with replicates (`comparison.chunks_withheld`)
    # 7 the same day: a count set by hand keeps Best's BIC beside it (`forced_bic`) and says where Best chooses fewer
    # 8 since 12 Nov 2026: a movie or cell interval carries each state's rate out and model dwell (`_exit_bands`)
    "hmm": 8,
    # the HMM section's Jump angles (`hmm_angles.compute_hmm_angles`, keyed on the condition's HMM result too); 1 since
    # 22 Nov 2026; 2 the same day (the chance optional, `compute_hmm_angles`' ``chance``; the data alone by default);
    # 3 the same day (the chance linked through the tracker's radius, finer sub-steps for a fast state, the wall's
    # shift taken off once); 4 the same day (the cells' width measured, the camera's exposure read, a state past the
    # linking limit without a chance)
    "hmm_angles": 4,
    # Clustering: `clustering_view.compute_clustering_view` (each condition alone) and
    # `clustering_compare.compute_shared_clustering` (one job over all of them, `cached_group_job`); 1 since 7 Nov 2026,
    # 2 the same day (the SQD-mixture chance per spot)
    "clusters": 2,
    # Kinetics' Populations over time: `induction_view.compute_induction` (the model, bands, bins) and
    # `induction_view.permutation_chunk` (the time-label permutation test's jobs); 1 since 7 Nov 2026,
    # 2 the same day (the test exact: every changing model refitted on every shuffle), 3 the same day (the experiment
    # designs: a result carries its design and its card's design lines, `kinetics_designs`), 4 on 8 Nov 2026 (the mean
    # mobility readout, the linking-limit caveat, the shapes drawn behind the chosen one)
    # 5 on 8 Nov 2026 (the mock addition: a result carries its fitted response for the Auto after window, and the
    # treated-against-mock job `kinetics_mock_view.compute_mock` is kept here, its key holding both conditions)
    # 6 the same day: a result says whether it holds baseline movies (``has_baseline``, the card's no-mock line)
    # 7 the same day: the linking-limit note says a drop near the limit reads late as well as small
    # (`induction_view.linking_limit_lines`)
    # 8 the same day: the treated-against-mock job says the smallest change it finds (`kinetics_mock.detectable_at`)
    # 9 the same day: the within-track line spells its ratios out ("D to fall 16.7 times", "fivefold"), never a digit
    # before "-fold" (`kinetics_designs.within_track_text`)
    # 10 the same day: a populations result keeps its half-time as numbers (`induction_view.half_time_record`), which
    # the card and the figure mark and the source data hold
    # 11 on 10 Nov 2026: a population's D on the fit's floor reads "not resolved" in the card's lines, which the
    # result keeps (`induction.d_at_floor`, `induction_view._d_words`)
    # 12 the same day: the mock comparison's bound-share lines end with what that readout cannot tell apart
    # (`kinetics_mock.ALIAS_SENTENCES`), and the result keeps its lines
    "kinetics": 12,
}


def _digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _rows_digest(table) -> str:
    """Every row in every column, order-independent (`provenance.hash_frame` over all columns, sorted by name) --
    except each spot's precision (`canonical.LOC_SIGMA_COLUMN`, 6 Nov 2026): added to every u-track table that day,
    it would have thrown away every result kept before, and only the per-spot precision fit reads it, which keys
    on it in its own settings."""
    from .canonical import LOC_SIGMA_COLUMN
    from .provenance import hash_frame

    columns = sorted((c for c in table.columns if c != LOC_SIGMA_COLUMN), key=str)
    try:
        return hash_frame(table, columns=columns)
    except TypeError:
        # a column of unhashable cells (lists): the same rows, pickled in the same order
        sort_keys = [c for c in ("uid", "frame") if c in table.columns] or columns
        ordered = table.sort_values(sort_keys, kind="stable")[columns].reset_index(drop=True)
        return _digest(pickle.dumps(ordered, protocol=4))


def _outlines_digest(project, table) -> str | None:
    """The outlines of the movies this condition's rows come from, in movie order (None: a project without any)."""
    meshes = getattr(project, "meshes", None)
    if not meshes:
        return None
    if not isinstance(meshes, dict):
        return _digest(pickle.dumps(meshes, protocol=4))
    movies = sorted({str(m) for m in table["movie"].unique()}) if "movie" in table.columns else []
    return _digest(pickle.dumps([(m, meshes.get(m)) for m in movies], protocol=4))


def _params(table) -> dict | None:
    params = table.attrs.get("params")
    if params is None:
        return None
    if is_dataclass(params):
        from .provenance import _clean

        values = asdict(params)
        # THE PER-CONDITION SEARCH RADII ARE KEYED ONLY WHEN THERE ARE ANY (tester D10, 4 Nov 2026): a project linked
        # at one radius keeps the key it had, so every result kept before stays valid; one linked at several gets new
        # keys (its ceilings and limits are now each condition's own)
        if values.get("search_radius_px_by_condition") is None:
            values.pop("search_radius_px_by_condition", None)
        return _clean(values)
    raise TypeError(f"acquisition parameters of type {type(params).__name__} have no stable form for a cache key")


def condition_inputs(project, condition: str) -> dict:
    """What one condition's result can depend on, as key material (see the module docstring)."""
    table = project.condition(condition)
    return {"rows": _rows_digest(table), "n_rows": int(len(table)),
            "outlines": _outlines_digest(project, table), "params": _params(table)}


def cached_job(cache, section: str, function, project, condition: str, settings: dict):
    """``function(project, condition, **settings)``, read from ``cache`` when the same condition's data was computed
    at the same settings before (the workspace's `WarmCache`), else computed and kept. ``cache`` None: computed, as
    before. Submitted to the workspace's process pool as it stands (every argument pickles; ``function`` by name)."""
    if section not in SECTION_CACHE_VERSIONS:
        raise ValueError(f"no cache version for section {section!r}; have {sorted(SECTION_CACHE_VERSIONS)}")
    if cache is None:
        return function(project, condition, **settings)
    from .provenance import capture

    prov = capture(
        settings={"section": section, "version": SECTION_CACHE_VERSIONS[section],
                  "function": f"{function.__module__}.{function.__qualname__}",
                  "condition": str(condition), "settings": dict(settings), **_chunk_rule(settings)},
        extra_inputs=condition_inputs(project, condition))
    return cache.get_or_compute(f"{section}_{condition}", prov, lambda: function(project, condition, **settings))


def _chunk_rule(settings: dict) -> dict:
    """Key material for a job on random chunks (Uncertainty "chunk": the apparent D's, the HMM's): the deal's rule, at
    least `comparison.PSEUDO_MIN_PER_CHUNK` units a chunk since 23 Nov 2026, so a result dealt under the rule of two is
    made again while every result without chunks keeps its key (no section's version raised for it)."""
    if str(settings.get("uncertainty")) != "chunk":
        return {}
    from .comparison import PSEUDO_MIN_PER_CHUNK

    return {"chunk_rule": {"min_per_chunk": PSEUDO_MIN_PER_CHUNK}}


def cached_group_job(cache, section: str, function, project, conditions, settings: dict):
    """``function(project, conditions, **settings)`` for a job that reads SEVERAL conditions at once (Clustering's shared
    centroids), kept as `cached_job` keeps one condition's: the key holds every condition's rows, outlines and parameters
    (`condition_inputs`, in the order given, which the result can depend on) and the settings. ``cache`` None: computed."""
    if section not in SECTION_CACHE_VERSIONS:
        raise ValueError(f"no cache version for section {section!r}; have {sorted(SECTION_CACHE_VERSIONS)}")
    conditions = [str(c) for c in conditions]
    if cache is None:
        return function(project, conditions, **settings)
    from .provenance import capture

    prov = capture(
        settings={"section": section, "version": SECTION_CACHE_VERSIONS[section],
                  "function": f"{function.__module__}.{function.__qualname__}",
                  "conditions": conditions, "settings": dict(settings)},
        extra_inputs={c: condition_inputs(project, c) for c in conditions})
    return cache.get_or_compute(f"{section}_group_{len(conditions)}", prov,
                                lambda: function(project, conditions, **settings))
