"""Controllers for the comparison-first browser, pure, headless, no Qt.

Each function takes ``(project, spec, *, cache=None)`` and returns
``(draw, status)``: ``draw(figure)`` paints a neutral comparison view (from
``smtracker.compare_plots``) and ``status`` is a one-line human summary carrying
the real numbers. All computation is in ``smtracker.comparison`` (or the analysis
modules it calls); the Qt shell only presses the button and shows the canvas, so
a wrong number fails a controller test, never subtly on screen.

Every view also gets a **provenance caption** stamped onto the figure and carried
into any export, so a shared figure stays traceable to the code + settings + data
that made it. The caption is drawn with ``figure.text``, and, like every other
text artist here, it must stay NEUTRAL (no interpretive word), which the gating
test enforces across the whole view.
"""

from __future__ import annotations

from . import compare_plots as cp
from . import hittest as _hittest
from . import comparison as C

__all__ = [
    "VIEWS",
    "assemble_view",
    "is_view_cached",
    "saspt_view",
    "spatial_view",
    "sqd_view",
    "appd_view",
    "kinetics_view",
    "sigma_view",
    "provenance_for",
    "stamp_caption",
    "condition_at_x",
    "count_warnings",
]


def count_warnings(project, spec: C.ComparisonSpec, conditions, *, cache=None) -> list:
    """"<condition>: Best (BIC) chooses 1 population here, ...; kept at the 2 you set" for every condition of the Run
    fitted at a count set by hand where Best chooses fewer (`comparison.forced_count_bic`,
    `bubble_view.forced_count_warning`; ruled 4 Nov 2026). The count is kept."""
    from .bubble_view import forced_count_warning

    out = []
    for condition in conditions:
        bic = C.forced_count_bic(project, condition, spec, cache)
        words = forced_count_warning(int(spec.model_for(condition)), bic) if bic else None
        if words:
            out.append(f"{condition}: {words}")
    return out


def condition_at_x(conditions, x):
    """The condition a click at data-x ``x`` lands on, or ``None`` if off-plot.

    Every comparison view lays conditions out at 0-based integer x positions, so
    the drill target is just the nearest column within half a step. Pure, so the
    Qt click handler stays a one-liner and the hit-testing is unit-tested.
    """
    if x is None or not conditions:
        return None
    index = int(round(x))
    if abs(x - index) > 0.5 or index < 0 or index >= len(conditions):
        return None
    return conditions[index]

#: (label, controller attribute). The order tabs appear in the window.
#: PER-condition only. The cross-condition comparison views, the fraction heatmap
#: (conditions × methods), the fraction SuperPlot (between-condition permutation p),
#: and the cross-method forest (Δ endpoint): were REMOVED: comparison and
#: interpretation belong in the manuscript, not the tool (pre-registration
#: Amendment 4). The tool describes each condition; it computes no Δ, p, or
#: threshold-based cross-condition slow-fraction.
VIEWS = [
    ("SQD populations", "sqd_view"),
    ("Apparent-D distribution", "appd_view"),
    ("saSPT spectra", "saspt_view"),
    ("HMM kinetics", "kinetics_view"),
    ("Spatial map", "spatial_view"),
    ("σ sensitivity", "sigma_view"),
]


#: The comparison cache-version lives in ``comparison`` so both the controller and
#: the per-condition fit caches key on the SAME value (see its definition there).
COMPARISON_CACHE_VERSION = C.COMPARISON_CACHE_VERSION


def provenance_for(project, spec: C.ComparisonSpec, *, view: str, extra=None):
    """A provenance stamp for a comparison view: settings + data + code + seeds.

    Timestamp/environment are excluded from the key by ``capture``; the settings
    that actually change the numbers (the conditions, the models, the mobility
    threshold, n_boot, and any ``extra`` such as the forest's ticked methods) are
    the key material.
    """
    from .provenance import capture

    settings = {
        "view": view,
        "cache_version": COMPARISON_CACHE_VERSION,     # invalidate on structural change
        "conditions": list(spec.conditions),
        "models": {c: spec.model_for(c) for c in spec.conditions},
        # "Best" over 1-3 and over 1-4 are different fits of the same data
        "best_max_pop": spec.best_max_pop,
        "threshold_um2_s": spec.threshold_um2_s,
        "reference": spec.reference_condition() if spec.conditions else None,
        "n_boot": spec.effective_n_boot(),
        # THE RESOLVED BUDGET (see comparison._ecdf_budget): the spec field is None by
        # default while the budget is not, so keying on the field would hand a thinned
        # result to a run that asked for the exact curve.
        "ecdf": C._ecdf_budget(spec),                  # in the key: it's a shortcut
        # D-sharing scope is ONE setting with three levels (Item 2). It IS key
        # material: a payload fit under "replicate" scope (per-replicate D) must
        # never be served under "condition" scope (shared D). Stamping it here is
        # what makes that stale-payload swap impossible.
        "d_sharing_scope": spec.d_sharing_scope,
        # A SHARED D PER CONDITION (1 Nov 2026): every condition's scope when they differ, nothing when the Run has
        # one (`ComparisonSpec.scope_settings`) -- so no key made before moves, and no version is raised: an entry
        # made before is exactly the Run it was made for.
        **spec.scope_settings(),
        # which conditions share a D under Shared D within a group (nothing otherwise)
        **spec.sharing_settings(),
        # the per-replicate fraction weighting (step|track) moves the fractions, so
        # it is key material, a step-weighted payload must not be served for track.
        "fraction_weighting": spec.fraction_weighting,
        # D-axis Berglund correction (item C) is DISPLAY-only but IS key material:
        # a corrected bubble figure must never be served for an uncorrected request
        # or vice versa (they show different D). Fenced like the fast-mode stamp.
        "d_correction": spec.d_correction,
        **spec.fast_settings(),                        # the honesty stamp
    }
    if extra:
        settings.update(extra)
    from .parallel import n_workers

    env_extra = {"workers": n_workers(spec.n_jobs),
                 "shortcuts": C.active_shortcuts(spec)}

    # Which replicate-day mapping produced the data, so a figure records it. It
    # is KEY material only for views whose picture changes with the mapping, the
    # per-day-coloured swarms (fraction, appD), and DISPLAY-ONLY for the rest
    # (an SQD/HMM/saSPT/spatial number does not change when day LABELS change), so
    # the warm cache for those expensive views is not busted by a mapping edit.
    day_mappings = getattr(project.canonical, "attrs", {}).get("day_mappings")
    if day_mappings:
        record = {c: {"source": m.get("source"), "n_days": m.get("n_days"),
                      "series_to_day": m.get("series_to_day")}
                  for c, m in day_mappings.items()}
        if view in ("fraction", "appd"):
            settings["day_mappings"] = record       # colouring depends on it → key
        else:
            env_extra["day_mappings"] = record      # recorded, but not key material

    return capture(
        settings=settings,
        seeds={"sqd": 0, "hmm": 0, "bootstrap": 0},
        canonical=project.canonical,
        unit="cell",
        # DISPLAY-ONLY: worker count and the active speed shortcuts record HOW it
        # ran (excluded from the key so 8/64 cores share the cache and the
        # numbers). The numbers-affecting shortcuts (model lock, fast) are also in
        # `settings` above, so the key already reflects them; this is the
        # human-readable summary that rides into the export.
        environment_extra=env_extra,
    )


def _cached_view(cache, provenance, kind, build):
    """Cache a WHOLE assembled view dict by its provenance, not just inner fits.

    This is what makes ``--browse`` instant off the warm cache: the entire view,
    every fit, bootstrap, saSPT run and confinement null inside it: is computed
    once and stored as one payload, so a warm hit returns it whole. Caching only
    the inner fits left the point estimates and the uncached bootstraps (SQD CI,
    saSPT, confinement) paying full cost on every open.
    """
    if cache is None:
        return build()
    return cache.get_or_compute(kind, provenance, build)


def stamp_caption(figure, provenance) -> None:
    """Draw the provenance one-liner at the foot of a view (neutral text)."""
    figure.text(0.01, 0.005, provenance.summary(), fontsize=6.0, color="#4a5568",
                family="monospace")


def _finish(figure, draw_fn, data, project, spec, view):
    """Paint the view, stamp provenance, and banner fast mode. Returns provenance.

    The provenance is also STASHED on the figure (``figure._smtracker_provenance``)
    so an export records the provenance of the data actually plotted, never a
    fresh re-derivation from a picker the user may have changed since, a
    data-integrity guarantee, not a cosmetic one.
    """
    draw_fn(figure, data)
    prov = provenance_for(project, spec, view=view)
    stamp_caption(figure, prov)
    _stamp_shortcuts(figure, spec)                 # fence the speed shortcuts
    if spec.is_fast:
        _banner_fast(figure, spec)
    figure._smtracker_provenance = prov            # type: ignore[attr-defined]
    return prov


def _stamp_shortcuts(figure, spec):
    """Name every active speed shortcut on the figure, so a fast/locked result
    carries its caveats into the export and can never pass as full rigor."""
    shortcuts = C.active_shortcuts(spec)
    if not shortcuts:
        return
    figure.text(0.01, 0.028, "speed shortcuts: " + "; ".join(shortcuts),
                fontsize=6.0, color="#c05621", family="monospace")


def _banner_fast(figure, spec):
    """A loud, unmissable fast-mode banner across the figure, the honesty guard's
    visible half (the provenance stamp is the machine-readable half). Neutral text,
    so it never trips the figure-neutrality gate."""
    figure.text(0.5, 0.5, spec.fast.banner, fontsize=26, color="#c53030",
                alpha=0.18, ha="center", va="center", rotation=18, zorder=1000,
                fontweight="bold")


# --------------------------------------------------------------------------- #
# Assembly, one place computes (and caches) each whole view dict
# --------------------------------------------------------------------------- #


def assemble_view(project, spec: C.ComparisonSpec, view: str, *, cache=None,
                  methods=None, with_ci=False, uncertainty="analytic",
                  spatial_source="appd", spatial_mode="enrichment", ci_unit="cell"):
    """The cached, whole-view assembled data for ``view``, the single source both
    the drawing controllers and the fixture generator read, so a warm hit serves
    exactly the payload a view draws.

    ``with_ci`` defaults to False, matching the tier policy's default (no bootstrap):
    a caller that omits it gets the instant point estimates, never a surprise
    50-refits-per-condition SQD bootstrap. Every real caller passes it explicitly
    (``sqd_view`` False, the fixture generator True), so this default is a footgun
    guard, not a behaviour change (see docs/uncertainty_tier_audit.md). ``ci_unit``
    is what ``with_ci`` resamples, ``"cell"`` or ``"movie"`` (`comparison.sqd_bubbles`).
    """
    # The build passes the cache INWARD so that when the whole-view cache misses
    # (e.g. the mobility threshold moved), the threshold-INDEPENDENT fits and
    # bootstraps are still warm hits, only the cheap below-threshold parts
    # recompute. That is what keeps the threshold slider instant on the SQD view.
    if view == "sqd":
        prov = provenance_for(project, spec, view="sqd", extra=_sqd_extra(with_ci, ci_unit))
        return _cached_view(cache, prov, "cmp_view_sqd",
                            lambda: C.sqd_bubbles(project, spec, cache=cache,
                                                  with_ci=with_ci, ci_unit=ci_unit))
    if view == "appd":
        prov = provenance_for(project, spec, view="appd")
        return _cached_view(cache, prov, "cmp_view_appd",
                            lambda: C.appd_points(project, spec))
    if view == "kinetics":
        prov = provenance_for(project, spec, view="kinetics")
        return _cached_view(cache, prov, "cmp_view_kinetics",
                            lambda: C.hmm_kinetics(project, spec, cache=cache))
    if view == "fraction":
        prov = provenance_for(project, spec, view="fraction",
                              extra={"uncertainty": uncertainty})
        return _cached_view(cache, prov, "cmp_view_fraction",
                            lambda: C.fraction_superplot(project, spec, cache=cache,
                                                         uncertainty=uncertainty))
    if view == "forest":
        prov = provenance_for(project, spec, view="forest",
                              extra={"methods": list(methods) if methods else "all"})
        return _cached_view(cache, prov, "cmp_view_forest",
                            lambda: C.forest(project, spec, methods=methods,
                                             cache=cache))
    if view == "heatmap":
        prov = provenance_for(project, spec, view="heatmap")
        return _cached_view(cache, prov, "cmp_view_heatmap",
                            lambda: C.fraction_matrix(project, spec, cache=cache))
    if view == "saspt":
        prov = provenance_for(project, spec, view="saspt")
        return _cached_view(cache, prov, "cmp_view_saspt",
                            lambda: C.saspt_spectra(project, spec, cache=cache))
    if view == "spatial":
        prov = provenance_for(project, spec, view="spatial",
                              extra={"source": spatial_source, "mode": spatial_mode})
        return _cached_view(cache, prov, "cmp_view_spatial",
                            lambda: C.spatial_maps(project, spec,
                                                   source=spatial_source,
                                                   mode=spatial_mode, cache=cache))
    raise ValueError(f"unknown view {view!r}")


def _sqd_extra(with_ci, ci_unit) -> dict:
    """The SQD view's cache settings: a cell interval keeps the key it always had."""
    return {"with_ci": with_ci} if ci_unit == "cell" else {"with_ci": with_ci, "ci_unit": ci_unit}


def _view_kind_prov(project, spec, view, *, methods=None, with_ci=False,
                    uncertainty="analytic", spatial_source="appd",
                    spatial_mode="enrichment", ci_unit="cell"):
    """(cache kind, provenance) for a view: the SAME key assemble_view uses, so
    Fit All can ask 'is this already cached?' without recomputing."""
    extra = {
        "sqd": _sqd_extra(with_ci, ci_unit),
        "fraction": {"uncertainty": uncertainty},
        "forest": {"methods": list(methods) if methods else "all"},
        "spatial": {"source": spatial_source, "mode": spatial_mode},
    }.get(view)
    return f"cmp_view_{view}", provenance_for(project, spec, view=view, extra=extra)


def is_view_cached(cache, project, spec, view, **kw):
    """Whether ``view`` is already cached and valid for this spec (Fit All skip)."""
    if cache is None or not hasattr(cache, "has"):
        return False
    kind, prov = _view_kind_prov(project, spec, view, **kw)
    try:
        return bool(cache.has(kind, prov))
    except Exception:                                  # pragma: no cover - defensive
        return False


# --------------------------------------------------------------------------- #
# The five views
# --------------------------------------------------------------------------- #


def saspt_view(project, spec: C.ComparisonSpec, *, cache=None):
    data = assemble_view(project, spec, "saspt", cache=cache)

    def draw(figure):
        _finish(figure, cp.draw_saspt, data, project, spec, "saspt")

    bits = ", ".join(
        f"{c}: {100 * d['occupancy_below']:.1f}% below D<{spec.threshold_um2_s:g}"
        for c, d in data["per_condition"].items())
    return draw, f"saSPT spectra - {bits}"


def spatial_view(project, spec: C.ComparisonSpec, *, cache=None, source="appd",
                 mode="enrichment"):
    data = assemble_view(project, spec, "spatial", cache=cache,
                         spatial_source=source, spatial_mode=mode)

    def draw(figure):
        _finish(figure, cp.draw_spatial, data, project, spec, "spatial")

    mapped = [c for c, rows in data["panels"].items()
              if rows and "error" not in rows[0]]
    sparse = sum(1 for rows in data["panels"].values()
                 for p in rows if p.get("sparse"))
    return draw, (f"spatial {data['mode']} ({data['source']}): {len(mapped)}/"
                  f"{len(data['conditions'])} conditions mapped, "
                  f"{data['n_pops']} populations, per-population robust scale, "
                  f"{sparse} panel(s) too sparse, {data['n_localizations']} localizations")


def heatmap_view(project, spec: C.ComparisonSpec, *, cache=None):
    data = assemble_view(project, spec, "heatmap", cache=cache)

    def draw(figure):
        _finish(figure, cp.draw_heatmap, data, project, spec, "heatmap")

    import numpy as np

    vals = np.asarray(data["values"], dtype=float)
    finite = np.isfinite(vals)
    span = (f"{100 * np.nanmin(vals):.0f}-{100 * np.nanmax(vals):.0f}%"
            if finite.any() else "n/a")
    return draw, (f"fraction @ D<{spec.threshold_um2_s:g} across "
                  f"{len(data['conditions'])} conditions x {len(data['methods'])} "
                  f"methods (range {span})")


def _shared_canonical(project, spec: C.ComparisonSpec, condition):
    """The data a replicate payload depends on: its condition's, and under a D shared with other conditions theirs
    too -- an exclusion in one condition moves the D a sibling condition's fractions are fitted at."""
    with_ = spec.sharing_settings(condition).get("d_sharing_with")
    if not with_ or len(with_) < 2:
        return project.condition(condition)
    import pandas as pd

    return pd.concat([project.condition(c) for c in spec.conditions if str(c) in with_], ignore_index=True)


#: The replicate payload's FIELDS VERSION, in its cache key (`replicate_fit_payload`): 2 since 30 Oct 2026, when
#: ``pooled`` and each replicate gained ``n_movies``.
REPLICATE_PAYLOAD_FIELDS = 2


def replicate_fit_payload(project, spec: C.ComparisonSpec, condition, *,
                          cache=None, compute=False):
    """The bubble inspector's per-replicate simultaneous-fit payload, CACHED.

    ``compute=True`` computes + stores it (called during Fit/Plot All / the SQD
    view build). ``compute=False`` is a PEEK: it returns the cached payload if
    present, else None, never a synchronous fit, so a bubble click is a cache
    read, never a wait. The payload is small (reuses the cached pooled fit + a
    cheap per-replicate EM), so a hit renders in well under 100 ms.
    """
    from .provenance import capture

    prov = capture(
        settings={"view": "replicate_fit", "condition": condition,
                  "cache_version": COMPARISON_CACHE_VERSION,
                  "model": spec.model_for(condition),
                  "best_max_pop": spec.best_max_pop,
                  "threshold_um2_s": spec.threshold_um2_s,
                  # scope IS in the key: a per-replicate-D payload (replicate scope)
                  # must never be served under condition/global scope (Item 1/2 trap).
                  # THE CONDITION'S OWN SCOPE (`ComparisonSpec.scope_for`, 1 Nov 2026), the one
                  # `condition_replicate_fit` fits it at: under one scope for the Run it is that
                  # scope, so every key made before holds; a condition Not shared beside others
                  # Across replicates shares its entry with the Run where all are Not shared --
                  # rightly, it is fitted from its own data alone under both.
                  "d_sharing_scope": spec.scope_for(condition),
                  "fraction_weighting": spec.fraction_weighting,
                  # THE ECDF BUDGET WAS MISSING FROM THIS KEY, so with a cache wired
                  # in, a Full run would be served the Fast payload -- the pooled fit
                  # inside it is made at the budget. Latent until the workspace began
                  # passing a cache.
                  "ecdf": C._ecdf_budget(spec),
                  **spec.fast_settings(),
                  # RANDOM CHUNKS (`comparison.PSEUDO_CHUNKS`): the key hashes the cells' content and not their
                  # replicate labels, so without this a chunk payload and a replicate payload would share an entry
                  **spec.replicate_settings(),
                  # THE CONDITIONS ITS D IS SHARED WITH (Across all conditions, or its group), and their data below
                  **spec.sharing_settings(condition),
                  # WHAT THE PAYLOAD HOLDS (30 Oct 2026: the movies counted, for the bubble chart's "n" line): a payload
                  # made before is made again -- cheaply, the pooled fit is cached on its own key -- never served
                  # without them. Only this key moves, so no fit is redone.
                  "payload_fields": REPLICATE_PAYLOAD_FIELDS},
        seeds={"sqd": 0}, canonical=_shared_canonical(project, spec, condition), unit="cell")
    kind = f"cmp_replfit_{condition}"
    build = lambda: C.condition_replicate_fit(project, condition, spec, cache=cache)
    if cache is None:
        return build() if compute else None
    if not compute and not cache.has(kind, prov):
        return None
    return cache.get_or_compute(kind, prov, build)


def sqd_view(project, spec: C.ComparisonSpec, *, cache=None, with_ci=False,
             payload_sink=None):
    # CIs are OPT-IN: the per-population bootstrap is 50 refits PER CONDITION, by
    # far the SQD view's biggest cost, and not needed for a first look. Off by
    # default → the bubble is just the point fits (fast); the browser's "SQD CIs"
    # box turns the whiskers on when wanted.
    data = assemble_view(project, spec, "sqd", cache=cache, with_ci=with_ci)
    # Precompute the bubble-inspector payloads so a bubble CLICK is INSTANT (Item
    # 3): compute+store on disk (cross-session) AND, when given a ``payload_sink``
    # dict, stash the payload in memory keyed by condition. The click then reads
    # that dict (a hash-free lookup) instead of re-capturing provenance, which is
    # what keeps it under the 100 ms budget.
    for column in data["columns"]:
        if not getattr(column, "error", None):
            try:
                payload = replicate_fit_payload(project, spec, column.condition,
                                                cache=cache, compute=True)
                if payload_sink is not None:
                    payload_sink[column.condition] = payload
            except Exception:                          # pragma: no cover - payload is optional
                pass

    # SEVERAL FITS AT ONCE, and the existing single-arrangement drawing kept behind a
    # flag. The D-sharing combo picks ONE arrangement, so a reader saw the fit that was
    # chosen and never what choosing it cost -- measured on a lab project as up to
    # 9 percentage points, which is larger than most effects these panels are read for.
    overlay, overlay_note = _overlay_table(project, spec)

    def draw(figure):
        if overlay is None:
            _finish(figure, cp.draw_bubble, data, project, spec, "sqd")
            return
        _finish(figure, _draw_overlay, overlay, project, spec, "sqd")
        # THE DRILL-DOWN SURVIVES, BY CONDITION. With several series in one column a
        # single bubble is no longer an unambiguous target, so the click resolves to the
        # COLUMN -- which is the identity `replicate_fit_payload` is keyed on anyway.
        _hittest.attach(figure, _hittest.ColumnHitTest(list(spec.conditions)))

    if overlay is None:
        from .comparison import below_populations

        bits = ", ".join(
            f"{col.condition}: {col.n_pop}-pop, below-threshold "
            f"{100 * col.slow_fraction:.1f}% ({below_populations(col.D, spec.threshold_um2_s)})"
            for col in data["columns"])
        caption = f"SQD populations @ D<{spec.threshold_um2_s:g} - {bits}"
        if overlay_note:
            caption += "   [%s]" % overlay_note
        return draw, caption

    slow = overlay[overlay["population"] == 1]
    series = list(dict.fromkeys(overlay["series"]))
    bits = "; ".join(
        "%s: %s" % (name, ", ".join(
            "%s %.1f%%" % (row.condition, 100.0 * row.fraction)
            for row in slow[slow["series"] == name].itertuples()))
        for name in series)
    # A CONTROL THAT IS NOT BEING HONOURED IS NAMED. The per-condition model picker
    # chooses a state count per condition; the overlay draws EXPLICIT counts, because a
    # series is one count by construction. Saying so beats a silent divergence between
    # what the picker reads and what the figure shows.
    chosen = {c: spec.model_for(c) for c in spec.conditions}
    if any(v not in ("best", "") for v in chosen.values()):
        bits += ("   [the per-condition model picker is not driving this view; the "
                 "state counts drawn are %s]"
                 % ", ".join(str(n) for n in spec.bubble_n_pops))
    if overlay_note:
        bits += "   [%s]" % overlay_note
    return draw, "SQD populations, fits overlaid - %s" % bits


def _draw_overlay(figure, table):
    from . import plots

    ax = figure.add_subplot(111)
    plots.draw_bubble_panel(ax, table)
    series = list(dict.fromkeys(table["series"]))
    ax.legend(handles=plots.bubble_panel_legend(series),
              loc="upper center", ncol=min(len(series), 2),
              bbox_to_anchor=(0.5, -0.20), frameon=False, fontsize=7)
    figure.subplots_adjust(bottom=0.36)


def _overlay_table(project, spec):
    """The overlaid series, or ``None`` to keep the single-arrangement drawing.

    ``None`` is returned when the caller has asked for exactly one series AND one state
    count, i.e. when there is nothing to overlay -- the classic drawing is richer there
    (it carries the side table and the per-bubble hit-test), so it stays the better
    picture for that case.
    """
    import pandas as pd

    from .aggregate import AggregationError
    from .compare import ComparisonError, bubble_series
    from .observables import squared_displacements

    modes = tuple(getattr(spec, "bubble_fit_modes", ()) or ())
    counts = tuple(getattr(spec, "bubble_n_pops", ()) or ())
    if len(modes) * len(counts) <= 1:
        # ONE ARRANGEMENT HAS NOTHING TO OVERLAY, and the classic drawing is richer
        # there: it carries the side table and the per-bubble hit-test. Returned as a
        # PAIR like every other exit, because a function with two return shapes is one
        # the caller unpacks correctly on the path that was tested and crashes on the
        # other -- which is exactly what this line did.
        return None, ""

    frames = []
    for condition in spec.conditions:
        canon = project.condition(condition).copy()
        canon["condition"] = condition
        # THE TRACK KEY IS QUALIFIED BEFORE POOLING. Conditions from separate u-track
        # projects can share movie names, and displacements group by uid, so an
        # unqualified pool computes steps BETWEEN molecules in different conditions --
        # measured as 29 steps of up to 9.2 um in a 1 um cell.
        canon["uid"] = str(condition) + ":" + canon["uid"].astype(str)
        frames.append(canon)
    from .comparison import _sqd_observable

    pooled = pd.concat(frames, ignore_index=True)
    observable = _sqd_observable(pooled, spec)        # the Run's lag, or lags 1-4 together
    wanted = getattr(spec, "bubble_unit", "day")
    # THE DAY UNIT IS REFUSED OVER ONE SESSION, and rightly: pseudo-days are not
    # replicates and a per-day interval built from them is anti-conservative. The
    # documented fallback is the CELL, with the single-session caveat reported -- so it
    # is taken here and NAMED, never taken quietly.
    for unit, note in ((wanted, ""),
                       ("cell", "spread over CELLS: the replicate unit was refused, this is "
                                "one imaging session and pseudo-replicates are not "
                                "replicates")):
        if unit != wanted and wanted != "day":
            break
        try:
            # PER-SPOT PRECISION (6 Nov 2026): both series with each step's own noise, by maximum likelihood
            table = bubble_series(
                observable, fit_modes=modes, n_pops=counts, unit=unit,
                weighting=spec.fraction_weighting, **spec.sqd_fit_kwargs())
        except AggregationError:
            continue
        except ComparisonError as exc:
            # A REFUSAL FALLS BACK TO THE PICTURE THAT STILL WORKS rather than emptying
            # the tab, and the reason travels so nothing claims the overlay was drawn.
            return None, "overlay not drawn: %s" % str(exc).split(".")[0]
        table.attrs["unit_note"] = note
        return table, note
    return None, ("overlay not drawn: no usable unit for the per-series spread "
                  "(the replicate unit is refused over a single session and the cell unit "
                  "did not fit either)")


def appd_view(project, spec: C.ComparisonSpec, *, cache=None):
    data = assemble_view(project, spec, "appd", cache=cache)

    def draw(figure):
        _finish(figure, cp.draw_ridgeline, data, project, spec, "appd")

    bits = ", ".join(f"{c}: {d['n_tracks']} tracks / {d['n_cells']} cells"
                     for c, d in data["per_condition"].items())
    return draw, f"apparent-D distribution - {bits}"


def kinetics_view(project, spec: C.ComparisonSpec, *, cache=None):
    data = assemble_view(project, spec, "kinetics", cache=cache)

    def draw(figure):
        _finish(figure, cp.draw_kinetics, data, project, spec, "kinetics")

    parts = []
    for row in data["rows"]:
        on = f"{row.on_rate:.2f}" if row.on_identifiable else "unidentifiable"
        off = f"{row.off_rate:.2f}" if row.off_identifiable else "unidentifiable"
        parts.append(f"{row.condition}: occ {100 * row.occupancy_slow:.1f}%, "
                     f"k(2->1) {on}, k(1->2) {off} 1/s")
    return draw, " | ".join(parts)


def fraction_view(project, spec: C.ComparisonSpec, *, cache=None,
                  uncertainty="analytic"):
    data = assemble_view(project, spec, "fraction", cache=cache,
                         uncertainty=uncertainty)

    def draw(figure):
        _finish(figure, cp.draw_fraction, data, project, spec, "fraction")

    bits = ", ".join(
        f"{c}: {d['mean']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}] "
        f"({d['n_cells']} cells)" for c, d in data["per_condition"].items())
    pv = "; ".join(f"{a} vs {b}: p={cp.format_p(p)}"
                   for (a, b), p in data["pvalues"].items())
    tail = f" | cell-level permutation {pv}" if pv else ""
    return draw, f"fraction @ D<{spec.threshold_um2_s:g} - {bits}{tail}"


def forest_view(project, spec: C.ComparisonSpec, *, methods=None, cache=None):
    data = assemble_view(project, spec, "forest", cache=cache, methods=methods)

    def draw(figure):
        _finish(figure, cp.draw_forest, data, project, spec, "forest")

    shown = ", ".join(
        f"{r.method} {r.delta:+.3f} [{r.ci[0]:+.3f}, {r.ci[1]:+.3f}]"
        for r in data["rows"] if r.delta == r.delta)   # skip NaN
    return draw, (f"forest {data['target']} - {data['reference']} "
                  f"({data['n_methods']} methods): {shown}")


def sigma_view(project, spec: C.ComparisonSpec, *, cache=None):
    """The σ-sensitivity sweep as a browser panel (item D). Computed directly (a
    deliberate, expensive refit-per-σ action, not part of the warm view set)."""
    data = C.sigma_sensitivity(project, spec)

    def draw(figure):
        _finish(figure, cp.draw_sigma_sensitivity, data, project, spec, "sigma")

    parts = []
    for c in data["conditions"]:
        occ = data["slow_occ"][c]
        parts.append(f"{c}: {100 * occ[0]:.1f}%→{100 * occ[-1]:.1f}% (σ 0→"
                     f"{data['sigmas_nm'][-1]:g} nm)")
    tail = ""
    if data.get("delta_range"):
        lo, hi = data["delta_range"]
        tail = f" | Δ spread {100 * (hi - lo):.1f} pp over σ"
    return draw, (f"σ-sensitivity ({data['n_pop']}-state): " + "; ".join(parts) + tail)
