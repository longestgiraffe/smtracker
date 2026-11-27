"""WHICH D A FITTED SPATIAL MAP USES, AND WHY -- with no Qt anywhere in it (approved 4 Nov 2026).

Asked: "it should be a tick box on fits for all of these maps: ( ) use same D as sqd/appd/hmm fit (will wait for it
if it hasnt run or will run it itself if it hasnt started)". Until then every fitted Spatial Map (SQD, apparent D,
HMM) fitted its OWN D per condition whatever the section was set to, so a population's map in one condition could be
a different population than in another (on one lab project the free D differed clearly between two conditions on the maps, where the SQD
section, its D forced across the conditions, had one).

THE RULE (`plan`):

* ONE BOX PER FITTED METHOD in the map dialog, "Use the same D as the SQD fit" (`box_text`), TICKED for a new map
  (the dialog's default, `TICK_DEFAULT`). A config saved without the key (`TICK_KEY`) reads UNTICKED: maps made
  before keep their own fit, and so does a script's `spatial_map_view.map_config`, whose default is unticked.
* TICKED, the map takes the section's LAST finished Run (`sqd_source`, `appd_source`, `hmm_source`): per condition
  its D (whatever its Force same D across: replicates, conditions, the whole data set), its count, and what the fit
  was made at (SQD: its fractions' weighting and its time lag; HMM: its sigma; apparent D: its shortest track). The
  map does not refit D: its fractions (SQD, apparent D) or its transitions (HMM) are fitted AT those D on the map's
  own table, and its jumps are assigned (soft or hard, as the map is set) under them. The D is the one the section
  DRAWS: the shared D, or under Don't force anywhere what its bubbles show (the replicates' mean under Uncertainty
  Replicates, the pooled fit otherwise). A condition fitted at several models side by side gives the map ONE, the
  section's own primary rule (`workspace_view.primary_model`: Best where lit, else the smallest count of two or more),
  named in the map's caption.
* While the section's Run is in progress the map WAITS for it (`waiting_text`) and draws when it lands; a section that
  has never run is STARTED with its current settings (as its dialog shows them) and waited for (`starting_text`).
* A hand split uses no fit, so it neither waits nor starts anything (`uses_section`).
* A condition the section has no usable fit for is REFUSED by name in its own row (`refusal`): not run, failed, or a
  count the map cannot draw (SQD and HMM maps 1-3, apparent-D maps 2-3).
* A map keeps the fit it was drawn with (``payload["section_fit"]``, `FIT_KEY`). When the section is run again and
  its fit for a drawn condition differs, the map says "Made with an earlier SQD fit" and offers Run
  (`stale_conditions`, `stale_text`): it is never redrawn in silence. A redraw without a hidden track keeps the fit it
  was drawn with (`job_config(..., fit=payload's)`).

A fit is plain data (`fit` dicts: lists, numbers, strings), so it travels to the process pool and into the section
cache's key: a map made at one section fit is never served for another. ``page`` names the section page the fit came
from ("SQD", or an added page's name: "SQD 2") and ``page_key`` its key (`subpages`: ``"sqd"``, ``"sqd~2"``), which is
what a map remembers and what staleness compares, so renaming a page never makes a map look out of date. The map
dialog's drop-down picks the page (``config["fit_page"]``, `PAGE_KEY`); a map whose page is deleted says so
(`gone_text`) and its Run takes the section's own page instead.
"""
from __future__ import annotations

import json

__all__ = [
    "MAP_THIS_FIT", "map_this_fit_tip", "map_page_name",
    "FIT_KEY", "FIT_NAMES", "METHODS", "PAGE_KEY", "PAGE_NAMES", "SECTION_KEYS", "TICK_DEFAULT", "TICK_KEY",
    "appd_source", "box_help", "box_text", "fit_name", "fit_page_key", "fit_row_text", "fit_source_text",
    "fit_short_text", "gone_text", "hmm_source", "job_config", "not_finished_text", "plan", "populations_note",
    "refusal", "same_fit", "sqd_source", "stale_conditions", "stale_text", "starting_text", "ticked",
    "uses_section", "waiting_text",
]

#: The fitted map methods, each with a section whose fit it can take.
METHODS = ("sqd", "appd", "hmm")
#: The workspace job each method's section runs as (`WorkspaceWindow._running`).
SECTION_KEYS = {"sqd": "sqd", "appd": "appd", "hmm": "hmm"}
#: The fit as a sentence names it ("the SQD fit").
FIT_NAMES = {"sqd": "SQD", "appd": "apparent-D", "hmm": "HMM"}
#: The section page a fit comes from, today each section's one page; the + subpages will add "SQD 2" and so on.
PAGE_NAMES = {"sqd": "SQD", "appd": "Apparent Diffusion", "hmm": "HMM"}
#: The map config's key for the box, and the key a condition's job (and its payload) carries the fit under.
TICK_KEY = "same_D"
FIT_KEY = "section_fit"
#: The map config's key for WHICH page of the section the D comes from (`subpages`: "sqd", "sqd~2" ...); absent, the
#: section's own page. The Run's, never a job's (`spatial_map_view.condition_config` drops it).
PAGE_KEY = "fit_page"
#: A NEW map's box: ticked (approved 4 Nov 2026). A config without `TICK_KEY` reads unticked (`ticked`).
TICK_DEFAULT = True

#: MAP THIS FIT (23 Nov 2026, asked: "would it make sense even though its kind of duplicate... if we have a spatial map
#: in each section where we can compute it?", then "'Map this fit' button"): one map place, the Spatial Map, and from
#: each fitted section's tab row a button that opens its settings with the method chosen, the box ticked and this page
#: chosen for the D. Nothing runs until Run is pressed there. SINCE THE SAME EVENING A PAGE OF ITS OWN (asked: "default
#: should be map this fit creates a new page in spatial map called HMM ... 1 for example... press again HMM 2.... prss
#: in sqd SQD 1"): every press adds a Spatial Map page named after the fit's page (`map_page_name`: "HMM map 1", "HMM
#: map 2", "SQD 2 map 1"; "map" in the name, because a page's name is also its job's in the Running list, its toasts'
#: and its files', where "HMM 1" would read as an HMM fit), and a page closed without a Run goes again.
MAP_THIS_FIT = "Map this fit"


def map_page_name(source: str, taken) -> str:
    """The new Spatial Map page's name for a map of the fit on the page called ``source`` ("HMM", "SQD 2"):
    "<source> map <n>", ``n`` one past the highest such page among ``taken`` (the Spatial Map's page names), from 1."""
    stem = f"{source} map "
    numbers = [int(name[len(stem):]) for name in taken
               if str(name).startswith(stem) and str(name)[len(stem):].isdigit()]
    return f"{stem}{max(numbers, default=0) + 1}"


def map_this_fit_tip(method: str) -> str:
    """The button's hover: what it opens and what it sets ("an" before all three: SQD, HMM and Apparent Diffusion are
    each said with a vowel first)."""
    name = PAGE_NAMES[method]
    return (f"A new Spatial Map page for an {name} map whose D is this page's fit (“Use the same D as the "
            f"{name} fit” ticked, this page chosen), its settings open. Press Run there; closed without a Run, the "
            f"page goes again.")


# ---------------------------------------------------------------------------------------------------------------------
# The box and its words
# ---------------------------------------------------------------------------------------------------------------------

def fit_name(method: str, page: str | None = None) -> str:
    """How the fit is named in a sentence: the page's own name ("SQD 2") where one is given, else the section's
    ("SQD")."""
    return str(page) if page else FIT_NAMES[method]


def fit_page_key(config: dict | None, method: str | None = None) -> str:
    """The page key a map config takes D from: its ``fit_page``, else the section's own page (the method's key)."""
    config = config or {}
    return str(config.get(PAGE_KEY) or method or config.get("method") or "sqd")


def box_text(method: str) -> str:
    """The box's label: "Use the same D as the SQD fit"."""
    return f"Use the same D as the {FIT_NAMES[method]} fit"


def box_help(method: str) -> str:
    """The line under the box: what ticking it does."""
    name, section = FIT_NAMES[method], PAGE_NAMES[method]
    noun = "states" if method == "hmm" else "populations"
    return (f"Each condition takes the D, {noun} and settings from the {name} section's last Run, so a population "
            f"means the same here as in {section}. The map fits only its "
            f"{'transitions' if method == 'hmm' else 'fractions'} at those D. The maps wait for a running {section}; "
            f"if it has not run, Run starts it with its current settings.")


def populations_note(method: str) -> str:
    """Said on the Populations page while the box is ticked, where the counts are greyed."""
    noun = "state count" if method == "hmm" else "population count"
    what = f"{noun} and its time lag come" if method == "sqd" else f"{noun} comes"
    return f"Ticked: each condition's {what} from the {FIT_NAMES[method]} fit."


def ticked(config: dict | None) -> bool:
    """Whether a map config takes a section's D: its method fits populations and its box is ticked. A config saved
    without the key reads UNTICKED, so a map made before keeps its own fit."""
    config = config or {}
    return config.get("method") in METHODS and bool(config.get(TICK_KEY, False))


def uses_section(config: dict | None) -> bool:
    """Whether a map Run needs the section's fit: ticked, and not a split by hand (whose ranges use no D)."""
    return ticked(config) and (config or {}).get("split", "fitted") != "manual"


def waiting_text(method: str, page: str | None = None) -> str:
    return f"waiting for the {fit_name(method, page)} fit; the maps draw when it finishes…"


def starting_text(method: str, page: str | None = None) -> str:
    return (f"the {fit_name(method, page)} fit has not run: started with its current settings; the maps draw when "
            f"it finishes…")


def not_finished_text(method: str, page: str | None = None) -> str:
    section = page or PAGE_NAMES[method]
    return (f"The {fit_name(method, page)} fit did not finish, so the maps were not drawn. Run {section}, then the "
            f"maps (or untick “{box_text(method)}”).")


def gone_text(method: str, page: str, instead: str | None = None) -> str:
    """The map's line once the page it took D from has been deleted: said, never redrawn in silence, with the Run
    that takes D from ``instead`` (the section's own page) offered."""
    instead = instead or PAGE_NAMES[method]
    return (f"“{page}” was deleted, so these maps have no fit to take D from. Run the maps again to take D from "
            f"{instead} instead, or pick another page in the map settings.")


# ---------------------------------------------------------------------------------------------------------------------
# A section's last finished Run, per condition
# ---------------------------------------------------------------------------------------------------------------------

def _floats(values) -> list:
    return [float(v) for v in values]


def _sqd_drawn_D(payload: dict, uncertainty: str) -> list:
    """The D the SQD bubbles draw for one condition (`bubble_view._compute_bubble_view`): under Uncertainty
    Replicates (and Chunks, not withheld) the column's D -- the shared D, or under Don't force anywhere the mean of the
    replicates' own (`comparison._column_from_payload`) -- else the pooled fit's."""
    from .comparison import _column_from_payload

    column = _column_from_payload(payload, float(payload.get("threshold_um2_s") or 0.0))
    day_like = uncertainty in ("day", "chunk") and not payload.get("chunks_withheld")
    D = column.D if day_like else (column.pooled_D or column.D)
    return _floats(D)


def sqd_source(plot, payloads: dict, spec, uncertainty: str, page: str | None = None,
               page_key: str = "sqd") -> dict:
    """The SQD section's last finished Run as ``{"method", "page", "page_key", "conditions": {condition: fit or
    {"error"}}}``: from its figure (``plot``, a `BubblePlotData`), its per-condition payloads
    (`condition_replicate_fit`'s, keyed as the figure's columns), its `ComparisonSpec` and its Uncertainty. ``page``,
    ``page_key``: the page it is of ("SQD 2", "sqd~2"; the section's own by default)."""
    page = page or PAGE_NAMES["sqd"]
    from .spatial_map_view import map_lag
    from .workspace_view import primary_model

    names = [str(c) for c in plot.conditions]
    errors = dict(getattr(plot, "errors", None) or {})
    columns: dict = {}
    for i, name in enumerate(names):
        meta = (getattr(plot, "fits", None) or [])
        if meta:
            condition, model = str(meta[i]["condition"]), str(meta[i].get("model") or meta[i].get("n_pop"))
        else:
            condition, model = name, str(spec.model_for(name))
        columns.setdefault(condition, []).append((model, (payloads or {}).get(name), errors.get(i)))
    lag = map_lag(getattr(spec, "sqd_lag", 1))
    # A RUN WITH PER-SPOT PRECISION (6 Nov 2026): its D hold only with each step's own noise, so the fit carries its
    # calibration and the map labels its steps with it (`spatial_map_view.held_calibration`); a condition whose tracks
    # carry none (`ComparisonSpec.sqd_precision_off`) was fitted without it, and so is its map
    precision = getattr(spec, "sqd_precision", None)
    off = set(getattr(spec, "sqd_precision_off", ()) or ())
    out = {}
    for condition, cols in columns.items():
        models = [m for m, _p, _e in cols]
        model = primary_model(models)
        _m, payload, error = cols[models.index(model)]
        if payload is None or error:
            out[condition] = {"error": str(error or "no fit")}
            continue
        D = _sqd_drawn_D(payload, uncertainty)
        scope = str(payload.get("scope") or spec.d_sharing_scope)
        # the fit of an ADDED page names its key; the section's own page's fit is the dict it always was (it is in
        # every map job's cache key, which a new field would change for every map made before subpages)
        out[condition] = {"method": "sqd", "page": page, **({"page_key": page_key} if page_key != "sqd" else {}),
                          "D": D, "n_pop": len(D), "model": model,
                          "models": models, "scope": scope,
                          "factor": str(getattr(spec, "d_sharing_factor", "") or "") if scope == "group" else "",
                          "weighting": str(payload.get("weighting") or spec.fraction_weighting),
                          "lag_frames": lag,
                          **({"precision": [float(precision[0]), float(precision[1]), str(precision[2])]}
                             if precision is not None and condition not in off else {})}
    return {"method": "sqd", "page": page, "page_key": page_key, "conditions": out}


def appd_source(results: dict, joint, config: dict, page: str | None = None, page_key: str = "appd") -> dict:
    """The Apparent Diffusion section's last finished Run (`appd_fits.design_for`: the design its bubbles draw), per
    condition. ``page``, ``page_key``: the page it is of ("Apparent Diffusion 2", "appd~2"; the section's own by
    default, whose fit is the dict it always was, as `sqd_source`)."""
    page = page or PAGE_NAMES["appd"]
    from .appd_fits import design_for, models_of
    from .workspace_view import primary_model

    out = {}
    for condition, result in (results or {}).items():
        condition = str(condition)
        if not result or "error" in result:
            out[condition] = {"error": str((result or {}).get("error") or "no fit")}
            continue
        models = [str(m) for m in models_of(config, condition)]
        model = primary_model(models)
        design = design_for(results, joint, condition, model)
        if design is None or "failed" in design:
            out[condition] = {"error": str((design or {}).get("failed") or "no fit")}
            continue
        # PER SPOT (6 Nov 2026): the fit carries its calibration, the map's tracks their noise offsets with it; a
        # condition whose tracks carry none was fitted without it, and so is its map
        own = (result.get("settings") or {})
        per_spot = (config.get("precision") == "per_spot" and config.get("calibration")
                    and not own.get("precision_off"))
        scope = str(design.get("scope") or config.get("scope"))
        out[condition] = {"method": "appd", "page": page, **({"page_key": page_key} if page_key != "appd" else {}),
                          "D": _floats(design["D"]),
                          "n_pop": int(design["n_pop"]), "model": model, "models": models, "scope": scope,
                          "factor": str(config.get("factor") or "") if scope == "group" else "",
                          "min_steps": int(config.get("min_steps", 4)),
                          **({"precision": [float(v) for v in config["calibration"][:2]]
                              + [str(config["calibration"][2])]} if per_spot else {})}
    return {"method": "appd", "page": page, "page_key": page_key, "conditions": out}


def hmm_source(results: dict, config: dict, page: str | None = None, page_key: str = "hmm") -> dict:
    """The HMM section's last finished Run (each card's `hmm_view.compute_hmm_view` result, None where it failed), per
    condition: the condition's fit, its D and state count, its Force same D across and its sigma. ``page``,
    ``page_key``: the page it is of ("HMM 2", "hmm~2"; the section's own by default, whose fit is the dict it always
    was, as `sqd_source`)."""
    page = page or PAGE_NAMES["hmm"]
    out = {}
    for condition, result in (results or {}).items():
        condition = str(condition)
        if not result or "error" in result or "fit" not in result:
            out[condition] = {"error": str((result or {}).get("error") or "the HMM fit failed")}
            continue
        fit, settings = result["fit"], result.get("settings") or {}
        sigma, precision = str(settings.get("sigma", "off")), None
        if sigma == "per_spot":
            # PER SPOT (6 Nov 2026): the fit carries its calibration, the map's fit and decoding each step's noise
            # with it; a condition whose tracks carry none was fitted without the model, and so is its map
            if settings.get("precision_off"):
                sigma = "off"
            elif settings.get("precision"):
                precision = [float(v) for v in settings["precision"][:2]] + [str(settings["precision"][2])]
            else:
                out[condition] = {"error": "its Per spot fit recorded no calibration: run the HMM again"}
                continue
        scope = str(settings.get("shared_d") or config.get("scope") or "condition")
        out[condition] = {"method": "hmm", "page": page, **({"page_key": page_key} if page_key != "hmm" else {}),
                          "D": _floats(fit["D"]),
                          "n_pop": int(fit["n_states"]), "model": str(settings.get("states", fit["n_states"])),
                          "models": [str(settings.get("states", fit["n_states"]))], "scope": scope,
                          "factor": str(config.get("factor") or "") if scope == "group" else "",
                          "sigma": sigma, **({"precision": precision} if precision is not None else {})}
    return {"method": "hmm", "page": page, "page_key": page_key, "conditions": out}


# ---------------------------------------------------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------------------------------------------------

def refusal(method: str, fit: dict | None) -> str | None:
    """Why a condition's map cannot take the section's fit, or None when it can."""
    from .spatial_map_view import map_n_pops

    name, section = FIT_NAMES[method], PAGE_NAMES[method]
    if fit is not None and fit.get("page") and "error" not in fit:
        name = section = str(fit["page"])
    if fit is None:
        return (f"the {name} fit has no result for this condition: run {section} with it, or untick "
                f"“{box_text(method)}”")
    if "error" in fit:
        return f"the {name} fit failed for this condition ({fit['error']})"
    counts = map_n_pops(method)
    n = int(fit["n_pop"])
    if n not in counts:
        noun = "state" if method == "hmm" else "population"
        return (f"the {name} fit has {n} {noun}{'' if n == 1 else 's'} here, and a map draws "
                f"{counts[0]} to {counts[-1]}: fit {section} at another count, or untick “{box_text(method)}”")
    return None


def plan(config: dict | None, *, conditions, source: dict | None, running: bool) -> dict:
    """What a map Run does about D: ``{"action": "own"}`` (the map fits its own: unticked, a method that fits nothing,
    a split by hand); ``"wait"`` (the section's Run is in progress); ``"start"`` (the section has not run: start it
    with its current settings, then wait); or ``"draw"`` with ``"fits"`` (condition -> fit) and ``"refused"``
    (condition -> why). ``source`` is the section's last finished Run (`sqd_source` ...), None when it has none."""
    config = config or {}
    if not uses_section(config):
        return {"action": "own"}
    method = config["method"]
    page = (source or {}).get("page") if source else config.get("fit_page_name")
    if running:
        return {"action": "wait", "method": method, "text": waiting_text(method, page)}
    if source is None:
        return {"action": "start", "method": method, "text": starting_text(method, page)}
    fits, refused = {}, {}
    for condition in conditions:
        fit = (source.get("conditions") or {}).get(str(condition))
        why = refusal(method, fit)
        if why:
            refused[str(condition)] = why
        else:
            fits[str(condition)] = dict(fit)
    return {"action": "draw", "method": method, "fits": fits, "refused": refused}


def job_config(config: dict, condition: str, fit: dict | None = None, chosen=None) -> dict:
    """The config ONE condition's map job is handed (`spatial_map_view.condition_config`, which drops the box), with
    the section's ``fit`` where the map takes it: its count, and for SQD its time lag (D is the apparent D at the lag
    it was fitted at), and the fit itself under `FIT_KEY`. ``chosen``: a redraw's count (`condition_config`)."""
    from .spatial_map_view import condition_config, map_lag

    out = condition_config(config, condition, chosen=chosen)
    if fit is None:
        return out
    out["n_pop"] = int(fit["n_pop"])
    if out.get("method") == "sqd":
        out["lag_frames"] = map_lag(fit.get("lag_frames", 1))
    out[FIT_KEY] = json.loads(json.dumps(fit))            # plain, hashable for the cache key, never shared
    return out


def same_fit(a: dict | None, b: dict | None) -> bool:
    """Whether two fits are the same fit (every value, D to the last digit)."""
    if a is None or b is None:
        return a is b
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def stale_conditions(payloads: dict, source: dict | None) -> list:
    """The drawn conditions whose map took a fit of ``source``'s method that the section's last Run no longer has:
    run again with other settings (a different D, count, scope ...), or no longer holding the condition."""
    if not source:
        return []
    out = []
    for condition, payload in (payloads or {}).items():
        fit = (payload or {}).get(FIT_KEY)
        if (not fit or fit.get("method") != source.get("method")
                or fit.get("page_key", fit.get("method")) != source.get("page_key", source.get("method"))):
            continue
        now = (source.get("conditions") or {}).get(str(condition))
        if now is None or "error" in now or not same_fit(now, fit):
            out.append(str(condition))
    return out


def stale_text(method: str, conditions, labels: dict | None = None, page: str | None = None) -> str:
    """The map's line once the section it took D from has been run again with other settings: it is never redrawn
    in silence."""
    labels = labels or {}
    names = ", ".join(str(labels.get(c, c)) for c in conditions)
    name, section = fit_name(method, page), page or PAGE_NAMES[method]
    return (f"Made with an earlier {name} fit: {section} was run again with other settings"
            f"{f' ({names})' if names else ''}. Run the maps again to take its D.")


# ---------------------------------------------------------------------------------------------------------------------
# What the map says it used
# ---------------------------------------------------------------------------------------------------------------------

def _count_words(fit: dict) -> str:
    n = int(fit["n_pop"])
    noun = "state" if fit.get("method") == "hmm" else "population"
    return f"{n} {noun}{'' if n == 1 else 's'}"


def fit_source_text(fit: dict) -> str:
    """Which D a map used, as its caption and receipt say it: "D from the SQD fit: Force same D across Conditions,
    3 populations", then (after " · ") what else it was fitted at."""
    from .workspace_view import SHARED_D_NAME, shared_d_pill

    method = fit["method"]
    head = (f"D from the {fit_name(method, fit.get('page') if fit.get('page_key', method) != method else None)} fit: "
            f"{SHARED_D_NAME} {shared_d_pill(fit['scope'], fit.get('factor', ''))}"
            f", {_count_words(fit)}")
    models = list(fit.get("models") or ())
    if len(models) > 1:
        def word(m):
            return "Best (BIC)" if m == "best" else f"{m}-pop"
        head += f" (its {word(fit['model'])} fit, of {', '.join(word(m) for m in models)} side by side)"
    details = []
    if method == "sqd":
        lag = fit.get("lag_frames", 1)
        if str(lag) != "1":
            details.append(f"time lag {lag}" + ("" if str(lag) == "1-4" else " frames"))
        details.append("fractions at those D, " + ("each track once" if fit.get("weighting") == "track"
                                                    else "every jump once"))
    elif method == "appd":
        details.append(f"fractions at those D, one value per track of {int(fit.get('min_steps', 4))} steps or more")
    else:
        details.append("transitions at those D" + ("" if fit.get("sigma", "off") in ("off", "per_spot")
                                                   else ", the tracker's sigma corrected"))
    if fit.get("precision"):
        details.append(f"each {'track' if method == 'appd' else 'step'}'s own localisation error (per spot, "
                       f"{fit['precision'][2]})")
    return " · ".join([head, *details])


def fit_short_text(fit: dict) -> str:
    """The receipt's short value: "FROM THE SQD FIT"."""
    return f"FROM THE {fit_name(fit['method'], fit.get('page') if fit.get('page_key', fit['method']) != fit['method'] else None).upper()} FIT"


def fit_row_text(payloads: list, unit: str = "condition") -> str | None:
    """The caption's Fit row where the maps took a section's D, else None (the map's own fit, said by its caller):
    one sentence where every condition's is the same, else one per condition."""
    fits = [(p.get("condition"), p.get(FIT_KEY)) for p in payloads or ()]
    if not any(f for _c, f in fits):
        return None
    texts = [(c, fit_source_text(f) if f else "the map's own fit") for c, f in fits]
    if unit == "day" or len({t for _c, t in texts}) == 1:
        text = texts[0][1]
    else:
        text = "; ".join(f"{c}: {t}" for c, t in texts)
    return text + (" · shared by every replicate" if unit == "day" else "")
