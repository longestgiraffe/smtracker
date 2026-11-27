"""The workspace window's own words and settings rules, with no Qt anywhere in it.

`ui/workspace/window.py` is the Qt shell; this holds what it prints that is decided from the project rather than
drawn, and the rules that turn the Compute dialog's staged values into a `comparison.ComparisonSpec`, so a wrong
number or a wrong mapping fails in a headless test (docs/gui_tool_views.md, `tests/test_ui_thin_qt.py`).

`spec_from_values` is the ONE place the dialog's values (`ComputePanel.values()` / `.condition_values()`) become a
`ComparisonSpec`. It takes them as plain dicts, so it is tested without building a dialog; `window._spec_from_dialog`
only reads the two dicts off the panel and hands them over. `ECDF_FAST_POINTS` and `PER_CONDITION` are defined here
once, and `ui/workspace/compute.py` imports them.
"""
from __future__ import annotations

__all__ = [
    "compute_toast",
    "factor_order",
    "ARRANGEMENT_NAMES",
    "CHUNKS_DEFAULT",
    "GROUP_PREFIX",
    "RunSettingsError",
    "SCOPE_SAMPLE_WORDS",
    "SHARED_D_NAME",
    "SHARED_D_ORDER",
    "SHARED_D_PILLS",
    "SHARED_D_WORDS",
    "SET_WIDE_HELP",
    "SharingGroupError",
    "every_models",
    "is_set_wide",
    "primary_model",
    "set_wide_models",
    "set_wide_note",
    "group_label",
    "group_tip",
    "resolve_shared_d",
    "scope_words",
    "setting_words",
    "shared_d_pill",
    "toggle_every_model",
    "toggle_model",
    "CHUNKS_RANGE",
    "ECDF_FAST_POINTS",
    "PER_CONDITION",
    "ROW_SHARED_D",
    "SCOPE_ARRANGEMENT",
    "shared_d_rows",
    "ticked_models",
    "OverlayCountError",
    "overlay_count",
    "spec_from_values",
    "workspace_subtitle",
]

#: The ECDF point budget "Fast" uses -- the measured knee (CLAUDE.md, "THIN THE
#: CURVE"): 2000 points is 13.8x faster than every point on the reference project and moves the
#: slow fraction by 0.0008 pp, while below ~1000 the time stops falling and the error
#: grows. Also the value a Custom field opens on.
ECDF_FAST_POINTS = 2000

#: Uncertainty "Chunks": how many random chunks each condition is dealt into (asked 25 Oct 2026: "for plotting in
#: chunks, maybe we can choose the chunk size somewhere?"), the Compute dialog's field under the settings. The
#: default is `comparison.PSEUDO_CHUNKS` and the range what `ComparisonSpec` accepts; spelled here so the dialog reads
#: them without importing the fit, and `tests/test_pseudo_replicates.py` holds them equal.
CHUNKS_DEFAULT = 5
CHUNKS_RANGE = (2, 20)

#: The Compute dialog's spelling of "no lock, each row decides" (`compute.DEFAULT_LOCK`).
#: `ComparisonSpec.lock_model` spells the same thing `None`.
PER_CONDITION = "per_condition"

#: Shared D -> the single fit arrangement it corresponds to, used only to name the
#: arrangement when the overlay is off (one arrangement draws the classic figure,
#: which reads Shared D directly -- so this never decides what is fitted).
SCOPE_ARRANGEMENT = {"condition": "independent", "replicate": "independent",
                     "global": "simultaneous", "group": "simultaneous"}

#: A SHARED D PER CONDITION (asked 1 Nov 2026): the two values a row of the Compute dialog's Fits page can take on its
#: own, Replicates ("condition") and Don't force anywhere ("replicate") -- `comparison.CONDITION_SCOPES`, spelled here
#: so the dialog reads them without importing the fit (`tests/test_shared_d_per_condition.py` holds them equal). The
#: General page's Conditions and Whole data set force one D across conditions, a property of the whole set: under them
#: a row has none. (The labels were Within Condition, All Conditions and Within <factor> until 1 Nov 2026, then Across
#: replicates, Across all conditions and Across <factor> until the evening; the values never changed.)
ROW_SHARED_D = ("condition", "replicate")

#: SHARED D WITHIN A GROUP (asked 25 Oct 2026: "shared D should also have within same group ... color choosable --
#: per group, (per protein, per medium, etc"): the Compute dialog offers a "Conditions" pill for a folder design's
#: factor (one per factor where there are several, `group_label`), whose value is this prefix and the factor's name;
#: `spec_from_values` turns it into the scope "group" with each condition's value of that factor as its group.
GROUP_PREFIX = "group:"


# -- THE NAMES OF "FORCE SAME D ACROSS" ------------------------------------------------------------------------------ #
# Ruled by the experimenter on 1 Nov 2026: "yes we need a good naming convention. Force same D across / Replicates
# (default) | Conditions | Whole data set | Don't force anywhere". THE ONE PLACE these words are spelled: the Compute
# dialog's pills, its Help popovers and column title, the classic browser's combo, and every caption, legend, source
# sheet, receipt, refusal and error message read them from here (`bubble_view.scope_words` is `scope_words` below).
# The values ("condition", "group:<factor>", "global", "replicate") and every spec field are unchanged.

#: The setting's name: the General page's control, the Fits page's column title, the receipt's row
SHARED_D_NAME = "Force same D across"

#: Each value's pill, but the folder design's "Conditions" (`group_label`)
SHARED_D_PILLS = {"condition": "Replicates", "global": "Whole data set", "replicate": "Don't force anywhere"}

#: The pills' order, the experimenter's: Replicates (the default) · Conditions · Whole data set · Don't force
#: anywhere -- "group" standing for a folder design's Conditions pill (one per factor), offered only with a design
SHARED_D_ORDER = ("condition", "group", "global", "replicate")

#: Each value in words, for a sentence: the short form every figure, table, receipt and refusal says
SHARED_D_WORDS = {"condition": "same D across replicates", "global": "same D across the whole data set",
                  "replicate": "D not forced: each replicate its own"}

#: The end of a column's sample line ("3 replicates · 1,271 tracks · D free per replicate"), said only where a
#: figure's conditions were fitted differently (`bubble_view.mixed_scopes`); a single condition is one of these two
SCOPE_SAMPLE_WORDS = {"condition": "same D across replicates", "replicate": "D free per replicate"}

#: The overlay's two arrangements (`ComparisonSpec.bubble_fit_modes`), as its series are named: "Independent" and
#: "Simultaneous" until 1 Nov 2026, which clashed once "Replicates" named the simultaneous fit of a condition's
#: replicates. "independent" fits each condition on its own (its replicates at its one D), "simultaneous" every
#: condition together.
ARRANGEMENT_NAMES = {"independent": "Same D across replicates", "simultaneous": "Same D across the whole data set"}


def group_label(factor: str, n_factors: int = 1) -> str:
    """The pill for one D per value of ``factor``: "Conditions" where the folder design has one factor, "Conditions,
    per <factor>" where it has ``n_factors`` two or more (one pill each). Its Help line (`group_tip`) names the factor
    either way. Not offered without a design: it would then be the same fit as Whole data set."""
    return "Conditions" if n_factors <= 1 else f"Conditions, per {factor}"


def group_tip(factor: str) -> str:
    """The Help popover's line for a Conditions pill. The ratio is the warning's own (`bubble_view.GROUP_D_RATIO`),
    spelled as every ratio is (`chart_text.fold_words`)."""
    from .bubble_view import GROUP_D_RATIO
    from .chart_text import fold_words

    return (f"One D per {factor}: conditions sharing a value of {factor} are fitted together with one D and one "
            f"state count; each condition's replicates keep their own fractions. Use it when the {factor} is the "
            f"molecule (one protein under several media), not across proteins. A condition whose own D "
            f"and its group's are {fold_words(GROUP_D_RATIO)} or more apart is named under the figure.")


def factor_order(factors: dict, names=()) -> list:
    """The folder design's factors in the DESIGN's order (``names``, `Project.factor_names`, outer level first), those
    that some condition carries; without ``names``, in the order the conditions' own factor dicts first name them. The
    order a Run dialog offers its "Conditions, per <factor>" pills in, as SQD's (found by the overnight tester, 4 Nov
    2026: the apparent-D and HMM dialogs sorted them alphabetically)."""
    present = list(dict.fromkeys(f for row in (factors or {}).values() for f in (row or {})))
    ordered = [f for f in (names or ()) if f in present]
    return ordered + [f for f in present if f not in ordered]


def shared_d_pill(value: str, factor: str = "") -> str:
    """The pill a Force same D across value is offered as, for a sentence that names one ("pick Replicates"): a
    ``"group:<factor>"`` value or the scope "group" with its ``factor`` reads "Conditions, per <factor>" (a sentence
    has no pill beside it to name the factor)."""
    value = str(value)
    if value.startswith(GROUP_PREFIX):
        value, factor = "group", value[len(GROUP_PREFIX):]
    if value == "group":
        return f"Conditions, per {factor}" if factor else "Conditions"
    return SHARED_D_PILLS.get(value, value)


def scope_words(scope: str, factor: str = "") -> str:
    """A Force same D across value in words (`SHARED_D_WORDS`), for every caption, table, receipt and refusal that
    names it: "same D across replicates", "same D across conditions, per protein", "same D across the whole data
    set", "D not forced: each replicate its own". A ``"group:<factor>"`` value reads as the scope "group"."""
    scope = str(scope)
    if scope.startswith(GROUP_PREFIX):
        scope, factor = "group", scope[len(GROUP_PREFIX):]
    if scope == "group":
        return f"same D across conditions, per {factor or 'group'}"
    return SHARED_D_WORDS.get(scope, scope)


def setting_words(*values, factor: str = "") -> str:
    """The setting as a sentence names it, for what to pick: "Force same D across: Replicates" or, given several,
    "Force same D across: Replicates or Don't force anywhere"."""
    pills = [shared_d_pill(v, factor) for v in values]
    listed = pills[0] if len(pills) == 1 else ", ".join(pills[:-1]) + " or " + pills[-1]
    return f"{SHARED_D_NAME}: {listed}"


# -- THE MODELS, CLICKED (1 Nov 2026) ------------------------------------------------------------------------------- #
def toggle_model(models, model, order) -> list:
    """A row's models after a click on ``model``'s pill, in ``order`` (`MODEL_CHOICES`' values): on if it was off, off
    if it was on -- except the row's LAST model, which stays lit (a condition is fitted at one model at least; its
    include box leaves it out). The rule the Jump distances page's Show follows (`jumps_view.toggle_show`); the
    Compute dialog's `Toggles` trays are controlled and light what this returns."""
    on = {str(m) for m in models}
    model = str(model)
    if model in on:
        if len(on) > 1:
            on.discard(model)
    else:
        on.add(model)
    return [m for m in order if m in on]


def every_models(rows) -> list:
    """The models the Every condition tray lights: those every included row has (``rows``, each a list of models, in
    order), none while no row is included."""
    rows = [list(r) for r in rows]
    if not rows:
        return []
    common = set(rows[0])
    for r in rows[1:]:
        common &= set(r)
    return [m for m in rows[0] if m in common]


def toggle_every_model(rows, model, order) -> list:
    """Every included row's models after a click on ``model``'s pill on the Every condition line (``rows``, each a
    list of models; returned in the same order, each in ``order``): a LIT pill -- every row has it -- takes it off
    every row, but never empties one (a row keeps its last model); an unlit pill adds it to every row."""
    rows = [list(r) for r in rows]
    model = str(model)
    if rows and model in every_models(rows):
        return [[m for m in order if m in set(r) - {model}] if len(r) > 1 else r for r in rows]
    return [[m for m in order if m in set(r) | {model}] for r in rows]


def primary_model(models):
    """THE ONE MODEL of a condition fitted at several side by side that a single reader takes (the apparent-D
    section's Per cell and Checks pages, `appd_fits.primary_count`; a Spatial Map that takes a section's D,
    `map_fit_source`): Best (BIC) where it is lit, else the smallest count of two or more (a slow share needs two
    populations), else the smallest count, else the first model (saSPT). None for no model."""
    models = [str(m) for m in models or ()]
    if not models:
        return None
    if "best" in models:
        return "best"
    counts = sorted(int(m) for m in models if m.isdigit())
    two = [k for k in counts if k >= 2]
    if two or counts:
        return str(two[0] if two else counts[0])
    return models[0]


# -- ONE D ACROSS CONDITIONS, SET ON THE FITS PAGE (1 Nov 2026, the night) --------------------------------------------- #
# Asked, answering the offer to put the set-wide choice on the Fits page: "true. but then there should be a button on
# top. clicking it, disables everything else in the fit screen." WHAT THE FIT READS, checked in `comparison` before
# this was built: under Whole data set ("global") the state count is chosen ONCE for every condition --
# `comparison._global_joint_D`: the lock if it is a number, else Best (BIC) on the pooled displacements of every
# condition in the Run (saSPT included: a lock "saspt" is counted by BIC there too) -- and a row's own model was never
# read for it: every row at 2 with nothing locked ran BIC on the pooled set. Under Conditions ("group") each group's
# count is `comparison._group_count`: the lock, else the number every member's row names, else Best on that group's
# pooled data. With fits side by side every model is its own joint fit (`bubble_view.at_model` locks it). So under a
# set-wide value the conditions share one fit and its models: the Fits page's Every condition tray is the one control
# that sets them, the rows show them disabled, and `spec_from_values` locks the one number they name.

def is_set_wide(shared_d) -> bool:
    """Whether a Force same D across value forces one D across CONDITIONS -- Whole data set ("global"), Conditions
    (``"group:<factor>"`` or the scope "group") -- rather than within each condition (`ROW_SHARED_D`)."""
    return str(shared_d) not in ROW_SHARED_D


def set_wide_note(shared_d) -> str:
    """The line under the Fits page's Force same D across while it forces one D across conditions ("" otherwise):
    why every row's controls are disabled, in what the fit does (see above)."""
    shared_d = str(shared_d)
    if not is_set_wide(shared_d):
        return ""
    if shared_d.startswith(GROUP_PREFIX) or shared_d == "group":
        factor = shared_d[len(GROUP_PREFIX):] if shared_d.startswith(GROUP_PREFIX) else ""
        each = f"each {factor}'s conditions" if factor else "the conditions of a group"
        return f"One D per {factor or 'group'}: {each} share one fit, so they share its models."
    return "One D for every condition: the conditions share one fit, so they share its models."


#: The Help popover's line for the Fits page's Force same D across while it is set-wide: what the Every condition
#: tray then sets, Best's count included (`set_wide_note`'s longer form)
SET_WIDE_HELP = (f"{shared_d_pill('group')} and {SHARED_D_PILLS['global']} fit conditions together, so each "
                 f"row's models and its own {SHARED_D_NAME} are disabled and show the set's; pick the models on "
                 f"the Every condition line. Best (BIC) chooses one count on the pooled data: for every "
                 f"condition under {SHARED_D_PILLS['global']}, for each group under {shared_d_pill('group')}.")


def set_wide_models(rows, default) -> list:
    """The models a set-wide Force same D across starts from, when it is picked: those every included row lights
    (``rows``, each a list of models), else ``default`` alone -- Best (BIC), what the joint fit chose before a row's
    model reached it."""
    return every_models(rows) or [str(default)]


def _model_word(model) -> str:
    """A model as an error message names it."""
    return {"best": "Best (BIC)", "saspt": "saSPT"}.get(str(model), str(model))


def _set_wide_lock(values: dict, conditions: dict, side, lock: str) -> str:
    """THE ONE COUNT A SET-WIDE FIT IS MADE AT, from the Fits page (see the section above): with nothing side by side
    and nothing locked, the one model every included row names -- locked when it is a number, since the joint fit
    reads only the lock (`comparison._global_joint_D`); Best and saSPT leave it unlocked, the joint fit's Best. Raises
    `RunSettingsError` naming the rows that differ, or the rows that draw fewer fits side by side: under one D across
    conditions every condition takes the set's models, and the dialog stages nothing per row."""
    shown = shown_fits(values, side, conditions)
    if shown:
        raise RunSettingsError(f"{setting_words(values['shared_d'])} fits the conditions together, so every condition "
                               f"draws every model side by side; {', '.join(sorted(shown))} asked for fewer. Pick the "
                               f"models on the Fits page's Every condition line.")
    if side or lock != PER_CONDITION:
        return lock
    named = {}
    for condition, (on, model) in conditions.items():
        if on:
            named.setdefault(str(model), []).append(str(condition))
    if len(named) > 1:
        listed = "; ".join(f"{_model_word(m)} for {', '.join(cs)}" for m, cs in named.items())
        raise RunSettingsError(f"{setting_words(values['shared_d'])} fits the conditions together, with one set of "
                               f"models; the Fits page names {listed}. Pick the models on its Every condition line.")
    if len(named) == 1:
        (only,) = named
        if only.isdigit():
            return only
    return lock


def ticked_models(values: dict) -> list:
    """The models every condition is fitted at side by side (``fit_models``, in `MODEL_CHOICES` order): none = each
    condition its own row's model, one = every condition locked to it, two or more = every condition fitted with
    each, drawn side by side (asked 30 Oct 2026: "we should move all of this on the 2nd page and just tick boxes of
    what to fit and show"). Since 1 Nov 2026 the Compute dialog's rows tick their models themselves and it stages
    none (each row its own) or two or more (the union of the rows', `ComputePanel.side_by_side`); one is a staging
    from 30 Oct - 1 Nov, when "Fit every condition at" locked the rows. Staged values from before the ticks read as
    what they meant: a lock as that one model, the 28-29 Oct Overlay "Counts" as its ticked counts. Raises
    `RunSettingsError` naming a model that is not one."""
    from .comparison import MODEL_CHOICES

    raw = values.get("fit_models")
    if raw is None:
        if values.get("overlay") == "counts":
            raw = values.get("side_by_side") or (2, 3)
        else:
            lock = values.get("lock_model", PER_CONDITION)
            raw = [] if lock == PER_CONDITION else [lock]
    ticked = {str(m) for m in raw}
    bad = sorted(ticked - set(MODEL_CHOICES))
    if bad:
        raise RunSettingsError(f"{bad[0]!r} is not a model; the models are {', '.join(MODEL_CHOICES)}")
    return [m for m in MODEL_CHOICES if m in ticked]


def shown_fits(values: dict, ticked, conditions: dict) -> dict:
    """WHICH OF THE TICKED MODELS EACH CONDITION SHOWS, from its row's ticks (``shown_fits``, ``{condition: models}``;
    asked 31 Oct 2026: "real data 1 and 2 pop fit, simulation only 1 pop fit ... choose for every protein /
    condition"): only under fits side by side (``ticked`` two or more), only the conditions in the Run, each kept to
    the models ticked. A condition that shows them all, or none of those ticked now (its row's ticks were made under
    other ones), is left out: it shows every model."""
    ticked = [str(m) for m in ticked or ()]
    if len(ticked) < 2:
        return {}
    out = {}
    for condition, models in dict(values.get("shown_fits") or {}).items():
        if condition not in conditions or not conditions[condition][0]:
            continue
        mine = [m for m in ticked if m in {str(x) for x in models}]
        if mine and len(mine) < len(ticked):
            out[condition] = tuple(mine)
    return out


def shared_d_rows(values: dict, shared_d: str, conditions: dict) -> dict:
    """EACH ROW'S OWN SHARED D, for the conditions in the Run (``shared_d_rows``, ``{condition: "condition" |
    "replicate"}``, the rows fitted otherwise than the General page's ``shared_d``; asked 1 Nov 2026). A row naming the
    General page's own value is no override and is left out. Under a value that forces one D across conditions
    (Conditions, Whole data set) the dialog stages none -- the set has one D -- and one that arrives anyway is a
    contradiction, refused by name rather than dropped. Raises `RunSettingsError` for a value that is not a row's."""
    rows = dict(values.get("shared_d_rows") or {})
    bad = {str(c): v for c, v in rows.items() if v not in ROW_SHARED_D}
    if bad:
        raise RunSettingsError(f"a condition's own {SHARED_D_NAME} is {shared_d_pill(ROW_SHARED_D[0])} or "
                               f"{shared_d_pill(ROW_SHARED_D[1])}, got {bad}")
    rows = {str(c): v for c, v in rows.items() if c in conditions and conditions[c][0] and v != shared_d}
    if rows and shared_d not in ROW_SHARED_D:
        raise RunSettingsError(f"{SHARED_D_NAME} is {shared_d_pill(shared_d)} on the General page, one D across "
                               f"conditions, so no condition has its own; {', '.join(sorted(rows))} asked for one. "
                               f"Pick {shared_d_pill(ROW_SHARED_D[0])} or {shared_d_pill(ROW_SHARED_D[1])} there "
                               f"to set it per condition.")
    return rows


class RunSettingsError(ValueError):
    """The Compute dialog's settings cannot make a Run, and the message says why (shown in place of the figure)."""


class OverlayCountError(RunSettingsError):
    """The overlay was asked for, but the Fits page names no ONE state count."""


class SharingGroupError(RunSettingsError):
    """Force same D across Conditions (one D per value of a factor) was asked for, and a condition has no value of
    that factor, or an overlay is on."""


def workspace_subtitle(project, conditions: list) -> str:
    """The sidebar header's second line: how many tracks the project holds, or how many conditions it has when it
    carries no canonical table to count them in (a hand-built project, or a table without a ``uid`` column).

    Only those two absences fall back. Any other error, a table that cannot be counted, is raised rather than
    shown as a different sentence."""
    try:
        n_tracks = int(project.canonical["uid"].nunique())
    except (AttributeError, KeyError):
        return f"{len(conditions)} condition(s)"
    cap = (project.canonical.attrs.get("track_cap") or {}).get("max_steps")
    if cap is not None:                     # a user's cap is said wherever the count is
        return f"{n_tracks:,} tracks · max {cap} steps"
    return f"{n_tracks:,} tracks"


def overlay_count(lock: str, conditions: dict) -> int:
    """The ONE state count both overlaid arrangements are drawn at, from the
    Fits page (the Conditions tab until 1 Nov 2026) -- never from a control of its own.

    An `Overlay States` pill once supplied it, and silently beat the per-condition
    model choice whenever the overlay was on. Two controls for one number, with the
    one further from the Conditions tab winning. Now the count comes from where
    counts are chosen, and when that place names more than one -- or names a rule
    (Best, saSPT) rather than a number -- the overlay is REFUSED with the reason,
    since two arrangements drawn at different counts would put fractions of
    different quantities side by side.
    """
    if lock not in (PER_CONDITION, None):
        if str(lock).isdigit():
            return int(lock)
        raise OverlayCountError(
            f"Overlay needs one fixed state count, and every condition is fitted at {lock!r}, which "
            f"picks a count per condition. Pick one number (1-4) on every row or turn Overlay off.")
    chosen = {m for _c, (on, m) in conditions.items() if on}
    if len(chosen) == 1:
        (only,) = chosen
        if str(only).isdigit():
            return int(only)
    raise OverlayCountError(
        "Overlay needs every included condition at the same fixed state count "
        f"(1-4); the Fits page has {sorted(map(str, chosen))}. Pick one "
        "number on every row (the Every condition line does it at once), or turn Overlay off.")


def _sqd_lag(value) -> "int | str":
    """The Compute dialog's Time lag as `ComparisonSpec.sqd_lag`: a number of frames, or "1-4" (lags 1-4 together)."""
    from .comparison import JOINT_LAG

    return JOINT_LAG if str(value) == JOINT_LAG else int(value)


def resolve_shared_d(shared_d: str, chosen, factors: dict | None) -> tuple:
    """``(scope, groups, factor)`` for a Shared D value: a `ComparisonSpec` scope as it is, or ``"group:<factor>"``
    -> ``("group", {condition: its value of the factor}, factor)``. ONE reading for every window that offers it (the
    Compute dialog, the classic browser). Raises `SharingGroupError` naming a chosen condition without the factor."""
    shared_d = str(shared_d)
    if not shared_d.startswith(GROUP_PREFIX):
        return shared_d, None, ""
    factor = shared_d[len(GROUP_PREFIX):]
    missing = [str(c) for c in chosen if factor not in ((factors or {}).get(c) or {})]
    if missing:
        raise SharingGroupError(f"{setting_words(shared_d)}: no {factor} is known for {', '.join(missing)}. "
                                f"Pick another {SHARED_D_NAME}, or give the folders a design that names it.")
    return "group", {c: str(factors[c][factor]) for c in chosen}, factor


#: The Compute dialog's Localisation error (6 Nov 2026, per-spot precision): its values.
PRECISION_OFF, PRECISION_PER_SPOT = "off", "per_spot"


def run_precision(project, chosen) -> tuple:
    """``(c_nm, p, source)`` of the calibration a Run with per-spot precision uses for the ``chosen`` conditions: the
    one CHOSEN for the project (`calibrations.resolve`, 6 Nov 2026: built-in, the project's file, or one from the
    library; never chosen, as before -- the project's file, else the built-in one of their tracker), or
    `RunSettingsError` saying why there is none."""
    from .calibrations import resolve
    from .precision import PrecisionError

    trackers = (getattr(project, "canonical", None) is not None
                and project.canonical.attrs.get("trackers")) or {}
    # ONLY THE CONDITIONS WHOSE TRACKS CARRY PRECISION choose it (6 Nov 2026): a track table -- a truth condition --
    # has none and is fitted without a localisation-error model (`precision_off`), never a reason to refuse the Run
    carrying = [c for c in chosen if c not in set(precision_off(project, chosen))]
    if not carrying:
        raise RunSettingsError("Localisation error Per spot: no condition of this Run carries the tracker's per-spot "
                               "precision (u-track's tracks do; TrackMate's when its spots were fitted; track tables "
                               "do not)")
    try:
        cal = resolve(project, carrying, {trackers.get(c, "utrack") for c in carrying})
    except PrecisionError as exc:
        raise RunSettingsError(f"Localisation error Per spot: {exc}") from exc
    # MEASURED ELSEWHERE IS SAID (10 Nov 2026, the per-spot audit's finding 5): the curve is not free of scale, so a
    # Run on another pixel size, frame interval or PSF width says so wherever the calibration is named
    from .precision import acquisition_mismatch

    note = acquisition_mismatch(cal, getattr(project, "params", None))
    return (cal.c_nm, cal.p, f"{cal.source} ({note})" if note else cal.source)


def precision_off(project, chosen) -> tuple:
    """The ``chosen`` conditions whose tracks carry NO per-spot precision (`precision.carries_precision`): a truth
    condition (exact positions), a track table. A Per spot Run fits them with no localisation-error model -- exact for
    a truth condition, which has none -- and says so by name (`bubble_view.precision_of`)."""
    from .precision import conditions_with_precision

    canonical = getattr(project, "canonical", None)
    carrying = conditions_with_precision(canonical) if canonical is not None else set()
    return tuple(str(c) for c in chosen if str(c) not in carrying)


def spec_from_values(values: dict, conditions: dict, factors: dict | None = None, precision=None,
                     precision_off=()):
    """The Compute dialog's staged values, as a real `ComparisonSpec`.

    ``values`` is `ComputePanel.values()` and ``conditions`` is `ComputePanel.condition_values()`
    (``{key: (included, model)}``), both plain dicts.

    Mirrors `browser.ComparisonBrowser.spec()` field-by-field -- the SAME
    value vocabulary (`shared_d` values ARE `D_SHARING_SCOPES`, `weighting`
    values ARE `fraction_weighting`'s) means this is a direct carry-over,
    never a re-derivation that could drift from the browser's own mapping.

    Raises `OverlayCountError` when the overlay is on and the Fits page does
    not name one state count -- see `overlay_count`.

    Each row's own Shared D (``values["shared_d_rows"]``, `shared_d_rows`) becomes
    `ComparisonSpec.d_sharing_by_condition`; conditions fitted differently are refused
    with the overlay on, since it draws its own two arrangements for every condition.

    Localisation error Per spot (``values["precision"]``) needs ``precision``, the Run's calibration
    (`run_precision`); refused by name without it, and the spec refuses the settings it is not built for.
    """
    from .comparison import ComparisonError, ComparisonSpec

    per_spot = values.get("precision", PRECISION_OFF) == PRECISION_PER_SPOT
    if per_spot and precision is None:
        raise RunSettingsError("Localisation error Per spot needs a calibration, and none was found for this Run.")
    off = tuple(c for c in (precision_off or ()) if (conditions.get(c) or (False,))[0]) if per_spot else ()
    if off and values.get("overlay") == "both":
        raise RunSettingsError(f"Localisation error Per spot: Overlay Both fits one D across every condition, and "
                               f"{', '.join(off)} carry no per-spot precision; switch Overlay off.")

    # THE MODELS TICKED decide both: one is a lock, two or more are fits side by side, none leaves each row its own
    ticked = ticked_models(values)
    lock = ticked[0] if len(ticked) == 1 else PER_CONDITION
    side = tuple(ticked) if len(ticked) >= 2 else ()
    # SHARED D WITHIN A GROUP: "group:<factor>" -> the scope "group", each condition's value of the factor its group
    shared_d, groups, factor = resolve_shared_d(values["shared_d"],
                                                [c for c, (on, _m) in conditions.items() if on], factors)
    overlay = " with ".join(ARRANGEMENT_NAMES[m] for m in ("independent", "simultaneous"))
    if shared_d == "group" and values["overlay"] == "both":
        raise SharingGroupError(f"{setting_words(values['shared_d'])} draws one fit: the overlay compares "
                                f"{overlay}. Switch Overlay off.")
    # A SHARED D PER CONDITION: the rows fitted otherwise than the General page says (`shared_d_rows`)
    rows = shared_d_rows(values, shared_d, conditions)
    if is_set_wide(shared_d):
        # ONE D ACROSS CONDITIONS: one set of models for every condition, the one number named locked (the joint fit
        # reads only the lock); rows that differ, or draw fewer fits side by side, are refused by name
        lock = _set_wide_lock(values, conditions, side, lock)
    included = [c for c, (on, _m) in conditions.items() if on]
    if values["overlay"] == "both" and len({rows.get(c, shared_d) for c in included}) > 1:
        # the overlay makes its own two arrangements for every condition at once, so conditions fitted differently
        # would not be what is drawn: named, never ignored (every row alike is the uniform Run, which it accepts)
        raise RunSettingsError(f"Overlay Both compares {overlay} for every condition; "
                               f"{', '.join(sorted(rows))} has its own {SHARED_D_NAME} on the Fits page. Switch "
                               f"Overlay off, or give every row the General page's.")
    # FULL IS SENT AS 0, NOT None. `comparison._ecdf_budget` reads `None` as "use the
    # measured default budget" and `0` as exact, so the old `None` made Full run the
    # very same 2000-point fit as Fast.
    ecdf_points = {"fast": ECDF_FAST_POINTS, "full": 0,
                   "custom": int(values["ecdf_points"])}[values["ecdf"]]
    compare = values["overlay"] == "both"
    if compare and side:
        raise OverlayCountError("Overlay Both compares two arrangements at one model: pick one model on every row "
                                "(Fits page), or switch Overlay off to see the picked models side by side.")
    try:
        return _spec(values, conditions, lock, shared_d, groups, factor, rows, ecdf_points, compare, side,
                     precision if per_spot else None, off)
    except ComparisonError as exc:
        if per_spot:                                     # what per-spot is not built for, said in the dialog's words
            raise RunSettingsError(f"Localisation error Per spot: {exc}") from exc
        raise


def _spec(values, conditions, lock, shared_d, groups, factor, rows, ecdf_points, compare, side, precision, off=()):
    from .comparison import ComparisonSpec

    return ComparisonSpec(
        sqd_precision=precision,
        sqd_precision_off=tuple(off),
        conditions=[c for c, (on, _m) in conditions.items() if on],
        models={c: m for c, (_on, m) in conditions.items()},
        # `PER_CONDITION` is the dialog's spelling of "no lock, each row decides" --
        # `ComparisonSpec.lock_model` spells that `None`. Never pass the raw sentinel
        # through as if it were a model.
        lock_model=None if lock == PER_CONDITION else lock,
        fraction_weighting=values["weighting"],
        d_sharing_scope=shared_d,
        d_sharing_groups=groups,
        d_sharing_factor=factor,
        # each row's own Shared D; the spec puts it in its one form (every row alike is the uniform Run)
        d_sharing_by_condition=rows or None,
        ecdf_subsample=ecdf_points,
        best_max_pop=int(values.get("best_max_pop", 3)),
        # (`SCOPE_ARRANGEMENT` names "condition" and "replicate" alike, so a row's own Shared D never moves this)
        bubble_fit_modes=(("independent", "simultaneous") if compare
                          else (SCOPE_ARRANGEMENT[shared_d],)),
        # One arrangement at one count draws the classic figure, which takes each
        # condition's count from the Fits page; the count below is only read
        # when two arrangements are overlaid.
        bubble_n_pops=((overlay_count(lock, conditions),) if compare else (2,)),
        side_by_side=side,
        side_by_side_shown=shown_fits(values, side, conditions) or None,
        sqd_gradient=values.get("gradient", "numeric"),
        sqd_lag=_sqd_lag(values.get("lag", "1")),
        # Uncertainty "Chunks": the replicate fits are made on random chunks, as many as the dialog's field says
        pseudo_replicates=(int(values.get("chunks", CHUNKS_DEFAULT)) if values.get("uncertainty") == "chunk"
                           else 0),
    )



# -- the toasts a section's Run posts --------------------------------------- #
#: How much of a failure's message a toast carries: its first line, trimmed. The
#: whole message is on the section's own page, which `Show` opens.
TOAST_ERROR_CHARS = 140


def compute_toast(section_title: str, outcome: str, *, seconds: float | None = None,
                  error=None, here: bool, note: str | None = None) -> dict:
    """The words of one toast, as a plain dict: ``title``, ``description``, ``action``.

    ``outcome`` is ``"done"`` (a Run finished), ``"failed"`` (it raised, or the spine
    refused it), or ``"ask"`` (Run was pressed again while this section is still
    computing). ``here`` says whether the reader is looking at that section: a finished
    or failed toast offers ``Show`` only when they are NOT, since on the section itself
    the figures changing IS the notification and a button taking you where you already
    are does nothing (the prototype's rule). The prompt's action is ``Proceed``; its
    close button means keep what is running. ``note`` is appended to a finished
    toast's description (e.g. how many conditions of a per-condition Run failed).
    """
    if outcome == "done":
        parts = [f"in {seconds:.0f} s"] if seconds is not None and seconds >= 1 else []
        if note:
            parts.append(note)
        description = " · ".join(parts) or None
        return {"title": f"{section_title} computed", "description": description,
                "action": None if here else "Show"}
    if outcome == "failed":
        first = str(error or "").strip().splitlines()
        text = first[0].strip() if first else ""
        if len(text) > TOAST_ERROR_CHARS:
            text = text[:TOAST_ERROR_CHARS - 1].rstrip() + "…"
        return {"title": f"{section_title} could not be computed",
                "description": text or None, "action": None if here else "Show"}
    if outcome == "ask":
        return {"title": f"{section_title} is still computing",
                "description": "Cancel it and start the new run?", "action": "Proceed"}
    raise ValueError(f"outcome must be 'done', 'failed' or 'ask', got {outcome!r}")
