"""THE HMM SECTION'S RUN, as SQD's -- with no Qt anywhere in it (3 Nov 2026).

Asked 3 Nov 2026: "can we do a simultanous fit over conditions for HMM as well?", then "maybe we build HMM simul fit
as well? the HMM run window is a bit outdated compared to how advanced the sqd run window is. we should step our game
up at hmm run window". Until then every condition was fitted at ONE state count, and under Uncertainty Replicates each
replicate was fitted with its OWN D -- so a rate "into the middle state" compared states that were not the same state
from one replicate (or condition) to the next. Now the Run is SQD's, in SQD's words (`workspace_view`):

* FORCE SAME D ACROSS: **Replicates** (the default, as SQD's: the condition's pooled fit gives the states' D and every
  replicate's transitions and occupancy are fitted AT those D, `hmm.fit_hmm`'s ``fixed_D``); **Don't force anywhere**
  (each replicate its own D, as before 3 Nov 2026); **Conditions** (a folder design's factor) and **Whole data set**
  -- ONE set of D for every condition of the group or the set, from the pooled tracks of all of them (`joint_D`, one
  pool job before the conditions'), each condition and each of its replicates fitted at it. SQD's design: a joint D,
  then everything else with D held. A row may set Replicates or Don't force anywhere for itself.
* STATES per condition (2, 3, 4 or Best), an Every condition line setting them all; Best looks up to 3 states, or 4
  when picked (`best_max`). Under a set-wide value the group shares one count, the rows' are greyed.
* UNCERTAINTY: Replicates, Movies, Cells (one session only), Chunks (SQD's own deal of the movies or cells into
  pseudo-replicates, `comparison.pseudo_replicate_deal`), None.
* σ correction, as before.

`run_config` checks the dialog's values into one form (a config from before reads as it meant: each replicate its own
D); `condition_settings` is what one condition's pool job is given -- and its cache key -- so another row's change does
not make it again; `groups_of` the groups a set-wide value fits together.

THE SHARED FIT RUNS AS MANY POOL JOBS, ONE PER START (4 Nov 2026). Reported: Best with Whole data set on one user's
protein F (four conditions, about 9,800 tracks) sat over ten minutes with ONE of seven workers busy, saying only
"computing... the states shared across conditions": `joint_D` fitted 1, 2 and 3 states, four EM starts each, one after
another in one job. The starts are independent (`hmm.HMMPlan`: every start's guess drawn up front, in the order the
loop drew them), so `joint_D` now makes the plans (`joint_plans`, one job), runs every start as a job of its own
(`joint_tasks`, the most states first) through the ``submit`` it is given -- the window's `_submit_bulk`, so a job
somebody waits on keeps its worker -- and keeps each count's best start (`joint_result`): EXACTLY the D, BIC rows and
choice the one job made (`tests/test_hmm_joint_parallel.py`). As each start lands it says how far it is
(`joint_progress`: the section's status line and the sidebar's Running list); a set ``cancelled`` event stops it
submitting and drops what has not started.
"""
from __future__ import annotations

from .hmm_view import SIGMA_MODES
from .workspace_view import SHARED_D_ORDER, SHARED_D_PILLS as _PILLS, setting_words as _setting_words

__all__ = ["BEST_MAX", "DEFAULTS", "HELP", "HMMRunError", "STATE_MODELS", "STATE_MODEL_LABELS", "UNCERTAINTY",
           "UNCERTAINTY_LABELS", "condition_settings", "groups_of", "joint_D", "joint_plans", "joint_progress",
           "joint_result", "joint_tasks", "run_config", "states_of", "with_precision"]

#: The state counts a condition can be fitted with, in their pills' order.
STATE_MODELS = ("2", "3", "4", "best")
STATE_MODEL_LABELS = ("2 states", "3 states", "4 states", "Best (BIC)")
#: Uncertainty, SQD's words and order.
UNCERTAINTY = ("day", "movie", "cell", "chunk", "none")
UNCERTAINTY_LABELS = ("Replicates", "Movies", "Cells", "Chunks", "None")
#: How far Best looks: 3 states (the default, as before), or 4.
BEST_MAX = (3, 4)
DEFAULTS = {"conditions": None, "states": "2", "shared_d": "condition", "uncertainty": "day", "sigma": "off",
            "best_max": 3, "chunks": 5, "rows": {}}
SET_SCOPES = ("global", "group")

#: The Run dialog's words per setting: (the line under the control, the hover text).
HELP = {
    "shared_d": ("Where the states' D are held the same.",
                 f"{_PILLS['condition']} (the default): the condition's pooled fit gives the states' D, and each "
                 "replicate's rates and occupancy are fitted with those D. "
                 f"{_PILLS['replicate']}: each replicate is fitted alone with its own D. "
                 f"Conditions and {_PILLS['global']}: one set of D for every condition of a group or of the set. "
                 "This is a stronger assumption: if every D of one condition is lower, the shared D moves that shift "
                 "into the occupancy and rates."),
    "states": ("How many diffusive states each condition's model has.",
               "Two states is the standard model. Three lets BIC test whether a third state is supported; its "
               "slowest D often sits at the localisation floor. Best fits one to three states (or four) and picks "
               "by BIC; a difference under 2 is inconclusive."),
    "uncertainty": ("What the ± is over.",
                    "Replicates: one fit per replicate, shown as mean ± SD. Movies and Cells: percentile intervals "
                    "from 100 refitted resamples, for a single replicate only. Chunks: movies (or cells) are dealt "
                    "at random into chunks and fitted as replicates; for data imaged without replicates, it "
                    "measures variation within one culture. None: the pooled fit alone."),
    "sigma": ("Correct D for localisation error, or not.",
              "Off (the default) reports apparent D, comparable with the SQD fit. The tracker's sigma subtracts the "
              "localisation offset; if the slow state disappears with it, it was localisation noise. Per spot "
              "gives every step its own localisation error, from the tracker's per-spot precision and a "
              "calibration measured on immobile spots."),
    "best_max": ("How many states Best (BIC) may choose.", "Up to 3, the default; 4 lets BIC choose four."),
}


class HMMRunError(ValueError):
    """The Run's settings cannot make a Run, and the message says why (the dialog shows it, nothing runs)."""


def _state(value, where: str) -> str:
    value = str(value)
    if value not in STATE_MODELS:
        raise HMMRunError(f"{where}: {value!r} is not a state count; pick 2, 3, 4 states or Best (BIC)")
    return value


def run_config(values: dict | None = None, *, conditions=(), factors: dict | None = None) -> dict:
    """The Run dialog's values, checked and in one form: ``{"conditions", "states", "shared_d", "scope", "groups",
    "factor", "uncertainty", "sigma", "best_max", "chunks", "rows"}``; ``rows`` maps a condition to its own
    ``states`` and (under Replicates / Don't force anywhere) its own ``shared_d``. RAISES `HMMRunError` naming what is
    wrong: no condition, a value that is not one, a row's own Force same D or count under a set-wide value that
    differs from the set's, a factor a condition lacks.

    A config from before 3 Nov 2026 (no ``shared_d``) reads as it ran: each replicate its own D ("replicate")."""
    from .workspace_view import (CHUNKS_RANGE, ROW_SHARED_D, SHARED_D_NAME, resolve_shared_d, setting_words,
                                 shared_d_pill)

    given = dict(values or {})
    v = {**DEFAULTS, **given}
    if "states" in given and "shared_d" not in given and "rows" not in given:
        v["shared_d"] = "replicate"                     # a config from before: each replicate its own D
    conditions = [str(c) for c in conditions]
    picked = v.get("conditions")
    chosen = [str(c) for c in (conditions if picked is None else picked)]
    chosen = [c for c in conditions if c in chosen] if conditions else chosen
    if not chosen:
        raise HMMRunError("Pick at least one condition.")
    states = _state(v["states"], "Every condition")
    shared_d = str(v["shared_d"])
    if not (shared_d in ("condition", "replicate", "global") or shared_d.startswith("group:")):
        raise HMMRunError(f"{_setting_words(*SHARED_D_ORDER)}, not {shared_d!r}")
    if str(v["uncertainty"]) not in UNCERTAINTY:
        raise HMMRunError(f"uncertainty must be one of {UNCERTAINTY}, got {v['uncertainty']!r}")
    if str(v["sigma"]) not in SIGMA_MODES:
        raise HMMRunError(f"sigma must be one of {SIGMA_MODES}, got {v['sigma']!r}")
    if int(v["best_max"]) not in BEST_MAX:
        raise HMMRunError(f"Best looks at up to 3 or 4 states, not {v['best_max']!r}")
    lo, hi = CHUNKS_RANGE
    if not lo <= int(v["chunks"]) <= hi:
        raise HMMRunError(f"Chunks deals each condition into {lo} to {hi} chunks, not {v['chunks']!r}")
    try:
        scope, groups, factor = resolve_shared_d(shared_d, chosen, factors)
    except ValueError as exc:
        raise HMMRunError(str(exc)) from exc
    rows = {}
    for condition, row in dict(v.get("rows") or {}).items():
        if str(condition) not in chosen:
            continue
        row, own = dict(row or {}), {}
        if row.get("states") not in (None, "", states):
            own["states"] = _state(row["states"], str(condition))
        if row.get("shared_d") not in (None, "", shared_d):
            if str(row["shared_d"]) not in ROW_SHARED_D:
                raise HMMRunError(f"{condition}'s own {SHARED_D_NAME} is {shared_d_pill(ROW_SHARED_D[0])} or "
                                  f"{shared_d_pill(ROW_SHARED_D[1])}, not {row['shared_d']!r}")
            own["shared_d"] = str(row["shared_d"])
        if own:
            rows[str(condition)] = own
    if scope in SET_SCOPES:
        bad = sorted(c for c, row in rows.items() if "shared_d" in row)
        if bad:
            raise HMMRunError(f"{setting_words(shared_d)} fits the conditions together, so no condition has its own "
                              f"{SHARED_D_NAME}; {', '.join(bad)} asked for one.")
        differ = sorted(c for c, row in rows.items() if "states" in row)
        if differ:
            raise HMMRunError(f"{setting_words(shared_d)} fits the conditions together at one state count; "
                              f"{', '.join(differ)} asked for another. Pick it on the Every condition line.")
    return {"conditions": chosen, "states": states, "shared_d": shared_d, "scope": scope, "groups": groups,
            "factor": factor, "uncertainty": str(v["uncertainty"]), "sigma": str(v["sigma"]),
            "best_max": int(v["best_max"]), "chunks": int(v["chunks"]), "rows": rows}


def with_precision(config: dict, project) -> dict:
    """``config`` with the Run's per-spot calibration chosen ONCE for all its conditions (``"precision"``:
    ``(c_nm, p, source)``, `workspace_view.run_precision`, as the SQD Run chooses its own), so every condition and
    every shared fit models the noise alike and the calibration is in each condition's cache key; ``config`` as it is
    under another sigma. RAISES `HMMRunError` saying why there is none."""
    if config.get("sigma") != "per_spot":
        return config
    from .workspace_view import RunSettingsError, run_precision

    try:
        precision = run_precision(project, list(config["conditions"]))
    except RunSettingsError as exc:
        raise HMMRunError(str(exc)) from exc
    return {**config, "precision": (float(precision[0]), float(precision[1]), str(precision[2]))}


def states_of(config: dict, condition: str) -> str:
    """The state count ``condition`` is fitted at: its row's, else the Every condition line's."""
    return str(((config.get("rows") or {}).get(str(condition)) or {}).get("states") or config["states"])


def condition_settings(config: dict, condition: str, fixed_D=None) -> dict:
    """What ``condition``'s pool job is given (`hmm_view.compute_hmm_view`'s keywords, and the job's cache key): its
    count, its own Force same D (a row's, else the Run's), Uncertainty, σ, Best's range, the chunks and, under a
    set-wide value, the group's D (``fixed_D``, `joint_D`)."""
    row = (config.get("rows") or {}).get(str(condition)) or {}
    scope = config["scope"] if config["scope"] in SET_SCOPES else str(row.get("shared_d") or config["scope"])
    out = {"states": states_of(config, condition), "uncertainty": config["uncertainty"], "sigma": config["sigma"],
           "shared_d": scope, "best_max": int(config["best_max"]), "chunks": int(config["chunks"])}
    if fixed_D is not None:
        out["fixed_D"] = tuple(float(d) for d in fixed_D)
    if config.get("precision") is not None:
        out["precision"] = tuple(config["precision"])
    return out


def groups_of(config: dict) -> dict:
    """``{group: [conditions]}`` a set-wide Force same D across fits together: every condition under Whole data set,
    each value of the factor under Conditions; ``{}`` under Replicates / Don't force anywhere."""
    if config["scope"] not in SET_SCOPES:
        return {}
    if config["scope"] == "global":
        return {"": list(config["conditions"])}
    out: dict = {}
    for c in config["conditions"]:
        out.setdefault(str((config.get("groups") or {}).get(c, "")), []).append(c)
    return out


def joint_plans(project, groups: dict, *, states: str, sigma: str, best_max: int = 3, precision=None) -> dict:
    """Each group's fits BEFORE any start runs: ``{group: {"members", "best", "plans": {count: hmm.HMMPlan}}}`` -- the
    POOLED tracks of the group's conditions, a plan per state count (Best: 1 up to ``best_max``; else ``states``),
    each at `fit_hmm`'s own defaults (its measured starts, seed 0). Plain data: one pool job makes it and hands it on.
    RAISES `HMMError` when a group's tracks cannot be fitted at a count (too few steps, nothing to fit). Under sigma
    "per_spot" every step carries its own noise (``precision``: the Run's calibration, `with_precision`)."""
    from .hmm import plan_hmm
    from .hmm_view import _inputs, _sigma_um, precision_of

    tau = float(project.params.tau_s)
    out = {}
    best = str(states) == "best"
    counts = list(range(1, int(best_max) + 1)) if best else [int(states)]
    from .hmm import HMMError
    from .precision import carries_precision

    for name, members in groups.items():
        if sigma == "per_spot":
            off = [c for c in members if not carries_precision(project.condition(c))]
            if off and len(off) < len(members):
                raise HMMError(f"One set of D across conditions with and without per-spot precision cannot model "
                               f"the error in only some of them ({', '.join(off)}: their tracks carry none). "
                               f"Force D across Replicates, or leave {', '.join(off)} out.")
        calibration = (None if sigma == "per_spot" and not carries_precision(project.condition(members[0]))
                       else precision_of(project, members, sigma, given=precision))
        sigma_um = 0.0 if calibration is not None else _sigma_um(project, sigma)[0]
        sequences, noise = [], []
        for c in members:
            one, kw = _inputs(project.condition(c), calibration)
            sequences += one
            noise += kw.get("noise_sequences", [])
        extra = {"noise_sequences": noise, "calibration": calibration} if calibration is not None else {}
        out[name] = {"members": list(members), "best": best,
                     "plans": {k: plan_hmm(sequences, n_states=k, tau_s=tau, localization_sigma_um=sigma_um, **extra)
                               for k in counts}}
    return out


def joint_tasks(plans: dict) -> list:
    """Every start of every group's every count, ``(group, count, start)``, THE MOST STATES FIRST: a start's EM costs
    more the more states it has, so the longest go to the workers first and the pool ends soon after the slowest
    one, never waiting for it to begin last. The order changes no number (`joint_result` keeps each count's starts in
    start order)."""
    tasks = [(name, k, start) for name, entry in plans.items() for k, plan in entry["plans"].items()
             for start in range(plan.n_starts)]
    return sorted(tasks, key=lambda task: -task[1])          # stable: groups, then starts, in their order


def joint_result(plans: dict, outcomes: dict) -> dict:
    """``{group: {"D", "n_states", "selection", "members", "n_tracks"}}`` from every start's outcome
    (``{(group, count, start): hmm.run_hmm_start(...)}``): each count's best start (`hmm.finish_hmm`), and under Best
    BIC on the pooled tracks, as SQD chooses one count for a joint fit. RAISES `HMMError` when a count of a group had
    no start that converged."""
    import math

    from .hmm import finish_hmm
    from .sqd import EVIDENCE_THRESHOLDS, NPopSelection

    out = {}
    for name, entry in plans.items():
        fits = [finish_hmm(plan, [outcomes[(name, k, start)] for start in range(plan.n_starts)])
                for k, plan in sorted(entry["plans"].items())]
        selection = None
        if entry["best"]:
            chosen = NPopSelection(fits=fits)
            best = min(f.bic for f in chosen.fits)
            rows = [(int(f.n_states), float(f.bic), float(f.bic - best)) for f in chosen.fits]
            fit = chosen.recommended
            runner_up = sorted(r[2] for r in rows)[1] if len(rows) > 1 else math.inf
            selection = {"rows": rows, "recommended": int(fit.n_states), "margin": float(runner_up),
                         "inconclusive": bool(runner_up < EVIDENCE_THRESHOLDS[0][0])}
        else:
            fit = fits[0]
        out[name] = {"D": [float(d) for d in fit.D], "n_states": int(fit.n_states), "selection": selection,
                     "members": list(entry["members"]), "n_tracks": int(fit.n_sequences)}
    return out


def joint_progress(plans: dict, finished) -> dict:
    """How far the shared fit is, ``finished`` being the `joint_tasks` landed: ``{"done", "total", "counts_done",
    "counts_total", "status", "stage", "unit"}``. ``status`` is the section's line ("computing... the states shared
    across conditions: 1 of 3 state counts fitted (7 of 12 fits)"; at one count "... 3 of 4 fits"); ``stage`` and
    ``unit`` are the sidebar's (`run_progress.sidebar_job`: "shared states: 7 of 12 fits")."""
    finished = set(finished)
    tasks = joint_tasks(plans)
    pairs = {(name, k): plan.n_starts for name, entry in plans.items() for k, plan in entry["plans"].items()}
    counts_done = sum(1 for (name, k), n in pairs.items()
                      if all((name, k, start) in finished for start in range(n)))
    done, total = sum(1 for task in tasks if task in finished), len(tasks)
    fits = f"{done} of {total} fit{'' if total == 1 else 's'}"
    best = any(entry["best"] for entry in plans.values())
    head = "computing… the states shared across conditions: "
    status = head + (f"{counts_done} of {len(pairs)} state counts fitted ({fits})" if best else fits)
    return {"done": done, "total": total, "counts_done": counts_done, "counts_total": len(pairs), "status": status,
            "stage": "shared states", "unit": "fit"}


def _run_here(function, *args, **kwargs):
    """`joint_D`'s ``submit`` when none is given: the job run now, in this process, as a finished future."""
    from concurrent.futures import Future

    future = Future()
    try:
        future.set_result(function(*args, **kwargs))
    except BaseException as exc:                     # noqa: BLE001 - handed on through the future, as a pool does
        future.set_exception(exc)
    return future


def _stop_if(cancelled) -> None:
    from concurrent.futures import CancelledError

    if cancelled is not None and cancelled.is_set():
        raise CancelledError("the shared fit was stopped")


def joint_D(project, groups: dict, *, states: str, sigma: str, best_max: int = 3, submit=None, progress=None,
            cancelled=None, precision=None) -> dict:
    """``{group: {"D", "n_states", "selection", "members", "n_tracks"}}``: each group's states' D from the POOLED
    tracks of its conditions (one HMM over all their tracks), at ``states`` -- Best: BIC on the pooled tracks, as SQD
    chooses one count for a joint fit. A group of one condition is fitted the same way (its own pooled fit). RAISES
    `HMMError` when a group cannot be fitted.

    ``submit(function, *args, **kwargs) -> Future`` runs each job: the plans (`joint_plans`), then every start
    (`hmm.run_hmm_start`, in `joint_tasks`' order), side by side on a pool; None runs them here, one after another.
    The numbers do not depend on it. ``progress(payload)`` (`joint_progress`) is called once the plans are made and
    whenever more starts have landed, ALWAYS FROM THIS THREAD and in order, the last (every start landed) before this
    returns. A set ``cancelled`` (a `threading.Event`) stops the submitting, cancels what has not started, and raises
    `concurrent.futures.CancelledError`."""
    from concurrent.futures import FIRST_COMPLETED, wait

    from .hmm import run_hmm_start

    submit = submit or _run_here
    plans = submit(joint_plans, project, groups, states=states, sigma=sigma, best_max=best_max,
                   precision=precision).result()
    _stop_if(cancelled)
    tasks = joint_tasks(plans)
    futures: dict = {}
    said = [-1]

    def say() -> None:
        finished = {task for task, f in futures.items() if f.done() and not f.cancelled() and f.exception() is None}
        if progress is not None and len(finished) != said[0]:
            said[0] = len(finished)
            progress(joint_progress(plans, finished))

    try:
        say()
        for task in tasks:
            _stop_if(cancelled)
            name, k, start = task
            futures[task] = submit(run_hmm_start, plans[name]["plans"][k], start)
            say()
        pending = set(futures.values())
        while pending:
            _stop_if(cancelled)
            _landed, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            say()
        outcomes = {task: future.result() for task, future in futures.items()}
    except BaseException:
        for future in futures.values():
            future.cancel()                              # what has not started never will
        raise
    return joint_result(plans, outcomes)
