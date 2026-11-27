"""The Dynamics tab's numbers, caveats and table rows, with no Qt anywhere in it.

`ui/workspace/dynamics_chart.py` is the Qt shell (Kinetics, State count and Clustering cards); this is
what it draws from. They are split because of the Thin Qt rule (docs/gui_tool_views.md;
`tests/test_ui_thin_qt.py` walks the whole GUI layer): a wrong number or a wrong sentence must fail in
a headless test, not look subtly wrong on screen, and a process-pool worker should import this module
without importing Qt.

Why Kinetics, State count and Clustering sit in ONE payload: Kinetics tells you whether State
count's own mode count can be trusted at all (CLAUDE.md: "saSPT resolves two because of switching,
not because there are two"). `smtracker.saspt_adapter` never cross-references a switching rate, so
`saspt_switching_caveat` composes that warning HERE from the Kinetics rate of the same condition.

`compute_dynamics_evidence` is picklable and runs on the process pool, one condition at a time. It
calls the SAME analysis functions the legacy controllers call (`hmm.fit_hmm`/`bootstrap_hmm`,
`saspt_adapter.run_saspt`, `clustering.select_k`) and each of the three fails INDEPENDENTLY
(`HMMError` / `SASPTError` / `ClusteringError`, all raised rather than returning a degenerate
result), so one unavailable analysis never blanks the other two cards. `D_um2_s=0.3` and
`feature="log"` for Clustering are the legacy controller's own deliberate deviations from
`select_k`'s defaults ("the FEATURE SCALE decides the verdict"); `n_replicates=20`, not the legacy's
3, because the chance SD is "too poorly determined to quote" below
`MIN_REPLICATES_FOR_A_QUOTABLE_SD` and this chart DRAWS that SD as a band.

The payload is plain lists, floats and strings only. The rules that live here: the LOW-SWITCHING
flag, the saSPT spectrum keeps only positive D (a log axis), the clustering null band is
chance +/- `EXCEEDS_NULL_SD` x chance SD (its top edge IS the decision boundary behind
`KSelection.exceeds_null`), and the axis bounds and table captions follow from the numbers. The
three charts' tooltip sentences are built here too (`kinetics_hover_rows`, `saspt_hover`,
`clustering_hover`), so the words on a hover are tested without Qt like every other number.
"""
from __future__ import annotations

__all__ = [
    "clustering_hover",
    "clustering_payload",
    "compute_dynamics_evidence",
    "kinetics_hover_rows",
    "kinetics_payload",
    "saspt_hover",
    "saspt_payload",
    "saspt_switching_caveat",
]

LOW_SWITCHING_FLAG = "Low switching: rates unreliable"

_NO_HMM = "HMM did not converge."
_NO_SASPT = "saSPT unavailable."
_NO_CLUSTERING = "Clustering unavailable."


def _ci_text(ci) -> str:
    return f"[{ci[0]:.3g}, {ci[1]:.3g}]" if ci is not None else "–"


def saspt_switching_caveat(hmm_ok: bool, kin_identifiable, kin_on, kin_off,
                           at_grid_floor: bool) -> str:
    """Composed from the Kinetics rate of the SAME condition (see the module docstring)."""
    if hmm_ok and kin_identifiable:
        text = (f"saSPT assumes one D per trajectory. At this condition's "
                f"measured switching rate (on {kin_on:.2g}/s, off {kin_off:.2g}/s), "
                f"a two-state system can read as one broad mode (see Kinetics).")
    elif hmm_ok:
        text = ("saSPT assumes one D per trajectory and cannot see "
                "switching. This condition's rates are unreliable (low-switching), so it is unknown "
                "whether that applies here.")
    else:
        text = ("saSPT assumes one D per trajectory and cannot see "
                "switching. This condition's HMM did not converge, so "
                "switching could not be cross-checked.")
    if at_grid_floor:
        text += (" Weight also piles up at the grid's lowest D, so read the "
                 "slow end as a floor and not as a resolved peak.")
    return text


# --------------------------------------------------------------------------- #
# Card payloads
# --------------------------------------------------------------------------- #


def kinetics_payload(condition: str, *, ok: bool, error: str | None = None, on=None,
                     off=None, on_ci=None, off_ci=None, identifiable=None, n_degenerate=None,
                     n_units=None, D=(), occupancy=()) -> dict:
    """The Kinetics (HMM) card: two bars with bootstrap-percentile whiskers and its table."""
    if not ok:
        return {"ok": False, "error": error, "bars": [], "caveat": None, "axis_top": None,
                "table": {"columns": [], "rows": [], "caption": error or _NO_HMM}}
    bars = [{"name": "On", "value": float(on), "ci": _pair(on_ci), "colour_key": "on"},
            {"name": "Off", "value": float(off), "ci": _pair(off_ci), "colour_key": "off"}]
    bounds = [float(on), float(off), *(on_ci or ()), *(off_ci or ())]
    columns = ["Rate", "1/s", "95% CI"]
    rows = [[name, f"{val:.3g}", _ci_text(ci)]
            for name, val, ci in (("On (binding)", on, on_ci), ("Off (release)", off, off_ci))]
    for i, (d, occ) in enumerate(zip(D, occupancy)):
        rows.append([f"State {i + 1} (D={d:.3g} µm²/s)", f"{100 * occ:.1f}% occupancy", "–"])
    if identifiable is None:
        caption = f"{condition}: bootstrap unavailable, rates shown without a CI."
    elif identifiable:
        caption = f"{condition}: rates identifiable ({n_units} units)."
    else:
        caption = (f"{condition}: low switching: {n_degenerate} of "
                   f"{n_units} bootstrap replicates had no observed "
                   f"transition; rates are unreliable.")
    return {
        "ok": True, "error": None, "bars": bars,
        "caveat": LOW_SWITCHING_FLAG if identifiable is False else None,
        "axis_top": max(max(bounds) * 1.15, 0.01),
        "table": {"columns": columns, "rows": rows, "caption": caption},
    }


def _pair(ci):
    return None if ci is None else [float(ci[0]), float(ci[1])]


def saspt_payload(*, ok: bool, error: str | None = None, grid=(), occupations=(), modes=(),
                  summary: str = "", caveat: str | None = None) -> dict:
    """The State count (saSPT) card: the posterior-occupancy spectrum over a log D axis, and its
    resolved modes. Only positive D can be drawn on that axis, so only those are kept."""
    if not ok:
        return {"ok": False, "error": error, "caveat": None, "grid": [], "occupancy": [],
                "occupancy_pct": [], "modes": [], "y_top": None,
                "table": {"columns": [], "rows": [], "caption": error or _NO_SASPT}}
    kept = [(float(g), float(o)) for g, o in zip(grid, occupations) if g > 0]
    kept_modes = [{"D": float(d), "occupancy": float(o)} for d, o in modes if d > 0]
    occupancy = [o for _, o in kept]
    return {
        "ok": True, "error": None, "caveat": caveat,
        "grid": [g for g, _ in kept], "occupancy": occupancy,
        "occupancy_pct": [100 * o for o in occupancy],
        "modes": kept_modes,
        "y_top": max(max(occupancy) * 1.15, 1e-6) if occupancy else None,
        "table": {"columns": ["D (µm²/s)", "Occupancy"],
                  "rows": [[f"{d:.3g}", f"{100 * o:.1f}%"] for d, o in modes],
                  "caption": summary},
    }


def clustering_payload(*, ok: bool, error: str | None = None, ks=(), observed=(), chance=(),
                       chance_sd=(), excess=(), verdict: str = "") -> dict:
    """The Clustering card: observed Calinski-Harabasz against the free-Brownian null, with the
    null band the legacy chart never drew (chance +/- `EXCEEDS_NULL_SD` x chance SD)."""
    if not ok:
        return {"ok": False, "error": error, "rows": [], "band_lo": [], "band_hi": [],
                "lo": None, "hi": None,
                "table": {"columns": [], "rows": [], "caption": error or _NO_CLUSTERING}}
    from .clustering import EXCEEDS_NULL_SD
    band_hi = [float(c) + EXCEEDS_NULL_SD * float(s) for c, s in zip(chance, chance_sd)]
    band_lo = [float(c) - EXCEEDS_NULL_SD * float(s) for c, s in zip(chance, chance_sd)]
    lo = min(min(observed), min(band_lo)) * 0.9
    hi = max(max(observed), max(band_hi)) * 1.1
    if hi <= lo:
        lo, hi = lo - 1, hi + 1
    rows = [{"k": float(k), "observed": float(o), "chance": float(c), "chance_sd": float(s),
             "excess": float(e)}
            for k, o, c, s, e in zip(ks, observed, chance, chance_sd, excess)]
    return {
        "ok": True, "error": None, "rows": rows, "band_lo": band_lo, "band_hi": band_hi,
        "lo": float(lo), "hi": float(hi),
        "table": {"columns": ["k", "Observed", "Chance", "± SD", "Excess"],
                  "rows": [[str(int(r["k"])), f"{r['observed']:.1f}", f"{r['chance']:.1f}",
                            f"{r['chance_sd']:.1f}", f"{r['excess']:+.1f}"] for r in rows],
                  "caption": verdict},
    }


# --------------------------------------------------------------------------- #
# Hover sentences: what each chart's tooltip says about the item under the pointer
# --------------------------------------------------------------------------- #


def kinetics_hover_rows(value: float, ci) -> tuple:
    """The tooltip rows of one rate bar: the rate, and its 95 % interval or a dash.

    The interval is the whole reason that chart has a linear y axis, so it belongs in the hover; where
    there is none its absence is stated rather than left as a missing row. Same interval text as the
    card's table (`_ci_text`), so the two can never disagree."""
    return (("Rate", f"{value:.3g} /s"), ("95% CI", _ci_text(ci)))


def saspt_hover(d: float, occupancy_pct: float) -> tuple:
    """``(heading, rows)`` for one point of the saSPT spectrum: its D and the occupancy (%) there."""
    return f"D = {d:.3g} µm²/s", (("Occupancy", f"{occupancy_pct:.2f}%"),)


def clustering_hover(k, observed: float, chance: float, chance_sd: float, excess: float) -> tuple:
    """``(heading, rows)`` for one k column: BOTH series at this k, which is the comparison the chart
    exists to make (the observed Calinski-Harabasz score against its chance rate), and the excess."""
    return (f"k={int(k)}",
            (("Observed", f"{observed:.1f}"),
             ("Chance", f"{chance:.1f} ± {chance_sd:.1f}"),
             ("Excess", f"{excess:+.1f}")))


# --------------------------------------------------------------------------- #
# Compute: one condition, on the process pool
# --------------------------------------------------------------------------- #


def compute_dynamics_evidence(project, condition: str) -> dict:
    """One condition's Kinetics / State-count / Clustering evidence. Returns plain data only."""
    from .clustering import ClusteringError, select_k
    from .hmm import HMMError, bootstrap_hmm, fit_hmm, step_sequences
    from .saspt_adapter import SASPTError, run_saspt

    part = project.condition(condition)
    tau = project.params.tau_s

    # -- Kinetics: fit and bootstrap fail independently -- a converged fit with a failed bootstrap
    # still draws bars, just without whiskers or identifiability. -- #
    hmm_ok = False
    kin: dict = {}
    try:
        hmm_fit = fit_hmm(step_sequences(part), n_states=2, tau_s=tau, n_starts=8)
        rates = hmm_fit.rates_per_s
        # States are sorted by ascending D: 0 = slow/bound, 1 = fast/free. "on" (binding) is
        # fast->slow; "off" (release) is slow->fast.
        kin = {"on": float(rates[1, 0]), "off": float(rates[0, 1]),
               "D": [float(d) for d in hmm_fit.D],
               "occupancy": [float(o) for o in hmm_fit.occupancy]}
        hmm_ok = True
        try:
            boot = bootstrap_hmm(part, n_states=2, tau_s=tau, unit="cell", n_boot=50)
            kin.update(
                on_ci=tuple(float(v) for v in boot["rate_2_to_1"].quantile([0.025, 0.975])),
                off_ci=tuple(float(v) for v in boot["rate_1_to_2"].quantile([0.025, 0.975])),
                identifiable=boot.attrs["n_degenerate"] == 0,
                n_degenerate=int(boot.attrs["n_degenerate"]),
                n_units=int(boot.attrs["n_units"]),
            )
        except HMMError:
            pass
        kinetics = kinetics_payload(condition, ok=True, **kin)
    except HMMError as exc:
        kinetics = kinetics_payload(condition, ok=False, error=str(exc))

    # -- State count (saSPT) -- #
    try:
        result = run_saspt(part)
        modes = result.modes()
        saspt = saspt_payload(
            ok=True, grid=[float(g) for g in result.diffusion_grid],
            occupations=[float(o) for o in result.occupations],
            modes=[(float(r.D_um2_per_s), float(r.occupancy)) for r in modes.itertuples()],
            summary=result.summary(),
            caveat=saspt_switching_caveat(hmm_ok, kin.get("identifiable"), kin.get("on"),
                                          kin.get("off"), bool(result.at_grid_floor())))
    except SASPTError as exc:
        saspt = saspt_payload(ok=False, error=str(exc))

    # -- Clustering -- #
    try:
        selection = select_k(part, D_um2_s=0.3, k_values=(2, 3), n_lags=4,
                             feature="log", n_replicates=20)
        table = selection.table
        clustering = clustering_payload(
            ok=True, ks=table["k"].tolist(), observed=table["observed"].tolist(),
            chance=table["chance"].tolist(), chance_sd=table["chance_sd"].tolist(),
            excess=table["excess"].tolist(), verdict=selection.verdict)
    except ClusteringError as exc:
        clustering = clustering_payload(ok=False, error=str(exc))

    # Each card's drawer table sits at the top level (the card reads it by key).
    return {"condition": condition,
            "kinetics_table": kinetics.pop("table"), "kinetics": kinetics,
            "saspt_table": saspt.pop("table"), "saspt": saspt,
            "clustering_table": clustering.pop("table"), "clustering": clustering}
