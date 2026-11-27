"""What the HMM section's Jump angles page shows, in numbers and words (22 Nov 2026).

The page (`ui.workspace.hmm_angles_page`) and its export (`figure_export` kind "hmm_angles") read only this module;
every number comes from `hmm_angles.compute_hmm_angles`' results, one per condition, nothing recomputed here.

TWO READINGS (asked the same day: "we should make comparison with simulation optional, and also just show pure data.
then optional to include against simulation"). A result made without chance (``result["chance"]`` False, the page's
standard) is read as the DATA: turning back per state, 1 for no preference, each replicate's own. A result made with
chance (the page's Against chance) is read beyond chance: observed minus free molecules at the same places, 0 for a
free molecule. `mode` says which every result on the page is; the words, the strip's reference line and the tables
follow it.

The comparison between conditions is Welch on the replicates' values (RULE 0: replicates are independent, never
paired), every pair of conditions that has two replicates a side or more, Holm over the pairs of one state's table.
"""
from __future__ import annotations

import math

from .hmm_angles import RESOLUTION, WINDOW_DEG

__all__ = ["TITLE", "READING_NOTES", "CHANCE_BOX", "CHANCE_TIP", "mode", "card_caption", "state_line", "card_notes",
           "histogram", "state_order", "strip_payload", "comparison_tables", "export_payload", "waiting_text",
           "strip_title"]

TITLE = "Jump angles"
#: The page's box that adds the simulation, and its tooltip.
CHANCE_BOX = "Against chance"
CHANCE_TIP = ("Simulate free molecules at the same places (same cells, the state's step size, each spot's localisation "
              "error, the camera's blur) and show turning back beyond them. Takes longer to compute.")

READING_NOTES = (
    "The angle between two consecutive steps of a track, at the position they share: 0° means the molecule went "
    "straight on, 180° means it turned back. A pair counts towards an HMM state in proportion to the chance that both "
    "steps were in that state; pairs that span a switch are left out.\n\n"
    f"Turning back is the number of pairs within {WINDOW_DEG:g}° of 180° divided by the number within "
    f"{WINDOW_DEG:g}° of 0°. 1 means no preference; above 1 means more turning back.\n\n"
    "A free molecule also turns back, near the cell wall and where its steps are about as small as the localisation "
    "error, and a fast one goes forward more because the camera averages its position over the frame. Tick Against "
    "chance to simulate free molecules at the same places; the page then shows turning back beyond them (0 is a free "
    f"molecule; the method itself reads within ±{RESOLUTION:g}). The simulated molecules go through the tracker's "
    "linking radius, so a pair counts only if both steps are shorter. The simulated cells are as wide as the space "
    "the fastest state's molecules fill (an outline is often wider), and the simulated camera is open as long as the "
    "camera files give. A fast state far past the linking radius can still read too high beyond chance, because "
    "u-track's search radius adapts to each track; compare such states between conditions, or with a synthetic set "
    "tracked the same way.\n\n"
    "Each dot is one replicate and the bar is the mean ± SD. Conditions are compared with Welch's t-test on the "
    "replicates (independent samples, never paired), with Holm correction over the pairs of one state.")


def _fmt(value, digits: int = 2, sign: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):+.{digits}f}" if sign else f"{float(value):.{digits}f}"


def _p_cell(p: float) -> str:
    """A p in a table cell: "0.034", "< 0.001" (`significance.p_text` without its "p")."""
    from .significance import p_text

    text = p_text(p)
    return text[4:] if text.startswith("p = ") else text[2:]


def mode(results: dict) -> str:
    """"chance" where every result was made against chance, else "data" (a page in the middle of switching reads as
    the data until every condition is in)."""
    good = [r for r in results.values() if "error" not in r]
    return "chance" if good and all(r.get("chance", True) for r in good) else "data"


def _against(result: dict) -> bool:
    return bool(result.get("chance", True))


def _values(state: dict, chance: bool) -> tuple:
    """``(replicate values, mean, sd, n)`` of the number shown: the excess against chance, else turning back."""
    if chance:
        reps = [float(r["excess"]) for r in state["replicates"] if r.get("ok")]
        return reps, state.get("excess_mean", math.nan), state.get("excess_sd", math.nan), len(reps)
    reps = [float(r["observed"]) for r in state["replicates"] if r.get("ok_observed", r.get("ok"))]
    mean = state.get("observed_mean")
    sd = state.get("observed_sd")
    if mean is None:                                   # a result made before the two readings: from its replicates
        mean = sum(reps) / len(reps) if reps else math.nan
        sd = (math.sqrt(sum((v - mean) ** 2 for v in reps) / (len(reps) - 1)) if len(reps) > 1 else math.nan)
    return reps, mean, sd, len(reps)


def waiting_text(have_hmm: bool) -> str:
    """The page's line before its numbers exist."""
    return ("Computing the jump angles from the HMM fits…" if have_hmm
            else "Run the HMM first; the jump angles are read from its fit.")


def strip_title(page_mode: str) -> str:
    return ("Across conditions · turning back beyond chance" if page_mode == "chance"
            else "Across conditions · turning back")


def card_caption(result: dict) -> str:
    """Under a condition's title: its tracks and pairs, and the share across a switch."""
    if "error" in result:
        return ""
    share = 100.0 * float(result.get("switch_share", 0.0))
    return (f"{int(result['n_tracks']):,} tracks · {int(result['n_pairs']):,} pairs of steps · "
            f"{share:.0f} % across a switch, left out")


def state_line(state: dict, chance: bool = True) -> str:
    """One state in one line: turning back (and chance and beyond it), and the replicates' mean ± SD."""
    if chance and state.get("past_limit"):
        return (f"{state['name']}: turns back {_fmt(state['observed'])} · past the linking limit, no chance (the "
                "data alone)")
    if chance:
        words = (f"{state['name']}: turns back {_fmt(state['observed'])} · chance {_fmt(state['chance'])} · "
                 f"beyond chance {_fmt(state['excess'], sign=True)}")
    else:
        words = f"{state['name']}: turns back {_fmt(state['observed'])} times as often as it goes forward"
    _reps, mean, sd, n = _values(state, chance)
    if n >= 2:
        words += f" (replicates {_fmt(mean, sign=chance)} ± {_fmt(sd)}, n = {n})"
    elif n == 1:
        words += " (1 replicate)"
    return words


def _linking_note(result: dict) -> str:
    """How the chance's pairs were linked (`hmm_angles.linking_for`)."""
    if "link_words" not in result:                     # made before the chance went through the linking radius
        return ("This result was made without the linking radius, so a fast state reads too high beyond chance. Run it "
                "again.")
    if result.get("link_um") is None:
        return ("No linking radius is recorded, so every simulated step was linked and a fast state reads too high beyond "
                "chance.")
    return (f"Linking: {result['link_words']}; the simulated pairs are linked like the tracked ones. A fast state far "
            "past the radius can still read too high, because u-track's radius adapts to each track.")


def card_notes(result: dict) -> list:
    """The lines under a card's half circles: against chance, the D the chance walked at, the error it used and what
    it assumed; for the data alone, what a free molecule would do."""
    states = result.get("states") or []
    if not _against(result):
        return ["Without chance, the numbers include what a free molecule does: it turns back near the wall and where "
                "its steps are as small as the localisation error, and goes forward more when fast. Tick Against "
                "chance to compare with free molecules at the same places."]
    notes = ["Chance walked at D " + ", ".join(
        f"{s['name']} {'n/a' if s.get('past_limit') else _fmt(s['D_chance'], 3)}" for s in states)
        + " µm²/s (the size of its own steps; the HMM's "
        + result.get("d_word", "apparent") + " D: " + ", ".join(f"{_fmt(s['D_fit'], 3)}" for s in states) + ")."]
    camera = result.get("exposure_words") or "the whole frame"
    cells = result.get("cell_words") or "each cell's outline"
    notes.append(f"Localisation error: {result.get('noise_words', 'not stated')}. Camera open: {camera}. Cells: "
                 f"straight capsules, each as long as its outline, {cells}.")
    notes.append(_linking_note(result))
    past = [s["name"] for s in states if s.get("past_limit")]
    if past:
        notes.append(f"{', '.join(past).capitalize()}: past the linking limit. Its steps are as long as a free molecule's can be "
                     "inside the linking radius, so they do not give its D and no chance is drawn.")
    floor = [s["name"] for s in states if s.get("at_floor")]
    if floor:
        notes.append(f"{', '.join(floor).capitalize()}: the steps are at most as long as the localisation error, so its "
                     "chance is that of an immobile molecule.")
    short = [s["name"] for s in states if int(s.get("n_replicates", 0)) < len(s.get("replicates") or [])]
    if short:
        notes.append(f"{', '.join(short).capitalize()}: a replicate with fewer than 20 pairs has no number of its own.")
    return notes


def histogram(result: dict, index: int) -> dict:
    """A state's observed angle density per bin (per degree), and chance's where the result has it (else None)."""
    state = result["states"][index]
    return {"name": state["name"], "edges": list(result["edges_deg"]),
            "observed": list(state["observed_density"]),
            "chance": list(state["chance_density"]) if _against(result) else None}


def state_order(results: dict) -> list:
    """Every state name the results hold, slowest first ("slow", "middle", "fast"; then the rest by name)."""
    rank = {"all": 0, "slow": 1, "middle": 2, "fast": 3}
    names = {s["name"] for r in results.values() if "error" not in r for s in r.get("states") or []}
    return sorted(names, key=lambda n: (rank.get(n, 9), n))


def _state(result: dict, name: str):
    return next((s for s in result.get("states") or [] if s["name"] == name), None)


def strip_payload(results: dict, order: list, labels: dict, colours: dict) -> dict:
    """The strip across conditions: one panel a state, one column a condition, its replicates' values and their
    mean ± SD, on one y range for every panel. The value is turning back (reference line at 1) or, against chance,
    beyond chance (reference at 0, with the band free molecules read within)."""
    chance = mode(results) == "chance"
    reference = 0.0 if chance else 1.0
    panels, values = [], [reference] + ([-RESOLUTION, RESOLUTION] if chance else [])
    for name in state_order(results):
        columns = []
        for key in order:
            result = results.get(key)
            state = _state(result, name) if result and "error" not in result else None
            if state is None:
                continue
            reps, mean, sd, n = _values(state, chance)
            if not reps:
                mean = float(state["excess"] if chance else state["observed"])
            columns.append({"key": key, "label": labels.get(key, key), "colour": colours.get(key, "#888888"),
                            "replicates": reps, "mean": float(mean), "sd": float(sd) if n > 1 else math.nan,
                            "n": n})
            values += reps + [mean] + ([mean - sd, mean + sd] if n > 1 and math.isfinite(sd) else [])
        if columns:
            panels.append({"name": name, "columns": columns})
    finite = [v for v in values if math.isfinite(v)]
    lo, hi = min(finite), max(finite)
    pad = 0.08 * (hi - lo or 1.0)
    return {"panels": panels, "lo": lo - pad, "hi": hi + pad, "reference": reference,
            "resolution": RESOLUTION if chance else None, "mode": "chance" if chance else "data",
            "y_label": "Turning back beyond chance" if chance else "Turning back (back / forward)"}


def comparison_tables(results: dict, order: list, labels: dict) -> list:
    """Per state, every pair of conditions with two replicates a side or more: the difference (second minus first in
    the reading shown), its 95 % interval, Welch's p and Holm's over the state's pairs.
    ``[{"name", "columns", "rows", "caption"}]``."""
    from .uncertainty import adjust, welch_test

    chance = mode(results) == "chance"
    what = "turning back beyond chance" if chance else "turning back"
    tables = []
    for name in state_order(results):
        have = []
        for key in order:
            result = results.get(key)
            state = _state(result, name) if result and "error" not in result else None
            reps = _values(state, chance)[0] if state else []
            if len(reps) >= 2:
                have.append((key, reps))
        tests = []
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                (a, va), (b, vb) = have[i], have[j]
                test = welch_test(va, vb)
                if math.isfinite(test.p):
                    tests.append((a, b, test))
        if not tests:
            continue
        holm = adjust([t.p for _a, _b, t in tests], "holm")
        rows = [[f"{labels.get(b, b)} vs {labels.get(a, a)}", _fmt(t.difference, sign=True),
                 f"[{_fmt(t.ci_low, sign=True)}, {_fmt(t.ci_high, sign=True)}]", _p_cell(t.p), _p_cell(p)]
                for (a, b, t), p in zip(tests, holm)]
        tables.append({"name": name, "columns": ["Pair", "Difference", "95 % interval", "p (Welch)", "p (Holm)"],
                       "rows": rows,
                       "caption": (f"{name.capitalize()} state: the first condition's {what} minus the second's, "
                                   f"replicates as independent samples; Holm over these {len(tests)} pair"
                                   f"{'' if len(tests) == 1 else 's'}.")})
    return tables


def export_payload(results: dict, order: list, labels: dict, colours: dict) -> dict:
    """What the Export dialog draws (`figure_export` kind "hmm_angles"): the strip, every condition's half circles and
    the tables, from the same results the page shows, in the page's reading."""
    shown = [k for k in order if k in results and "error" not in results[k]]
    chance = mode({k: results[k] for k in shown}) == "chance"
    return {"mode": "chance" if chance else "data",
            "strip": strip_payload(results, shown, labels, colours),
            "histograms": [{"key": k, "label": labels.get(k, k), "colour": colours.get(k, "#888888"),
                            "states": [histogram(results[k], i) for i in range(len(results[k]["states"]))]}
                           for k in shown],
            "tables": comparison_tables(results, shown, labels),
            "notes": {k: card_notes(results[k]) for k in shown},
            "lines": {k: [state_line(s, chance) for s in results[k]["states"]] for k in shown},
            "entries": [{"key": k, "label": labels.get(k, k)} for k in shown],
            "unlinked": [labels.get(k, k) for k in shown if chance and results[k].get("link_um") is None]}
