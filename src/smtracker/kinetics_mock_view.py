"""The Kinetics card against a mock addition: the job, the block on the card, the chart, the export rows.

(8 Nov 2026; `smtracker.kinetics_mock` is the statistics, this is its door in the app.) Headless; imports no Qt.

THE JOB (`compute_mock`, one pool job per treated condition, kept in the section cache by `cached_group_job` so the
key holds BOTH conditions' rows): each condition's steps on its OWN clock (`induction_view.clock_steps_for`: its own
movie starts, the per-spot noise where the Run has it), the after window (typed, else Auto: `kinetics_mock.after_start`
with the treated condition's fitted response, which the model's result carries as ``result["response"]``), then
`kinetics_mock.compare`. The mock is a condition like the others: ticked in the same Run, with its own addition times.
Treated against mock is two independent samples of replicates (Rule 0: Welch, never paired by replicate number).

A REFUSAL (`kinetics_mock.MockError`: too few replicates with both windows, no step after the addition) is a result,
not a failure: the block carries ``refused`` and the card writes the words. Any other exception is kept as ``error``
in the block (the treated condition's own card stays whole). Nothing is filled in.

THE BLOCK (``result["mock"]``), plain lists and numbers so a cache and a session hold it: ``readout``, ``mock``
(the condition's key), ``after_from_s``, ``after_why``, ``lines`` (`kinetics_mock.lines`), ``treated`` / ``mock_rows``
(one dict per replicate: before, after, change, steps in each window, why it was left out), ``treated_change`` /
``mock_change`` (mean, low, high), ``welch`` (difference, ci_low, ci_high, p, n_a, n_b, df, reason). Changes are logs of
a ratio for the mean mobility (shown as percentages) and percentage points for the bound share.

THE CARD LEADS WITH IT (ruled 8 Nov 2026, "1 yes"): wherever a mock was chosen, the block stands first, above the
model table, and the held-out verdict follows under `HELD_OUT_HEADING` with `held_out_caution` (a drift over the
session reads there as a response; the drift set showed it naming a D change the addition never made). A refusal leads
too, so the reason stands first. A legend draft and the notes follow the same order (`lead_sentences`). Display only:
the result is unchanged, so no section cache version moves.
"""
from __future__ import annotations

import math

import numpy as np

from . import kinetics_mock as km

BLOCK_HEADING = "Against the mock addition"


class _Response:
    """The treated fit's response as the card kept it (``{"change", "curve": [(t, h)]}``): what
    `kinetics_mock.after_start` reads of a fit (`change`, `response`)."""

    def __init__(self, record: dict):
        self.change = record.get("change", "none")
        curve = np.asarray(record.get("curve") or [(0.0, 0.0)], float)
        self._t, self._h = curve[:, 0], curve[:, 1]

    def response(self, t) -> np.ndarray:
        return np.interp(np.asarray(t, float), self._t, self._h)


def response_record(fit, grid) -> dict:
    """The fit's response on ``grid``, small and plain, for ``result["response"]`` (the Auto after window)."""
    return {"change": str(fit.change), "curve": [(float(t), float(h)) for t, h in zip(grid, fit.response(grid))]}


def _rows(table) -> list:
    out = []
    for _i, r in table.iterrows():
        out.append({"replicate": int(r["replicate"]), "n_before": int(r["n_before"]), "n_after": int(r["n_after"]),
                    "before": float(r["before"]), "after": float(r["after"]), "change": float(r["change"]),
                    "left_out": str(r["left_out"])})
    return out


def compute_mock(project, conditions, *, readout: str, treated_clock: dict, mock_clock: dict,
                 after_from_s: float | None = None, response: dict | None = None, n_pop: int = 2) -> dict:
    """``conditions`` ``[treated, mock]``. ``treated_clock`` / ``mock_clock``: `induction_view.clock_steps_for`'s
    keywords of each (``trigger_frame``, ``additions``, ``samples``, ``precision``). ``after_from_s`` None: Auto.
    ``n_pop``: the Run's populations, for the bound share (the slowest's share; 2 by default).
    Returns the block (module docstring)."""
    from .induction_view import clock_steps_for

    treated, mock = conditions
    block = {"readout": readout, "mock": str(mock), "after_from_s": None, "after_why": ""}
    try:
        steps_t = clock_steps_for(project, treated, **treated_clock)[0]
        steps_m = clock_steps_for(project, mock, **mock_clock)[0]
        if after_from_s is None:
            after, why = km.after_start(steps_t, _Response(response) if response else None)
        else:
            after, why = float(after_from_s), f"from {float(after_from_s):g} s, as typed"
        block.update(after_from_s=float(after), after_why=why)
        got = km.compare(steps_t, steps_m, readout=readout, after_from_s=after, n_pop=int(n_pop))
    except km.MockError as exc:
        return dict(block, refused=str(exc))
    except Exception as exc:                            # noqa: BLE001 - the card names it, the treated card stays
        return dict(block, error=f"{type(exc).__name__}: {exc}")
    w = got["welch"]
    return dict(block, lines=list(got["lines"]), treated=_rows(got["treated"]), mock_rows=_rows(got["mock"]),
                treated_change=[float(v) for v in got["treated_change"]],
                mock_change=[float(v) for v in got["mock_change"]],
                welch={"difference": float(w.difference), "ci_low": float(w.ci_low), "ci_high": float(w.ci_high),
                       "p": float(w.p), "n_a": int(w.n_a), "n_b": int(w.n_b), "df": float(w.df),
                       "reason": str(w.reason)})


def attach_mock(result: dict, block: dict) -> dict:
    """``result`` with the block attached (``result["mock"]``) and its words in the card's text (``text["mock"]``):
    the lines, or the refusal, or the error, each led by where the after window starts."""
    if "error" in result:
        return result
    return dict(result, mock=block, text=dict(result["text"], mock=block_lines(block)))


def block_lines(block: dict) -> list:
    """The card's words for a block: the after window, then the sentences, or the refusal / error."""
    out = []
    if block.get("after_why"):
        out.append(f"After window {block['after_why']}; the baseline is every step before the addition.")
    if "refused" in block:
        out.append(f"Not read against the mock: {block['refused']}.")
    elif "error" in block:
        out.append(f"The mock comparison failed: {block['error']}")
    else:
        out.extend(block["lines"])
    return out


def _pct(v: float) -> float:
    return 100.0 * (math.exp(v) - 1.0) if math.isfinite(v) else math.nan


def mock_chart(block: dict) -> dict | None:
    """The small chart's numbers, or None where there is nothing to draw (a refusal or an error).

    ``{"unit": "%" | "pp", "columns": [{"role": "treated" | "mock", "values": [...], "mean", "lo", "hi"}],
    "difference": {"value", "lo", "hi", "p", "text"}}``; the mean mobility's log ratios are shown as percentages
    (``exp - 1``), the bound share's changes as they are, in percentage points. Each column's interval is a t
    interval over its replicates (`kinetics_mock._one_sample`); a replicate left out is not a dot."""
    if "welch" not in block:
        return None
    mobility = block["readout"] == "mean_D"

    def conv(v):
        return _pct(v) if mobility else v

    def column(role, rows, change):
        mean, lo, hi = change
        return {"role": role, "values": [conv(r["change"]) for r in rows if math.isfinite(r["change"])],
                "mean": conv(mean), "lo": conv(lo), "hi": conv(hi)}

    w = block["welch"]
    p = w["p"]
    ptext = f"p = {p:.2g}" if math.isfinite(p) else "no p"
    value = conv(w["difference"])
    known = math.isfinite(w["ci_low"]) and math.isfinite(w["ci_high"])       # no interval without spread
    if mobility:
        span = f" [{conv(w['ci_low']):+.0f}, {conv(w['ci_high']):+.0f}]" if known else ""
        text = f"beyond the mock {value:+.0f} %{span}, {ptext}"
    else:
        span = f" [{w['ci_low']:+.1f}, {w['ci_high']:+.1f}]" if known else ""
        text = f"beyond the mock {value:+.1f} pp{span}, {ptext}"
    return {"unit": "%" if mobility else "pp",
            "columns": [column("treated", block["treated"], block["treated_change"]),
                        column("mock", block["mock_rows"], block["mock_change"])],
            "difference": {"value": value, "lo": conv(w["ci_low"]), "hi": conv(w["ci_high"]), "p": p, "text": text}}


def replicate_table(block: dict, treated_name: str, mock_name: str) -> list:
    """The per-replicate rows for the export's source data, treated first: condition, role, replicate, the steps in
    each window, the value before and after, the change (its unit in the column's name) and why one was left out."""
    unit = "log ratio (after / before)" if block["readout"] == "mean_D" else "pp (after minus before)"
    out = []
    for role, name, rows in (("treated", treated_name, block["treated"]), ("mock", mock_name, block["mock_rows"])):
        for r in rows:
            out.append({"condition": name, "role": role, "replicate": r["replicate"],
                        "steps before": r["n_before"], "steps after": r["n_after"], "before": r["before"],
                        "after": r["after"], f"change, {unit}": r["change"], "left out": r["left_out"]})
    return out


#: The y title of the values a block draws, in the readout's own unit (the chart draws values, not changes, since
#: 8 Nov 2026)
VALUE_TITLES = {"mean_D": "mean D (µm²/s)", "bound_share": "bound share (%)"}


def axis_title(block: dict) -> str:
    """The y title of a block's chart, card and export alike: the readout in its own unit, "mean D (µm²/s)" or
    "bound share (%)" (each replicate's value before and after the addition is drawn, `mock_lines`)."""
    return VALUE_TITLES["mean_D" if block.get("readout") == "mean_D" else "bound_share"]


# --------------------------------------------------------------------------- #
# The chart: each replicate BEFORE and AFTER, in the readout's own unit (8 Nov 2026)
# --------------------------------------------------------------------------- #
# Asked 8 Nov 2026 for Fig. 6D: the change dots ("+22 pp", a change of a change) were unclear; "why not just show the D
# value itself instead of percentage". Each replicate is a thin line from its value before the addition to its value
# after it, the treated condition beside its mock, the replicates' mean as a bold line, on an axis in the readout's own
# unit from 0. The heading says each condition's own mean change in words and the p of their difference, Welch over
# replicates (Rule 0: the lines of one condition carry no pairing with the other's). The size of the difference is
# not in the heading: for the mean D it is a ratio of ratios, which confused more than it told; the card's sentences
# and the legend keep it with its interval.

#: Where a chart's four columns stand: the treated condition's before and after, then the mock's
LINE_X = {"treated": (0.0, 1.0), "mock": (2.2, 3.2)}
#: The p under which the heading says "different" (the brackets' first tier)
DIFFERENT_BELOW = 0.05


def _signed(v: float, unit: str) -> str:
    """"+27 %" / "-13 pp": whole numbers from 10 on, one decimal under it (a change of 0.4 pp is not "+0")."""
    if not math.isfinite(v):
        return "no value"
    return f"{v:+.0f} {unit}" if abs(v) >= 9.95 else f"{v:+.1f} {unit}"


def mock_heading(block: dict, treated_name: str, mock_name: str = "mock") -> list:
    """The chart's two lines: each condition's own mean change, then the verdict and p. ``["D drop -43 %, mock +27 %",
    "different, p = 0.0069"]``; the bound share in pp. Under `DIFFERENT_BELOW` "different", else "no difference
    shown" (three replicates a side cannot show that there is none). [] where there is nothing to draw."""
    chart = mock_chart(block)
    if chart is None:
        return []
    treated, mock = chart["columns"]
    unit = chart["unit"]
    first = f"{treated_name} {_signed(treated['mean'], unit)}, {mock_name} {_signed(mock['mean'], unit)}"
    p = chart["difference"]["p"]
    if not math.isfinite(p):
        return [first, "no p (too few replicates)"]
    return [first, f"{'different' if p < DIFFERENT_BELOW else 'no difference shown'}, p = {p:.2g}"]


def mock_lines(block: dict, treated_name: str = "treated", mock_name: str = "mock") -> dict | None:
    """What the chart draws, card and export alike, or None where there is nothing to draw (a refusal, an error).

    ``{"title", "unit", "y_range", "heading", "columns": [{"role", "x": (before, after), "rows": [(replicate, before,
    after)], "mean": (before, after)}]}``: every replicate with both values (one left out is not drawn), the mean of
    the drawn replicates' values (the bold line), the y range from 0 to 15 % above the highest value (the bound share
    capped at 100)."""
    if mock_chart(block) is None:
        return None
    columns, top = [], 0.0
    for role, rows in (("treated", block["treated"]), ("mock", block["mock_rows"])):
        drawn = [(int(r["replicate"]), float(r["before"]), float(r["after"])) for r in rows
                 if not r.get("left_out") and math.isfinite(r["before"]) and math.isfinite(r["after"])]
        mean = ((sum(b for _r, b, _a in drawn) / len(drawn), sum(a for _r, _b, a in drawn) / len(drawn))
                if drawn else (math.nan, math.nan))
        columns.append({"role": role, "x": LINE_X[role], "rows": drawn, "mean": mean})
        top = max([top, *(v for _r, b, a in drawn for v in (b, a))])
    share = block["readout"] != "mean_D"
    high = 1.15 * top if top > 0 else 1.0
    return {"title": axis_title(block), "unit": "%" if share else "µm²/s",
            "y_range": (0.0, min(100.0, high) if share else high),
            "heading": mock_heading(block, treated_name, mock_name), "columns": columns}


def export_lines(block: dict, mock_name: str) -> list:
    """The block's words for a legend draft and notes (the mock named, the lines as the card has them)."""
    return [f"Read against the mock addition {mock_name}."] + block_lines(block)


# --------------------------------------------------------------------------- #
# The card LEADS with the mock comparison (ruled 8 Nov 2026)
# --------------------------------------------------------------------------- #

#: The held-out verdict's heading on a card read against a mock, where it comes second, under the mock's block.
HELD_OUT_HEADING = "Held-out verdict: after against before"

#: What the held-out verdict cannot see and the mock comparison can (the SynK course drift data set, 8 Nov 2026: every moving
#: state's D fell 20 % over the session; read after against before, the binding rise was called "the D's and the
#: shares changed", its fast D 0.575 to 0.446, where the set without the drift read it flat).
HELD_OUT_DRIFT = ("Read on one clock, so a drift over the session counts here as a response: this verdict can "
                  "report a D change that the addition did not cause.")
HELD_OUT_READ = HELD_OUT_DRIFT + " The comparison with the mock above takes the drift out."
HELD_OUT_UNREAD = HELD_OUT_DRIFT + " Here the mock could not take it out (above)."


def leads(text: dict) -> bool:
    """Whether a card LEADS with the mock block: whenever a mock was chosen for it, a refusal or a failure included,
    so the reason it could not be read stands first."""
    return bool(text.get("mock"))


def held_out_caution(block: dict) -> str:
    """The line under `HELD_OUT_HEADING`: the drift the held-out verdict reads as a response, and whether the mock
    above took it out (a block with its Welch result) or could not (a refusal, an error)."""
    return HELD_OUT_READ if "welch" in (block or {}) else HELD_OUT_UNREAD


def lead_sentences(result: dict) -> list:
    """The sentences a legend draft and the notes put FIRST, in the card's order: the mock named and its block, then
    the held-out verdict's heading and caution; ``[]`` without a mock (the export then reads as before)."""
    if not leads(result.get("text") or {}):
        return []
    block = result.get("mock") or {}
    return [*export_lines(block, str(block.get("mock", ""))), HELD_OUT_HEADING + ".", held_out_caution(block)]


# --------------------------------------------------------------------------- #
# The card says so when no mock was chosen (item 4)
# --------------------------------------------------------------------------- #

#: The line a card carries when a design with baseline movies has no mock chosen (8 Nov 2026 measurements, MSD project).
NO_MOCK_LINE = ("Without a mock addition, a drift over the session reads as a response: in measured sessions "
                "with nothing added, mobility fell by tens of per cent per hour. Choose a mock condition in Run "
                "to read the change against it.")

#: The designs that image baseline movies before an addition and are read against time in the session.
DESIGNS_NEEDING_A_MOCK = ("culture", "sample")


def no_mock_line(design: str | None, has_baseline: bool, mock_chosen: bool) -> list:
    """``[NO_MOCK_LINE]`` for the designs "Added to the culture" / "Added to each sample" with baseline movies before
    the addition and no mock chosen; else ``[]``."""
    return [NO_MOCK_LINE] if (design in DESIGNS_NEEDING_A_MOCK and has_baseline and not mock_chosen) else []


def attach_no_mock(result: dict) -> dict:
    """``result`` with `NO_MOCK_LINE` among the card's flags where it applies (a populations result of the design
    "Added to the culture" / "Added to each sample" that holds baseline movies); else ``result`` as it is. The Run calls
    it only when NO mock was chosen, so a mock condition itself, in a Run that has one, says nothing."""
    if "error" in result or result.get("readout") != "populations":
        return result
    line = no_mock_line((result.get("settings") or {}).get("design"), bool(result.get("has_baseline")), False)
    if not line:
        return result
    text = result["text"]
    return dict(result, text=dict(text, flags=list(text["flags"]) + line))
