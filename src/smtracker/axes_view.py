"""Typed axis ranges: the ONE rule behind every chart's Axes button -- headless, plain values.

Asked 30 Oct 2026, of the Jump distances page's Axes popover: "the axes button is cool, we can use it in other plots
like sqd for y axis etc?". A chart's Axes button (`ui.widgets.axes_popover.AxesPopover`) shows one row of from / to
fields per axis the chart lets you set (`axis_rows`); the text is read here (`parse`), checked here (`check`), and
what a chart draws is its automatic range with the typed ends put in (`apply_range`). The rules, the same for every
chart:

* an empty field is AUTOMATIC: that end follows what is drawn, refitted at every change;
* a typed range STAYS until Auto (the popover's button, or every field emptied) -- through a change of what is shown
  and through a new Run of the same section -- unless the chart's unit or scale changes (the Jump distances y unit),
  where the owner drops it;
* a range is REFUSED BY NAME: "from" not below "to"; a log axis from or to at or below 0; an axis that cannot go below
  a floor (a count, a share, a jump distance) below it.

Which axes a chart offers (`CHART_AXES`, `axis_rows`): categorical axes never (the SQD and apparent-D bubbles' x, the
HMM comparison's conditions). In print a chart's typed range is the Export dialog's own y / x from-to (the window
fills `FigureSpec.y_min` ... from it), or, for a chart of several panels with one range each (the HMM comparison,
Kinetics' "Populations over time" y), the payload's ``"axes"``.

Invariants: no Qt; a row is ``{"key", "label", "log", "low"}``; a range is ``(from, to)`` with either end None
(automatic), or None for the whole axis.
"""
from __future__ import annotations

__all__ = ["CHART_AXES", "axis_rows", "number", "check_range", "check", "parse", "apply_range", "empty"]


def _row(key: str, label: str, *, log: bool = False, low: float | None = None) -> dict:
    return {"key": key, "label": label, "log": bool(log), "low": low}


#: The axes each chart offers: its rows in the popover's order. ``low`` is the floor an end may not go under (None:
#: anything); a log axis is above 0 in any case.
CHART_AXES = {
    "jumps": (_row("x", "x, jump distance (nm)", low=0.0), _row("y", "y", low=0.0)),
    "sqd_bubbles": (_row("y", "y, D (µm²/s)", log=True),),
    "sqd_crossed": (_row("x", "x, fraction (%)", low=0.0), _row("y", "y, D (µm²/s)", log=True)),
    "appd_bubbles": (_row("y", "y, apparent D (µm²/s)", log=True),),
    "appd_histogram": (_row("x", "x, apparent D (µm²/s)", log=True), _row("y", "y, tracks", low=0.0)),
    "msd": (_row("x", "x, lag time (s)", low=0.0), _row("y", "y, MSD (µm²)")),
    "clusters": (_row("x", "x, lag time (s)", low=0.0), _row("y", "y, MSD (µm²)")),
    "kinetics_populations": (_row("x", "x, time since the induction (s)"),
                             _row("D", "D panel, D (µm²/s), each population", low=0.0),
                             _row("share", "share panel, share (%)", low=0.0)),
}


def axis_rows(chart: str, *, unit: str | None = None, panels=()) -> list:
    """The popover's rows for ``chart`` (`CHART_AXES`): the Jump distances' y named in its current ``unit``; the HMM
    comparison's one row per panel, ``panels`` ``[(key, title, unit), ...]`` (each from 0)."""
    if chart == "hmm_comparison":
        return [_row(str(key), f"{title} ({u})", low=0.0) for key, title, u in panels]
    if chart not in CHART_AXES:
        raise ValueError(f"no typed axes for {chart!r}")
    rows = [dict(r) for r in CHART_AXES[chart]]
    if chart == "jumps" and unit:
        rows[1]["label"] = f"y ({unit})"
    return rows


def number(text, what: str):
    """A typed end: "" is None (automatic), "0,5" reads as 0.5, a minus sign as "-"; anything else is refused by
    name."""
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return float(raw.replace(",", ".").replace("−", "-"))
    except ValueError:
        raise ValueError(f"{what}: “{raw}” is not a number") from None


def check_range(pair, row: dict):
    """One axis's ``(from, to)`` checked against its ``row``; None when both ends are automatic. Refused by name: a
    log axis at or below 0, an end below the row's floor, from not below to."""
    if pair is None or (pair[0] is None and pair[1] is None):
        return None
    lo, hi = (None if v is None else float(v) for v in pair)
    name = row["key"]
    for v in (lo, hi):
        if v is None:
            continue
        if row.get("log") and v <= 0:
            raise ValueError(f"{row['label']} is a log axis: from and to must be above 0 ({v:g})")
        if row.get("low") is not None and v < row["low"]:
            raise ValueError(f"{row['label']} cannot start or end below {row['low']:g} ({v:g})")
    if lo is not None and hi is not None and lo >= hi:
        raise ValueError(f"{name} from must be below {name} to ({lo:g} is not below {hi:g})")
    return (lo, hi)


def check(rows, axes) -> dict:
    """Every row's range checked (`check_range`): ``{key: (from, to) or None}``."""
    axes = axes or {}
    return {r["key"]: check_range(axes.get(r["key"]), r) for r in rows}


def parse(rows, texts) -> dict:
    """The popover's fields, ``texts`` ``{(key, "from" | "to"): text}``, read and checked (`check`)."""
    axes = {}
    for r in rows:
        key = r["key"]
        axes[key] = (number(texts.get((key, "from"), ""), f"{key} from"),
                     number(texts.get((key, "to"), ""), f"{key} to"))
    return check(rows, axes)


def empty(axes) -> bool:
    """Whether nothing is typed (every axis automatic)."""
    return not any(pair is not None and (pair[0] is not None or pair[1] is not None)
                   for pair in (axes or {}).values())


def apply_range(lo: float, hi: float, pair, *, log: bool = False) -> tuple:
    """The range a chart draws: its automatic ``(lo, hi)`` with the typed ends of ``pair`` put in. A typed end
    beyond the automatic other end (from 5 typed alone over an automatic 3) moves that other end out of its way:
    a decade on a log axis, the span (at least 1) on a linear one."""
    if pair is None:
        return float(lo), float(hi)
    a = float(lo) if pair[0] is None else float(pair[0])
    b = float(hi) if pair[1] is None else float(pair[1])
    if a >= b:
        if pair[0] is None:
            a = b / 10.0 if log else b - max(abs(b), 1.0)
        else:
            b = a * 10.0 if log else a + max(abs(a), 1.0)
    return a, b
