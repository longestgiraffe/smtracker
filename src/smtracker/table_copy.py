"""Tables to the clipboard, ready to paste into Excel. Headless.

Asked 15 Oct 2026: "in sqd data tables, and other data tables. a copy table button where u copy the values to
directly paste in excel". Excel pastes TAB-separated lines into cells, so that is what is produced, with three
things done so the cells arrive as NUMBERS rather than text:

* the decimal mark is the system's (`decimal`): a German Excel reads "37.6" as text, or as a date;
* no thousands separator: the app writes "3,511 tracks", which a German Excel would read as 3.511;
* a displayed "38 ± 4%" becomes two columns, value and spread, the "%" moving into the heading; "38% [30–45]"
  becomes three (value, low, high); a dash (nothing to show) becomes an empty cell.

A table whose numbers are rounded for display can hand over its own full-precision rows instead
(`bubble_view.copy_rows`); `excel_text` writes a float with up to seven significant digits.
"""
from __future__ import annotations

import re

__all__ = ["excel_rows", "excel_text"]

_NUM = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
_PM = re.compile(rf"^\s*({_NUM})\s*±\s*({_NUM})\s*(%?)\s*$")
_CI = re.compile(rf"^\s*({_NUM})\s*(%?)\s*\[\s*({_NUM})\s*[–-]\s*({_NUM})\s*\]\s*$")
_ONE = re.compile(rf"^\s*({_NUM})\s*(%?)\s*$")
_DASHES = {"", "—", "–", "-", "n/a"}


def _kind(text: str):
    """``(kind, parts, percent)``: kind is pm / ci / num / dash / text."""
    if text.strip() in _DASHES:
        return "dash", (), False
    m = _PM.match(text)
    if m:
        return "pm", (m.group(1), m.group(2)), bool(m.group(3))
    m = _CI.match(text)
    if m:
        return "ci", (m.group(1), m.group(3), m.group(4)), bool(m.group(2))
    m = _ONE.match(text)
    if m:
        return "num", (m.group(1),), bool(m.group(2))
    return "text", (text,), False


def excel_rows(columns, rows) -> tuple[list, list]:
    """``(headings, rows)`` with every cell as ``(text, is_number)``, a column split where its cells are
    "value ± spread" or "value [low–high]". ``rows`` hold strings, or numbers (written as numbers)."""
    columns = [str(c) for c in columns]
    width = len(columns)
    rows = [list(r) + [""] * (width - len(r)) for r in rows]
    out_head: list = []
    out_rows: list = [[] for _ in rows]
    for j, heading in enumerate(columns):
        cells = [r[j] for r in rows]
        kinds = [("num", (repr(v),), False) if isinstance(v, (int, float)) and not isinstance(v, bool)
                 else _kind(str(v)) for v in cells]
        has_pm = any(k == "pm" for k, _p, _pc in kinds)
        has_ci = any(k == "ci" for k, _p, _pc in kinds)
        percent = any(pc for _k, _p, pc in kinds)
        unit = " (%)" if percent and "%" not in heading else ""
        # value, then "±" where any cell has a spread, then low / high where any has an interval
        out_head += ([heading + unit] + ([f"{heading} ±"] if has_pm else [])
                     + ([f"{heading} low", f"{heading} high"] if has_ci else []))
        empty = ("", False)
        for i, (cell, (kind, parts, _pc)) in enumerate(zip(cells, kinds)):
            if isinstance(cell, float):
                value, spread, interval = (f"{cell:.7g}", True), empty, [empty, empty]
            elif isinstance(cell, int) and not isinstance(cell, bool):
                value, spread, interval = (str(cell), True), empty, [empty, empty]
            elif kind in ("pm", "ci", "num"):
                nums = [(p.replace(",", ""), True) for p in parts]
                value = nums[0]
                spread = nums[1] if kind == "pm" else empty
                interval = nums[1:3] if kind == "ci" else [empty, empty]
            elif kind == "dash":
                value, spread, interval = empty, empty, [empty, empty]
            else:
                value, spread, interval = (str(cell), False), empty, [empty, empty]
            out_rows[i] += [value] + ([spread] if has_pm else []) + (interval if has_ci else [])
    return out_head, out_rows


def excel_text(columns, rows, *, decimal: str = ".") -> str:
    """The table as TAB-separated lines for the clipboard, numbers with ``decimal`` as their decimal mark."""
    head, body = excel_rows(columns, rows)

    def clean(text: str) -> str:
        return str(text).replace("\t", " ").replace("\r", " ").replace("\n", " ")

    lines = ["\t".join(clean(h) for h in head)]
    for row in body:
        lines.append("\t".join(clean(t.replace(".", decimal) if is_number else t) for t, is_number in row))
    return "\n".join(lines) + "\n"
