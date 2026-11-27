"""The Metadata panel's receipt: a section's settings as a scientific print-out.

Asked for by the UI designer: the Spatial Map's 17-19-row settings list sat under every grid and
"screams for attention" on a page meant to be quiet, while most readers need it rarely.
It moved into a toolbar tool, **Metadata**, laid out like the receipt an instrument
prints -- a stamp, ruled sections, ``label . . . . VALUE`` lines and a total -- in IBM
Plex Mono. The prototype the layout was chosen from is the UI designer's
``notes/metadata-panel.html``.

**Nothing is recomputed here.** A section hands over the (name, value) rows it already
builds (the Spatial Map's are `spatial_map_view.matrix_caption_rows`) and this module
only arranges them: which heading a row goes under, the short value shown on the line
against its longer explanation (shown on hover), and the plain text Copy puts on the
clipboard. The widget (`ui.workspace.metadata`) paints this payload and decides nothing.

**A receipt line is SHORT, as the design is** ("I think you included way more
information compared to the design"): a section may hand a ``short(name, value)`` that
gives the line its own label and value -- ``Min. jumps/bin``, ``h 50 nm · σ 19 nm``,
``1 PER CONDITION`` -- and otherwise a value is cut at its first " · " ("OFF · every bin
drawn" prints ``OFF``). Whatever the line leaves out is its hover DETAIL. The full text is
never lost: each line keeps its row's own name and full value, the copied text carries
every detail, and a test rebuilds the original rows from a receipt exactly.

Qt-free: imported with Qt blocked, like the other ``*_view`` modules.
"""

from __future__ import annotations

__all__ = ["COMPUTATION", "RENDERING", "SPLIT", "NOT_RUN_TEXT", "NO_SECTION_TEXT",
           "split_value", "receipt_from_rows", "receipt_rows", "receipt_text"]

COMPUTATION = "COMPUTATION"
RENDERING = "RENDERING"
#: A RESULT a section can file its own rows under (the Apparent-D split, per condition):
#: not a setting, so neither heading above. Extra headings sit between the two.
SPLIT = "SPLIT"
#: The empty states: a section that has not run, and one with no receipt yet.
NOT_RUN_TEXT = "Run the map to print its receipt."
NO_SECTION_TEXT = "No metadata for this section yet."
#: The row that becomes the footer.
TOTAL_NAME = "Total"
#: The receipt's last line, as printed under the total.
FOOTER_TEXT = "* every value as computed *"
_SEP = " · "


def split_value(text: str) -> tuple[str, str]:
    """``(value, detail)``: the text before the first " · " and the rest ("" if none)."""
    head, sep, rest = str(text).partition(_SEP)
    return (head, rest) if sep else (str(text), "")


def _lines(name: str, value: str, short=None) -> list:
    """A row's receipt lines. Usually one; a ``short`` returning a LIST itemises the row,
    one line per entry ``(label, value, detail)`` -- the Apparent-D split, one line per
    condition -- and only the first carries the row (name, full), so nothing is counted
    twice when the rows are rebuilt."""
    picked = short(name, str(value)) if short is not None else None
    if isinstance(picked, list):
        out = []
        for i, entry in enumerate(picked):
            label, shown, detail = entry[:3]
            line = {"label": label, "value": shown, "detail": detail,
                    "name": name if i == 0 else None,
                    "full": str(value) if i == 0 else None}
            if len(entry) > 3:
                line["section"] = entry[3]
            out.append(line)
        return out
    return [_line(name, value, short)]


def _line(name: str, value: str, short=None) -> dict:
    """One receipt line: what it SHOWS (label, value), what hovering adds (detail), and the
    row it came from (name, full) -- the last two only so nothing is ever lost."""
    value = str(value)
    picked = short(name, value) if short is not None else None
    if picked is None:
        label, shown = name, split_value(value)[0]
    else:
        label, shown = picked
    if value.lower() == shown.lower():      # only the case changed: nothing to explain
        detail = ""
    elif value.startswith(shown + _SEP):
        detail = value[len(shown) + len(_SEP):]
    else:
        detail = value
    return {"label": label, "value": shown, "detail": detail, "name": name, "full": value}


def receipt_from_rows(rows, *, stamp, rendering_names, short=None,
                      footer: str | None = FOOTER_TEXT) -> dict:
    """The receipt payload for ``rows`` (``(name, value)`` pairs, in display order).

    ``stamp`` is the list of header lines (already worded). Rows named in
    ``rendering_names`` go under RENDERING, the "Total" row becomes the footer, and
    everything else goes under COMPUTATION; order within a heading is the rows' own.
    A heading with no rows is left out.
    """
    rendering_names = set(rendering_names)
    groups: dict[str, list] = {COMPUTATION: []}
    total = None
    for name, value in rows:
        if name == TOTAL_NAME:
            total = _line(name, value, short)
            total["label"] = name.upper()
            continue
        default = RENDERING if name in rendering_names else COMPUTATION
        for line in _lines(name, value, short):
            groups.setdefault(line.pop("section", default), []).append(line)
    # COMPUTATION first, any heading a section named next (in order of first use), and
    # RENDERING last -- how a map is drawn comes after everything that decided it.
    order = [COMPUTATION] + [h for h in groups if h not in (COMPUTATION, RENDERING)] + [RENDERING]
    sections = [{"heading": h, "lines": groups[h]} for h in order if groups.get(h)]
    return {"stamp": [str(s) for s in stamp if s], "sections": sections, "total": total,
            "footer": footer or ""}


def receipt_rows(receipt: dict) -> list:
    """The ``(name, value)`` rows a receipt was built from, rebuilt -- the check that
    arranging them lost nothing. The order is the receipt's (headings, then total)."""
    out = [(line["name"], line["full"]) for section in receipt["sections"]
           for line in section["lines"] if line.get("name") is not None]
    if receipt.get("total"):
        out.append((receipt["total"]["name"], receipt["total"]["full"]))
    return out


def _wrap(text: str, width: int) -> list[str]:
    """Words onto lines of at most ``width`` characters (a longer word stands alone)."""
    lines, cur = [], ""
    for word in text.split(" "):
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        lines.append(cur)
    return lines or [""]


def _leader(label: str, value: str, width: int) -> list[str]:
    """``label . . . value`` at ``width`` characters, the value wrapping right-aligned."""
    room = max(8, width - len(label) - 3)
    parts = _wrap(value, room)
    first = parts[0]
    gap = width - len(label) - len(first)
    out = [f"{label} {'.' * max(1, gap - 2)} {first}"]
    out += [p.rjust(width) for p in parts[1:]]
    return out


def receipt_text(receipt: dict, width: int = 48) -> str:
    """The receipt as plain aligned text, for Copy: what a notebook or a methods
    section can hold. Every detail is printed (indented under its line), so the
    copy carries the full settings, not the abbreviated screen."""
    out = []
    for s in receipt["stamp"]:
        out.append(s.center(width).rstrip())
    out.append("=" * width)
    for section in receipt["sections"]:
        out.append("")
        out.append(section["heading"])
        out.append("-" * width)
        for line in section["lines"]:
            out += _leader(line["label"], line["value"], width)
            if line["detail"]:
                out += ["    " + d for d in _wrap(line["detail"], width - 4)]
    if receipt.get("total"):
        out.append("=" * width)
        total = receipt["total"]
        out += _leader(total["label"], total["value"], width)
        if total["detail"]:
            out += ["    " + d for d in _wrap(total["detail"], width - 4)]
        out.append("=" * width)
    if receipt.get("footer"):
        out.append("")
        out.append(receipt["footer"].center(width).rstrip())
    return "\n".join(out) + "\n"
