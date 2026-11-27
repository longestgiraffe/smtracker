"""THE FIGURE COMPOSER's model: which panels, in what order, how wide, how lettered.

Asked for: "composer can be a new tab. u run the functions and in composer u can put it
together and export". Headless, so the Composer page (`ui.workspace.composer_page`) is thin
Qt: it shows `rows`, forwards a click to one method here, and hands `build_inputs` to a
preview worker. The drawing is `figure_export.compose`.

A PANEL IS A SNAPSHOT. It is added from a chart's Export dialog ("Add to composer") with that
chart's payload AS IT WAS, and the dialog's words (axis titles, names, numbers / grid /
legend, the fits drawn, a jump figure's residual plots: `figure_export.PANEL_WORDS`). A
later re-run does not change a panel already added -- the composed figure must
not move under you because something else was recomputed -- and each row says when it was
added, so a stale panel can be seen and replaced. Journal, width, font, files, columns and
letters belong to the COMPOSITION, so every panel in one figure shares them.

A PANEL'S WORDS CAN BE EDITED (30 Oct 2026, asked: "in COMPOSER; edit button for single images -
for example if title u wanna edit, etc basically it gets u to the same page u would have on sole
export screen for this image to change stuff"): its row's Edit opens that chart's own Export
dialog on `edit_spec` -- the composition's journal, font and files, this panel's words, at the
width it is drawn in the figure -- and Apply to panel writes the dialog's words back with
`set_words`, to that panel only. Its payload (the data) never changes; the composition's settings
are not the panel's to change (the dialog greys them). `edit_refusal` names a panel whose kind has
no Export dialog (none today: `add` takes only exportable kinds).

THE ROWS ARE DEFINED, NOT DERIVED (asked for: "the width setting in the composer is a bit
useless, instead we need to define the grid (4x2, etc)", and "if images are not the same size
... a way to rescale the single images"). Rows (Auto or fixed) make the grid; each panel has a
WIDTH, a fraction of the figure's full width (`figure_export.PANEL_WIDTHS`: 1/4, 1/3, 1/2, 2/3,
3/4, Full), spans rows, and has a SIZE in its place (30-100 %), drawn again at that size so text
keeps its point size and a vector file stays vector. A panel added to a full fixed grid grows it
by the rows it needs and says so (`note`).

WIDTHS, NOT COLUMNS (3 Nov 2026, asked: "1 seems to be 1/2 of the full width of the page. 1.5 seems
to be 2/3, and 2 full width in normal export, while composer had 1 - 3 ... if i put something with
1.5 to composer from export, it automatically gets 1.5 there as well", "why not 1/3, 1/2, 2/3, 1",
"all scaling to the width of one full figure"): the Composer had a Columns grid and spans over it,
the Export dialog columns of the journal, so "1" named half a page in one and a cell in the other,
and a chart exported at 1.5 columns had no place. Both now name a fraction of the journal's full
width, the same fraction in both (`width_for`), and panels of 1/2, 1/3 and 2/3 share one figure.

Not persisted across sessions: a payload can hold objects (the SQD bubble data) that are
not plain JSON, and a figure rebuilt from yesterday's numbers would be a figure of a run
nobody has open.

THE COMPOSITION FILE (30 Oct 2026, asked: "if a composer file is saved with export and u could open
it with same settings if u just wanna change small things?"): ``<name>.composer``, written beside
every figure the Composer exports (`figure_export.write_figure`'s ``composition``; Export every
figure's composed figure too) and by Save composition, holds the figure WHOLE: every panel in its
order (kind, its payload as drawn, words, title, when added, width, rows, size, snap), the figure's
own settings (journal, width, height, font, files, resolution: `FigureSpec.record`), rows and
letters, the build that wrote it and a format version (2 since 3 Nov 2026; a version 1 file's spans
over its columns are read as the widths they drew, `figure_export.legacy_width`). It is the session file's route
(`session_file.write_bundle` / `read_bundle`: a zip of a JSON manifest and a pickle read back by the
RESTRICTED reader, which refuses anything but plain types, numpy and SMTracker's own dataclasses), and
a file is read back before it is kept (``check``), so one that could not be opened is never written.
`read_composition` builds the WHOLE composition before anything uses it: a file of another format or
a newer version, a corrupt one, or one naming a kind this build cannot draw is refused by name
(`ComposerError`) and the open Composer is left as it was. It needs no project: the data are in it --
which is why it is shared like the figure's source data, and opened only if you made it (MANUAL).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from . import figure_export as fx

__all__ = ["COMPOSITION_FORMAT", "COMPOSITION_SUFFIX", "COMPOSITION_VERSION", "Composition", "ComposerError",
           "Panel", "read_composition", "width_for", "width_tip", "write_composition"]

COMPOSITION_FORMAT = "smtracker-composition"
#: 2 since 3 Nov 2026: each panel a ``width`` (a fraction of the figure), no ``columns``; version 1 is still read.
COMPOSITION_VERSION = 2
COMPOSITION_SUFFIX = ".composer"


class ComposerError(ValueError):
    """A request the composition cannot honour, named."""


def width_for(spec: fx.FigureSpec) -> str:
    """The width a chart takes when its Export dialog adds it (a key of `figure_export.PANEL_WIDTHS`), from the width
    set there (3 Nov 2026, asked: "when i add something as 2 columns to composer it should have 2 columns there
    automatically", then "if i put something with 1.5 to composer from export, it automatically gets 1.5 there"):
    the SAME fraction of the full width (1/4, 1/3, 1/2, 2/3, 3/4, Full); a custom width the fraction nearest its share of the
    journal's full width (`figure_export.fraction_mm` turned round). Changed in the Composer as before."""
    if spec.width in fx.WIDTH_FRACTIONS:
        return fx.WIDTH_FRACTIONS[spec.width]
    full = float(spec.preset_values["double_mm"])
    gap, _gap_y = fx.compose_gutters(spec)
    share = (float(spec.width_mm or full) + gap) / (full + gap)
    return min(fx.PANEL_WIDTHS, key=lambda w: abs(float(fx.width_fraction(w)) - share))


def width_tip(spec: fx.FigureSpec) -> str:
    """What a panel's width pills say on pointing: every fraction in mm at ``spec``'s full width."""
    gap, _gap_y = fx.compose_gutters(spec)
    full = spec.width_value()
    sizes = ", ".join(f"{fx.PANEL_WIDTH_LABELS[w]} {fx.fraction_mm(w, full, gap):.0f} mm" for w in fx.PANEL_WIDTHS)
    return (f"Its width, a fraction of the figure's full {full:g} mm (the {gap:g} mm gaps between panels "
            f"counted): {sizes}. A row takes panels until the next would pass the full width.")


def _normal_words(words: dict) -> dict:
    """A panel's words as a spec takes them: the condition names as pairs of a tuple."""
    out = dict(words or {})
    if "names" in out:
        out["names"] = tuple(tuple(pair) for pair in out["names"] or ())
    return out


@dataclass
class Panel:
    uid: int
    kind: str
    payload: dict
    #: The Export dialog's words for this chart (`figure_export.PANEL_WORDS`).
    words: dict
    title: str
    #: When it was added, "HH:MM" -- a snapshot says when it was taken.
    added: str
    #: Its width, a fraction of the figure's full width (`figure_export.PANEL_WIDTHS`).
    width: str = "1/2"
    #: Rows spanned.
    row_span: int = 1
    #: Its size inside its cell, per cent (`figure_export.PANEL_SCALE`).
    scale: int = 100
    #: ``("height" | "width", uid of another panel)``: snapped to that panel's plot or image
    #: area instead of a size (asked for: "snap A to height of B, or width of C").
    match: tuple | None = None

    def record(self) -> dict:
        """The shape `figure_export.compose` draws (a match by uid; `build_inputs` turns
        it into the other panel's place in the list), with when it was added (a composition
        file keeps it; the drawing ignores it)."""
        return {"kind": self.kind, "payload": self.payload, "words": dict(self.words),
                "title": self.title, "width": self.width, "row_span": self.row_span,
                "scale": self.scale, "match": self.match, "added": self.added}


class Composition:
    """The panels of one composed figure, in reading order."""

    def __init__(self, *, rows=None, letters: str = "journal"):
        self.panels: list[Panel] = []
        self.rows_fixed = None
        self.letters = "journal"
        #: What the last change did that nobody asked for in words (a grid that grew).
        self.note = ""
        self._next = 1
        self.set_rows(rows)
        self.set_letters(letters)

    # -- what is in it ------------------------------------------------------------ #
    def add(self, kind: str, payload: dict, words: dict | None = None,
            title: str | None = None, *, width: str = "1/2", now: datetime | None = None) -> Panel:
        """``width``: its fraction of the figure (`width_for` the width its Export dialog was set to)."""
        if kind not in fx.KINDS:
            raise ComposerError(f"no exporter for {kind!r}, so it cannot be composed")
        width = _checked_width(width)
        words = {k: v for k, v in (words or {}).items() if k in fx.PANEL_WORDS}
        panel = Panel(uid=self._next, kind=kind, payload=payload, words=words,
                      title=title or fx.panel_title(kind, payload),
                      added=(now or datetime.now()).strftime("%H:%M"), width=width)
        self._next += 1
        self.panels.append(panel)
        self.note = ""
        self._fit_rows(f"panel {fx.panel_letter(len(self.panels) - 1, self.letters, 'nature') or len(self.panels)}")
        return panel

    def _places(self, rows):
        """``(places, n_rows)`` as `figure_export.compose_places` lays the panels out."""
        places, n_rows, _units = fx.compose_places([p.width for p in self.panels],
                                                   [p.row_span for p in self.panels], rows)
        return places, n_rows

    def _fit_rows(self, why: str) -> None:
        """A fixed grid too small for its panels grows by the rows they need, and says so
        (`note`); past the largest grid offered it becomes Auto."""
        if self.rows_fixed is None or not self.panels:
            return
        _places, needed = self._places(None)
        if needed <= self.rows_fixed:
            return
        grown = needed if needed in fx.COMPOSE_ROWS else None
        self.note = (f"The grid grew to {needed} rows for {why}." if grown else
                     f"The grid is set to Auto: {why} needs {needed} rows.")
        self.rows_fixed = grown

    def _index(self, uid: int) -> int:
        for i, panel in enumerate(self.panels):
            if panel.uid == uid:
                return i
        raise ComposerError(f"no panel {uid} in this composition")

    def remove(self, uid: int) -> None:
        del self.panels[self._index(uid)]
        for panel in self.panels:                      # a snap to it has nothing to snap to
            if panel.match and panel.match[1] == uid:
                panel.match = None

    def clear(self) -> None:
        self.panels.clear()

    # -- a panel's words, edited in its own Export dialog --------------------------- #
    def panel(self, uid: int) -> Panel:
        return self.panels[self._index(uid)]

    def edit_refusal(self, uid: int) -> str:
        """Why ``uid`` cannot be edited in an Export dialog, or "" (every exportable kind can)."""
        kind = self.panel(uid).kind
        if kind not in fx.KINDS:
            return (f"No Export dialog for {kind!r}: this panel cannot be edited, only removed "
                    f"and added again")
        return ""

    def panel_width(self, uid: int, spec: fx.FigureSpec) -> float:
        """How wide ``uid`` is drawn in the figure at ``spec``, mm: its fraction of the full width
        (`figure_export.fraction_mm`) at its size in it."""
        panel = self.panel(uid)
        gap, _gap_y = fx.compose_gutters(spec)
        lo, hi = fx.PANEL_SCALE
        return (fx.fraction_mm(panel.width, spec.width_value(), gap)
                * max(lo, min(hi, panel.scale)) / 100.0)

    def edit_spec(self, uid: int, spec: fx.FigureSpec) -> fx.FigureSpec:
        """What ``uid``'s Export dialog opens on: the COMPOSITION's journal, font and files
        (``spec``), this panel's words, at the width it is drawn in the figure (20-600 mm, what
        a figure may be), its height automatic. A title never typed stays None (automatic)."""
        words = _normal_words(self.panel(uid).words)
        width = max(20.0, min(600.0, round(self.panel_width(uid, spec), 1)))
        return fx.with_changes(spec, width="custom", width_mm=width, height_mm=None, **words)

    def set_words(self, uid: int, words: dict) -> None:
        """``uid``'s words as its Export dialog left them (`figure_export.panel_words`); only
        that panel changes, and only its words: the payload, its place and size stay."""
        panel = self.panel(uid)
        if not isinstance(words, dict):
            raise ComposerError(f"a panel's words are a dict, not {type(words).__name__}")
        panel.words = _normal_words({k: v for k, v in words.items() if k in fx.PANEL_WORDS})

    def move(self, uid: int, step: int) -> None:
        """Earlier (``step`` < 0) or later in the reading order; held at the ends."""
        i = self._index(uid)
        j = max(0, min(len(self.panels) - 1, i + int(step)))
        self.panels.insert(j, self.panels.pop(i))

    def set_width(self, uid: int, width: str) -> None:
        """``uid``'s width, a fraction of the figure (`figure_export.PANEL_WIDTHS`)."""
        self.panels[self._index(uid)].width = _checked_width(width)
        self.note = ""
        self._fit_rows("that width")

    def set_row_span(self, uid: int, rows: int) -> None:
        top = self.rows_fixed or max(fx.COMPOSE_ROWS[1:])
        self.panels[self._index(uid)].row_span = max(1, min(int(rows), top))
        self.note = ""
        self._fit_rows("that span")

    def set_match(self, uid: int, dimension: str | None = None, target: int | None = None):
        """Snap ``uid`` to ``target``'s height or width, or (``None``) back to its size. A
        snap to itself, or one closing a loop (b to a while a is snapped to b), is refused."""
        panel = self.panels[self._index(uid)]
        if dimension is None or target is None:
            panel.match = None
            return
        if dimension not in ("height", "width"):
            raise ComposerError(f"a panel matches a height or a width, not {dimension!r}")
        self._index(target)
        seen, step = {uid}, target
        while step is not None:
            if step in seen:
                raise ComposerError("that would snap these panels to each other in a loop; "
                                    "undo the other panel's match first")
            seen.add(step)
            nxt = self.panels[self._index(step)].match
            step = nxt[1] if nxt else None
        panel.match = (dimension, int(target))

    def match_options(self, uid: int, preset: str) -> list:
        """``[(key, words)]`` for one panel's Match choice: none, then the height and the
        width of every other panel by its letter."""
        out = [("", "No snap")]
        for i, other in enumerate(self.panels):
            if other.uid == uid:
                continue
            name = fx.panel_letter(i, self.letters, preset) or str(i + 1)
            out.append((f"height:{other.uid}", f"Height of {name}"))
            out.append((f"width:{other.uid}", f"Width of {name}"))
        return out

    def set_scale(self, uid: int, percent) -> None:
        lo, hi = fx.PANEL_SCALE
        self.panels[self._index(uid)].scale = int(round(max(lo, min(hi, float(percent)))))

    # -- how it is laid out -------------------------------------------------------- #
    def set_rows(self, rows) -> None:
        """A fixed number of rows, or None for as many as the panels need. Too few for the
        panels already placed is refused, with how many they take."""
        if rows not in fx.COMPOSE_ROWS:
            raise ComposerError(f"rows must be one of {fx.COMPOSE_ROWS}, got {rows!r}")
        if rows is not None and self.panels:
            _places, needed = self._places(None)
            if needed > rows:
                raise ComposerError(
                    f"these panels need {needed} rows at their widths; remove or narrow some first, or "
                    f"keep Auto")
        self.rows_fixed = rows
        for panel in self.panels:
            panel.row_span = min(panel.row_span, rows or panel.row_span)
        self.note = ""

    def n_rows(self) -> int:
        """The rows as drawn: a fixed grid's, or as many as the panels need."""
        if not self.panels:
            return self.rows_fixed or 0
        _places, n_rows = self._places(self.rows_fixed)
        return n_rows

    def set_letters(self, case: str) -> None:
        if case not in fx.LETTER_CASES:
            raise ComposerError(f"letters must be one of {fx.LETTER_CASES}, got {case!r}")
        self.letters = case

    def letter_of(self, uid: int, preset: str) -> str:
        return fx.panel_letter(self._index(uid), self.letters, preset)

    def rows(self, preset: str) -> list:
        """What the page lists, one plain dict per panel, in order."""
        out = []
        for i, panel in enumerate(self.panels):
            letter = fx.panel_letter(i, self.letters, preset)
            out.append({"uid": panel.uid, "letter": letter or str(i + 1),
                        "title": panel.title,
                        "detail": f"added {panel.added}",
                        "width": panel.width, "row_span": panel.row_span,
                        "scale": panel.scale,
                        "match": (f"{panel.match[0]}:{panel.match[1]}" if panel.match
                                  else ""),
                        # why its Edit is off ("" : it opens the panel's Export dialog)
                        "edit_why": self.edit_refusal(panel.uid),
                        "first": i == 0,
                        "last": i == len(self.panels) - 1})
        return out

    def summary(self, spec: fx.FigureSpec) -> str:
        """One line under the preview: the panels, their rows, and each width used in mm."""
        if not self.panels:
            return "No panels yet."
        n_rows = self.n_rows()
        gap, _gap_y = fx.compose_gutters(spec)
        used = [w for w in fx.PANEL_WIDTHS if any(p.width == w for p in self.panels)]
        widths = ", ".join(f"{fx.PANEL_WIDTH_LABELS[w]} {fx.fraction_mm(w, spec.width_value(), gap):.0f} mm"
                           for w in used)
        return (f"{len(self.panels)} panel{'s' if len(self.panels) != 1 else ''} in {n_rows} "
                f"row{'s' if n_rows != 1 else ''} · {widths} · {fx.PRESET_LABELS[spec.preset]}")

    # -- the figure ---------------------------------------------------------------- #
    def build_inputs(self, spec: fx.FigureSpec) -> tuple:
        """``(records, spec, rows, letters)``: a SNAPSHOT for a worker, so a panel
        added or moved while it draws cannot change the figure under it."""
        where = {p.uid: i for i, p in enumerate(self.panels)}
        records = []
        for panel in self.panels:
            record = panel.record()
            if panel.match and panel.match[1] in where:
                record["match"] = (panel.match[0], where[panel.match[1]])
            else:
                record["match"] = None
            records.append(record)
        return records, spec, self.rows_fixed, self.letters

    def build(self, spec: fx.FigureSpec) -> fx.Built:
        records, spec, rows, letters = self.build_inputs(spec)
        return fx.compose(records, spec, rows=rows, letters=letters)

    def write(self, path, spec: fx.FigureSpec, *, build: str = ""):
        """This composition, with the figure's settings ``spec``, as a composition file (`write_composition`)."""
        records, spec, rows, letters = self.build_inputs(spec)
        return write_composition(path, {"panels": records, "rows": rows, "letters": letters}, spec, build=build)


def _checked_width(width) -> str:
    """``width`` as a key of `figure_export.PANEL_WIDTHS`, or refused by name."""
    try:
        fx.width_fraction(width)
    except fx.FigureExportError as exc:
        raise ComposerError(str(exc)) from exc
    return str(width)


# -- the composition file ------------------------------------------------------------------------------------------
def composition_path(path):
    """``path`` with the composition file's suffix (``<name>.composer``)."""
    from pathlib import Path

    path = Path(path)
    return path if path.suffix.lower() == COMPOSITION_SUFFIX else path.with_name(path.name + COMPOSITION_SUFFIX)


def write_composition(path, composed: dict, spec: fx.FigureSpec, *, build: str = ""):
    """``composed`` -- ``{"panels": [`Panel.record`, match by place], "rows", "letters"}``, the
    Composer's snapshot (`Composition.build_inputs`; Export every figure's composed job) -- with the figure's
    settings ``spec``, written as ONE composition file at ``path`` (``.composer`` added when missing). Read back
    before it is kept: refused (`ComposerError`) naming what a composition file cannot carry. Returns the path."""
    from . import session_file

    panels = [dict(p) for p in composed["panels"]]
    state = {"panels": panels, "rows": composed.get("rows"), "letters": str(composed["letters"]),
             "spec": spec.record()}
    manifest = {"format": COMPOSITION_FORMAT, "version": COMPOSITION_VERSION,
                "saved_at": datetime.now().isoformat(timespec="seconds"), "build": build,
                "figure": {"preset": spec.preset, "width_mm": spec.width_value(), "height_mm": spec.height_mm,
                           "font": spec.font, "font_pt": spec.pt(), "rows": state["rows"],
                           "letters": state["letters"]},
                "panels": [{"place": i + 1, "kind": p.get("kind"), "title": p.get("title")}
                           for i, p in enumerate(panels)]}
    try:
        return session_file.write_bundle(composition_path(path), manifest, state, what="composition",
                                         check=True)
    except session_file.SessionError as exc:
        raise ComposerError(f"the composition file was not written: {exc}") from exc


def read_composition(path) -> tuple:
    """``(Composition, FigureSpec, manifest)`` from a composition file, built WHOLE before it is returned: the
    panels in their order with their words, places, sizes and snaps, the grid and letters, the figure's settings.
    A file of another format or a newer version, a corrupt one, or one this build cannot draw is refused by name
    (`ComposerError`); nothing is half-made."""
    from pathlib import Path

    from . import session_file

    name = Path(path).name
    try:
        manifest, state = session_file.read_bundle(path, fmt=COMPOSITION_FORMAT, version=COMPOSITION_VERSION,
                                                   what="composition", oldest=1)
    except session_file.SessionError as exc:
        raise ComposerError(str(exc)) from exc

    def refuse(why):
        raise ComposerError(f"{name}: {why}; the composition was not opened")

    if not isinstance(state, dict) or not isinstance(state.get("panels"), list):
        refuse("it holds no panels")
    # version 1 drew spans over a grid of columns: each span is read as the width it drew (`legacy_width`)
    legacy = int(manifest.get("version", COMPOSITION_VERSION)) < 2
    if legacy and state.get("columns") not in fx.COMPOSE_COLUMNS:
        refuse(f"{state.get('columns')!r} columns is not a grid this build draws")
    if state.get("rows") not in fx.COMPOSE_ROWS:
        refuse(f"{state.get('rows')!r} rows is not a grid this build draws")
    if state.get("letters") not in fx.LETTER_CASES:
        refuse(f"letters {state.get('letters')!r} are not ones this build knows")
    spec = fx.FigureSpec.from_dict(state.get("spec"))
    problem = spec.problem()
    if problem:
        refuse(f"its figure settings cannot be drawn ({problem})")
    records = state["panels"]
    c = Composition(letters=state["letters"])
    panels = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict) or not isinstance(rec.get("payload"), dict):
            refuse(f"panel {i + 1} is not a panel")
        kind = rec.get("kind")
        if kind not in fx.KINDS:
            refuse(f"panel {i + 1} is a {kind!r}, which this build cannot draw")
        words = {k: v for k, v in (rec.get("words") or {}).items() if k in fx.PANEL_WORDS}
        try:
            row_span, scale = int(rec.get("row_span", 1)), int(rec.get("scale", 100))
            width = (fx.legacy_width(int(rec.get("span", 1)), state["columns"]) if legacy
                     else _checked_width(rec.get("width")))
        except (TypeError, ValueError):
            refuse(f"panel {i + 1}'s place in the figure is not one this build reads")
        lo, hi = fx.PANEL_SCALE
        panels.append(Panel(uid=i + 1, kind=kind, payload=rec["payload"], words=_normal_words(words),
                            title=str(rec.get("title") or fx.panel_title(kind, rec["payload"])),
                            added=str(rec.get("added") or "before"),
                            width=width, row_span=max(1, row_span),
                            scale=max(lo, min(hi, scale))))
    for i, rec in enumerate(records):
        match = rec.get("match")
        if match:
            try:
                dimension, place = str(match[0]), int(match[1])
            except (TypeError, ValueError, IndexError):
                refuse(f"panel {i + 1}'s snap is not one this build reads")
            if dimension not in ("height", "width") or not 0 <= place < len(panels) or place == i:
                refuse(f"panel {i + 1} snaps to a panel the file does not hold")
            panels[i].match = (dimension, panels[place].uid)
    c.panels = panels
    c._next = len(panels) + 1
    c.rows_fixed = state["rows"]
    if c.rows_fixed is not None and panels:
        _places, needed = c._places(None)
        if needed > c.rows_fixed:
            refuse(f"its {len(panels)} panels need {needed} rows, not the {c.rows_fixed} it fixes")
    return c, spec, manifest
