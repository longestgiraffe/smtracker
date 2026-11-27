"""Text rules a chart obeys on screen AND on paper, written once.

The workspace paints its charts with Qt and the figure exporter draws them with
matplotlib; a rule written into each would drift (the condition colours already did,
three ways, across the manuscript scripts -- `figstyle.CONDITION_COLORS`). So the two
renderers hand their own text MEASURE to these functions and keep nothing of their own:

* `wrap_text` -- a condition name under its column wraps at ``_ . - /`` or a space, up to
  three lines, and past that is elided at the FRONT, so the ending that tells
  ``...mNG.d2.1`` from ``...mNG.d2.2`` is what stays;
* `shrink_to_fit` -- the one rule for a text wider than its room (asked 30 Oct 2026: "when something gets cut ->
  scale down text size"): set smaller in steps down to a floor, never below it; what still does not fit is WRAPPED
  in print (`figure_export._fit_texts`, at spaces: `SPACE_BREAKS`) and ELIDED on screen (`fit_title`,
  `elide_title`: a title's last " · " part kept whole, the full text the hover), never cut at an edge;
* `short_form` -- tried FIRST, before any of that (asked the same evening: "first try the SHORT form of a fit label
  ('1-pop' instead of '1 population') ... then shrink, then wrap"): a title too wide for its room is shown in its
  short form where it has one, then set smaller, then wrapped (print) or elided (screen);
* `readable_ink` -- white or near-black on a fill, by its WCAG relative luminance;
* `scale_number` / `scale_ends` -- the numbers at a colour bar's two ends, every bar's, on screen and in print;
* `log_ticks` -- the D axis's 1-2-5 ticks per decade, thinned to whole decades (then every second ...) where the axis
  spans more than its height can label (`log_tick_limit`), and `tick_label`, each tick a plain decimal;
* `D_AXIS_LABEL`, `MSD_AXIS_LABEL`, `LAG_AXIS_LABEL` -- the axis titles, the same on screen
  and in an exported figure (an exported figure once said "Apparent D" where the screen
  said "D": asked to match).

Pure: no Qt, no matplotlib.
"""
from __future__ import annotations

import math
import re

__all__ = ["UNIT_WORDS", "unit_word", "LABEL_MAX_LINES", "INK_DARK", "INK_LIGHT", "D_AXIS_LABEL", "MSD_AXIS_LABEL",
           "APPD_AXIS_LABEL", "LAG_AXIS_LABEL",
           "SLOW_PCT_LABEL", "SLOW_SHARE_CELL_LABEL", "SLOW_SHARE_LABEL", "D_LINEAR_LABEL", "TRACK_STEPS_LABEL", "TRIGGER_TIME_LABEL", "INDUCTION_TIME_LABEL", "induction_time_label", "SHARE_PCT_LABEL", "QC_TRACKS_LABEL", "QC_STEPS_LABEL", "QC_CELL_LENGTH_LABEL",
           "QC_NEIGHBOUR_LABEL", "QC_INSIDE_LABEL", "FLAG_COLOR", "wrap_text", "elide_front", "log_ticks", "readable_ink",
           "bubble_axis_range", "ZERO_D_BELOW", "zero_d_stand_in", "log_drawn", "zero_d_ticks", "JUMPS_COUNT_LABEL", "JUMP_DISTANCE_LABEL", "PEARSON_LABEL",
           "ECDF_RESIDUAL_LABEL", "SPACE_BREAKS", "TITLE_FLOOR_PX", "TITLE_JOIN", "shrink_to_fit", "elide_middle",
           "elide_title", "fit_title", "short_form", "scale_number", "scale_ends", "axis_title_lines", "fold_words"]

#: The word a reader sees for an internal grouping unit (asked 24 Oct 2026: "maybe we call it replicate instead of day
#: everywhere?"; "internal column day is fine"): the column, the keys and `day_map.json` keep ``day``; every sentence,
#: label and heading says replicate -- an independent culture, which two of can share a calendar day.
UNIT_WORDS = {"day": "replicate", "days": "replicates"}


def unit_word(unit) -> str:
    """The word shown for the grouping unit ``unit`` ("day" -> "replicate"; "movie", "cell" unchanged)."""
    return UNIT_WORDS.get(str(unit), str(unit))


#: A whole fold change as a word (the vocabulary rule: fold changes are SPELLED OUT, "about twofold", never "2-fold"
#: or "2x"; 4 Nov 2026: the fit warning read "under 2-fold apart", the group warning "(2.2-fold)", and 35
#: messages, tooltips and captions wrote "2.28x", "1.5x the median", "within 2x the floor").
_FOLD_WORDS = {2: "twofold", 3: "threefold", 4: "fourfold", 5: "fivefold", 6: "sixfold", 7: "sevenfold",
               8: "eightfold", 9: "ninefold", 10: "tenfold"}
#: The same before what it multiplies ("twice the median", "three times the spread"), where "twofold the" is no English.
_TIMES_WORDS = {2: "twice", 3: "three times", 4: "four times", 5: "five times", 6: "six times", 7: "seven times",
                8: "eight times", 9: "nine times", 10: "ten times"}


def fold_words(ratio, digits: int | None = None, *, of: bool = False) -> str:
    """A ratio as a user reads it: "twofold" for a whole number two to ten, else "1.5 times" (``digits`` decimals, or
    as short as it is), as QC words its own ("the longest median is 1.25 times the shortest"). ``of``: the ratio
    stands before what it multiplies, so a whole number reads "twice" / "three times" ("twice the median"). Never a
    digit before "-fold" or "x". Every sentence that says how far apart two values are, or by how much one scales
    another, reads it; a magnification ("100x objective") and a playback speed ("4×") are not ratios and keep their
    sign."""
    value = float(ratio)
    if math.isfinite(value) and value == round(value) and int(round(value)) in _FOLD_WORDS:
        return (_TIMES_WORDS if of else _FOLD_WORDS)[int(round(value))]
    return f"{value:.{digits}f} times" if digits is not None else f"{value:g} times"


LABEL_MAX_LINES = 3
_BREAKS = re.compile(r"(?<=[_.\-/\s])")
#: Break a TITLE at spaces, and after a hyphen between letters ("short-" / "axis position"): a condition name breaks
#: after ``_ . - /`` (`wrap_text`'s default), but a title broke "Probability (per 7.2 nm bin)" after "7." -- a number
#: cut in two.
SPACE_BREAKS = re.compile(r"(?<=\s)|(?<=[A-Za-z]-)(?=[A-Za-z])")
#: Break a sentence that names a FOLDER at spaces and after a path's separators too: a path has no spaces, so a
#: word-wrapped label cut it mid-word at its edge ("in C:\Users\name\AppData\Local\Temp\cache\s"; found by the
#: overnight tester, 4 Nov 2026, in the toasts that say where files went).
PATH_BREAKS = re.compile(r"(?<=[\s\\/])")
#: The smallest a chart's title is set on screen before it is elided (`fit_title`), in pixels: the card title's 16 px
#: down to the size of a muted caption, and no smaller.
TITLE_FLOOR_PX = 11
INK_DARK, INK_LIGHT = "#18181b", "#ffffff"
D_AXIS_LABEL = "D (µm²/s, log)"
MSD_AXIS_LABEL = "MSD (µm²)"
APPD_AXIS_LABEL = "apparent D (µm²/s, log)"
SLOW_PCT_LABEL = "slow fraction (%)"
#: The apparent-D section's Per cell chart and its track-length check.
SLOW_SHARE_CELL_LABEL = "slow share per cell (%)"
#: D on a linear axis (the MSD checks: one ensemble D per fit window or length class).
D_LINEAR_LABEL = "D (µm²/s)"
SLOW_SHARE_LABEL = "slow share (%)"
TRACK_STEPS_LABEL = "track length (steps)"
TRIGGER_TIME_LABEL = "time since the trigger (s)"
#: The Kinetics section's "Populations over time" (`induction_view`): its time axis and its share axis.
INDUCTION_TIME_LABEL = "time since the induction (s)"


def induction_time_label(unit: str = "s") -> str:
    """The time axis' title in the unit it is drawn in ("s" seconds, the default, or "min" minutes)."""
    if unit not in ("s", "min"):
        raise ValueError(f"time unit {unit!r}: choose 's' or 'min'")
    return f"time since the induction ({unit})"


SHARE_PCT_LABEL = "share (%)"
#: The QC Detail charts' axis titles (`qc_chart`), on screen and on paper.
QC_TRACKS_LABEL = "tracks"
QC_STEPS_LABEL = "steps per track"
QC_CELL_LENGTH_LABEL = "cell length (µm)"
QC_NEIGHBOUR_LABEL = "% with a neighbor"
QC_INSIDE_LABEL = "% inside"
#: What a flag is drawn in: a QC unit unlike the rest, a fit that could not be made.
FLAG_COLOR = "#c05621"
LAG_AXIS_LABEL = "lag time (s)"
#: The SQD section's Jump distances charts (`jumps_chart`), on screen and on paper.
JUMPS_COUNT_LABEL = "Jumps per bin"
JUMP_DISTANCE_LABEL = "Jump distance (nm)"
PEARSON_LABEL = "Pearson residual"
ECDF_RESIDUAL_LABEL = "Model − ECDF"


def elide_front(text: str, measure, width: float) -> str:
    """``text`` shortened from the FRONT to fit ``width``, marked with an ellipsis."""
    if measure(text) <= width:
        return text
    for cut in range(1, len(text)):
        candidate = "…" + text[cut:]
        if measure(candidate) <= width:
            return candidate
    return "…"


def elide_middle(text: str, measure, width: float) -> str:
    """``text`` shortened in the MIDDLE to fit ``width``, marked with an ellipsis: a chart title keeps the start of its
    condition's name and the end that says which fit ("mNeonGreen, me…2 populations")."""
    text = str(text)
    if measure(text) <= width:
        return text
    for keep in range(len(text) - 1, 0, -1):
        head = text[:(keep + 1) // 2].rstrip()
        tail = text[len(text) - keep // 2:].lstrip() if keep // 2 else ""
        candidate = head + "…" + tail
        if measure(candidate) <= width:
            return candidate
    return "…"


def shrink_to_fit(text, measure, width: float, size: float, floor: float, step: float = 0.5) -> tuple:
    """``(size, fits)``: the largest size from ``size`` down to ``floor`` in ``step``s at which ``text`` is no wider
    than ``width`` -- the one rule for a text wider than its room (asked 30 Oct 2026: "when something gets cut ->
    scale down text size"). ``measure(text, size)`` is the renderer's own width of ``text`` at ``size`` (points in
    print, pixels on screen). A text that fits keeps ``size``; one that does not fit even at ``floor`` returns
    ``(floor, False)`` and its caller wraps it (print) or elides it (screen). Never above ``size``, never below
    ``floor`` (a ``floor`` above ``size`` is ``size``: a text already under the floor is never made larger)."""
    size, floor = float(size), min(float(floor), float(size))
    if width > 0 and measure(text, size) <= width:
        return size, True
    k = 1
    while True:
        s = round(size - k * step, 4)
        if s < floor + 1e-9:
            break
        if width > 0 and measure(text, s) <= width:
            return s, True
        k += 1
    return floor, width > 0 and measure(text, floor) <= width


#: What joins a chart title's parts ("Glucose · 2 populations", "Signed jumps · Glucose").
TITLE_JOIN = " · "

#: A FIT LABEL in a title: "1 population", "2 populations", "1 and 3 populations", "1, 2 and 3 populations".
_FIT_LABEL = re.compile(r"\b(\d+(?:(?:, | and )\d+)*) populations?\b")


def short_form(text) -> str:
    """``text`` with every fit label in its SHORT form -- "1 population" -> "1-pop", "2 populations" -> "2-pop", "1 and
    3 populations" -> "1 and 3-pop", the SQD section's own words for a fit (`bubble_view.model_label`) -- or ``text``
    unchanged where it has none. Asked 30 Oct 2026 (the Composer's one-plot-per-fit tiles wrapped "measured · 1" /
    "population"): a title too wide for its room tries this FIRST, then is set smaller, then wrapped or elided
    (`fit_title`, `figure_export._fit_one`) -- the experimenter's order: the short form reads as the same label, a
    smaller or broken title less well."""
    return _FIT_LABEL.sub(lambda m: f"{m.group(1)}-pop", str(text))


def elide_title(text: str, measure, width: float) -> str:
    """A title shortened to ``width``. A title of parts joined by " · " keeps its LAST part whole and shortens the
    ones before it at the FRONT, as a condition name is (`elide_front`: its end tells conditions apart) --
    "…, measured · 2 populations" beside "…, synthetic · 2 populations". Middle elision had made it
    "mNeonGreen,…populations", the fit count gone from a tile whose point it is; cut at the name's end, both tiles
    read "mNeonGreen, m… · 2 populations". One part, or a last part that alone leaves no room, is shortened in the
    middle (`elide_middle`)."""
    text = str(text)
    if measure(text) <= width:
        return text
    parts = text.split(TITLE_JOIN)
    if len(parts) > 1:
        head, tail = TITLE_JOIN.join(parts[:-1]), TITLE_JOIN + parts[-1]
        for cut in range(1, len(head)):
            candidate = "…" + head[cut:].lstrip() + tail
            if measure(candidate) <= width:
                return candidate
    return elide_middle(text, measure, width)


def fit_title(text, measure, width: float, size: float, floor: float = TITLE_FLOOR_PX, step: float = 1.0,
              elide=None) -> tuple:
    """A title on SCREEN in ``width``: ``(size, shown, elided)``. A title that fits is left whole; one that does not
    is shown in its SHORT form where it has one (`short_form`: "2 populations" -> "2-pop"), then `shrink_to_fit` down
    to ``floor``, and what still does not fit there is elided (`elide_title`, or ``elide(text, width, size)``).
    ``elided`` is True whenever the title is not shown whole (short or elided): the caller shows the whole text on
    hover. ``measure(text, size)`` as in `shrink_to_fit`."""
    text = str(text)
    if not text or (width > 0 and measure(text, size) <= width):
        return size, text, False
    base = short_form(text)
    size, fits = shrink_to_fit(base, measure, width, size, floor, step)
    if fits:
        return size, base, base != text
    shown = (elide(base, width, size) if elide is not None
             else elide_title(base, lambda s: measure(s, size), max(0.0, width)))
    return size, shown, shown != text


def elide_end(text: str, measure, width: float) -> str:
    """``text`` shortened from the END to fit ``width``, marked with an ellipsis ("Replicate…")."""
    text = str(text)
    if measure(text) <= width:
        return text
    for cut in range(len(text) - 1, 0, -1):
        candidate = text[:cut].rstrip() + "…"
        if measure(candidate) <= width:
            return candidate
    return "…"


def name_lines(name, measure, width: float, max_lines: int = LABEL_MAX_LINES) -> list:
    """A condition's name under a narrow column: broken at its SPACES only (`SPACE_BREAKS`), each word kept whole; a
    word, or a last line, still wider than ``width`` is shortened at its END with an ellipsis, and the whole name is
    the column's hover. It was broken wherever a piece stopped fitting ("mNG" / ", D2" / "…ted", "Replicate" / "s"),
    fragments that name nothing (found 4 Nov 2026, rail review B, the HMM comparison at 1366 px)."""
    lines = wrap_text(name, measure, width, max_lines=max_lines, breaks=SPACE_BREAKS, cut_words=False,
                      elide=lambda s, w: elide_end(s, measure, w))
    return [elide_end(line, measure, width) for line in lines]


def wrap_text(text, measure, width: float, max_lines: int = LABEL_MAX_LINES,
              elide=None, breaks=None, cut_words: bool = True) -> list:
    """``text`` as at most ``max_lines`` lines no wider than ``width``.

    ``measure(s)`` is the renderer's own width of ``s``; ``elide(s, width)`` its own
    front-elision (default `elide_front`); ``breaks`` where a line may break (default after
    ``_ . - /`` or a space, a condition name's; `SPACE_BREAKS` for a title). ``cut_words``
    False keeps a word wider than ``width`` whole on a line of its own (a title: "Pea" /
    "rson" over a 6 mm plot read worse than "Pearson" a millimetre past it)."""
    text = str(text)
    if width <= 0 or measure(text) <= width:
        return [text]
    lines, line = [], ""
    for piece in (p for p in (breaks or _BREAKS).split(text) if p):
        if measure(line + piece) <= width:
            line += piece
            continue
        if line:
            lines.append(line)
        # One piece wider than the column on its own: cut it where it stops fitting.
        while cut_words and len(piece) > 1 and measure(piece) > width:
            cut = len(piece) - 1
            while cut > 1 and measure(piece[:cut]) > width:
                cut -= 1
            lines.append(piece[:cut])
            piece = piece[cut:]
        line = piece
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        tail = "".join(lines[max_lines - 1:])
        cut = (elide(tail, width) if elide is not None
               else elide_front(tail, measure, width))
        lines = lines[:max_lines - 1] + [cut]
    return [ln.rstrip() for ln in lines]


def axis_title_lines(text, measure, width: float) -> list:
    """A y-axis title as ONE line where it fits the plot's height ``width``, else TWO: broken before its unit in
    brackets where both halves fit ("Probability" / "(per 10 nm bin)", "median apparent D" / "(µm²/s)"), else at the
    spaces (`wrap_text`, words kept whole), the end elided only where two lines still do not hold it (3 Nov 2026,
    reported: on a Jump distances tile at Size S and QC's per-cell outliers the title ran past both ends of the plot,
    "robability (per 10 nm bin")."""
    text = str(text)
    if width <= 0 or measure(text) <= width:
        return [text]
    head, sep, rest = text.rpartition(" (")
    if sep and head and measure(head) <= width and measure("(" + rest) <= width:
        return [head, "(" + rest]
    lines = wrap_text(text, measure, width, max_lines=2, breaks=SPACE_BREAKS, cut_words=False,
                      elide=lambda s, w: elide_title(s, measure, w))
    return [ln if measure(ln) <= width else elide_title(ln, measure, width) for ln in lines]


#: THE MOST TICKS A LOG AXIS CARRIES (6 Nov 2026, reported with a screenshot: "fix the y axis labels when so many
#: numbers are shown. especially when exporting, impossible to see"). A per-spot fit puts the slow D at the floor,
#: 0.0001 um2/s, the axis then spans five decades, and 1, 2 and 5 in each gave sixteen labels. Up to this many the
#: 1-2-5 ticks stand as before (a typical 0.005-5 axis has ten); beyond it, or beyond what the axis's height can hold
#: (`log_tick_limit`), 1 and 3 per decade, then whole decades, then every second, third ... decade, 1 always among
#: them. More than `DENSE_DECADES_MAX` decades always read in whole decades: ten labels over five decades are still
#: too many to read.
LOG_TICKS_MAX = 10
DENSE_DECADES_MAX = 3.2


def log_tick_limit(length: float, label_height: float) -> int:
    """How many tick labels an axis ``length`` long holds with ``label_height`` each (one unit: pixels on screen,
    points in print), a label's height of air between two: at least 2, at most `LOG_TICKS_MAX`."""
    if label_height <= 0:
        return LOG_TICKS_MAX
    return max(2, min(LOG_TICKS_MAX, int(length / (2.0 * label_height))))


def log_ticks(lo: float, hi: float, max_ticks: int | None = None) -> list:
    """Ticks covering [lo, hi] on a log axis: the bubble chart's D axis, on screen and on paper, and every other log
    axis. Decade + {1, 2, 5} while there are at most ``max_ticks`` of them (`LOG_TICKS_MAX` when None), else {1, 3},
    else whole decades, else every second, third, fifth ... decade (anchored at 1) -- the densest set that fits; an
    axis over more than `DENSE_DECADES_MAX` decades starts at whole decades."""
    import math

    if lo <= 0 or hi <= 0 or hi <= lo:
        return [v for v in (lo, hi) if v > 0]
    limit = max(2, int(max_ticks if max_ticks is not None else LOG_TICKS_MAX))
    first, last = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    dense = math.log10(hi / lo) <= DENSE_DECADES_MAX

    def ticks_at(multiples, every: int) -> list:
        out = []
        for e in range(first, last + 1):
            if e % every:
                continue
            for m in multiples:
                v = m * (10.0 ** e)
                if lo * 0.999 <= v <= hi * 1.001:
                    out.append(v)
        return out

    chosen = None
    for multiples, every in (((1, 2, 5), 1), ((1, 3), 1), ((1,), 1), ((1,), 2), ((1,), 3), ((1,), 5), ((1,), 10)):
        if len(multiples) > 1 and not dense:
            continue                            # a wide axis: whole decades at most
        ticks = ticks_at(multiples, every)
        if not ticks:
            break                               # thinner still names nothing: keep the last set that named something
        chosen = ticks
        if len(ticks) <= limit:
            break
    return chosen or [lo, hi]


def tick_label(value: float) -> str:
    """A tick's number as a plain decimal, never in scientific notation ("0.00005", not "5e-05"; "20", "0.1")."""
    value = float(value)
    if value == 0:
        return "0"
    text = f"{value:.12f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else f"{value:g}"


#: A D OF 0 ON A LOG D AXIS: the immobile population of exact-zero jumps (`sqd`'s module head, 4 Nov 2026) has
#: D = 0, which no log axis holds. It is drawn this factor below the lowest positive D on the axis, at a tick that
#: reads "0" (`zero_d_ticks`); the ticks between it and the lowest positive D are left out, so the axis visibly
#: breaks there. Display only: every number, table and export keeps D = 0.
ZERO_D_BELOW = 10.0


def zero_d_stand_in(values) -> "float | None":
    """Where a D of 0 among ``values`` is drawn on a log axis (`ZERO_D_BELOW` under the lowest positive one), or
    None when no value is 0."""
    values = [float(v) for v in values]
    if not any(v == 0 for v in values):
        return None
    positive = [v for v in values if v > 0]
    return (min(positive) if positive else 0.1) / ZERO_D_BELOW


def log_drawn(value: float, stand_in: "float | None") -> float:
    """``value`` where a log D axis draws it: a D of 0 at ``stand_in`` (`zero_d_stand_in`), any other as it is."""
    return float(stand_in) if (stand_in is not None and float(value) == 0) else float(value)


def zero_d_ticks(ticks, stand_in: "float | None") -> list:
    """``[(value, label), ...]``: a log D axis's ticks, and with a D of 0 drawn at ``stand_in`` its "0" tick first,
    the ticks within half a decade over it left out (the break). Without one, every tick labelled as before."""
    if stand_in is None:
        return [(v, tick_label(v)) for v in ticks]
    gap = float(stand_in) * ZERO_D_BELOW ** 0.5
    return [(float(stand_in), "0"), *[(v, tick_label(v)) for v in ticks if v > gap]]


def bubble_axis_range(lo: float, hi: float, plot: float, top: float, bottom: float,
                      least: float = 1.6) -> tuple:
    """The D axis's ends for a bubble chart: ``lo`` and ``hi`` (the fitted Ds) padded so
    the largest bubble fits above the top one and a bubble plus the number printed UNDER
    it fits below the bottom one. ``plot``, ``top`` and ``bottom`` are in one unit (points
    in print, pixels on screen): a bubble is sized in that unit, so its room is found in it,
    never as a fixed factor (x1.6 clipped the top bubble of a short panel). Never less than
    ``least`` either way. The screen chart and the exported figure both call this, so the
    two leave the same room."""
    import math

    lo, hi = float(lo), float(hi)
    span = max(math.log10(hi / lo), 0.3)
    total = span / max(0.2, 1.0 - (top + bottom) / max(plot, 1e-9))
    pad_top = max(math.log10(least), top * total / max(plot, 1e-9))
    pad_bottom = max(math.log10(least), bottom * total / max(plot, 1e-9))
    return lo / 10.0 ** pad_bottom, hi * 10.0 ** pad_top


#: Where a bubble's number may stand, in the order tried (`place_numbers`): under its bubble (the convention), over
#: it only where under HITS something. NEVER BESIDE (1 Nov 2026, asked of the export: "the numbers look a bit random
#: moved arond cause it doesnt fit anymore"): a number to the right in one column, over in the next and under in a
#: third read as noise. Where a number is wider than its column the export sets every number to fit
#: (`figure_export._number_style`) instead of moving one aside.
NUMBER_SIDES = ("below", "above")


def place_numbers(bubbles, sizes, *, gap: float, reach: float, bounds=None) -> list:
    """Where each bubble's number stands, in one unit (pixels on screen, points in print) with y DOWN.

    ``bubbles`` are ``(x, y, r)``, ``sizes`` each number's ``(width, height)`` (0 wide: no number), ``gap`` between a
    bubble's edge and its number, ``reach`` how far the D ticks stand out past the edge (a number may not cross
    another bubble's), ``bounds`` the plot's ``(left, top, right, bottom)``.

    Asked for (1 Nov 2026): a bubble drawn over the number under its neighbour hid it ("83±0%" under a 17 % bubble of
    the same column). Each number stands UNDER its own bubble, and OVER it only where under would HIT another bubble,
    its D ticks, a number already placed or the plot's edge (`NUMBER_SIDES`; both axes keep a number's room past the
    outermost bubbles, so over the top one fits). "Default should be under, if text hits a bubble, only then it's
    moved" (asked the same evening, of a version that put the top bubble's number over it in every column holding
    several, and before that moved a number off a neighbour a margin away): another bubble is kept only the gap a
    number keeps from its own, so a number never stands nearer a neighbour than its own bubble. Where neither side is
    clear, the side less covered, with ``clear`` False: the painter backs that number in the page's colour, and every
    number is drawn after every bubble, so none is hidden. The screen chart and the exported figure both call this.

    Returns per bubble ``{"side", "x", "y", "rect": (left, top, right, bottom), "clear"}``, ``x``/``y`` the number's
    centre; ``side`` and ``rect`` None where there is no number.
    """
    import math

    bubbles = [(float(x), float(y), float(r)) for x, y, r in bubbles]
    margin = float(gap)
    tick = max(1.0, 0.5 * margin)

    def blockers(i):
        for j, (x, y, r) in enumerate(bubbles):
            if j != i:
                yield "circle", (x, y, r + margin)
                yield "rect", (x - r - reach, y - tick, x + r + reach, y + tick)

    def hits(kind, shape, px, py):
        if kind == "circle":
            return math.hypot(px - shape[0], py - shape[1]) < shape[2]
        return shape[0] < px < shape[2] and shape[1] < py < shape[3]

    def touches(kind, shape, rect):
        left, top, right, bottom = rect
        if kind == "circle":
            cx, cy, r = shape
            return math.hypot(cx - min(max(cx, left), right), cy - min(max(cy, top), bottom)) < r
        return left < shape[2] and shape[0] < right and top < shape[3] and shape[1] < bottom

    def covered(rect, shapes):
        """0 where ``rect`` is clear; else 1 + how many of its sample points are covered (or outside the plot)."""
        outside = bounds is not None and (rect[0] < bounds[0] or rect[1] < bounds[1]
                                          or rect[2] > bounds[2] or rect[3] > bounds[3])
        if not outside and not any(touches(k, s, rect) for k, s in shapes):
            return 0
        count = 1
        for a in range(9):
            for b in range(5):
                px = rect[0] + (rect[2] - rect[0]) * (a + 0.5) / 9.0
                py = rect[1] + (rect[3] - rect[1]) * (b + 0.5) / 5.0
                off = bounds is not None and not (bounds[0] <= px <= bounds[2] and bounds[1] <= py <= bounds[3])
                count += off or any(hits(k, s, px, py) for k, s in shapes)
        return count

    placed, out = [], []
    for i, ((x, y, r), (w, h)) in enumerate(zip(bubbles, sizes)):
        w, h = float(w), float(h)
        if w <= 0.0:
            out.append({"side": None, "x": x, "y": y, "rect": None, "clear": True})
            continue
        shapes = list(blockers(i)) + [("rect", p) for p in placed]
        best = None
        for side in NUMBER_SIDES:
            cx, cy = x, (y + r + gap + h / 2.0 if side == "below" else y - r - gap - h / 2.0)
            rect = (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
            cost = covered(rect, shapes)
            if best is None or cost < best[0]:
                best = (cost, side, cx, cy, rect)
            if cost == 0:
                break
        cost, side, cx, cy, rect = best
        placed.append((rect[0] - gap, rect[1] - gap, rect[2] + gap, rect[3] + gap))
        out.append({"side": side, "x": cx, "y": cy, "rect": rect, "clear": cost == 0})
    return out


def number_reach(bubble, place) -> tuple:
    """``(above, below)``: how far a bubble and its number (`place_numbers`) stand out over and under its centre, in
    the unit they were placed in -- where a significance bracket at that bubble starts."""
    x, y, r = bubble
    rect = place.get("rect") if place else None
    if rect is None:
        return float(r), float(r)
    return max(float(r), y - rect[1]), max(float(r), rect[3] - y)


def readable_ink(fill: str) -> str:
    """`INK_LIGHT` or `INK_DARK`, whichever reads on ``fill`` (``#rrggbb``)."""
    value = fill.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))

    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return INK_DARK if lum > 0.36 else INK_LIGHT


# --------------------------------------------------------------------------- #
# The numbers at a colour bar's ends
# --------------------------------------------------------------------------- #

#: Plain decimals between these magnitudes; exponent notation only outside them, where plain decimals would be a row
#: of zeros ("0.00000939") or of digits.
SCALE_PLAIN_FROM, SCALE_PLAIN_TO = 1e-5, 1e6


def _scale_exponent(value: float, digits: int) -> str:
    """``value`` in compact exponent notation, ``digits`` significant figures: "9.4e-6", "1.2e6"."""
    mantissa, exponent = f"{value:.{max(0, digits - 1)}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def scale_number(value, digits: int = 3) -> str:
    """A number at the end of a COLOUR BAR -- every colour bar, on screen and in print (asked 30 Oct 2026: a Spatial
    Map's bar read "9.39e-06" and "0.0105"; "end labels are formatted with sensible significant digits and no
    scientific notation for ordinary ranges"). 0 is "0"; ``digits`` significant figures in plain decimals, trailing
    zeros dropped ("0.0105", "34.2", "0.01"); from 1,000 a whole number with a thousands comma ("1,234"); under 0.001
    two figures ("0.000037"); exponent notation only where plain decimals would be a row of zeros or digits (under
    1e-5, from 1e6: "9.4e-6")."""
    v = float(value)
    if v == 0.0 or not math.isfinite(v):
        return "0" if v == 0.0 else str(v)
    a = abs(v)
    if a < SCALE_PLAIN_FROM or a >= SCALE_PLAIN_TO:
        return _scale_exponent(v, min(digits, 2))
    if a >= 1000.0:
        return f"{v:,.0f}"
    figures = digits if a >= 1e-3 else min(digits, 2)
    text = f"{v:.{max(0, figures - 1 - math.floor(math.log10(a)))}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def scale_ends(lo, hi, fits=None, *, percent: bool = False) -> tuple:
    """A colour bar's two end labels, ``(lo text, hi text)``: `scale_number` at three figures, and where ``fits(pair)``
    (the renderer's own test: do the two stand side by side under the bar?) says they do not, shorter in turn -- a
    share as a percentage (``percent``), two figures, exponent notation when that is shorter, one figure. Without
    ``fits``, the first. ONE formatter for every colour bar (the Spatial Map's, SDA's, Images & tracks' apparent-D),
    on screen and in print."""
    lo, hi = float(lo), float(hi)
    tries = [(scale_number(lo), scale_number(hi))]
    if percent:
        tries.append((f"{scale_number(lo * 100.0, 2)}%", f"{scale_number(hi * 100.0, 2)}%"))
    two = (scale_number(lo, 2), scale_number(hi, 2))
    tries.append(two)
    short = tuple(_scale_exponent(v, 2) if v != 0.0 and len(_scale_exponent(v, 2)) < len(t) else t
                  for v, t in zip((lo, hi), two))
    if short != two:
        tries.append(short)
    tries.append((scale_number(lo, 1), scale_number(hi, 1)))
    if fits is None:
        return tries[0]
    return next((t for t in tries if fits(t)), tries[-1])
