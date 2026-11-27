"""Neutral comparison figures: states by COLOUR + numeric D, never a word.

The hard rule for this whole comparison layer: **no figure prints an interpretive
word** ("bound", "free", "slow", "fast", "mobile", "immobile", "binding",
"diffusing"). A state is encoded only by its Okabe–Ito colour (from
``figstyle.STATE_COLORS``) and its numeric D (a legend entry reads
``"state 1) D = 0.06 µm²/s"``, states ordered slowest→fastest by D.
Interpretation is the user's, and it lives in their caption, never on the axes.

``FORBIDDEN_WORDS`` is the enforced list; ``assert_neutral(figure)`` walks every
text artist and raises if any appears, and the gating test calls it on each view.
Keeping the list here (next to the drawing) means the discipline travels with the
figures, not in a distant test file.

Each ``draw_*`` takes an assembled dict from ``smtracker.comparison`` and paints a
``Figure``. They compute nothing: a wrong number fails a comparison test, never
subtly on screen.
"""

from __future__ import annotations

from .chart_text import unit_word  # the word shown for a grouping unit

import numpy as np

from . import hittest as _hittest
from .figstyle import STATE_COLORS

__all__ = [
    "FORBIDDEN_WORDS",
    "assert_neutral",
    "assert_no_day_claim",
    "has_real_days",
    "state_label",
    "format_p",
    "draw_bubble",
    "draw_ridgeline",
    "draw_kinetics",
    "draw_fraction",
    "draw_forest",
    "draw_heatmap",
    "draw_saspt",
    "draw_spatial",
    "draw_sigma_sensitivity",
]


def format_p(p: float) -> str:
    """Neutral, never-zero p display. A permutation p is floored at 1/(n_perm+1),
    so rounding it to ``0.000`` would misread as an impossible exact zero, show
    ``"< 0.001"`` instead."""
    if p != p:                                    # NaN
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"

#: Interpretive words banned from every figure in the comparison layer.
FORBIDDEN_WORDS = (
    "bound", "free", "slow", "fast", "mobile", "immobile", "binding", "diffusing",
)

#: A stable colour per replicate day for the SuperPlot swarms (matches plots.py).
_DAY_COLORS = ["#3182ce", "#e53e3e", "#38a169", "#805ad5", "#dd6b20", "#319795"]
#: Neutral point colour when there is NO real replicate-day structure to encode.
_SINGLE_SESSION_COLOR = "#4a5568"


def _day_color(day, days):
    return _DAY_COLORS[list(days).index(day) % len(_DAY_COLORS)]


def has_real_days(days) -> bool:
    """Whether there are ≥2 distinct replicate days worth encoding by colour.

    A single session (one day) must NOT be coloured by day: the day-level
    statistics guard stops a NUMBER asserting replicate structure, and colouring
    per day is the FIGURE asserting it: the same violation in a different medium.
    So every per-day swarm asks this first and falls back to one neutral colour,
    a cell-unit caption, and no day legend when it is single-session.
    """
    return len({d for d in days if d is not None}) >= 2


def state_label(index: int, D: float) -> str:
    """The neutral legend label for a state: index + numeric D, no word."""
    return f"state {index}: D = {D:.3g} µm²/s"


def _state_color(index: int) -> str:
    return STATE_COLORS.get(index, STATE_COLORS[-1])


def assert_neutral(figure) -> None:
    """Raise if any text artist in ``figure`` contains a forbidden word.

    Case-insensitive substring match over every ``Text`` artist (titles, labels,
    tick labels, legend entries, annotations). This is the gate the whole task
    hangs on, so it is deliberately strict: a word inside a longer token still
    trips it, which is the safe direction.

    When this fires on a legitimate use, rename: do not add an exception.
    It has already happened once: an occupancy annotation read "CI arm at bound",
    where "bound" was mathematically ordinary (the [0, 1] range limit) and had
    nothing to do with a molecular state. The fix was to rename it "range limit",
    not to whitelist the phrase. Exceptions accumulate, and each one erodes the
    gate until it stops catching the thing it was built for; a rename costs one
    line and leaves the gate at full strength. Same rule for field names that end
    up pasted into captions (hence ``occupancy_ci_at_limit``, not ``..._at_bound``).
    """
    import matplotlib

    offenders = []
    for ax in figure.get_axes():
        # matplotlib keeps THREE title artists. ``ax.title`` is only the CENTRED
        # one; loc="left"/"right" titles live in separate artists that are not in
        # ``ax.texts`` either, so a gate reading ``ax.title`` alone is blind to
        # every left-aligned panel title, which is how nearly every panel here is
        # titled. Read all three through the public accessor.
        texts = [ax.xaxis.label, ax.yaxis.label]
        titles = [ax.get_title(loc=loc) for loc in ("left", "center", "right")]
        texts += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        texts += [t for t in ax.texts]
        legend = ax.get_legend()
        if legend is not None:
            texts += list(legend.get_texts())
            # get_texts() returns the ENTRY labels only: the legend's own title is a
            # separate artist, and it is exactly where a heading like "bound states"
            # would sit. The four sibling copies of this gate already read it.
            if legend.get_title() is not None:
                texts.append(legend.get_title())
        for artist in texts:
            if not isinstance(artist, matplotlib.text.Text):
                continue
            content = artist.get_text().lower()
            for word in FORBIDDEN_WORDS:
                if word in content:
                    offenders.append((word, artist.get_text()))
        for title in titles:
            for word in FORBIDDEN_WORDS:
                if word in title.lower():
                    offenders.append((word, title))
    # Figure-level texts (suptitle, figure.text captions).
    for artist in figure.texts:
        content = artist.get_text().lower()
        for word in FORBIDDEN_WORDS:
            if word in content:
                offenders.append((word, artist.get_text()))
    if offenders:
        raise AssertionError(
            "figure prints interpretive word(s): "
            + "; ".join(f"{w!r} in {t!r}" for w, t in offenders))


#: Phrases where "day" is allowed: they DENY replicate structure, not assert it.
_DAY_OK_PHRASES = ("single session", "single-session", "single imaging", "1 replicate",
                   "one replicate", "no replicate")


def assert_no_day_claim(figure) -> None:
    """Raise if a figure asserts a replicate-DAY structure, in TEXT or in COLOUR.

    The aggregate guard stops a NUMBER claiming days on single-session data; this
    stops the FIGURE claiming them, which is the same violation in a different
    medium. Two channels are checked:

    - **Text**, every title / label / tick label / legend entry / annotation /
      figure caption. The word "day" trips it unless the surrounding text is one
      of ``_DAY_OK_PHRASES`` (which deny day structure, e.g. "single session").
    - **Colour mapping**, a per-day swarm draws one scatter collection per day,
      each in a distinct ``_DAY_COLORS`` hue. Using ≥2 of those palette colours is
      a per-day encoding, so it is refused too; a single-session figure must use
      one neutral colour.

    Meant to VERIFY a single-session figure (tests + the preliminary export). A
    genuinely multi-day figure legitimately encodes days and would not call this.
    """
    from matplotlib.colors import to_rgba

    def _strings():
        for ax in figure.get_axes():
            # all THREE title positions, loc='left' titles live outside ax.title
            for loc in ("left", "center", "right"):
                yield ax.get_title(loc=loc)
            yield ax.get_xlabel()
            yield ax.get_ylabel()
            for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                yield t.get_text()
            for t in ax.texts:
                yield t.get_text()
            legend = ax.get_legend()
            if legend is not None:
                for t in legend.get_texts():
                    yield t.get_text()
        for t in figure.texts:
            yield t.get_text()

    for raw in _strings():
        content = (raw or "").lower()
        # "replicate" is the word shown since 24 Oct 2026 (`chart_text.unit_word`); "day" is still caught
        if ("day" in content or "replicate" in content) and not any(p in content for p in _DAY_OK_PHRASES):
            raise AssertionError(
                f"figure asserts a replicate structure in text: {raw!r}")

    palette = {tuple(np.round(to_rgba(c), 3)) for c in _DAY_COLORS}
    used = set()
    for ax in figure.get_axes():
        for coll in ax.collections:
            for fc in coll.get_facecolors():
                key = tuple(np.round(to_rgba(fc), 3))
                if key in palette:
                    used.add(key)
    if len(used) >= 2:
        raise AssertionError(
            f"figure encodes ≥2 replicate colours ({len(used)} of the replicate "
            "palette): a per-replicate colour mapping on data that must not assert replicates")


# --------------------------------------------------------------------------- #
# SQD bubble plot
# --------------------------------------------------------------------------- #


def focal_pop(data: dict) -> int:
    """The FOCAL state (Item 6c): the SLOWEST state (lowest D), PRE-NOMINATED,
    chosen by a data-independent rule (not by which state differs most), so it
    carries no post-hoc selection bias. 1-based; 0 if nothing is fit. Under global
    scope D is shared, so this is the same state in every condition."""
    for col in data["columns"]:
        if getattr(col, "D", None):
            return int(np.argmin(col.D)) + 1            # D is slowest→fastest
    return 0


def bubble_table_rows(data: dict) -> list:
    """The bubble plot's side-table rows, ONE source of truth for the drawn table
    AND any export (Item 4). Each row is a dict built from the SAME BubbleColumn the
    bubbles are drawn from, so the table can never disagree with the plot.

    Columns: condition, population (1-based index), D, fraction_mean, fraction_sd,
    n_replicates, per_replicate_fractions, scope, focal. NEUTRAL: population by index.
    """
    scope = data.get("d_sharing_scope", "condition")
    focal = focal_pop(data)
    rows = []
    for col in data["columns"]:
        if not col.D:
            continue
        means = col.fraction_mean or col.fractions
        sds = col.fraction_sd or [0.0] * len(col.D)
        n_rep = getattr(col, "n_replicates", 1)
        per_rep = getattr(col, "per_replicate_fractions", None)
        for k, D in enumerate(col.D):                    # k is 0-based population
            per_rep_k = ([round(pr[k], 3) for pr in per_rep] if per_rep else None)
            rows.append({
                "condition": col.condition, "pop": k + 1, "D": float(D),
                "fraction_mean": float(means[k]),
                "fraction_sd": float(sds[k]) if k < len(sds) else 0.0,
                "n_replicates": int(n_rep), "focal": bool(k + 1 == focal),
                "per_replicate_fractions": per_rep_k, "scope": scope})
    return rows


def _sel_matches(selection, condition, pop) -> bool:
    """Does a Selection/(condition, pop) name this (condition, pop=1-based)?"""
    if selection is None:
        return False
    sc = getattr(selection, "condition", selection[0] if isinstance(selection, tuple)
                 else None)
    sp = getattr(selection, "pop", selection[1] if isinstance(selection, tuple)
                 and len(selection) > 1 else None)
    if sc != condition:
        return False
    return sp is None or sp == pop                       # pop=None highlights the column


def draw_bubble(figure, data: dict, highlight=None) -> "object":
    """Bubble plot + side data table (Item 4): x = condition, y = D (log), area ∝
    mean fraction, colour = state. The table beside it lists every drawn bubble
    (condition, population, D, fraction mean ± SD, n replicates, per-replicate
    fractions, scope): built from ``bubble_table_rows`` so it is the SAME numbers.
    Both export together (one figure) and share the figure's provenance.

    ``highlight`` (a Selection or ``(condition, pop)``) rings the matching bubble
    and shades its table row; it and the bubble use ONE combined hit-test so a
    click on either resolves to the same identity. ``data`` is
    ``comparison.sqd_bubbles(...)``.
    """
    figure.clear()
    columns = data["columns"]
    gridspec = figure.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.05)
    ax = figure.add_subplot(gridspec[0, 0])
    tax = figure.add_subplot(gridspec[0, 1])
    xs = np.arange(len(columns))
    column_pops = []                                     # per column: [(pop, D), …]
    focal = focal_pop(data)                              # slowest state, pre-nominated (6c)

    # D-AXIS CORRECTION (item C), default OFF. When on, EVERY state's D is mapped to
    # D_true = (3/2)·(D_app − σ²/τ) (all-or-none, never a subset); a corrected value
    # at/below the floor is no longer resolvable, so it renders as a BOUND, not a
    # point. The transform is affine, so CI/SD whisker endpoints map the same way.
    dc = data.get("d_correction") or {}
    correct_on = bool(dc.get("on"))
    floor = dc.get("floor_D")
    if correct_on:
        from .msd import berglund_correct_apparent_D
        _sig_um, _tau, _R = dc["sigma_nm"] / 1000.0, dc["tau_s"], dc["R"]

        def _corr(d):
            return float(berglund_correct_apparent_D(d, _tau, sigma_um=_sig_um, R=_R))
    else:
        def _corr(d):
            return float(d)

    for x, col in zip(xs, columns):
        if not col.D:                                    # a condition that failed to fit
            column_pops.append([])
            reason = getattr(col, "error", "not fit")
            ax.annotate(reason, (x, 1.0), fontsize=6.5, ha="center", va="center",
                        color="#c05621", rotation=90, wrap=True)
            continue
        means = col.fraction_mean or col.fractions
        sds = col.fraction_sd or [0.0] * len(col.D)
        n_rep = getattr(col, "n_replicates", 1)
        # hit-test on the DISPLAYED D (corrected or not) so a click maps to the dot
        column_pops.append([(k + 1, max(_corr(float(D)), 1e-6))
                            for k, D in enumerate(col.D)])
        for k, (D, frac) in enumerate(zip(col.D, means), start=1):
            Dc = _corr(D)
            is_bound = correct_on and floor is not None and Dc <= floor
            # a bound sits AT the floor (its true value is unresolvable, ≤ floor)
            Dplot = float(floor) if is_bound else max(Dc, 1e-6)
            size = 200 + 4000 * frac
            hot = _sel_matches(highlight, col.condition, k)
            is_focal = k == focal                        # the focal (slowest) state
            ax.scatter(x, Dplot, s=size, color=_state_color(k),
                       alpha=(0.7 if is_focal else 0.45),
                       edgecolors=("#e53e3e" if hot else "k"),
                       linewidths=(2.2 if hot else (1.4 if is_focal else 0.5)),
                       linestyle=("--" if is_bound else "-"),
                       zorder=(5 if hot else 3))
            # "X ± Y%": Y is the SD of per-replicate fractions (spread conditional
            # on the shared D̂), shown only when there are ≥2 replicates to spread over.
            sd = sds[k - 1] if k - 1 < len(sds) else 0.0
            label = (f"{100 * frac:.0f}±{100 * sd:.0f}%" if n_rep >= 2
                     else f"{100 * frac:.0f}%")
            ax.annotate(label, (x, Dplot), ha="center", va="center",
                        fontsize=7, zorder=6)
            if is_bound:                                 # mark it as a bound, not a point
                ax.annotate("≤ floor", (x, Dplot), xytext=(0, -11),
                            textcoords="offset points", ha="center", va="top",
                            fontsize=5.5, color="#c05621", zorder=6)
                ax.plot(x, Dplot, marker="v", ms=6, color="#c05621", zorder=6)
            # Under replicate scope D itself has a spread (per-replicate D); draw it
            # as a whisker. Under condition/global there is deliberately NO ± on D.
            # Endpoints map through the SAME correction so the whisker stays honest.
            if getattr(col, "D_sd", None) and col.D_sd[k - 1] > 0 and not is_bound:
                lo_d, hi_d = _corr(D - col.D_sd[k - 1]), _corr(D + col.D_sd[k - 1])
                ax.plot([x, x], [max(lo_d, 1e-6), max(hi_d, 1e-6)],
                        color=_state_color(k), lw=1.2, alpha=0.5, zorder=2)
            if col.D_ci and col.D_ci[k - 1] and not is_bound:
                lo, hi = col.D_ci[k - 1]
                ax.plot([x, x], [max(_corr(lo), 1e-6), max(_corr(hi), 1e-6)],
                        color=_state_color(k), lw=1.2, alpha=0.8, zorder=2)

    # FOCAL state marker (6c): the slowest state is the pre-nominated primary. A
    # faint band + a left-edge tag; the others are accompanying description.
    if focal and columns and getattr(columns[0], "D", None):
        fd = max(_corr(columns[0].D[focal - 1]), 1e-6)
        if correct_on and floor is not None and fd <= floor:
            fd = float(floor)
        ax.annotate("◄ focal state (lowest D, pre-nominated)", (-0.55, fd),
                    fontsize=6.5, va="center", ha="left", color="#2d3748",
                    annotation_clip=False)

    # THRESHOLD LINE REMOVED (Amendment 4): the tool computes no cross-condition
    # slow-fraction, so D < 0.15 has no line here. Instead the LOCALIZATION FLOOR
    # σ²/τ is drawn for REFERENCE: states at or below it are unresolved. Hidden
    # when the D axis is corrected (mixing corrected/apparent scales). floor_D in
    # the stamp is σ²/(4τ), so the physical resolution limit is 4× it.
    floor_phys = 4.0 * floor if (floor is not None) else None
    if floor_phys and not correct_on:
        ax.axhline(floor_phys, color="#a0aec0", lw=0.8, ls=":", zorder=1, alpha=0.8)
        ax.annotate(f"floor σ²/τ = {floor_phys:.3g}", (len(columns) - 0.5, floor_phys),
                    fontsize=6, va="bottom", ha="right", color="#a0aec0")

    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([col.condition for col in columns])
    ax.set_ylabel("D true (µm²/s, log): Berglund-corrected" if correct_on
                  else "D apparent (µm²/s, log)")
    ax.set_xlim(-0.6, len(columns) - 0.4)
    scope = data.get("d_sharing_scope", "global")
    n_pop = max((c.n_pop for c in columns if getattr(c, "n_pop", 0)), default=0)
    from .workspace_view import scope_words

    scope_clause = {
        "global": f": {scope_words('global')} ({n_pop}-state shared model)",
        "condition": f": {scope_words('condition')}",
        "replicate": f": {scope_words('replicate')}",
        "group": f": {scope_words('group', data.get('d_sharing_factor') or '')}",
    }.get(scope, "")
    ax.set_title(f"SQD states per condition{scope_clause}", fontsize=9, loc="left")
    _bubble_legend(ax, columns)

    # One clause on what the ± is, why D carries none, and the fraction weighting.
    n_rep = max((getattr(c, "n_replicates", 1) for c in columns), default=1)
    wt = data.get("fraction_weighting", "track")
    wt_note = ("  fraction = molecule fraction (track)" if wt == "track"
               else "  fraction = displacement/time fraction (step)")
    if scope == "replicate":
        note = ("area ∝ mean per-replicate fraction; ± = SD of per-replicate D "
                "and fraction across replicates")
    elif n_rep >= 2:
        note = (f"area ∝ mean fraction; ± = SD of the per-replicate fractions "
                f"(n = {n_rep} replicates, conditional on the shared D̂: "
                "excludes common-mode D̂ uncertainty). D is shared, so it has no ±.")
    else:
        note = ("area ∝ fraction (single replicate: no spread). D is shared "
                "across replicates, so it carries no ±.")
    note += wt_note
    # Item 6: the primary readout is PER-STATE (D + fraction per state); the focal
    # state is the slowest (pre-nominated, no post-hoc selection), the others are
    # accompanying description. The threshold is only a cross-method device.
    note += "  primary = per-state; focal = lowest-D state (pre-nominated)"
    # Item C: fence the D-axis correction in the caption, same register as the
    # fast-mode stamp, so a corrected figure can never pass as an apparent-D one.
    if correct_on:
        note += (f"  D CORRECTED: D_true=(3/2)(D_app−σ²/τ), σ={dc['sigma_nm']:.1f} nm "
                 f"({dc['sigma_source']}), R={dc['R']:.3g}; states ≤ floor "
                 f"({dc['floor_D']:.3g}) shown as ≤floor limits; threshold hidden "
                 "(apparent-D)")
    else:
        note += "  D = apparent (uncorrected; Berglund toggle OFF)"
    # The opt-in D whisker is a CELL bootstrap, hard-gated to single-session data
    # (Report A): a cell interval is the wrong unit on a day-level figure, so it is
    # structurally withheld there, and the reason is stated, never a silent drop.
    if any(getattr(c, "D_ci", None) for c in columns):
        note += "  D whisker = cell-level bootstrap CI (single-session)"
    if data.get("ci_gated_multiday"):
        note += "  cell CI withheld: replicate-level figure; use the per-replicate ± SD"
    figure.text(0.01, 0.005, note, fontsize=6.5, color="#4a5568", va="bottom")

    table_rows = _draw_bubble_table(tax, data, highlight)

    # ONE combined hit-test for the bubbles AND the table (Item 4).
    _hittest.attach(figure, _hittest.BubbleTableHitTest(
        conditions=[c.condition for c in columns], column_pops=column_pops,
        bubble_axes_id=id(ax), table_axes_id=id(tax),
        table_rows=tuple(table_rows)))
    # Stash the data so the tab can redraw with a highlight in place (no recompute).
    figure._smtracker_bubble_data = data                 # type: ignore[attr-defined]
    return ax


def _draw_bubble_table(tax, data, highlight):
    """Render the side table into ``tax`` and return the row hit-positions
    ``[(y_center, condition, pop), …]`` in the table axes' data coords."""
    tax.set_xlim(0, 1)
    tax.set_ylim(0, 1)
    tax.axis("off")
    rows = bubble_table_rows(data)
    if not rows:
        return []
    # The table exports WITH the figure, so its D column must be on the SAME scale
    # as the plotted axis: corrected too when the correction is on (never mix).
    dc = data.get("d_correction") or {}
    if dc.get("on"):
        from .msd import berglund_correct_apparent_D
        _s, _t, _R, _flr = dc["sigma_nm"] / 1000.0, dc["tau_s"], dc["R"], dc["floor_D"]
        for r in rows:
            dcorr = float(berglund_correct_apparent_D(r["D"], _t, sigma_um=_s, R=_R))
            r["D"] = max(dcorr, 0.0)
            r["d_bound"] = dcorr <= _flr
    dcol = "D≤flr" if dc.get("on") else "D"
    header = f"{'cond':>10} {'pop':>3} {dcol:>7} {'frac':>6} {'±SD':>5} {'n':>2}  {'per-rep':>14} {'':>6}"
    n = len(rows)
    top, bottom = 0.97, 0.03
    pitch = (top - bottom) / (n + 1)                     # +1 for the header row
    tax.text(0.0, top, header, fontsize=6.2, family="monospace", va="center",
             color="#2d3748")
    hit = []
    for i, r in enumerate(rows, start=1):
        y = top - i * pitch
        per = ("/".join(f"{v:.2f}" for v in r["per_replicate_fractions"])
               if r["per_replicate_fractions"] else "-")
        if len(per) > 14:
            per = per[:13] + "…"
        tag = "◄focal" if r.get("focal") else ""
        line = (f"{r['condition'][:10]:>10} {r['pop']:>3d} {r['D']:>7.3g} "
                f"{r['fraction_mean']:>6.2f} {r['fraction_sd']:>5.2f} "
                f"{r['n_replicates']:>2d}  {per:>14} {tag:>6}")
        hot = _sel_matches(highlight, r["condition"], r["pop"])
        if hot:                                          # shade the highlighted row
            tax.axhspan(y - pitch / 2, y + pitch / 2, color="#fed7d7", alpha=0.7,
                        zorder=0)
        tax.text(0.0, y, line, fontsize=6.0,
                 family="monospace", va="center",
                 color=("#c53030" if hot else "#4a5568"))
        hit.append((y, r["condition"], r["pop"]))
    tax.set_title("data table (exports with the figure)", fontsize=7.5, loc="left")
    return hit


def rehighlight(figure, selection) -> bool:
    """Redraw the bubble figure with ``selection`` highlighted, IN PLACE, from the
    data stashed on the figure: no recompute. Returns False if the figure is not a
    bubble figure (so a generic caller can try it harmlessly on any view)."""
    data = getattr(figure, "_smtracker_bubble_data", None)
    if data is None:
        return False
    draw_bubble(figure, data, highlight=selection)
    return True


def _bubble_legend(ax, columns):
    import matplotlib.lines as mlines

    seen = {}
    for col in columns:
        for k, D in enumerate(col.D, start=1):
            seen.setdefault(k, D)
    handles = [
        mlines.Line2D([], [], marker="o", linestyle="none",
                      color=_state_color(k), markeredgecolor="k",
                      label=state_label(k, D))
        for k, D in sorted(seen.items())
    ]
    if handles:
        ax.legend(handles=handles, fontsize=7, frameon=False, loc="best",
                  title="area ∝ fraction")


# --------------------------------------------------------------------------- #
# appD ridgeline / violin + per-cell SuperPlot swarm
# --------------------------------------------------------------------------- #


def draw_ridgeline(figure, data: dict) -> "object":
    """Per-condition violin of log10(apparent-D) + a per-cell SuperPlot swarm.

    ``data`` is ``comparison.appd_points(...)``. The violin shows the per-track
    distribution; the overlaid points are per-cell means coloured by replicate
    day (SuperPlot style), so the pseudoreplication structure is visible.
    """
    figure.clear()
    ax = figure.add_subplot(111)
    per_condition = data["per_condition"]
    conditions = list(per_condition)
    positions = np.arange(len(conditions))       # 0-based: uniform with drill-down

    violin_data = [per_condition[c]["track_log_appd"] for c in conditions]
    nonempty = [(p, v) for p, v in zip(positions, violin_data) if len(v)]
    if nonempty:
        parts = ax.violinplot([v for _, v in nonempty],
                              positions=[p for p, _ in nonempty],
                              showmeans=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("#a0aec0")
            body.set_alpha(0.35)

    all_days = [d for c in conditions for d in per_condition[c]["cells"].get(
        "day", [])]
    by_day = has_real_days(all_days)

    rng = np.random.default_rng(0)
    for pos, cond in zip(positions, conditions):
        cells = per_condition[cond]["cells"]
        if not len(cells):
            continue
        if by_day:
            days = sorted(cells["day"].unique())
            for day in days:
                sub = cells[cells["day"] == day]
                vals = sub["log_appd"].to_numpy(dtype=float)
                x = pos + rng.uniform(-0.18, 0.18, len(vals))
                ax.scatter(x, vals, s=22, color=_day_color(day, days), alpha=0.7,
                           edgecolors="none", zorder=3)
        else:
            vals = cells["log_appd"].to_numpy(dtype=float)
            x = pos + rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(x, vals, s=22, color=_SINGLE_SESSION_COLOR, alpha=0.65,
                       edgecolors="none", zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("log₁₀ apparent D (µm²/s)")
    point_phrase = ("points = per-cell means by replicate" if by_day
                    else "points = per-cell means (single session)")
    ax.set_title(f"Per-track apparent-D distribution; {point_phrase}",
                 fontsize=9, loc="left")
    _hittest.attach(figure, _hittest.ColumnHitTest(list(conditions)))
    return ax


# --------------------------------------------------------------------------- #
# HMM kinetics, occupancy + rates, unidentified rate as a hatched span
# --------------------------------------------------------------------------- #


def draw_kinetics(figure, data: dict) -> "object":
    """Grouped per condition: occupancy (state 1) + on/off rates with CIs.

    ``data`` is ``comparison.hmm_kinetics(...)``. A rate the data cannot pin
    renders as a hatched span over the full plausible range, terminating in NO
    arrowhead at either end, never a point with error bars, which would imply it
    was measured, and never an arrow, which would assert a direction. Colour
    encodes the two rate directions; no word appears.

    AMENDMENT 5. The rate out of the low-D state is **not a bound in either
    direction**: it is an apparent, censored rate with two opposing censoring
    mechanisms and an unestablished net sign. So nothing here calls it a bound,
    in a label, a docstring or a comment: the drawn quantity is the *identified
    end* of the interval where one is quoted at all, and the SPAN says the quantity
    is unidentified on this design. The geometry asserts nothing: an arrowhead,
    one or two: is read as a value, so there is none. ``tests/test_comparison.py`` asserts this module's
    source contains no directional-limit phrasing anywhere, because Amendment 5's
    scope includes code comments and the rendered-text gate cannot see those. The
    banned phrases are listed in that test, not here: naming them in this module
    would trip its own check.
    """
    figure.clear()
    rows = data["rows"]
    conditions = [r.condition for r in rows]
    xs = np.arange(len(rows))
    grid = figure.add_gridspec(1, 2, wspace=0.32)
    ax_occ = figure.add_subplot(grid[0, 0])
    ax_rate = figure.add_subplot(grid[0, 1])

    # occupancy of state 1 (the below-threshold state), colour = state 1
    ax_occ.bar(xs, [100 * r.occupancy_slow for r in rows],
               color=_state_color(1), width=0.6)
    # the occupancy CI comes from the SAME unit resamples as the rates. It is
    # asymmetric on purpose (a percentile interval near 0/1 must be), so it is drawn
    # as explicit lo/hi arms, never as a symmetric ±.
    ci_rows = [(x, r) for x, r in zip(xs, rows) if getattr(r, "occupancy_ci", None)]
    if ci_rows:
        cx = [x for x, _ in ci_rows]
        mid = np.array([100 * r.occupancy_slow for _, r in ci_rows])
        lo = np.array([100 * r.occupancy_ci[0] for _, r in ci_rows])
        hi = np.array([100 * r.occupancy_ci[1] for _, r in ci_rows])
        ax_occ.errorbar(cx, mid,
                        yerr=np.vstack([np.clip(mid - lo, 0, None),
                                        np.clip(hi - mid, 0, None)]),
                        fmt="none", ecolor="#333333", elinewidth=1.2, capsize=4)
        # an arm ON the [0, 1] range limit is NOT a short arm: it says the interval ran
        # into the parameter's edge. Marked, never left to look like precision.
        for x, r in ci_rows:
            if getattr(r, "occupancy_ci_at_limit", False):
                ax_occ.annotate("CI arm at range limit", (x, 100 * r.occupancy_ci[1]),
                                xytext=(0, 6), textcoords="offset points",
                                fontsize=6, ha="center", color="#c05621")
    for x, r in zip(xs, rows):
        if getattr(r, "error", None):                # a condition that couldn't fit
            ax_occ.annotate(r.error, (x, 2), fontsize=6, ha="center", va="bottom",
                            rotation=90, color="#c05621", wrap=True)
    ax_occ.set_xticks(xs)
    ax_occ.set_xticklabels(conditions)
    ax_occ.set_ylabel("state-1 occupancy (%)")
    ax_occ.set_ylim(0, 100)
    unit = data.get("unit", "unit")
    ax_occ.set_title(f"HMM occupancy (state 1, lowest D): CI over {unit_word(unit)}s",
                     fontsize=9, loc="left")

    width = 0.35
    _rate_group(ax_rate, xs - width / 2, [r.on_rate for r in rows],
                [r.on_ci for r in rows], [r.on_identifiable for r in rows],
                color="#0072B2", width=width, label="k: state 2→1")
    _rate_group(ax_rate, xs + width / 2, [r.off_rate for r in rows],
                [r.off_ci for r in rows], [r.off_identifiable for r in rows],
                color="#D55E00", width=width, label="k: state 1→2")
    ax_rate.set_xticks(xs)
    ax_rate.set_xticklabels(conditions)
    ax_rate.set_ylabel("transition rate (1/s)")
    ax_rate.set_title("HMM switching rates (hatched span = unidentified on this design)",
                      fontsize=9, loc="left")
    if ax_rate.get_legend_handles_labels()[1]:       # only if something was drawn
        ax_rate.legend(fontsize=7, frameon=False)
    # both sub-axes share the integer-x condition layout, so one column hit-test
    # covers a click in either the occupancy or the rate panel
    _hittest.attach(figure, _hittest.ColumnHitTest(list(conditions)))
    return (ax_occ, ax_rate)


def _rate_group(ax, positions, values, cis, identifiable, *, color, width, label):
    for i, (x, val, ci, ok) in enumerate(zip(positions, values, cis, identifiable)):
        first = i == 0
        if not np.isfinite(val):
            # a condition the HMM could not fit, say so, don't draw a phantom bar
            ax.annotate("n/a", (x, 0), fontsize=6, ha="center", va="bottom",
                        color="#a0aec0")
            continue
        if ok and ci is not None:
            lo, hi = ci
            ax.bar(x, val, width=width, color=color, alpha=0.8,
                   yerr=[[val - lo], [hi - val]], capsize=4,
                   label=label if first else None)
        else:
            # Unidentifiable: a HATCHED SPAN over the full plausible range, with NO
            # ARROWHEAD AT EITHER END and no fill implying a preferred region.
            #
            # WHY NOT AN ARROW (Amendment 5, ruled). An open arrow pointing upward
            # from the interval's identified end draws a one-sided claim about the
            # magnitude, the same directional assertion the words used to make, in
            # a different medium, and Amendment 5 says the rate out of low-D is not a
            # bound in EITHER direction. A double-headed marker is no better: two
            # arrowheads are two assertions, not zero, and a reader takes any
            # terminator as a value. So the span simply covers the range and
            # terminates in nothing.
            identified_end = ci[0] if ci is not None else 0.0
            top = max(identified_end,
                      max([v for v in values if np.isfinite(v)],
                          default=identified_end))
            span_top = top * 1.35 + 0.5
            ax.bar(x, span_top, width=width, color=color, alpha=0.22, hatch="///",
                   edgecolor=color, linewidth=0.0,
                   label=label if first else None, zorder=2)
            ax.annotate("?", (x, span_top), fontsize=8, ha="center", va="bottom",
                        color=color)


# --------------------------------------------------------------------------- #
# Fraction SuperPlot, per-cell swarm + CI + cell-level permutation p
# --------------------------------------------------------------------------- #


def draw_fraction(figure, data: dict) -> "object":
    """Per-cell threshold-basis fraction per condition: swarm + CI + permutation p.

    ``data`` is ``comparison.fraction_superplot(...)``. Cells are points coloured
    by replicate day; the per-condition mean and cell-bootstrap CI are drawn; a
    cell-level permutation p between conditions is annotated. Never pooled-step
    significance: the test input is one value per cell.
    """
    figure.clear()
    ax = figure.add_subplot(111)
    per_condition = data["per_condition"]
    conditions = list(per_condition)
    positions = np.arange(len(conditions))       # 0-based: uniform with drill-down
    rng = np.random.default_rng(0)

    # Decide the day encoding ONCE, over all conditions: colour by day only when
    # there are genuine replicate days; a single session gets one neutral colour
    # so the figure never asserts a replicate structure the data lacks.
    all_days = [d for b in per_condition.values() for d in b["per_cell"]["day"]]
    by_day = has_real_days(all_days)

    for pos, cond in zip(positions, conditions):
        block = per_condition[cond]
        per_cell = block["per_cell"]
        if by_day:
            days = sorted(per_cell["day"].unique())
            day_means = block.get("day_means", {})
            for day in days:
                sub = per_cell[per_cell["day"] == day]
                vals = sub["slow_fraction"].to_numpy(dtype=float)
                x = pos + rng.uniform(-0.16, 0.16, len(vals))
                colour = _day_color(day, days)
                ax.scatter(x, vals, s=24, color=colour, alpha=0.5,
                           edgecolors="none", zorder=3, label=None)
                # the day is the UNIT: its mean is a large marker (the SuperPlot
                # convention: cells are the cloud, the per-day means are the data)
                if day in day_means:
                    ax.plot(pos, day_means[day], "o", ms=9, color=colour,
                            markeredgecolor="k", zorder=4)
        else:
            vals = per_cell["slow_fraction"].to_numpy(dtype=float)
            x = pos + rng.uniform(-0.16, 0.16, len(vals))
            ax.scatter(x, vals, s=24, color=_SINGLE_SESSION_COLOR, alpha=0.6,
                       edgecolors="none", zorder=3)
        ci = block.get("ci")
        if ci is not None:
            lo, hi = ci
            ax.plot([pos, pos], [lo, hi], color="#2d3748", lw=6, alpha=0.25,
                    solid_capstyle="round", zorder=1)
        ax.plot(pos, block["mean"], "D", ms=11, color="#2d3748",
                markeredgecolor="k", zorder=4)
        ax.annotate(f"{block['mean']:.2f}", (pos, block["mean"]), xytext=(10, 0),
                    textcoords="offset points", fontsize=7, va="center")
        # flag where the analytic interval may mislead (→ bootstrap worth it)
        if block.get("escalation"):
            ax.annotate("⚠", (pos, 0.01), fontsize=11, color="#c53030",
                        ha="center", va="bottom", zorder=5)

    _annotate_pvalues(ax, data["pvalues"], conditions, positions, per_condition)

    thr = data.get("threshold_um2_s")
    ax.set_xticks(positions)
    if by_day:
        ax.set_xticklabels(conditions)
    else:
        # single session: the unit is the CELL, and the axis says so per condition
        ax.set_xticklabels([f"{c}\nn={per_condition[c]['n_cells']} cells"
                            for c in conditions])
    ax.set_ylabel(f"per-cell fraction with D < {thr:g} µm²/s")
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, len(conditions) - 0.4)
    tier = data.get("tier", "analytic")
    unit = data.get("unit", "cell")
    # name the UNIT the interval is computed over, honestly
    if tier == "analytic":
        tier_label = ("replicate-level mean ± SD/√n_days" if unit == "day"
                      else "cell mean ± SD/√n_cells")
    else:
        tier_label = {"bootstrap": "cell bootstrap CI",
                      "none": "no interval"}.get(tier, tier)
    flagged = any(b.get("escalation") for b in per_condition.values())
    suffix = "  (⚠ = few units)" if flagged else ""
    if by_day:
        unit_phrase = ("large markers = per-replicate means (the unit)" if unit == "day"
                       else "by replicate")
    else:
        unit_phrase = "by cell (single session)"
    ax.set_title(f"Fraction below the mobility threshold: {unit_phrase}: "
                 f"{tier_label}{suffix}", fontsize=9, loc="left")
    _hittest.attach(figure, _hittest.ColumnHitTest(list(conditions)))
    return ax


def _annotate_pvalues(ax, pvalues, conditions, positions, per_condition):
    pos_of = {c: p for c, p in zip(conditions, positions)}
    top = 1.0
    for offset, ((a, b), p) in enumerate(pvalues.items()):
        if not np.isfinite(p):
            continue
        xa, xb = pos_of[a], pos_of[b]
        y = 0.90 - 0.08 * offset
        ax.plot([xa, xa, xb, xb], [y, y + 0.02, y + 0.02, y], color="#2d3748",
                lw=0.8)
        ax.annotate(f"permutation p = {format_p(p)} (cell-level)",
                    ((xa + xb) / 2, y + 0.02), ha="center", va="bottom",
                    fontsize=7, color="#2d3748")


# --------------------------------------------------------------------------- #
# Cross-method forest
# --------------------------------------------------------------------------- #


#: Okabe–Ito hues (skipping black) for per-condition curves.
_CONDITION_COLORS = OKABE_ITO_CYCLE = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]


def _condition_color(i):
    return _CONDITION_COLORS[i % len(_CONDITION_COLORS)]


def draw_saspt(figure, data: dict) -> "object":
    """saSPT posterior spectra, one curve per condition on a log-D axis.

    ``data`` is ``comparison.saspt_spectra(...)``. Each condition's posterior
    occupancy over D is overlaid so a shift in where the mass sits reads directly;
    the mobility threshold is marked. NEUTRAL: x = D, y = posterior occupancy,
    curves labelled by condition: no interpretive word, no asserted state count.
    """
    figure.clear()
    ax = figure.add_subplot(111)
    per_condition = data["per_condition"]
    drawn = 0
    curves = []                                   # (condition, xs, ys) as drawn
    for i, (cond, block) in enumerate(per_condition.items()):
        grid = np.asarray(block["grid"], dtype=float)
        occ = np.asarray(block["occupations"], dtype=float)
        if block.get("error") or not len(grid):
            # honest edge state: a condition too sparse to infer is named, not drawn
            ax.plot([], [], "-", color=_condition_color(i), lw=1.6,
                    label=f"{cond}: {block.get('error', 'no data')}")
            continue
        ax.plot(grid, occ, "-", color=_condition_color(i), lw=1.6, label=cond)
        curves.append((cond, grid, occ))
        drawn += 1
    thr = data.get("threshold_um2_s")
    if thr:
        ax.axvline(thr, color="#2d3748", lw=1.0, ls=":")
        ax.annotate(f"D = {thr:g} µm²/s", (thr, ax.get_ylim()[1]), fontsize=7,
                    va="top", ha="right", color="#2d3748")
    ax.set_xscale("log")
    ax.set_xlabel("D (µm²/s, log)")
    ax.set_ylabel("saSPT posterior occupancy")
    ax.set_title(f"saSPT posterior spectra: {len(per_condition)} conditions",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=min(len(per_condition), 3))
    # x is D, not a condition: identity is the nearest curve, read off the curves
    # actually drawn, so a click never resolves via a conditions-on-X assumption.
    _hittest.attach(figure, _hittest.CurveHitTest(curves=curves))
    return ax


def draw_spatial(figure, data: dict) -> "object":
    """Folded per-population maps for ALL conditions (the legacy speed map, fixed).

    ``data`` is ``comparison.spatial_maps(...)``. A grid of folded-cell heatmaps,
    rows = conditions, columns = populations (slowest→fastest by D). ONE colour
    scale per population (shared across conditions, its own colorbar), so
    abundances are never mixed on one scale. ``mode="enrichment"`` draws density/
    null on a log scale symmetric about 1 (blue↔red); ``mode="density"`` draws raw
    density (magma). A panel too sparse to map is greyed with an honest ``n = …``
    note, not confetti. NEUTRAL: populations by index + numeric D, d-axis is a
    LOCATION.
    """
    figure.clear()
    conditions = data["conditions"]
    n_pops = data.get("n_pops", 0)
    if not n_pops:
        ax = figure.add_subplot(111)
        ax.text(0.5, 0.5, "no localizations to map", ha="center", va="center")
        ax.set_axis_off()
        return ax

    mode = data.get("mode", "density")
    limits_by_pop = data.get("limits_by_pop", {})
    axes = figure.subplots(len(conditions), n_pops, squeeze=False)
    col_mappable = {}
    panels = {}                                   # id(axes) -> Selection, per panel

    for r, condition in enumerate(conditions):
        rows = data["panels"].get(condition, [])
        errored = len(rows) == 1 and "error" in rows[0]
        for c in range(n_pops):
            ax = axes[r][c]
            pop = c + 1
            # the panel IS (this condition, this population), key the hit-test on
            # the subplot itself, so a click resolves by WHICH panel, never by an
            # (x, y) reading that a shared column convention would misinterpret
            panels[id(ax)] = _hittest.Selection(condition=condition, pop=pop,
                                                kind="spatial_panel")
            vmin, vmax = limits_by_pop.get(pop, (0.0, 1.0))
            cmap, norm = _spatial_cmap_norm(mode, vmin, vmax)
            if errored:
                _grey_panel(ax, f"not mappable\n({rows[0]['error'][:40]})")
            elif c < len(rows):
                panel = rows[c]
                if panel.get("sparse"):
                    _grey_panel(ax, f"n = {panel['n']}: too sparse to map")
                else:
                    im = ax.imshow(np.ma.masked_invalid(np.asarray(panel["values"])),
                                   origin="lower", aspect="auto",
                                   extent=panel["extent"], cmap=cmap, norm=norm)
                    col_mappable[pop] = im
                    D = panel.get("D")
                    if D is not None:
                        ax.annotate(f"D≈{D:.2g}", (0.02, 0.98),
                                    xycoords="axes fraction", fontsize=6, va="top",
                                    ha="left", color="#111111")
            else:
                _grey_panel(ax, "n/a")
            if r == 0:
                ax.set_title(f"population {pop}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{condition}\nd: 0=axis,1=membrane", fontsize=7)
    for c in range(n_pops):
        axes[-1][c].set_xlabel("l (pole→pole)", fontsize=7)

    # ONE colorbar per population column, at the BOTTOM of the column (a
    # right-side bar would land between columns and overlap the next panel).
    label = ("enrichment vs null (log, 1 = no preference)" if mode == "enrichment"
             else "smoothed density")
    for pop, im in col_mappable.items():
        cbar = figure.colorbar(
            im, ax=[axes[r][pop - 1] for r in range(len(conditions))],
            location="bottom", fraction=0.05, pad=0.12)
        cbar.set_label(label, fontsize=7)

    lim_txt = " | ".join(
        f"pop {p}: [{lo:.3g}, {hi:.3g}]" for p, (lo, hi) in sorted(limits_by_pop.items()))
    figure.suptitle(f"Spatial {mode} per population ({data['source']}, folded): "
                    f"per-population robust scale", fontsize=10)
    # The bottom-left corner (y≈0.005–0.03) is reserved on EVERY view for the
    # global provenance + speed-shortcut stamp; this view-specific scale note sits
    # above it so the two never overlap (they used to collide at 0.005).
    figure.text(0.01, 0.052, f"scale (2–98%): {lim_txt}", fontsize=6,
                color="#4a5568", family="monospace")
    # honest skip: cells with no Oufti mesh rib can't be projected, say how many
    skipped = data.get("skipped_no_rib") or {}
    if skipped:
        note = "; ".join(f"{c}: {n}" for c, n in skipped.items())
        figure.text(0.01, 0.078, f"skipped (no mesh rib): {note}", fontsize=6,
                    color="#c05621", family="monospace")
    _hittest.attach(figure, _hittest.AxesHitTest(panels))
    return axes[0][0]


def _spatial_cmap_norm(mode, vmin, vmax):
    import matplotlib
    from matplotlib.colors import LogNorm, Normalize

    if mode == "enrichment":
        cmap = matplotlib.colormaps["coolwarm"].with_extremes(bad="#cccccc")
        lo = max(vmin, 1e-3)
        return cmap, LogNorm(vmin=lo, vmax=max(vmax, lo * 1.001))
    cmap = matplotlib.colormaps["magma"].with_extremes(bad="#cccccc")
    return cmap, Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-9))


def _grey_panel(ax, message):
    ax.set_facecolor("#cccccc")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=6.5,
            color="#333333", wrap=True, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])


def draw_heatmap(figure, data: dict) -> "object":
    """Conditions × methods heatmap of the fraction below the mobility threshold.

    ``data`` is ``comparison.fraction_matrix(...)``. Rows are conditions, columns
    are methods; each cell's colour and printed % is that method's below-threshold
    fraction for that condition. A failed cell is blank (grey), never a fake 0.
    The overview across many conditions: a condition's shift is a row that lights
    up. NEUTRAL: colour + numeric fraction, axis says ``D < X``, no word.
    """
    figure.clear()
    ax = figure.add_subplot(111)
    values = np.asarray(data["values"], dtype=float)
    conditions = data["conditions"]
    methods = data["methods"]

    masked = np.ma.masked_invalid(values)
    import matplotlib

    # a failed cell is grey, never a fake 0
    cmap = matplotlib.colormaps["viridis"].with_extremes(bad="#cccccc")
    vmax = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=max(vmax, 1e-6),
                   aspect="auto")
    cbar = figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"fraction with D < {data['threshold_um2_s']:g} µm²/s")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions)
    for i in range(len(conditions)):
        for j in range(len(methods)):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{100 * v:.0f}%", ha="center", va="center",
                        fontsize=7,
                        color="white" if v > 0.55 * max(vmax, 1e-6) else "black")
    ax.set_title(f"Fraction below D = {data['threshold_um2_s']:g} µm²/s: "
                 f"{len(conditions)} conditions × {len(methods)} methods",
                 fontsize=9, loc="left")
    # conditions are the ROWS (Y), methods the COLUMNS (X), the hit-test reads
    # that off the plotted arrays, so a click resolves to the cell under it, not
    # to whatever a conditions-on-X assumption would have picked.
    _hittest.attach(figure, _hittest.GridHitTest(row_labels=list(conditions),
                                                 col_labels=list(methods)))
    return ax


#: A stable marker per method for the multi-condition forest.
_METHOD_MARKERS = {"SQD": "o", "HMM": "s", "appD": "^", "saSPT": "D",
                   "confinement": "v"}

#: Methods whose forest MAGNITUDE is known-biased and must carry a footnote, so a
#: meeting-facing figure read without a caption cannot be pointed at as a
#: disagreement. appD averages within-track switches → compressed Δ (confirmed on
#: ground truth; see the appD switching-blur note in CLAUDE.md / Methods). Purely
#: METHODOLOGICAL (no state word) so it passes the neutral-labels gate.
_MAGNITUDE_BIASED = {"appD"}
_MAGNITUDE_NOTE = "‡ = magnitude biased low under state switching (see Methods)"

#: Foot-of-figure note for the forest / Δ views (Amendment 2). The below-threshold
#: Δ is a cross-method device, NOT the pre-registered endpoint, which is per-state
#: occupancy under global D-sharing (the SQD populations view). Kept as a labelled
#: secondary; never deleted, never captioned as the endpoint.
_FOREST_SECONDARY_NOTE = (
    "secondary cross-method device (commensurable Δ), NOT the pre-registered "
    "endpoint: endpoint = per-state occupancy under global D-sharing (SQD "
    "populations view)")


def _forest_method_label(method: str, shares_machinery: bool) -> str:
    """A method's forest label with its footnote markers (shared-machinery † and
    switching-blur ‡), so both live in one place for the single/multi paths."""
    mark = ""
    if not shares_machinery:
        mark += " †"
    if method in _MAGNITUDE_BIASED:
        mark += " ‡"
    return method + mark


def draw_forest(figure, data: dict) -> "object":
    """Forest of Δ fraction (target − reference). Generalises to N conditions.

    ``data`` is ``comparison.forest(...)`` restricted to the TICKED methods. A
    line at 0 marks "no change"; methods sharing no machinery with the rest
    (saSPT, confinement) are marked with a dagger. With one target it is the
    single-column forest (rows = methods); with several, rows are GROUPED BY
    target so many conditions read against one reference at once, method encoded
    by marker. The caption counts the ticked methods, never more than are shown.
    """
    figure.clear()
    ax = figure.add_subplot(111)
    rows = data["rows"]
    targets = data.get("targets", [data.get("target")])
    methods = data.get("methods") or sorted({r.method for r in rows})

    # ENDPOINT REALIGNMENT (Amendment 2): the Δ of the below-threshold fraction is a
    # cross-method commensurability DEVICE, the one basis that makes SQD, appD, HMM
    # and saSPT report one number, NOT the pre-registered endpoint. Say so at the
    # foot of the figure so an exported forest can never be read as the endpoint.
    figure.text(0.01, 0.005, _FOREST_SECONDARY_NOTE, fontsize=6.0,
                color="#718096", va="bottom")

    if len(targets) <= 1:
        return _draw_forest_single(ax, data, rows)

    # Multi-target: one horizontal band per target, methods offset within it.
    reference = data.get("reference")
    n_methods = len(methods)
    yticks, ylabels = [], []
    marks = []                                    # (y, Selection) as each is drawn
    for ti, target in enumerate(targets):
        base = ti * (n_methods + 1)
        yticks.append(base + (n_methods - 1) / 2)
        ylabels.append(target)
        for mi, method in enumerate(methods):
            row = next((r for r in rows if r.target == target and r.method == method),
                       None)
            if row is None or not np.isfinite(row.delta):
                continue
            y = base + mi
            lo, hi = row.ci
            ax.plot([lo, hi], [y, y], color="#2d3748", lw=1.0, zorder=2)
            ax.plot(row.delta, y, _METHOD_MARKERS.get(method, "o"), ms=7,
                    color="#0072B2", markeredgecolor="k", zorder=3)
            marks.append((y, _hittest.Selection(
                condition=target, method=method, target=target,
                reference=reference, kind="forest_point")))

    ax.axvline(0.0, color="#c53030", lw=1.0, ls="--", zorder=1)
    _hittest.attach(figure, _hittest.RowHitTest(rows=marks))
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel(f"Δ fraction below D = {data['threshold_um2_s']:g} µm²/s "
                  f"(condition − {data['reference']})")
    n = data["n_methods"]
    note = f"  ({_MAGNITUDE_NOTE})" if any(m in _MAGNITUDE_BIASED for m in methods) else ""
    ax.set_title(f"Δ vs {data['reference']}: {len(targets)} conditions × "
                 f"{n} method{'s' if n != 1 else ''} shown{note}",
                 fontsize=9, loc="left")
    _method_marker_legend(ax, methods)
    return ax


def _draw_forest_single(ax, data, rows):
    reference, target = data.get("reference"), data.get("target")
    ys = np.arange(len(rows))[::-1]
    marks = []                                    # (y, Selection) per method row
    for y, row in zip(ys, rows):
        marks.append((float(y), _hittest.Selection(
            condition=target, method=row.method, target=target,
            reference=reference, kind="forest_point")))
        if not np.isfinite(row.delta):
            ax.annotate(f"{row.method}: n/a", (0, y), fontsize=7, va="center",
                        color="#a0aec0")
            continue
        lo, hi = row.ci
        ax.plot([lo, hi], [y, y], color="#2d3748", lw=1.4, zorder=2)
        ax.plot(row.delta, y, "o", ms=9, color="#0072B2", markeredgecolor="k",
                zorder=3)
    ax.axvline(0.0, color="#c53030", lw=1.0, ls="--", zorder=1)
    _hittest.attach(ax.figure, _hittest.RowHitTest(rows=marks))
    labels = [_forest_method_label(r.method, r.shares_machinery) for r in rows]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Δ fraction below D = {data['threshold_um2_s']:g} µm²/s "
                  f"({data['target']} − {data['reference']})")
    n = data["n_methods"]
    notes = "† = shares no machinery"
    if any(r.method in _MAGNITUDE_BIASED for r in rows):
        notes += f";  {_MAGNITUDE_NOTE}"
    ax.set_title(f"Cross-method Δ vs {data['reference']}: {n} method"
                 f"{'s' if n != 1 else ''} shown  ({notes})",
                 fontsize=9, loc="left")
    return ax


def draw_sigma_sensitivity(figure, data: dict) -> "object":
    """σ-sensitivity panel (item D): lowest-D state occupancy vs σ, per condition,
    plus the between-condition Δ. ``data`` is ``comparison.sigma_sensitivity(...)``.
    NEUTRAL: the y-axis is occupancy, the series are conditions + their Δ. A flat Δ
    means σ is not a threat to the endpoint; a rising Δ is the finding."""
    figure.clear()
    ax = figure.add_subplot(111)
    sig = data["sigmas_nm"]
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
    for i, c in enumerate(data["conditions"]):
        ax.plot(sig, [100 * v for v in data["slow_occ"][c]], "o-",
                color=palette[i % len(palette)], label=c)
    if data.get("delta") is not None:
        a, b = data["conditions"]
        ax.plot(sig, [100 * v for v in data["delta"]], "s--", color="#2d3748",
                label=f"Δ ({b} − {a})")
    px = data.get("pixelation_nm")
    if px is not None:
        ax.axvline(px, color="#a0aec0", ls=":", lw=1.0)
        ax.annotate(f"pixelation {px:.1f} ({data.get('sigma_source', '')})",
                    (px, ax.get_ylim()[1]), rotation=90, fontsize=6,
                    va="top", ha="right", color="#718096")
    ax.set_xlabel("localization σ (nm)")
    ax.set_ylabel("lowest-D state occupancy / Δ (%)")
    ax.set_title(f"σ-sensitivity of the lowest-D state ({data['n_pop']}-state fit) "
                 ": endpoint flat in σ ⇒ σ not a threat", fontsize=8, loc="left")
    ax.legend(fontsize=7, frameon=False)
    if data.get("delta_range"):
        lo, hi = data["delta_range"]
        figure.text(0.01, 0.005, f"Δ range over σ = [{100*lo:+.1f}, {100*hi:+.1f}] pp "
                    f"(spread {100*(hi-lo):.1f} pp)", fontsize=6.0, color="#4a5568",
                    va="bottom")
    return ax


def _method_marker_legend(ax, methods):
    import matplotlib.lines as mlines

    handles = [mlines.Line2D(
        [], [], marker=_METHOD_MARKERS.get(m, "o"), linestyle="none",
        color="#0072B2", markeredgecolor="k",
        label=(m + " ‡" if m in _MAGNITUDE_BIASED else m)) for m in methods]
    if handles:
        ax.legend(handles=handles, fontsize=7, frameon=False, loc="best",
                  ncol=min(len(methods), 3))
