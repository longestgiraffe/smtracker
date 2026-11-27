"""The workspace's exclusions, with no Qt in it (asked 16 Oct 2026: "Exclusions in the workspace -- exclude a track,
cell or movie with a reason ... the fractions shown with and without it").

`smtracker.exclusions` holds the rules (a written reason for every unit, a record of what was removed, never
everything); until now only the command line reached it. The workspace's Exclusions dialog
(`ui/workspace/exclusions_dialog.py`) draws from this module: what can be excluded (`choices`), whether a typed key
names a unit (`key_problem`), what QC already flagged (`suggestions`), the list with each exclusion's tracks
(`rows`), and the WITH-AND-WITHOUT preview (`preview`, a pool job: each affected condition's tracks and slow
fraction from one pooled two-population SQD fit, before and after). `project.with_exclusions` applies a set to the
whole table; the window rebuilds the workspace around it, as it does for the track cap, so every section and every
cache key sees the same data (a fit's provenance hashes the table it was made from, so a filtered condition never
hits an unfiltered fit).

Items travel as plain dicts ``{"kind", "key", "reason"}`` (`Exclusion.to_dict`), the shape the table's
``attrs["exclusions"]`` and the remembered set already use.
"""
from __future__ import annotations

__all__ = ["KIND_LABELS", "KINDS", "choices", "exclusion_set", "key_problem", "preview", "rows", "suggestions"]

from .exclusions import KINDS

#: How each kind reads in the dialog.
KIND_LABELS = {"movie": "Movie", "cell": "Cell", "uid": "Track"}


def exclusion_set(items):
    """The `ExclusionSet` of plain items. RAISES `ExclusionError` on a malformed one (no reason, unknown kind)."""
    from .exclusions import Exclusion, ExclusionSet

    return ExclusionSet([Exclusion(kind=i["kind"], key=str(i["key"]), reason=str(i["reason"])) for i in items])


def _short(key: str, condition: str) -> str:
    prefix = f"{condition}_"
    return key[len(prefix):] if key.startswith(prefix) else key


def choices(project) -> dict:
    """What can be excluded, per condition: ``{condition: {"movie": [(key, shown)], "cell": [(key, shown)]}}``,
    keys as the table holds them (qualified by condition) and shown without the condition's prefix. A track is
    typed (or picked elsewhere), never listed: there are thousands."""
    from .canonical import without_attrs

    table = without_attrs(project.canonical)
    out = {}
    for condition in project.conditions:
        part = table[table["condition"] == condition]
        movies = sorted(part["movie"].astype(str).unique())
        cells = sorted(part["cell"].dropna().astype(str).unique()) if "cell" in part.columns else []
        out[condition] = {"movie": [(m, _short(m, condition)) for m in movies],
                          "cell": [(c, _short(c, condition)) for c in cells]}
    return out


def key_problem(project, kind: str, key: str) -> str:
    """Why ``kind:key`` cannot be excluded, or "" when it names a unit of the loaded project."""
    if kind not in KINDS:
        return f"kind must be one of {', '.join(KINDS)}"
    key = str(key).strip()
    if not key:
        return "name the unit to exclude"
    column = project.canonical[kind] if kind in project.canonical.columns else None
    if column is None:
        return f"the table has no {kind} column"
    if not (column.astype(str) == key).any():
        return f"no {KIND_LABELS[kind].lower()} {key!r} in this project"
    return ""


def usual_range(low: float, high: float) -> str:
    """"usual range 0.065 to 0.252": QC's median +- k scale, held at 0 below (every quantity QC flags on -- an
    apparent D, a fraction, a track length -- is never negative), with "to", never a minus sign that reads as a dash
    (3 Nov 2026, reported: "usual range -0.0648-0.252" for an apparent D)."""
    return f"usual range {max(0.0, float(low)):.3g} to {float(high):.3g}"


def suggestions(qc_payloads: dict, existing=()) -> list:
    """What QC flagged, offered (never applied): ``[{"kind", "key", "shown", "condition", "why"}]`` for each flagged movie and
    cell of every condition QC has run on, minus those already excluded -- a cell of an excluded movie included (found by
    the overnight tester, 4 Nov 2026: they were still offered). The reason is the analyst's to write."""
    taken = {(i["kind"], str(i["key"])) for i in existing}
    gone = {str(i["key"]) for i in existing if i["kind"] == "movie"}
    out = []
    for condition, entry in (qc_payloads or {}).items():
        units = (entry or {}).get("units") or {}
        for kind in ("movie", "cell"):
            block = units.get(kind)
            if not isinstance(block, dict) or "names" not in block:
                continue
            for name, value, flagged in zip(block["names"], block["values"], block["flagged"]):
                if not flagged or (kind, str(name)) in taken:
                    continue
                if kind == "cell" and str(name).rsplit(":", 1)[0] in gone:
                    continue                                    # its movie is excluded already
                what = block.get("quantity", "value")
                why = (f"{what} {value:.3g}" if value is not None else "flagged") + (
                    f", {usual_range(block['low'], block['high'])}"
                    if block.get("low") is not None and block.get("high") is not None else "")
                out.append({"kind": kind, "key": str(name), "shown": _short(str(name), condition),
                            "condition": condition, "why": f"QC: {why}"})
    return out


def rows(project, items) -> list:
    """The dialog's list: per item ``{"kind", "kind_label", "key", "shown", "condition", "reason", "tracks"}``, the
    tracks it removes counted on ``project`` (the table before exclusions)."""
    table = project.canonical
    out = []
    for item in items:
        kind, key = item["kind"], str(item["key"])
        hit = table[table[kind].astype(str) == key] if kind in table.columns else table.iloc[0:0]
        condition = str(hit["condition"].iloc[0]) if len(hit) else ""
        out.append({"kind": kind, "kind_label": KIND_LABELS.get(kind, kind), "key": key,
                    "shown": _short(key, condition), "condition": condition, "reason": str(item["reason"]),
                    "tracks": int(hit["uid"].nunique()) if len(hit) else 0})
    return out


def preview(project, items) -> dict:
    """WITH AND WITHOUT, for the conditions the items touch: per condition its tracks and its slow fraction from
    ONE pooled two-population SQD fit (step-weighted, as the Compute dialog's default, exact derivatives, 2000
    ECDF points), on ``project`` as it is and with the items removed. Picklable; a pool job. A condition the items
    would empty, or a fit that fails, says so in its row. The fit is the quick pooled one, not the Run's: the
    preview says how much the exclusion moves the data, the Run then measures it with its own settings."""
    from .canonical import without_attrs
    from .exclusions import ExclusionError, apply_exclusions
    from .observables import squared_displacements
    from .sqd import SQDError, fit_sqd

    s = exclusion_set(items)
    touched = []
    for r in rows(project, items):
        if r["condition"] and r["condition"] not in touched:
            touched.append(r["condition"])

    def slow(table):
        fit = fit_sqd(squared_displacements(table), n_pop=2, weighting="step", gradient="analytic",
                      ecdf_max_points=2000)
        order = sorted(range(len(fit.D)), key=lambda k: fit.D[k])
        return float(fit.fractions[order[0]]), float(fit.D[order[0]])

    out = []
    for condition in touched:
        before = project.condition(condition)
        row = {"condition": condition, "tracks_before": int(before["uid"].nunique())}
        try:
            after = apply_exclusions(before, s)
        except ExclusionError as exc:
            row.update(tracks_after=0, error=str(exc))
            out.append(row)
            continue
        row["tracks_after"] = int(without_attrs(after)["uid"].nunique())
        try:
            row["slow_before"], row["D_before"] = slow(before)
            row["slow_after"], row["D_after"] = slow(after)
        except SQDError as exc:
            row["error"] = f"the fit failed: {exc}"
        out.append(row)
    lines = []
    for r in out:
        if "error" in r:
            lines.append(f"{r['condition']}: {r['tracks_before']:,} → {r['tracks_after']:,} tracks; {r['error']}")
        else:
            change = 100 * (r["slow_after"] - r["slow_before"])
            lines.append(f"{r['condition']}: {r['tracks_before']:,} → {r['tracks_after']:,} tracks; slow fraction "
                         f"{100 * r['slow_before']:.1f} % → {100 * r['slow_after']:.1f} % ({change:+.1f} points)")
    return {"rows": out, "lines": lines,
            "caption": "one pooled two-population SQD fit per condition, step-weighted: how much the exclusions "
                       "move the data; a Run then measures it with its own settings"}
