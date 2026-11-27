"""Comparing several groups: each free to have its own fit settings.

What a "group" is
-----------------
A group is one **subfolder of the loaded project**, the level at which the
experiment was organised. ``protein`` and ``condition`` are interchangeable
labels for it: sometimes those subfolders are one protein under several
conditions, sometimes several different proteins. The code does not presume
which, so the default grouping is ``("condition",)`` (the subfolder) and any
column combination can be used instead.

The decision the code cannot make for you
-----------------------------------------
**Whether the groups should share a state count is a scientific question.**

- *Different molecules* → let each group keep its own count. Protein A described
  by 2 states and protein B by 3 is a biological result, not an inconsistency.
  Forcing a common count would misdescribe at least one of them.
- *One molecule under different conditions* → the count must be **shared**. A
  fraction from a 2-state fit and one from a 3-state fit are different
  quantities: "38 % of one of two states" and "38 % of one of three states"
  cannot be subtracted, averaged, or plotted on one axis. Selecting per
  condition would silently destroy the primary readout, the shift in fractions
  between conditions.

So: model selection runs **per group** (``select_n_pop_by_group``), and you then
either leave the counts as they are, or call ``harmonize`` to impose one count
across the groups, with a recorded reason. ``compare_fractions`` refuses to
build a side-by-side table while the counts differ, rather than emitting one
whose columns mean different things in different rows.

Fail loud
---------
- A group too small to fit raises, naming the group.
- Comparing fractions across differing state counts raises.
- A group whose grouping columns are unassigned (``<NA>``) raises, instead of
  quietly pooling it with the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .sqd import NPopSelection, SQDError, SQDFit, bootstrap_sqd, fit_sqd, select_n_pop

__all__ = [
    "ComparisonError",
    "GroupFit",
    "GroupedSQD",
    "select_n_pop_by_group",
    "fit_by_group",
    "compare_fractions",
    "bubble_series",
    "BUBBLE_FIT_MODES",
]

#: A group is one subfolder of the loaded project. ``condition`` and ``protein``
#: are interchangeable labels for that level; ``condition`` is always assigned,
#: so it is the default. Override with ``group_by=("protein",)``,
#: ``("protein", "condition")``, or ``()`` to pool everything.
DEFAULT_GROUP_BY = ("condition",)


class ComparisonError(RuntimeError):
    """Raised when groups cannot be fitted or their results are not comparable."""


@dataclass
class GroupFit:
    """One fit group's model selection and (once chosen) its fit."""

    key: tuple
    label: str
    selection: NPopSelection
    n_observations: int

    @property
    def fit(self) -> SQDFit:
        """The fit in force for this group (analyst's choice, else BIC's)."""
        return self.selection.best

    @property
    def n_pop(self) -> int:
        return self.fit.n_pop

    def choose(self, n_pop: int, *, reason: str) -> SQDFit:
        """Record this group's state count. Per group: that is the point."""
        return self.selection.choose(n_pop, reason=reason)


@dataclass
class GroupedSQD:
    """Several fit groups, each free to have its own state count."""

    groups: list[GroupFit]
    group_by: tuple[str, ...]
    _bootstraps: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self):
        return iter(self.groups)

    def __getitem__(self, label: str) -> GroupFit:
        for group in self.groups:
            if group.label == label:
                return group
        raise ComparisonError(
            f"no fit group '{label}'; have {[g.label for g in self.groups]}"
        )

    def choose(self, label: str, n_pop: int, *, reason: str) -> SQDFit:
        """Set one group's state count without touching the others."""
        return self[label].choose(n_pop, reason=reason)

    def harmonize(self, n_pop: int | None = None, *, reason: str) -> dict[str, int]:
        """Impose ONE state count on every group: for one molecule, several conditions.

        Use when the groups are the same molecule under different conditions and
        you intend to compare their fractions. Without a shared count those
        fractions are not the same quantity.

        ``n_pop=None`` picks the count with the best **total** BIC across groups
        (the model the pooled evidence prefers), rather than letting whichever
        group happens to have the most data decide. Every group records the
        reason, and each group's own recommendation stays visible in
        ``result_table()`` so a forced count is never mistaken for a free one.
        """
        counts = self.candidate_counts()
        if n_pop is None:
            totals = {
                n: sum(g.selection.fit_for(n).bic for g in self.groups) for n in counts
            }
            n_pop = min(totals, key=totals.get)
        elif n_pop not in counts:
            raise ComparisonError(
                f"n_pop={n_pop} was not fitted in every group; available: {sorted(counts)}"
            )
        for group in self.groups:
            group.choose(n_pop, reason=reason)
        return self.n_pop_by_group

    def candidate_counts(self) -> set[int]:
        """State counts fitted in *every* group (the harmonizable options)."""
        per_group = [set(f.n_pop for f in g.selection.fits) for g in self.groups]
        return set.intersection(*per_group) if per_group else set()

    def disagreements(self) -> list[str]:
        """Groups whose own recommendation differs from the count in force."""
        return [
            f"{g.label}: using {g.n_pop}, BIC alone preferred "
            f"{g.selection.recommended.n_pop} (ΔBIC = {g.selection.delta_bic(g.n_pop):+.1f})"
            for g in self.groups
            if g.n_pop != g.selection.recommended.n_pop
        ]

    @property
    def n_pop_by_group(self) -> dict[str, int]:
        return {g.label: g.n_pop for g in self.groups}

    @property
    def share_state_count(self) -> bool:
        """True if every group ended up with the same number of states."""
        return len(set(self.n_pop_by_group.values())) <= 1

    def selection_table(self) -> pd.DataFrame:
        """Per group: every candidate count with BIC, ΔBIC and evidence."""
        frames = []
        for group in self.groups:
            table = group.selection.table()
            table.insert(0, "group", group.label)
            table.insert(1, "n_observations", group.n_observations)
            frames.append(table)
        return pd.concat(frames, ignore_index=True)

    def result_table(self) -> pd.DataFrame:
        """Per group: the fit in force, state count, D's and fractions."""
        rows = []
        for group in self.groups:
            fit = group.fit
            row = {
                "group": group.label,
                "n_pop": fit.n_pop,
                "n_observations": fit.n_observations,
                "chosen_by": (
                    "analyst" if group.selection.chosen_n_pop is not None else "BIC"
                ),
                "choice_reason": group.selection.choice_reason or "",
                "recommended_n_pop": group.selection.recommended.n_pop,
            }
            for i, (d, f) in enumerate(zip(fit.D, fit.fractions_percent), start=1):
                row[f"D_{i}"] = d
                row[f"fraction_{i}_percent"] = f
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> str:
        counts = self.n_pop_by_group
        head = "; ".join(f"{label}: {n} states" for label, n in counts.items())
        if not self.share_state_count:
            return (
                f"{head}. Groups differ in state count: this is fine if they are "
                "different molecules, but their fractions are not directly "
                "comparable. If they are one molecule under different conditions, "
                "call harmonize(reason=...)."
            )
        forced = self.disagreements()
        if forced:
            return (
                f"{head}. Shared count imposed; fractions are comparable. "
                "Note: " + "; ".join(forced) + "."
            )
        return f"{head}. All groups agree on the state count: fractions are comparable."


def _resolve_group_by(observable: pd.DataFrame, group_by) -> tuple[str, ...]:
    group_by = tuple(group_by) if group_by is not None else DEFAULT_GROUP_BY
    missing = [c for c in group_by if c not in observable.columns]
    if missing:
        raise ComparisonError(f"observable has no column(s) {missing} to group by")
    for col in group_by:
        if observable[col].isna().any():
            raise ComparisonError(
                f"'{col}' is unassigned for some rows, so fit groups cannot be formed. "
                "Assign it (grouping.assign_groups), or choose a different group_by: "
                "pooling unassigned data with the rest would hide which is which."
            )
    return group_by


def _iter_groups(observable: pd.DataFrame, group_by: tuple[str, ...]):
    """Yield ``(key, label, frame)`` per group; a single group when empty."""
    if not group_by:
        yield ((), "all", observable)
        return
    for key, frame in observable.groupby(list(group_by), observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        yield key, " / ".join(str(k) for k in key), frame


def select_n_pop_by_group(
    observable: pd.DataFrame,
    *,
    group_by=DEFAULT_GROUP_BY,
    max_pop: int = 3,
    min_pop: int = 1,
    method: str = "mle",
    **fit_kwargs,
) -> GroupedSQD:
    """Run BIC model selection **separately in each fit group**.

    Each group gets its own recommendation, which the analyst can override per
    group via ``.choose(label, n_pop, reason=...)``. Groups are never forced to
    agree: a state count is a property of the molecule.
    """
    group_by = _resolve_group_by(observable, group_by)
    tau_lag_s = observable.attrs.get("lag_time_s")

    groups: list[GroupFit] = []
    for key, label, frame in _iter_groups(observable, group_by):
        values = frame["sq_disp_um2"].to_numpy()
        try:
            selection = select_n_pop(
                values,
                tau_lag_s=tau_lag_s,
                max_pop=max_pop,
                min_pop=min_pop,
                method=method,
                **fit_kwargs,
            )
        except SQDError as exc:
            raise ComparisonError(
                f"fit group '{label}' ({len(values)} displacements) could not be "
                f"fitted: {exc}"
            ) from exc
        groups.append(
            GroupFit(key=key, label=label, selection=selection, n_observations=len(values))
        )

    if not groups:
        raise ComparisonError("no fit groups formed: the observable is empty")
    return GroupedSQD(groups=groups, group_by=group_by)


def fit_by_group(
    observable: pd.DataFrame,
    *,
    n_pop,
    group_by=DEFAULT_GROUP_BY,
    method: str = "ecdf_lsq",
    **fit_kwargs,
) -> pd.DataFrame:
    """Fit each group at a **given** state count, one int, or per-group counts.

    ``n_pop`` may be an ``int`` (same count everywhere, appropriate when the
    groups are conditions of one protein) or a ``{group_label: int}`` mapping
    (appropriate when the groups are different proteins).
    """
    group_by = _resolve_group_by(observable, group_by)
    tau_lag_s = observable.attrs.get("lag_time_s")

    rows = []
    for _, label, frame in _iter_groups(observable, group_by):
        if isinstance(n_pop, dict):
            if label not in n_pop:
                raise ComparisonError(
                    f"n_pop mapping does not cover fit group '{label}'; "
                    f"have {sorted(n_pop)}"
                )
            count = n_pop[label]
        else:
            count = int(n_pop)

        values = frame["sq_disp_um2"].to_numpy()
        try:
            fit = fit_sqd(
                values, n_pop=count, tau_lag_s=tau_lag_s, method=method, **fit_kwargs
            )
        except SQDError as exc:
            raise ComparisonError(f"fit group '{label}' could not be fitted: {exc}") from exc
        row = {"group": label, "n_pop": count, "n_observations": len(values)}
        for i, (d, f) in enumerate(zip(fit.D, fit.fractions_percent), start=1):
            row[f"D_{i}"] = d
            row[f"fraction_{i}_percent"] = f
        rows.append(row)
    return pd.DataFrame(rows)


def compare_fractions(
    grouped: GroupedSQD,
    *,
    observable: pd.DataFrame | None = None,
    bootstrap_unit: str | None = None,
    n_boot: int = 200,
    **bootstrap_kwargs,
) -> pd.DataFrame:
    """Fractions side by side: **only** when the groups share a state count.

    Raises rather than emitting a table whose columns mean different things in
    different rows. If the groups genuinely differ in state count, that *is* the
    result: report it (``grouped.result_table()``), do not force a comparison.

    Pass ``observable`` and ``bootstrap_unit`` (``"cell"`` / ``"movie"`` /
    ``"day"``) to attach honest percentile intervals per group.
    """
    if not grouped.share_state_count:
        raise ComparisonError(
            "fit groups have different state counts "
            f"({grouped.n_pop_by_group}), so their fractions are not comparable: "
            "'x % of 2 states' and 'x % of 3 states' are different quantities. "
            "Report the per-group results with result_table(), or choose a common "
            "count deliberately via .choose(label, n, reason=...)."
        )

    table = grouped.result_table()
    if observable is None or bootstrap_unit is None:
        return table

    group_by = list(grouped.group_by)
    intervals = []
    for group in grouped.groups:
        frame = _select_group(observable, group_by, group.key)
        frame.attrs.update(observable.attrs)
        boot = bootstrap_sqd(
            frame,
            n_pop=group.n_pop,
            unit=bootstrap_unit,
            n_boot=n_boot,
            **bootstrap_kwargs,
        )
        row = {"group": group.label, "bootstrap_unit": bootstrap_unit,
               "n_units": boot.attrs["n_units"], "caveat": boot.attrs["caveat"]}
        for i in range(1, group.n_pop + 1):
            lo, hi = boot[f"fraction_{i}"].quantile([0.025, 0.975])
            row[f"fraction_{i}_lo_percent"] = lo * 100
            row[f"fraction_{i}_hi_percent"] = hi * 100
        intervals.append(row)
    return table.merge(pd.DataFrame(intervals), on="group", how="left")


def _select_group(observable: pd.DataFrame, group_by: list[str], key: tuple) -> pd.DataFrame:
    if not group_by:
        return observable.copy()
    mask = np.ones(len(observable), dtype=bool)
    for col, value in zip(group_by, key):
        mask &= (observable[col] == value).to_numpy()
    return observable.loc[mask].copy()


#: The two arrangements a fraction can come from. They are DIFFERENT QUANTITIES and the
#: table always carries which one produced a row, because that is the distinction this
#: module exists to keep.
BUBBLE_FIT_MODES = ("independent", "simultaneous")


def _group_label_series(frame: pd.DataFrame, group_by: tuple[str, ...]) -> pd.Series:
    """The same label `_iter_groups` yields, as a column."""
    if not group_by:
        return pd.Series(["all"] * len(frame), index=frame.index)
    return frame[list(group_by)].astype(str).agg(" / ".join, axis=1)


def _qualified_unit(frame: pd.DataFrame, unit: str, group_by: tuple[str, ...]):
    """Make the unit key unique across groups, never joined.

    ``day`` is a WITHIN-GROUP ORDINAL: glucose day 1 and malate day 1 are independent
    measurements that share an integer and nothing else. ``aggregate.fit_by_unit``
    refuses a unit spanning several groups and is right to; the fix is to qualify the
    key, which is what makes n groups x m days into n*m independent samples rather than
    m paired ones.
    """
    out = frame.copy()
    out[unit] = _group_label_series(frame, group_by) + ":" + out[unit].astype(str)
    return out


def _refuse_shared_tracks(observable: pd.DataFrame, group_by: tuple[str, ...]) -> None:
    """Refuse a track identifier that appears under more than one group.

    MEASURED, not hypothetical. Two conditions loaded from separate u-track projects can
    both number their movies ``series001``-``series030``, so ``uid`` collides across
    them. Displacements are grouped by ``uid``, so pooling such groups computes steps
    BETWEEN molecules in different conditions -- on real data that produced 29
    displacements of up to 9.2 um in a cell about 1 um wide, which underflowed the
    fraction fit and made it refuse a whole condition. That refusal was very nearly
    written up as a biological finding.

    Only the pooled fit is exposed to this, and it fails silently, so it is CHECKED here
    rather than trusted. Same shape as the unit key above, one level down.
    """
    if not group_by or "uid" not in observable.columns:
        return
    counts = observable.groupby("uid", observed=True)[list(group_by)].nunique()
    spanning = counts[(counts > 1).any(axis=1)]
    if len(spanning):
        raise ComparisonError(
            "%d track identifier(s) appear under more than one group, so a pooled fit "
            "would compute displacements between molecules in different groups. First "
            "few: %s. Qualify the track id by group before pooling; separate projects "
            "often reuse the same movie names."
            % (len(spanning), list(spanning.index[:5])))


def _unit_spread(frame, n_pop, unit, weighting, fit_kwargs):
    """Per-unit fractions with D shared across the units of ONE group.

    Returns ``(sd_array, n_units)``, with ``sd`` NaN below two units: a single unit is
    not a spread, and reporting zero there would read as perfect agreement. The spread
    is formed the same way for both fit modes so the two are comparable.
    """
    from .aggregate import fit_by_unit

    if unit not in frame.columns:
        return None, 0
    try:
        per_unit = fit_by_unit(frame, n_pop=n_pop, unit=unit, shared_D=True,
                               weighting=weighting, group_columns=(), **fit_kwargs)
    except Exception:                        # noqa: BLE001 - a spread is optional
        return None, 0
    usable = per_unit[per_unit["ok"]] if "ok" in per_unit else per_unit
    if usable.empty:
        return None, 0
    values = np.column_stack([usable["fraction_%d" % i].to_numpy(float)
                              for i in range(1, n_pop + 1)])
    sd = (values.std(axis=0, ddof=1) if len(values) > 1
          else np.full(n_pop, float("nan")))
    return sd, len(values)


def bubble_series(
    observable: pd.DataFrame,
    *,
    fit_modes=BUBBLE_FIT_MODES,
    n_pops=(2,),
    group_by=DEFAULT_GROUP_BY,
    unit: str = "day",
    weighting: str = "track",
    method: str = "ecdf_lsq",
    **fit_kwargs,
) -> pd.DataFrame:
    """Several fits of the same data in ONE tidy table, for drawing together.

    Each **series** is one ``(fit_mode, n_pop)`` combination; every row is one population
    of one group under one series. Nothing is merged across series: a fraction always
    travels with the ``n_pop`` and the ``fit_mode`` that produced it, because "38 % of
    one of two states" and "38 % of one of three states" are different quantities, and
    keeping them apart is what this module is for. Drawing them side by side is a
    COMPARISON OF MODELS; putting them in one column would be the error
    ``compare_fractions`` refuses.

    ``fit_modes``
        ``"independent"`` fits every group on its own, so each keeps its own ``D``. That
        is honest about ``D`` and leaves the fractions as percentages of different
        things. ``"simultaneous"`` fits ``D`` ONCE on the pooled data and then fits only
        the fractions per unit -- the legacy simultaneous design, and the only
        arrangement in which the groups' fractions mean one quantity. Neither is right
        alone, which is exactly why they are drawn together.

    The SD is BIOLOGICAL SPREAD over units, never a confidence interval: the uncertainty
    of the mean is smaller by sqrt(n). It is formed identically for both modes.
    """
    from .aggregate import fit_by_unit                     # local: keeps import light

    group_by = _resolve_group_by(observable, group_by)
    tau_lag_s = observable.attrs.get("lag_time_s")
    modes = tuple(fit_modes)
    counts = tuple(int(n) for n in n_pops)
    if not modes:
        raise ComparisonError("no fit modes requested")
    if not counts:
        raise ComparisonError("no population counts requested")
    for mode in modes:
        if mode not in BUBBLE_FIT_MODES:
            raise ComparisonError(
                "unknown fit mode %r; expected one of %s" % (mode, BUBBLE_FIT_MODES))
    if "simultaneous" in modes:
        _refuse_shared_tracks(observable, group_by)
        if unit not in observable.columns:
            raise ComparisonError(
                "the simultaneous fit needs a %r column to form units from; have %s"
                % (unit, sorted(observable.columns)))

    rows: list[dict] = []
    for n_pop in counts:
        usable = None
        shared_D = None
        if "simultaneous" in modes:
            qualified = _qualified_unit(observable, unit, group_by)
            qualified.attrs.update(observable.attrs)
            per_unit = fit_by_unit(qualified, n_pop=n_pop, unit=unit, shared_D=True,
                                   weighting=weighting, group_columns=group_by, method=method,
                                   **fit_kwargs)
            usable = per_unit[per_unit["ok"]] if "ok" in per_unit else per_unit
            if usable.empty:
                raise ComparisonError(
                    "the simultaneous fit at n_pop=%d produced no usable unit; %d "
                    "unit(s) were formed and every one failed"
                    % (n_pop, len(per_unit)))
            shared_D = np.asarray(
                [float(usable["D_%d" % i].iloc[0]) for i in range(1, n_pop + 1)])
            usable = usable.assign(
                _label=_group_label_series(usable, group_by))

        for key, label, frame in _iter_groups(observable, group_by):
            for mode in modes:
                if mode == "independent":
                    try:
                        fit = fit_sqd(frame, n_pop=n_pop, tau_lag_s=tau_lag_s,
                                      method=method, weighting=weighting, **fit_kwargs)
                    except SQDError as exc:
                        raise ComparisonError(
                            "group %r could not be fitted at n_pop=%d: %s"
                            % (label, n_pop, exc)) from exc
                    D = np.asarray(fit.D, dtype=float)
                    fractions = np.asarray(fit.fractions, dtype=float).ravel()
                    sd, n_units = _unit_spread(frame, n_pop, unit, weighting, {**fit_kwargs, "method": method})
                else:
                    part = usable[usable["_label"] == label]
                    if part.empty:
                        continue
                    D = shared_D
                    per_day = np.column_stack(
                        [part["fraction_%d" % i].to_numpy(float)
                         for i in range(1, n_pop + 1)])
                    fractions = per_day.mean(axis=0)
                    sd = (per_day.std(axis=0, ddof=1) if len(per_day) > 1
                          else np.full(n_pop, float("nan")))
                    n_units = len(per_day)

                for population, idx in enumerate(np.argsort(D), start=1):
                    rows.append({
                        "series": "%s, %d pop" % (mode, n_pop),
                        "fit_mode": mode,
                        "n_pop": n_pop,
                        "condition": label,
                        "population": population,
                        "D_um2_per_s": float(D[idx]),
                        "fraction": float(fractions[idx]),
                        "fraction_sd": (float("nan") if sd is None
                                        else float(sd[idx])),
                        "n_units": int(n_units),
                        "n_observations": int(len(frame)),
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ComparisonError("no series formed -- the observable is empty")
    out.attrs["fit_modes"] = modes
    out.attrs["n_pops"] = counts
    out.attrs["unit"] = unit
    out.attrs["weighting"] = weighting
    return out
