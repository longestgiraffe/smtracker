"""The shape the onboarding screens read: Series, Day, Group and Project, with every total derived (no Qt, no colours).

The INTERFACE between the onboarding Confirm screens and the data. `ui/onboarding/adapt` builds these from a loaded
`smtracker.project.Project`, and the screens (`ConfirmPage`, `ConditionBody`, `ReplicateDays`, `GroupsPanel`, ...)
only read them. It lives outside `ui/` because it is plain data and rules: no Qt, no colours, no sizes, so it imports
with Qt blocked and its totals are tested headless (`tests/test_ui_thin_qt.py`, `tests/test_onboarding_model.py`).

Every number the Confirm screen prints comes from here and is DERIVED rather than typed twice: a lane footer sums its
own cards, the counts row sums the lanes, and the day table sums the series of that day. A figure that disagrees with
another is therefore impossible to author.

The rules kept here: a series is in the fit only when it has a group AND has not been held out (`is_included`);
holding a series out does not clear its group, so re-ticking restores the placement; `on_disk` is every series, the
denominator the counts row measures against; and `reset_to_days` is the factory reset, one group per imaging day.
What is held out or left without a group is REMEMBERED with the grouping (`left_out` / `apply_left_out`, saved in
``day_map.json`` as ``held_out`` and ``no_group``; tester D11, ruled 4 Nov 2026): before, it came back at the next open.

**PAIRED REPLICATES, OPT-IN** (5 Nov 2026, asked: "optional pair ... in day picker?", "a tick box for activate pairing
in day picker"). A group may carry a pair TAG (``Group.pair``, "A", "B", ...): a group of another condition with the
same tag is its partner (one culture split, one imaging session), and the significance brackets' paired test reads
them. Set only by a person on the board -- or proposed from shared imaging dates (`propose_pairs`) for a person to
keep or clear -- never from replicate numbers (Rule 0). A tag is used at most once in a condition (`free_tags`).

**A REPLICATE IS NOT AN IMAGING DAY** (30 Oct 2026: "we need to differ between replicate and days. sometimes two
different replicates are made on the same day and the user specified and flagged it already"). ``days`` are the imaging
sessions the files record (the Groups board's "Reset" goes back to them); ``replicates`` are the replicates SAVED in
``day_map.json`` -- what the workspace fits over -- as rows of the same shape, each dated by its earliest movie, so two
replicates imaged on one date both carry that date. The record view lists `record_rows`: the saved replicates where
there are any (tagged R1, R2 ...), else the imaging days, which are then what the board proposes as replicates. With
saved replicates the record view shows BOTH labels (asked: "maybe we can have two labels as well D1 for day 1 and R1
for the group it is in ... in the end the chosen group / replicate is what goes into analysis and not the day, the day
is basically a helper to sort it and automatically detect replicates"): each replicate row names the imaging days its
series were taken on (`record_days`).

The prototype's fixtures (`reference_project`, `undated_project`, `conditions`, `RECENTS`) stay in
`smtracker.ui.onboarding.model`, built from these classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

__all__ = ["Replicate", "Group", "Project", "Series", "PAIR_TAGS", "free_tags", "pair_note", "propose_pairs"]

#: The tags a pair can be given on the board, in the order offered.
PAIR_TAGS = tuple("ABCDEFGHIJKL")


@dataclass(frozen=True)
class Series:
    """One movie folder. ``day`` is read from the files; ``group`` is chosen."""

    name: str
    day: int | None          # None == the files recorded no date
    found: int               # tracks the tracker found
    kept: int                # tracks that survive outline + length filters
    cells: int               # segmented cells in this series
    localizations: int

    @property
    def keep_rate(self) -> float:
        return self.kept / self.found if self.found else 0.0


@dataclass
class Day:
    """A replicate day. Immutable record: this is what the files say."""

    index: int
    #: The imaging date, read from each movie's ``movieData.mat``. None only
    #: when the files really do not say -- see `adapt`.
    on: date | None
    #: True when the date came from ``createTime_``, u-track's IMPORT time,
    #: rather than the raw file or ``acquisitionDate_``. `daymap` flags this
    #: distinction and so does the screen: an import date is not when a movie
    #: was taken -- protein A/Malate's three sessions, eight months apart, were all
    #: imported on one date -- so it is drawn as a proxy and never decides a day.
    date_is_proxy: bool = False
    frames: int = 1501
    note: str = ""
    #: "D" for an imaging day, "R" for a replicate the user saved (`Project.replicates`)
    tag_prefix: str = "D"

    @property
    def tag(self) -> str:
        return f"{self.tag_prefix}{self.index}"


@dataclass
class Group:
    """An analysis group. Editable: this is the user's choice."""

    name: str
    members: list[str] = field(default_factory=list)
    #: The pair tag ("A", ...) this replicate shares with one replicate of another condition, or "" (no pair). Read
    #: only while the board's Pair replicates box is ticked.
    pair: str = ""


@dataclass
class Project:
    name: str
    path: str
    condition: str
    days: list[Day]
    series: list[Series]
    groups: list[Group]
    #: The tracker's own acquisition metadata (`canonical.AcquisitionParams`),
    #: or None on a screen drawn before a load. `panels.ParamRow` reads it; the
    #: mock had these five values typed into a fixture.
    params: object = None
    #: THIS condition's five acquisition cards, ``[(value, caption)]`` in `panels.FACTS` order
    #: (`onboarding_view.condition_cards`); None on a screen drawn before a load, which then reads ``params``.
    cards: list | None = None
    #: Where opening the workspace writes the canonical cache: ``<the folder opened>/.smtracker_warm``, which is
    #: the PROJECT folder, never a condition's (tester B15, 3 Nov 2026). "" on a screen drawn before a load.
    cache_dir: str = ""
    #: The line said under the cards when the conditions were not recorded alike
    #: (`onboarding_view.acquisition_differences`), else "".
    acquisition_note: str = ""
    #: Series held out by the user. Group membership is NOT cleared by this —
    #: the two are independent facts, so re-ticking restores the placement.
    excluded: set[str] = field(default_factory=set)
    #: THE REPLICATES SAVED in ``day_map.json``, one `Day` (tagged "R") per replicate, dated by its earliest movie;
    #: None when nothing was saved. ``replicate_series`` names each one's series. See the module docstring.
    replicates: list[Day] | None = None
    replicate_series: dict = field(default_factory=dict)

    # -- lookups ------------------------------------------------------------ #
    def series_by_name(self, name: str) -> Series:
        for s in self.series:
            if s.name == name:
                return s
        raise KeyError(name)

    def day(self, index: int) -> Day:
        for d in self.days:
            if d.index == index:
                return d
        raise KeyError(index)

    def group_of(self, name: str) -> Group | None:
        for g in self.groups:
            if name in g.members:
                return g
        return None

    def series_of_day(self, index: int | None) -> list[Series]:
        return [s for s in self.series if s.day == index]

    # -- the record view: the saved replicates, else the imaging days ------- #
    def record_rows(self) -> list[Day]:
        """What the record view lists: the saved replicates, else the imaging days."""
        return self.days if self.replicates is None else self.replicates

    def record_series(self, index: int) -> list[Series]:
        """The series of record row ``index`` (a saved replicate's, else an imaging day's)."""
        if self.replicates is None:
            return self.series_of_day(index)
        names = set(self.replicate_series.get(index, ()))
        return [s for s in self.series if s.name in names]

    def record_days(self, index: int) -> list[Day]:
        """The imaging days record row ``index``'s series were taken on: one per row of imaging days; for a saved
        replicate, the days it spans (R2 and R3 imaged on one day both name D2)."""
        on = {s.day for s in self.record_series(index) if s.day is not None}
        return [d for d in self.days if d.index in on]

    def record_totals(self, index: int) -> dict:
        rows = self.record_series(index)
        return {
            "series": len(rows),
            "cells": sum(r.cells for r in rows),
            "kept": sum(r.kept for r in rows),
            "found": sum(r.found for r in rows),
        }

    # -- what is actually in the fit ---------------------------------------- #
    def is_included(self, name: str) -> bool:
        """In the fit iff it has a group AND has not been held out."""
        return self.group_of(name) is not None and name not in self.excluded

    def included(self) -> list[Series]:
        return [s for s in self.series if self.is_included(s.name)]

    @property
    def n_series(self) -> int:
        """For the acquisition captions: "30 series, one value"."""
        return len(self.series)

    def undated(self) -> list[Series]:
        return [s for s in self.series if s.day is None]

    def unplaced(self) -> list[Series]:
        return [s for s in self.series if self.group_of(s.name) is None]

    # -- totals, all derived ------------------------------------------------ #
    def totals(self, names: list[str] | None = None) -> dict:
        rows = ([self.series_by_name(n) for n in names] if names is not None
                else self.included())
        return {
            "series": len(rows),
            "cells": sum(r.cells for r in rows),
            "kept": sum(r.kept for r in rows),
            "found": sum(r.found for r in rows),
            "localizations": sum(r.localizations for r in rows),
        }

    def on_disk(self) -> dict:
        """The denominator every bar on the counts row is measured against."""
        return {
            "series": len(self.series),
            "cells": sum(s.cells for s in self.series),
            "kept": sum(s.kept for s in self.series),
            "found": sum(s.found for s in self.series),
        }

    def group_totals(self, group: Group) -> dict:
        live = [m for m in group.members if m not in self.excluded]
        t = self.totals(live)
        t["held_out"] = len(group.members) - len(live)
        t["days"] = sorted({self.series_by_name(m).day for m in group.members
                            if self.series_by_name(m).day is not None})
        return t

    def day_totals(self, index: int) -> dict:
        rows = self.series_of_day(index)
        return {
            "series": len(rows),
            "cells": sum(r.cells for r in rows),
            "kept": sum(r.kept for r in rows),
            "found": sum(r.found for r in rows),
        }

    # -- edits -------------------------------------------------------------- #
    def assign(self, name: str, group_name: str | None) -> None:
        for g in self.groups:
            if name in g.members:
                g.members.remove(name)
        if group_name is None:
            return
        for g in self.groups:
            if g.name == group_name:
                g.members.append(name)
                g.members.sort()
                return
        raise KeyError(group_name)

    def set_included(self, name: str, on: bool) -> None:
        if on:
            self.excluded.discard(name)
        else:
            self.excluded.add(name)

    # -- the hold-outs, remembered in day_map.json (tester D11, ruled 4 Nov 2026) -- #
    def left_out(self) -> dict:
        """``{"held_out": [...], "no_group": [...]}``: the series held out and the series with no group, as
        `daymap.save_day_map` remembers them (`DayMapping.held_out` / ``no_group``)."""
        names = {s.name for s in self.series}
        return {"held_out": sorted(n for n in self.excluded if n in names),
                "no_group": sorted(s.name for s in self.unplaced())}

    def apply_left_out(self, held_out=(), no_group=()) -> None:
        """Put back what an earlier session left out: each ``no_group`` series off its group, each ``held_out`` one
        held out (its group kept, so re-ticking restores it). Names this condition does not hold are ignored: the
        series is gone from the folder."""
        names = {s.name for s in self.series}
        for name in no_group:
            if name in names:
                self.assign(name, None)
        self.excluded |= {name for name in held_out if name in names}

    def reset_to_days(self) -> None:
        """The factory reset: one group per imaging day, in date order."""
        self.groups = [Group(f"Group {d.index}",
                             [s.name for s in self.series_of_day(d.index)])
                       for d in self.days]

    def split(self, n_groups: int) -> list[list[str]]:
        """``n_groups`` groups cut along the SERIES NUMBERING: day-blind, consecutive.

        Asked for: "really just manually set 3 groups -> 3 groups, not caring
        about the date". The fixed-size `regroup` cannot do that whenever the
        days are unequal -- no ``3 x per`` cuts 29 series as 9 / 10 / 10 -- so
        here the count is the only input and the cuts go where the numbering
        itself breaks, in this order:

          1. a jump in the numbering (009 -> 031), the widest first;
          2. the start of a new block of ten (040 -> 041), the lab's
             ten-series-to-a-day convention;
          3. while more groups are asked for than those give, the largest group
             is halved.

        A condition of series 001-009 and 031-050 into three is therefore 001-009 / 031-040
        / 041-050 -- its three sessions -- without a date being read. Names with
        no series number have no breaks and are halved evenly. More groups than
        series leaves the surplus empty, drawn as empty, as `regroup` does.
        Returns the proposal WITHOUT applying it -- the dialog previews first.
        """
        from .grouping import GroupingError, series_number

        if n_groups < 1:
            raise ValueError(f"need at least one group, got {n_groups}")
        names = [s.name for s in self.series if s.day is not None]
        try:
            numbers = [series_number(n) for n in names]
        except GroupingError:
            numbers = None
        breaks = []                       # (tier, -width, position of the cut)
        if numbers is not None:
            for i in range(1, len(names)):
                a, b = numbers[i - 1], numbers[i]
                if b - a > 1:
                    breaks.append((0, -(b - a), i))
                elif (max(a, 1) - 1) // 10 != (max(b, 1) - 1) // 10:
                    breaks.append((1, 0, i))
        cuts = sorted(i for _tier, _width, i in sorted(breaks)[:n_groups - 1])
        blocks = [names[a:b] for a, b in zip([0, *cuts], [*cuts, len(names)])]
        while len(blocks) < n_groups:
            k = max(range(len(blocks)), key=lambda j: len(blocks[j]))
            big = blocks[k]
            if len(big) < 2:
                blocks.append([])
                continue
            half = (len(big) + 1) // 2
            blocks[k:k + 1] = [big[:half], big[half:]]
        return blocks

    def regroup(self, n_groups: int, per_group: int) -> list[list[str]]:
        """Linear, day-blind: consecutive blocks in file order.

        The last group takes whatever is left, so 28 into 6 x 5 is
        5·5·5·5·5·3 rather than a spread nobody asked for. Returns the
        proposal WITHOUT applying it — the dialog previews first.
        """
        names = [s.name for s in self.series if s.day is not None]
        blocks: list[list[str]] = []
        i = 0
        for k in range(n_groups):
            if i >= len(names):
                blocks.append([])
                continue
            take = len(names) - i if k == n_groups - 1 else per_group
            blocks.append(names[i:i + take])
            i += take
        return blocks

    def apply_regroup(self, blocks: list[list[str]]) -> None:
        self.groups = [Group(f"Group {i + 1}", list(b))
                       for i, b in enumerate(blocks)]


def free_tags(project: Project, group: Group) -> list[str]:
    """The tags ``group`` may take: every `PAIR_TAGS` tag not on another group of its condition (a replicate has one
    partner per other condition, so a tag names one replicate of each), its own included."""
    taken = {g.pair for g in project.groups if g is not group and g.pair}
    return [tag for tag in PAIR_TAGS if tag not in taken]


def _first_date(project: Project, group: Group):
    """The date ``group``'s earliest included movie was imaged, or None (no date, or only u-track's import date,
    which is not when a movie was taken: `Day.date_is_proxy`)."""
    dates = []
    for name in group.members:
        if not project.is_included(name):
            continue
        s = project.series_by_name(name)
        if s.day is None:
            continue
        day = project.day(s.day)
        if day.on is not None and not day.date_is_proxy:
            dates.append(day.on)
    return min(dates, default=None)


def propose_pairs(projects) -> dict:
    """``{condition: {group name: tag}}``: the board's PROPOSAL from the imaging dates, for a person to keep or clear.
    Replicates of different conditions whose earliest movies were imaged on one date share a tag, one per date in
    date order (A, B, ...); a date only one condition has, a condition with two replicates on one date (which of the
    two would the partner be?) and a replicate without a real date get none. Every group of every condition is
    listed, "" where it gets no tag, so applying the proposal replaces what was there."""
    projects = list(projects)
    on_date: dict = {}
    for p in projects:
        for g in p.groups:
            if not any(p.is_included(m) for m in g.members):
                continue
            when = _first_date(p, g)
            if when is not None:
                on_date.setdefault(when, []).append((p.condition, g.name))
    tags: dict = {}
    letters = iter(PAIR_TAGS)
    for when in sorted(on_date):
        found = on_date[when]
        per = {}
        for condition, name in found:
            per.setdefault(condition, []).append(name)
        single = {c: names[0] for c, names in per.items() if len(names) == 1}
        if len(single) < 2:
            continue
        tag = next(letters, None)
        if tag is None:
            break
        for condition, name in single.items():
            tags.setdefault(condition, {})[name] = tag
    return {p.condition: {g.name: tags.get(p.condition, {}).get(g.name, "") for g in p.groups} for p in projects}


def _date_words(when) -> str:
    return f"{when.day} {when.strftime('%b')}"


def pair_note(projects, project: Project, group: Group) -> tuple[str, str]:
    """``(note, tip)`` beside a replicate's pair tag on the board: which other conditions share its tag, and whether
    their replicates were imaged on its date (the earliest included movie's, as the proposal reads it: `_first_date`),
    so a wrong tag shows before the workspace opens. Asked 5 Nov 2026 (the date check, "yes"). A CHECK, never a rule:
    a culture split and imaged on two days is still a pair, so "dates differ" only asks a person to look."""
    tag = group.pair
    if not tag:
        return "", ""
    partners = [(p, g) for p in projects if p.condition != project.condition
                for g in p.groups if g.pair == tag]
    if not partners:
        return "no partner yet", f"No other condition has a replicate tagged {tag}."
    names = ", ".join(p.condition for p, _g in partners)
    mine = _first_date(project, group)
    theirs = [(p.condition, _first_date(p, g)) for p, g in partners]
    if mine is None or any(when is None for _c, when in theirs):
        return (f"with {names}",
                "No imaging date to compare: a movie has no date, or only u-track's import date.")
    if all(when == mine for _c, when in theirs):
        return f"with {names} · same date", f"Imaged on {_date_words(mine)}, as its partner{'s' * (len(theirs) > 1)}."
    listed = "; ".join(f"{c}: {_date_words(when)}" for c, when in theirs)
    return (f"with {names} · dates differ",
            f"This replicate: {_date_words(mine)}; {listed}. A pair is one culture or one imaging session, so "
            f"check the tag. A culture imaged on two days is still a pair.")
