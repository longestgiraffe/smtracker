"""A loaded project, in the shape the Confirm screens already read.

The screens are the prototype's, unchanged — `ConfirmPage`, `ConditionBody`,
`ReplicateDays`, `GroupsPanel` and the rest all read `model.Project`: a plain
dataclass of `Series`, `Day` and `Group` with every total derived from them. So
the port is one function, not a rewrite: build those three lists from
`smtracker.project.Project` and hand them over.

THE VIEWS ARE NOT TOUCHED, and that is the point. An earlier attempt
reimplemented the screen against real data and produced something that looked
nothing like the design; adapting the DATA leaves the design exactly as it was
argued out.

WHERE EACH NUMBER COMES FROM

    Series.name           the movie folder, as `discover_utrack_project` names it
    Series.kept           uid.nunique() on the canonical table — POST-filter
    Series.found          canonical.attrs["tracks_found"] — a VIEW of
                          attrs["cell_filter_by_movie"], which cellmesh.assign_cells
                          records where the filter runs, since that is the only
                          place the dropped tracks still exist to be counted. The
                          loader used to count them a second time off the raw
                          table; two loops over one quantity is what hard rule 5
                          forbids, and they would drift the first time either
                          changed what "a track" means.
    Series.cells          cell.nunique()
    Series.localizations  the row count
    Series.day            the IMAGING day: by acquisition time, else by the series naming
                          (`onboarding_view.imaging_days`);
                          the loaded day column is the confirmed GROUPING, which differs once
                          "Open workspace" has saved one, and the groups are built from it
    Day.frames            the highest frame index seen for that day, + 1
    Day.on                the earliest movie's date (`_stamps`): the raw `.nd2`/`.stk`'s own
                          time, else `movieData.mat`'s ``acquisitionDate_``, or ``createTime_``
                          (the u-track IMPORT time, flagged ``date_is_proxy``). None where
                          none can be read,
                          and `panels.ReplicateDays` prints an em dash: *the files do not
                          say*, where a file's modification time would be a plausible wrong
                          answer.

GROUPS DEFAULT TO ONE PER DAY, which is `model.Project.reset_to_days()`, the
model's own factory reset -- unless a grouping was confirmed and saved, which the
board then opens on. It is also what this project already asserts
elsewhere: the replicate day is the unit a fit is grouped over.

WHAT WAS HELD OUT STAYS HELD OUT (tester D11, ruled 4 Nov 2026). The series held
out or left with "No group" are saved in ``day_map.json`` beside the grouping
(`left_out_from`, written by the window's Open workspace), and the board opens
with them held out again (`_saved_left_out`); before, they came back at the next
open, since the file had to keep them with their replicate.

PAIRED REPLICATES (5 Nov 2026, opt-in). The pair tags the board gave its groups
are saved in ``day_map.json`` keyed by the replicate numbers `day_maps_from`
gives the same groups (`pairs_from`), and the board opens with them on their
groups again (`_saved_pairs`); with the box unticked nothing is saved.
"""
from __future__ import annotations

from ...grouping import GroupingError
from ...onboarding_model import Day, Project, Series
from ...onboarding_view import (
    acquisition_differences,
    condition_cards,
    found_by_movie,
    frames_by_day,
    group_blocks,
    imaging_days,
    odd_day_notes,
    series_rows,
)

__all__ = ["day_maps_from", "left_out_from", "pairs_from", "projects_from"]


def projects_from(loaded, discovery=None) -> dict:
    """`{condition: model.Project}` from a loaded `smtracker.project.Project`."""
    out = {}
    note = acquisition_differences(loaded)
    for name in loaded.conditions:
        out[name] = _one(loaded, name, discovery)
        # EACH CONDITION'S OWN acquisition, not the project's merged copy (tester B1, 3 Nov 2026), and the line saying
        # where the conditions differ, which only QC said before
        out[name].cards = condition_cards(loaded, name)
        out[name].acquisition_note = note
        out[name].cache_dir = _cache_dir(loaded, discovery)
    return out


def day_maps_from(projects) -> dict:
    """The group board's CONFIRMED arrangement, going back the other way: the
    shape `project.regroup` and `day_map.json` already use --
    ``{condition: {original movie name: group number}}``.

    ``projects`` is an iterable of `model.Project` (`ConfirmPage.projects`,
    which the group board mutates in place, so this reads whatever is on
    screen right now). Only INCLUDED series are covered -- a series with no
    group, or held out, is the user's own choice, and `project.regroup` reads
    its absence from the mapping as exactly that: dropped, not given some
    placeholder day. Group numbers are the group's own position among the
    groups actually holding an included series, so an empty group in between
    never shifts anyone else's number.
    """
    result = {}
    for p in projects:
        active = [g for g in p.groups if any(p.is_included(m) for m in g.members)]
        index_of = {g.name: i + 1 for i, g in enumerate(active)}
        mapping = {s.name: index_of[p.group_of(s.name).name]
                   for s in p.series if p.is_included(s.name)}
        if mapping:
            result[p.condition] = mapping
    return result


def pairs_from(projects, on: bool) -> dict:
    """The pair tags the board gave its groups, the way `day_map.json` remembers them and the workspace reads them:
    ``{condition: {replicate number: tag}}``, numbered as `day_maps_from` numbers the same groups (their position among
    the groups holding an included series). Empty when the board's Pair replicates box is not ticked (``on``), and a
    condition without a tag is left out."""
    if not on:
        return {}
    result = {}
    for p in projects:
        active = [g for g in p.groups if any(p.is_included(m) for m in g.members)]
        tags = {i + 1: g.pair for i, g in enumerate(active) if g.pair}
        if tags:
            result[p.condition] = tags
    return result


def _saved_pairs(root: str) -> dict:
    """``{replicate number: tag}`` remembered in the condition's ``day_map.json``; empty where there is none or the
    file cannot be read."""
    from ...daymap import DayMapError, load_day_map

    try:
        mapping = load_day_map(root) if root else None
    except DayMapError:
        mapping = None
    return dict(getattr(mapping, "pairs", None) or {})


def left_out_from(projects) -> dict:
    """The group board's hold-outs, the way `day_map.json` remembers them: ``{condition: {"held_out": [...],
    "no_group": [...]}}`` (`model.Project.left_out`), for every condition on the board."""
    return {p.condition: p.left_out() for p in projects}


def _saved_left_out(root: str) -> dict:
    """``{"held_out": [...], "no_group": [...]}`` remembered in the condition's ``day_map.json``; empty lists where
    there is no file or it cannot be read (the loader said why already: a corrupt file is reported and derived)."""
    from ...daymap import DayMapError, load_day_map

    try:
        mapping = load_day_map(root) if root else None
    except DayMapError:
        mapping = None
    return {"held_out": list(getattr(mapping, "held_out", ()) or ()),
            "no_group": list(getattr(mapping, "no_group", ()) or ())}


def _one(loaded, condition: str, discovery=None) -> Project:
    table = loaded.condition(condition)
    found = found_by_movie(loaded.canonical)
    root = str(loaded.roots.get(condition, "")) if hasattr(loaded, "roots") else ""

    # What each movie holds comes from the headless view; this only fits it into the model the screens read.
    rows = series_rows(table, condition, found)
    # TWO DIFFERENT "DAYS", READ APART. The loaded `day` column is the GROUPING the workspace fits over -- after "Open
    # workspace" it is the board's own arrangement, read back from `day_map.json`. The imaging days come from the
    # movies' own acquisition times (the series names where a time is missing), so "Reset to days" always has real
    # days to go back to (`imaging_days`).
    stamps = _stamps(loaded, condition, [row["name"] for row in rows])
    grouping = {row["name"]: row["day"] for row in rows}
    imaging = imaging_days(grouping, stamps)
    series: list[Series] = [Series(**{**row, "day": imaging[row["name"]]}) for row in rows]

    days: list[Day] = []
    indices = sorted({s.day for s in series if s.day is not None})
    frames_of = frames_by_day(table, indices, imaging, condition)
    for index in indices:
        frames = frames_of[index]
        # THE DAY'S DATE IS ITS EARLIEST MOVIE'S. A replicate day is a session,
        # and a session that ran past midnight would otherwise show two dates
        # for one row; the first movie is the one that names it.
        mine = [stamps[s.name] for s in series
                if s.day == index and s.name in stamps]
        when = min((m[0] for m in mine if m[0]), default=None)
        days.append(Day(index=int(index), on=None if when is None else when.date(),
                        date_is_proxy=any(m[1] == "createTime_" for m in mine),
                        frames=frames))
    for day, note in zip(days, odd_day_notes([d.frames for d in days]), strict=True):
        if note:
            day.note = note

    project = Project(
        name=_project_name(loaded, discovery),
        path=root,
        condition=condition,
        days=days,
        series=series,
        groups=[],
        params=getattr(loaded, "params", None),
    )
    if _grouping_was_saved(root):
        # A grouping was confirmed and saved: the board opens on it, and Reset still goes back to the days.
        project.apply_regroup(group_blocks(grouping))
        # ...and the record view lists THOSE replicates, each dated by its earliest movie: two replicates imaged on
        # one date (one user's exponential and stationary free mNG, both 21 March) are two rows with one date, never one
        # row (asked 30 Oct 2026: "the record view should show the replicates correctly")
        project.replicates, project.replicate_series = _replicate_rows(table, condition, grouping, stamps)
        # ...and what was held out or left without a group stays so (D11): the file keeps those series with their
        # replicate, so without this they came back in the fit at the next open
        project.apply_left_out(**_saved_left_out(root))
        # ...and the pair tags it was given (5 Nov 2026): group i is the i-th loaded replicate (`group_blocks`)
        saved = _saved_pairs(root)
        if saved:
            for group, day in zip(project.groups, sorted({d for d in grouping.values() if d is not None})):
                group.pair = str(saved.get(int(day), ""))
    else:
        # Nothing confirmed yet: the model's own factory reset, one group per imaging day, in date order. Checked by
        # the FILE, not by comparing the two "days": the loader numbers days by series blocks, so with dates in hand
        # the two can differ on a first open, and that difference is not a decision anybody made.
        project.reset_to_days()
    return project


def _replicate_rows(table, condition: str, grouping: dict, stamps: dict) -> tuple:
    """``([Day per saved replicate, tagged R], {replicate: [series]})`` from the loaded grouping (the saved one): the
    record view's rows when a grouping was saved. A replicate's date is its earliest movie's, as an imaging day's is."""
    indices = sorted({day for day in grouping.values() if day is not None})
    members = {index: sorted(name for name, day in grouping.items() if day == index) for index in indices}
    frames_of = frames_by_day(table, indices, grouping, condition)
    rows = []
    for index in indices:
        mine = [stamps[name] for name in members[index] if name in stamps]
        when = min((m[0] for m in mine if m[0]), default=None)
        rows.append(Day(index=int(index), on=None if when is None else when.date(),
                        date_is_proxy=any(m[1] == "createTime_" for m in mine), frames=frames_of[index],
                        tag_prefix="R"))
    for row, note in zip(rows, odd_day_notes([r.frames for r in rows]), strict=True):
        if note:
            row.note = note
    return rows, members


def _grouping_was_saved(root: str) -> bool:
    """Whether this condition carries a saved day grouping (`day_map.json`) -- a directory check, nothing parsed."""
    from ...daymap import day_map_path

    return bool(root) and day_map_path(root).is_file()


def _stamps(loaded, condition: str, names=None) -> dict:
    """`{movie: (timestamp, source)}`: the raw acquisition's own time, else ``movieData.mat``'s.

    `daymap.read_acquisition_stamps` reads ``raw/<movie>.nd2``/``.stk`` first,
    then ``acquisitionDate_`` when u-track recorded one and ``createTime_`` --
    the IMPORT time -- when it did not, and says which. The source travels with
    the value for that reason: an import time is NOT reliable about the day
    either (protein A/Malate: three sessions eight months apart, all imported on one
    date), so it never decides one (`imaging_days`) and is drawn as a proxy.

    Best-effort: a project whose movies cannot be read still opens, with dashes
    where the dates would be.

    THE MOVIES ARE THE ONES LOADED (``names``, the board's rows), FROM THE TRACKER CHOSEN.
    This listed u-track's output whatever had been read: a condition tracked by both
    (asked for: "if user chooses trackmate over utrack -> ignore utrack tracking data for
    replicate day picker, vice versa") was dated from the other tracker's movies, and a
    TrackMate-only condition got no dates at all although its raw files carry them. A
    TrackMate movie has no movie folder, so the condition root stands in and only the raw
    acquisition dates it.
    """
    from pathlib import Path

    from ...daymap import read_acquisition_stamps
    from ...grouping import discover_utrack_project

    try:
        root = loaded.roots[condition]
    except (KeyError, AttributeError):
        return {}
    canonical = getattr(loaded, "canonical", None)
    tracker = (getattr(canonical, "attrs", {}) or {}).get("trackers", {}).get(condition)
    folders = {}
    if tracker not in ("trackmate", "table"):
        try:
            # `read_movie_stamps` wants the MOVIE FOLDER -- the one holding
            # `movieData.mat` -- and discovery returns the tracks file three levels
            # inside it.
            folders = {movie: path.parents[2]
                       for movie, path in discover_utrack_project(root)}
        except (GroupingError, OSError):               # no u-track output here
            folders = {}
    if names is not None:
        folders = {name: folders.get(name, Path(root)) for name in names}
    if not folders:
        return {}
    # "unreadable" comes back for a movieData.mat it cannot decode, and a raw file it cannot read keeps the movieData
    # stamp, so nothing else is expected to fail.
    return {s.movie: (s.timestamp, s.source) for s in read_acquisition_stamps(root, folders)}


def _cache_dir(loaded, discovery) -> str:
    """Where `project.load_project` writes the canonical cache: ``<the folder opened>/.smtracker_warm``."""
    from pathlib import Path

    if discovery is not None:
        return str(Path(discovery.root) / ".smtracker_warm")
    roots = [Path(r) for r in getattr(loaded, "roots", {}).values()]
    if not roots:
        return ""
    import os

    try:
        common = Path(os.path.commonpath([str(r) for r in roots]))
    except ValueError:                                 # roots on different drives: no one folder holds them
        return ""
    if len(roots) == 1:
        common = common.parent if common == roots[0] else common
    return str(common / ".smtracker_warm")


def _project_name(loaded, discovery) -> str:
    if discovery is not None:
        from pathlib import Path

        return Path(discovery.root).name or str(discovery.root)
    roots = list(getattr(loaded, "roots", {}).values())
    return roots[0].parent.name if roots else "project"
