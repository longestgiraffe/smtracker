"""The onboarding prototype's FIXTURES: the reference project the Confirm screen's design was checked against.

The data model itself (Series, Day, Group, Project and every derived total) lives in the headless
`smtracker.onboarding_model`, which `adapt` builds real projects with; it is re-exported here so the fixtures below
read as they always did. Nothing in the application uses these fixtures, only tests and design checks do
(`tests/test_overlay_dialogs.py` reads `reference_project`).

The shape is Concept 4's: 28 series over 7 imaging days, six analysis groups -- five of them one whole day, the
sixth two days merged.
"""

from __future__ import annotations

from datetime import date

from ...onboarding_model import Day, Group, Project, Series

__all__ = ["RECENTS", "Replicate", "Group", "Project", "Series", "conditions", "reference_project", "undated_project"]


# --------------------------------------------------------------------------- #
# The reference project, matching Concept 4 exactly.
# --------------------------------------------------------------------------- #

_DAY_DATES = [date(2025, 3, 4), date(2025, 3, 11), date(2025, 3, 18),
              date(2025, 3, 25), date(2025, 4, 8), date(2025, 4, 15),
              date(2025, 4, 22)]

# (found, kept, cells) per series, four per day, in file order.
_ROWS = [
    (205, 148, 6), (176, 121, 5), (198, 139, 5), (213, 152, 6),   # D1  560/22
    (210, 151, 6), (193, 138, 5), (208, 149, 6), (198, 142, 5),   # D2  580/22
    (188, 133, 6), (179, 126, 5), (180, 131, 5), (201, 144, 5),   # D3  534/21
    (172, 125, 6), (180, 130, 5), (174, 122, 5), (176, 128, 5),   # D4  505/21
    (188, 130, 5), (189, 136, 5), (159, 113, 6), (170, 119, 5),   # D5  498/21
    (189, 131, 6), (191, 136, 6), (192, 139, 6), (189, 132, 5),   # D6  538/23
    (187, 133, 5), (189, 138, 6), (186, 130, 5), (184, 136, 5),   # D7  537/21
]


def reference_project() -> Project:
    days = [Day(i + 1, d) for i, d in enumerate(_DAY_DATES)]
    days[5].frames = 1001
    days[5].note = "session ended early: 1 001 frames, not 1 501"

    series: list[Series] = []
    for i, (found, kept, cells) in enumerate(_ROWS):
        series.append(Series(
            name=f"series{i + 1:03d}",
            day=i // 4 + 1,
            found=found, kept=kept, cells=cells,
            localizations=kept * 8,
        ))

    groups = [Group(f"Group {n}") for n in range(1, 7)]
    for i, s in enumerate(series):
        # Groups 1-5 are one day each; group 6 merges days 6 and 7.
        groups[min(i // 4, 5)].members.append(s.name)

    return Project(
        name="mNeonGreen_glucose",
        path=r"D:\Imaging\2025-03 campaign",
        condition="Glucose",
        days=days, series=series, groups=groups,
    )


def undated_project() -> Project:
    """The anomaly: three series whose files recorded no date."""
    p = reference_project()
    extra = [Series(f"series{29 + i:03d}", None, 184 - i, 131 - i, 5, 1045 - i)
             for i in range(3)]
    p.series.extend(extra)
    p.assign(extra[0].name, "Group 3")     # resolved
    return p                                # the other two stay unplaced


def _condition(name: str, path: str, n_days: int, per_day: int,
               tail: int, seed: int) -> Project:
    """A second and third condition, built by the same rules as the first.

    Generated rather than typed so every total stays derived: the tab strip
    prints each condition's own N, and those numbers have to be the ones its
    body will show when you switch to it.
    """
    from datetime import timedelta

    start = date(2025, 3, 4) + timedelta(days=seed)
    days = [Day(i + 1, start + timedelta(days=7 * i)) for i in range(n_days)]

    series: list[Series] = []
    for d in range(n_days):
        count = tail if (d == n_days - 1 and tail) else per_day
        for k in range(count):
            found = 168 + (seed * 7 + d * 11 + k * 13) % 46
            kept = int(found * (0.66 + ((d + k + seed) % 9) * 0.012))
            series.append(Series(
                name=f"series{len(series) + 1:03d}",
                day=d + 1, found=found, kept=kept,
                cells=5 + (d + k) % 2, localizations=kept * 8))

    groups = [Group(f"Group {i + 1}") for i in range(n_days)]
    for s in series:
        groups[s.day - 1].members.append(s.name)

    return Project(name=f"mNeonGreen_{name.lower()}", path=path,
                   condition=name, days=days, series=series, groups=groups)


def conditions() -> dict[str, Project]:
    """The three conditions the Confirm screen's tab strip offers.

    Glucose is the reference project — the one every number in the design was
    checked against. The other two exist so the tab strip is a real control
    rather than a picture of one.
    """
    root = r"D:\Imaging\2025-03 campaign"
    return {
        "Glucose": reference_project(),
        "Malate": _condition("Malate", root, 8, 4, 3, 3),
        "Succinate": _condition("Succinate", root, 6, 4, 0, 6),
    }


RECENTS = [
    ("Project A 2024", r"D:\data\project-a\2024-experiments",
     "3 conditions \u00b7 26 series", "opened yesterday", True),
    ("Titration run", r"D:\data\protein\titration-run2",
     "8 conditions \u00b7 61 series", "4 days ago", True),
    ("Fixed-cell control", r"D:\data\controls\fixed-2024-02",
     "1 condition \u00b7 6 series", "2 weeks ago", True),
    ("Malate pilot", r"D:\pilot\malate", "", "folder not found", False),
]
