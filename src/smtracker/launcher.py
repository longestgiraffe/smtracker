"""The first window: which STAGE of the work, not which analysis.

The experimenter's design, and the three doors are theirs rather than the code's:

    1. image preparation + tracking   organise the folder, set the crop range, run
                                      TrackMate over a whole condition (or import
                                      existing u-track results)
    2. cell outlines                  draw and correct the outlines the analysis
                                      assigns tracks to (the mesh editor). AUTOMATIC
                                      segmentation is a SEPARATE thing and is not
                                      built; this door is the editor.
    3. analysis                       the browser: fractions, kinetics, maps, QC

preparation is not a door of its own, and an earlier version of this menu had it as one.
That was the shape of the CODE, not of the work: tracking needs exactly one thing from
preparation (the frame range) and `crops.json` is the single record of it, so the two
were merged. Exporting cropped stacks still exists, because u-track cannot read a `.nd2`
and someone will need it; it is a button INSIDE the tracking window and `--prepare` still
opens it directly. Removing it from this list without rehoming it would have been the
defect this project keeps recording.

Tracking is not an analysis. Forcing it into the analysis catalogue would have run a
fifteen-minute batch job every time a panel opened, because that catalogue's defining
property is that it computes EVERY entry on EVERY loaded condition side by side. It also
PRODUCES the canonical table rather than consuming one, which every catalogue entry does.

What this must not do is make anything harder to reach. This project records that the
analysis catalogue and the track views once sat behind `--analyses --views`, flags the
no-argument flow never set, and nobody found them. A menu puts one click in front of the
analysis path, so analysis is the default: it is focused, it answers Return, and every
existing flag still goes straight where it always did (`--browse`, `--inspect`, `--qc`
never see this window).

A stage that is not built is shown disabled and says why. An enabled button that opens
nothing is worse than an absent one: the same rule as "no modal dialog on an empty
path", learned here when a `QMessageBox` on an empty path hung the suite for its whole
900 s timeout. Every door here is built today, so nothing is greyed.

The outlines door is the Editor, and its title says so. It read "Cell segmentation
+ Mesh editor" while it was disabled, which described a stage nobody could open fairly
enough. Enabling it under that title would have promised AUTOMATIC segmentation, which
is a separate thing and is not built, and a button that delivers half of what it names
is the failure this rule exists to prevent rather than an approximation of it. So the
door is named for what is behind it, and the card says plainly what is not.

Nothing is computed here. `choose_stage` returns a string and `app.main` acts on it, so
the routing is testable without Qt: the same thin-Qt rule as `widgets` and `statemaps`.
"""
from __future__ import annotations

__all__ = [
    "STAGES",
    "STAGE_PREPARE",
    "STAGE_SEGMENT",
    "STAGE_TRACK",
    "STAGE_ANALYSE",
    "STAGE_SIMULATE",
    "stage_keys",
    "enabled_stages",
    "describe_stages",
    "choose_stage",
]

STAGE_PREPARE = "prepare"
STAGE_TRACK = "track"
STAGE_SEGMENT = "segment"
STAGE_ANALYSE = "analyse"
#: The fourth door (25 Nov 2026, asked: "a fourth separate openable window after launching smtracker"): simulated
#: data with known states, from a recipe (`generator`, `ui.generator.window`).
STAGE_SIMULATE = "simulate"

#: ``(key, title, blurb, enabled)``, in WORKFLOW order rather than readiness order, a
#: user should see the shape of the whole tool, including the part not yet built.
#:
#: THREE DOORS, AS THE EXPERIMENTER DREW THEM. An earlier list offered "Tracking" and
#: "Export cropped stacks" as separate stages, which was the shape of the code and not
#: the shape of the work: preparation was merged INTO tracking (the crop range is the
#: only thing tracking needs from it, and `crops.json` stays the single record so the
#: two halves cannot disagree), so listing the export beside it advertised a stage that
#: no longer exists as one.
#:
#: EXPORTING CROPPED STACKS DID NOT DISAPPEAR, it moved. It is a button inside the
#: tracking window and `--prepare` still opens it directly, because it is genuinely
#: needed to feed u-track, which cannot read a `.nd2`. Dropping it from this menu
#: without rehoming it would be the defect this project keeps recording, a built,
#: tested capability reachable from nothing.
STAGES = (
    (
        STAGE_TRACK,
        "Image preparation + Tracking",
        # THE CARD SHOWS THE FIRST SENTENCE (`stages._summary`), so it must say what the stage does: it read "Raw
        # .nd2 or .stk." (tester B21, 3 Nov 2026); TIFF stacks named since 26 Nov 2026 (`rawmovie.discover_raw_movies` reads them)
        "Crop and calibrate raw movies (.nd2, .stk or TIFF), then track them with TrackMate (Fiji) or u-track (MATLAB)."
        "\n"
        "Track one condition at a time, or run many overnight in Batch. TrackMate takes about 30 s per movie.",
        True,
    ),
    (
        STAGE_SEGMENT,
        "Cell outlines: Segmentation + Mesh editor",
        "Find the cells automatically, then correct them by hand."
        "\n"
        "Segment with an Omnipose model or the classical segmenter, then correct outlines in Edit "
        "outlines. Choose the channel: phase contrast works for any protein, fluorescence only for a cytosolic label. "
        "You can train a model on your corrections, which are its labels.",
        True,
    ),
    (
        STAGE_ANALYSE,
        "Analysis",
        "Diffusive-state fractions, kinetics, spatial maps and QC, with every loaded condition side by side."
        "\n"
        "Choose a project folder and confirm its conditions and replicates; the analysis workspace opens next.",
        True,
    ),
    (
        STAGE_SIMULATE,
        "Simulated data",
        "Build a data set with known populations, D, places in the cell, switching and kinetics."
        "\n"
        "Set the cells, the camera and the replicates; build tracks for the analysis or movies for tracking, "
        "each with its counted truth to compare the results with.",
        True,
    ),
)


def stage_keys() -> tuple[str, ...]:
    return tuple(key for key, _title, _blurb, _enabled in STAGES)


def enabled_stages() -> tuple[str, ...]:
    return tuple(key for key, _title, _blurb, enabled in STAGES if enabled)


def describe_stages() -> str:
    """The menu as plain text, for a headless run or a `--help` line.

    One source of truth with the window: both read `STAGES`, so a stage cannot be
    enabled in one and absent from the other.
    """
    lines: list[str] = []
    for key, title, blurb, enabled in STAGES:
        mark = "" if enabled else "   [not built yet]"
        lines.append(f"  {key:<9} {title}{mark}")
        for part in blurb.splitlines():
            lines.append(f"            {part}")
    return "\n".join(lines)


def choose_stage(*, parent=None, default: str = STAGE_ANALYSE):
    """Show the launcher; return the chosen stage key, or ``None`` if cancelled.

    ``default`` is focused and answers Return, so the analysis path stays one
    keypress away: it is what every existing user expects `smtracker` to do.

    THE WINDOW LIVES IN `smtracker.ui.onboarding.stages`, and this module keeps
    the doors. That is the same split the rest of the project runs on -- the
    stages are data here, rendered as text by `describe_stages()` and as three
    cards by the window, and neither renderer owns the list. Qt is imported
    INSIDE the function, which keeps THIS module importable on the headless
    spine: `stage_keys()` and `describe_stages()` answer with no display, and
    `app.main`'s routing is testable without a widget.
    """
    from .ui.onboarding import stages
    return stages.choose_stage(parent=parent, default=default)
