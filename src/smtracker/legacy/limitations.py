"""One short line per legacy item, naming its principal limitation.

Why this exists. The policy is that version 3 fixes the defects and adds what the old
tool could not do, while the old outputs stay available because people need to recognise
their own figures. Supported, reproduced faithfully in design, not recommended for a
claim.

**That policy is only real if the distinction appears at the point of use.** A document
nobody opens while looking at a figure does not carry it. So every legacy figure prints
its own limitation, and the same string goes into the provenance.

One line, not a paragraph. A legacy figure covered in caveats stops looking like the
figure the user recognises, which defeats the purpose of reproducing it. The line names
the single largest reason the figure should not carry a claim, and nothing else. The
full account belongs in the provenance JSON and in the deposited tables.

ONE SOURCE. The figure and the provenance read the same string from here, so they cannot
drift apart.
"""
from __future__ import annotations

import re

__all__ = ["LIMITATION_LINES", "limitation_line", "LimitationError"]


class LimitationError(KeyError):
    """Raised when a legacy item has no limitation line.

    Not having one is a mistake rather than a default. A legacy figure without its line
    is exactly the figure this module exists to prevent.
    """


#: The principal limitation of each legacy item, in one line each.
#:
#: A LINE MAY NOT CARRY A HARD-CODED MEASUREMENT. Any number that varies between groups
#: is a ``{field}`` placeholder, filled from the group being drawn. A literal here is the
#: same defect class as a caption number that can drift from its figure, which this
#: project already fixed on the Python side. ``limitation_line`` RAISES when a field is
#: not supplied, so a figure cannot ship with an unfilled or stale number.
LIMITATION_LINES = {
    "sqd": (
        "legacy interval: coverage 3 to 4 of 200 seeded runs against 190 expected; "
        "point estimates consistent with no bias"),
    "speedmap": (
        "drawn from tracks of at least {min_length} localisations: "
        "{kept} of {total} tracks, {percent_kept:.0f} per cent"),
    # THE CONVENTION, NOT THE UP-SCALING, IS THE LARGEST REASON THIS MAP SHOULD NOT
    # CARRY A CLAIM. The earlier line named the sub-pixel division, which is real but is
    # a display defect: it adds no information and removes none. The midpoint convention
    # CHANGES THE NUMBER a reader takes off the map, and in the direction that flatters a
    # radial claim: measured on all twelve groups it reads higher than the step-start
    # convention in 12 of 12, median 2.92x against 1.63x. Those are the SDA map's OWN
    # numbers; the 3.90x against 2.70x pair recorded elsewhere belongs to the HMM
    # per-step state map and is not interchangeable with them.
    "sda": (
        "step positions are the legacy MIDPOINT, which reads a radial contrast of "
        "{midpoint_contrast:.2f} times where version 3's step-start convention reads "
        "{start_contrast:.2f} times on the same tracks; bin counts are not up-scaled here"),
    # TWO LIMITATIONS COMPETE HERE AND THE MEASURED ONE WINS. The chance rate is the
    # bigger effect on the CONFINED FRACTION, and it is already measured per group by
    # confinement_null, so it carries its own numbers. The classification's limitation is
    # different in kind: a pooled-curve F-test whose sub-diffusive verdict is predicted by
    # mixing and survivorship alone. The line names whichever is being drawn, so the
    # figure never carries the other one's caveat.
    "sla": (
        "confined {confined_pct:.0f} per cent of segments against a chance rate of "
        "{chance_pct:.0f} per cent on matched free Brownian tracks, excess "
        "{excess_pp:.0f} percentage points"),
    "sla_classification": (
        "an F-test on ONE POOLED MSD curve: mixing and survivorship bend it downward with "
        "no anomalous motion present, and at this scatter it calls a true exponent of 0.6 "
        "Brownian in {missed} of {tried} draws"),
    # THE STATEMENT THAT HOLDS ON BOTH FEATURE SCALES. The earlier line said the
    # partition does not exceed chance, which is the RAW-scale answer: on the log scale
    # the same data sits 19.2 SD ABOVE chance. Shipping one scale's conclusion as the
    # line would leave the corrected finding alive in the document and dead in the
    # output. The two scales disagree, and under the standing rule the disagreement is
    # the finding.
    "clustering": (
        "far closer to one population than to two separated states; the raw and log "
        "feature scales disagree on whether it exceeds chance at all"),
}


def limitation_line(item: str, **facts) -> str:
    """The one-line limitation for a legacy item, with its measured fields filled in.

    Raises if the item has no line, and raises if the line needs a measurement that was
    not supplied. Both are mistakes rather than defaults: a legacy figure without its
    line, or with a number that does not belong to the group being drawn, is exactly the
    figure this module exists to prevent.
    """
    try:
        template = LIMITATION_LINES[item]
    except KeyError:
        raise LimitationError(
            f"no limitation line for legacy item {item!r}. Every legacy figure prints "
            f"one, because the policy that legacy output is not recommended for a claim "
            f"is only real if it appears at the point of use. Known items: "
            f"{sorted(LIMITATION_LINES)}") from None
    try:
        return template.format(**facts)
    except (KeyError, IndexError) as exc:
        needed = sorted(set(re.findall(r"\{(\w+)", template)))
        raise LimitationError(
            f"the limitation line for {item!r} needs measured values {needed} and was "
            f"given {sorted(facts)}. The number must come from the group being drawn. A "
            f"hard-coded one can drift from its own figure.") from exc
