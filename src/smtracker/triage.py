"""Getting-started / triage, run the checks in order, give a plain verdict.

Opening a project, a newcomer should not have to know to run QC before
poolability before comparing. This runs the existing checks in that order and
returns a plain verdict: **clear to compare**, or **resolve these first** with a
concrete list. It computes nothing new: it orchestrates the tested QC and
poolability functions and reads their honest flags.

Functional skeleton: the exact flow and wording are meant to be refined with
the experimenter. The *gates* are real and tested; the phrasing is a placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["TriageError", "TriageReport", "triage_project"]


class TriageError(RuntimeError):
    """Raised when triage cannot run."""


@dataclass(frozen=True)
class TriageReport:
    """The ordered check results and a plain verdict."""

    conditions: list
    qc: dict                         # condition -> {flagged_cells, silent_movies, n}
    poolability: dict                # unit -> {ratio, verdict}
    blockers: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def clear_to_compare(self) -> bool:
        return not self.blockers

    def verdict(self) -> str:
        if self.clear_to_compare:
            head = "CLEAR TO COMPARE"
            if self.warnings:
                head += " (with notes)"
            return head
        return "RESOLVE THESE FIRST: " + "; ".join(self.blockers)

    def summary(self) -> str:
        lines = [self.verdict()]
        for name in self.conditions:
            q = self.qc.get(name, {})
            lines.append(
                f"  {name}: {q.get('n_cells', '?')} cells, "
                f"{len(q.get('flagged_cells', []))} flagged, "
                f"{len(q.get('silent_movies', []))} empty movie(s)")
        for note in self.warnings:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def triage_project(project, *, field_shape=(126, 130), min_cells: int = 3) -> TriageReport:
    """Run QC → poolability → ready-to-compare and return a verdict.

    Blockers (must resolve): a condition with too few cells to bootstrap; a
    condition where poolability says days differ as much as conditions (pooling
    would hide a batch effect). Warnings (worth a look, not blocking): flagged
    outlier cells, movies that contributed nothing.
    """
    from .observables import squared_displacements
    from .qc import group_counts, track_summary, unit_summary

    conditions = list(project.conditions)
    if len(conditions) < 1:
        raise TriageError("project has no conditions")

    qc: dict = {}
    blockers: list = []
    warnings: list = []

    for name in conditions:
        part = project.condition(name)
        summary = track_summary(part)
        cells = unit_summary(part, unit="cell", summary=summary)
        counts = group_counts(part, summary=summary)

        on_disk = sorted(m for m in getattr(project, "movie_source", {})
                         if m.startswith(f"{name}_"))
        contributing = set(counts["movie"])
        silent = [m for m in on_disk if m not in contributing]

        n_cells = int(part["cell"].nunique())
        flagged = cells.loc[cells["flagged"], "cell"].astype(str).tolist()
        qc[name] = {"n_cells": n_cells, "flagged_cells": flagged,
                    "silent_movies": silent}

        if n_cells < min_cells:
            blockers.append(
                f"{name} has only {n_cells} cell(s): too few to bootstrap")
        if flagged:
            warnings.append(
                f"{name}: {len(flagged)} outlier cell(s) flagged ({', '.join(flagged[:3])}"
                + ("…" if len(flagged) > 3 else "") + "): inspect before pooling")
        if silent:
            warnings.append(
                f"{name}: {len(silent)} movie(s) contributed no tracks "
                f"({', '.join(silent)}): check segmentation")

    # Poolability (needs >1 condition to compare between; within-condition spread
    # is the guard against pooling a batch effect).
    poolability: dict = {}
    if len(conditions) >= 2:
        from .aggregate import poolability as poolability_fn

        observable = squared_displacements(project.canonical)
        for unit in ("cell",):
            try:
                table = poolability_fn(observable, unit=unit)
                within = float(table.loc[table["comparison"] == "within_condition",
                                         "median_wasserstein_um2"].iloc[0])
                between = float(table.loc[table["comparison"] == "between_condition",
                                          "median_wasserstein_um2"].iloc[0])
                ratio = between / within if within else float("inf")
                verdict = "pooling defensible" if ratio > 1.5 else "DO NOT pool"
                poolability[unit] = {"ratio": ratio, "verdict": verdict}
                if ratio <= 1.5:
                    blockers.append(
                        f"poolability by {unit}: within-condition spread rivals the "
                        f"between-condition difference (ratio {ratio:.1f}): pooling "
                        "would hide a batch effect")
            except Exception as exc:                    # noqa: BLE001 - reported
                warnings.append(f"poolability by {unit} could not run: {exc}")

    return TriageReport(conditions=conditions, qc=qc, poolability=poolability,
                        blockers=blockers, warnings=warnings)
