"""What a long-running job says about itself while it runs, and when it is done.

Asked for (13 Oct 2026): "whenever something is running a more obvious way to show user
something is running and how long it takes (4 conditions with 10 movies per condition ->
condition done 5/10 ... total 5/40 or something) a bar or something filling", and "a popup
when something that's running for long finished".

HEADLESS: plain dicts for the windows to paint (`ui.widgets.progress`). Nothing here
knows Qt, so every window -- preparation, outlines, analysis -- words its jobs one way.

THE ESTIMATE IS A RATE, SAID AS "ABOUT". Time left = time so far / units done x units
left, once at least one unit is done; before that there is no estimate and none is
printed. Units are what the job counts (movies, fields, conditions), so a first unit that
also paid a start-up cost (MATLAB starting, a pool spawning) makes the first estimate
pessimistic; it settles as units land.
"""
from __future__ import annotations

import time

__all__ = ["NOTIFY_AFTER_S", "clock_words", "duration_words", "estimate_words", "finished_payload",
           "sidebar_job", "sidebar_jobs_words",
           "progress_payload"]

#: A finished pop-up appears only for a job that ran at least this long (seconds); a
#: few-second job announcing itself would be noise. The options screens change it.
NOTIFY_AFTER_S = 60.0


def duration_words(seconds: float) -> str:
    """``"45 s"``, ``"12 min"``, ``"2 h 05 min"``."""
    s = max(0.0, float(seconds))
    if s < 60:
        return f"{s:.0f} s"
    minutes = int(round(s / 60))
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


def estimate_words(done: int, total: int, elapsed_s: float) -> str:
    """``"about 12 min left"`` from the rate so far, ``""`` before the first unit."""
    if done <= 0 or total <= done or elapsed_s <= 0:
        return ""
    left = elapsed_s / done * (total - done)
    if left < 60:
        return "less than a minute left"
    return f"about {duration_words(left)} left"


def progress_payload(*, title: str, unit: str, done: int, total: int,
                     group: str = "", groups_done: int | None = None,
                     groups_total: int | None = None, current: str = "",
                     current_done: int | None = None, current_total: int | None = None,
                     started: float | None = None, now: float | None = None) -> dict:
    """One running job, as the progress card shows it.

    ``done`` / ``total`` count ``unit`` (e.g. "movie") over the WHOLE job and fill the
    bar; ``groups_*`` count ``group`` ("condition") when the job has several; ``current``
    names the group in hand with its own ``current_done`` / ``current_total``. ``total``
    of 0 means the size is not known yet: the bar is drawn busy, without a fill.
    """
    now = time.monotonic() if now is None else now
    elapsed = (now - started) if started is not None else 0.0
    total = max(int(total), 0)
    done = max(0, min(int(done), total)) if total else max(int(done), 0)
    parts = []
    if group and groups_total:
        parts.append(f"{group} {min(int(groups_done or 0) + 1, int(groups_total))} of "
                     f"{int(groups_total)}")
    if current:
        here = current
        if current_total:
            here += f": {unit} {min(int(current_done or 0), int(current_total))} of " \
                    f"{int(current_total)}"
        parts.append(here)
    if total:
        parts.append(f"{done:,} of {total:,} {unit}s in all")
    timing = []
    if started is not None:
        timing.append(f"running {duration_words(elapsed)}")
        eta = estimate_words(done, total, elapsed)
        if eta:
            timing.append(eta)
    return {"title": title, "fraction": (done / total) if total else None,
            "detail": "  ·  ".join(parts), "timing": "  ·  ".join(timing)}


def finished_payload(*, title: str, lines, started: float | None,
                     now: float | None = None, notify_after_s: float = NOTIFY_AFTER_S,
                     failed: bool = False) -> dict | None:
    """The pop-up for a job that just ended, or ``None`` when it was too short to need
    one. ``lines`` are the job's own summary lines (what it did, what failed)."""
    now = time.monotonic() if now is None else now
    if started is None:
        return None
    elapsed = now - started
    if elapsed < float(notify_after_s):
        return None
    return {"title": title, "lines": [str(x) for x in lines if str(x).strip()],
            "elapsed": f"took {duration_words(elapsed)}", "failed": bool(failed)}


# -- the analysis sidebar's Running list (2 Nov 2026) ---------------------------------- #
# Asked: "maybe if i run multiple things like sqd, hmm, appd etc, we can have on the left bar like an info that
# something (what and roughly percent) is running in the background, so people not wondering why its slow". Every
# section's Run lands in the workspace's one job register (`WorkspaceWindow._running`); each running job is a line at
# the foot of the sidebar (`ui.workspace.sidebar.JobsPanel`), said here.

def clock_words(seconds: float) -> str:
    """A running clock: ``"0:42"``, ``"12:05"``, ``"1:02:05"`` -- it moves every second, so a job that has finished
    nothing yet still shows that it is alive."""
    s = int(max(0.0, float(seconds)))
    hours, rest = divmod(s, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def sidebar_job(*, title: str, done: int, total: int, started: float | None, now: float | None = None,
                unit: str = "condition", stage: str = "") -> dict:
    """One running job as the sidebar lists it: ``{"title", "fraction", "percent", "detail", "line"}``.

    ``done`` / ``total`` count ``unit`` (a section's Run counts conditions: each is one job on the pool, so the
    percent is ROUGH -- conditions run side by side and land in bursts). ``total`` 0: the size is not known (QC's
    one job), ``fraction`` is None and the bar is drawn busy, with no percent. ``detail`` is "2 of 5 conditions ·
    1:12", ``line`` the whole job in one line for the collapsed rail's tooltip: "SQD 40 % (2 of 5 conditions, 1:12)".
    ``stage`` names a part of the Run counted on its own (4 Nov 2026: the HMM's shared fit, before its conditions):
    "shared states: 7 of 12 fits · 1:12", "shared states · 0:03" before its size is known.
    """
    now = time.monotonic() if now is None else now
    clock = clock_words(now - started) if started is not None else ""
    total = max(int(total), 0)
    done = max(0, min(int(done), total))
    fraction = (done / total) if total else None
    percent = f"{round(100 * fraction)} %" if fraction is not None else ""
    count = f"{done} of {total} {unit}{'' if total == 1 else 's'}" if total else "running"
    if stage:
        count = f"{stage}: {count}" if total else str(stage)
    detail = "  ·  ".join(part for part in (count, clock) if part)
    inside = ", ".join(part for part in (count if total or stage else "", clock) if part)
    line = " ".join(part for part in (title, percent) if part) + (f" ({inside})" if inside else "")
    return {"title": str(title), "fraction": fraction, "percent": percent, "detail": detail, "line": line}


def sidebar_jobs_words(jobs) -> str:
    """The collapsed rail's tooltip: every running job on a line under "Running", or "" when none is."""
    lines = [str(job["line"]) for job in jobs or ()]
    return "Running\n" + "\n".join(lines) if lines else ""
