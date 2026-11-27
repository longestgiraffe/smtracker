"""SMMTrack's dwell times, read back in (v2's File > Import, `impSMTrack.m` + `dwellTime.m`; added 17 Oct 2026).

Bernhard Schmidt's SMMTrack reads the tracks SMTracker exports (`trackexport.write_smmtrack`) and writes, per
condition, a text file of DWELL TIMES. v2 imported one such file per condition from a folder the user picked,
matched by name (the file's name without ``.txt`` equal to the condition's, ignoring case), read rows 4-34 and columns
1-5 (``csvread(filename, 3, 0, [3, 0, 33, 4])``: three header lines, then up to 31 rows ``[interval disjoint forward
sum ...]``), and drew them (`dwellTime.m`): each row's interval, in FRAMES, repeated as many times as column 2
counts events, times the frame interval in ms; a histogram on a log axis (bins one frame wide, up to 30 frames), the
cumulative distribution, and the mean +- SEM per condition.

Read the same here, with two changes:

- v2 started each condition's event list with a 0 (``events = 0; events = cat(1, events, a)``, `dwellTime.m:9-13`), so
  the histogram counted one zero-length event per condition and the SEM divided by one event too many. Not here:
  the events are the file's, no more (`legacy.deviations`).
- v2 printed a two-sample KS p (alpha 0.001) against the first condition. Not offered: SMMTrack's events carry no cell
  and no day, so the test can only treat every event as independent -- events of one molecule, cell and day are not
  (CLAUDE.md, hard rule 6). The distributions are drawn and described, never tested.

Headless (no Qt); `ui.workspace.dwell_dialog` draws what `dwell_payload` returns.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

__all__ = ["DwellImportError", "read_dwell_file", "match_dwell_files", "dwell_payload", "frame_interval_s",
           "HEADER_LINES",
           "MAX_ROWS", "MAX_FRAMES"]

#: `impSMTrack.m`'s ``csvread(filename, 3, 0, [3, 0, 33, 4])``: skip three lines, read rows 4-34 (31), columns 1-5.
HEADER_LINES = 3
MAX_ROWS = 31
MAX_COLUMNS = 5
#: `dwellTime.m`'s ``max_no_frames``: the histogram runs to 30 frames.
MAX_FRAMES = 30


class DwellImportError(RuntimeError):
    """Raised when a dwell-time file cannot be read as SMMTrack writes it."""


def read_dwell_file(path) -> dict:
    """One SMMTrack dwell-time file: ``{"interval_frames": [...], "counts": [...], "rows": [[...] x <= 5], "path"}``.
    v2's window exactly: the three header lines skipped, at most 31 rows, at most five columns; column 1 the dwell
    interval in frames, column 2 the number of events of that length. A row needs both numbers."""
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise DwellImportError(f"{path.name}: cannot be read ({exc})") from exc
    body = [ln for ln in lines[HEADER_LINES:HEADER_LINES + MAX_ROWS] if ln.strip()]
    if not body:
        raise DwellImportError(f"{path.name}: no rows after the {HEADER_LINES} header lines SMMTrack writes")
    rows = []
    for k, line in enumerate(body, start=HEADER_LINES + 1):
        cells = [c for c in re.split(r"[,;\t ]+", line.strip()) if c]
        try:
            values = [float(c) for c in cells[:MAX_COLUMNS]]
        except ValueError as exc:
            raise DwellImportError(f"{path.name}, line {k}: not numbers ({line.strip()[:60]!r})") from exc
        if len(values) < 2:
            raise DwellImportError(f"{path.name}, line {k}: an interval and an event count are needed, found "
                                   f"{len(values)} number(s)")
        if values[1] < 0 or not float(values[1]).is_integer():
            raise DwellImportError(f"{path.name}, line {k}: the event count {values[1]:g} is not a count")
        rows.append(values)
    return {"path": str(path), "rows": rows, "interval_frames": [r[0] for r in rows],
            "counts": [int(r[1]) for r in rows]}


def match_dwell_files(folder, conditions) -> dict:
    """The ``.txt`` files in ``folder`` matched to ``conditions`` (``[(key, label)]``) as v2 matched them: the file's
    name without ``.txt`` equal to the condition's key -- or here also its label -- ignoring case. Returns
    ``{"matched": {key: path}, "unmatched_conditions": [key], "unused_files": [name]}``."""
    folder = Path(folder)
    if not folder.is_dir():
        raise DwellImportError(f"no such folder: {folder}")
    files = {p.stem.lower(): p for p in sorted(folder.glob("*.txt"))}
    matched, missing = {}, []
    for key, label in conditions:
        hit = files.get(str(key).lower()) or files.get(str(label).lower())
        if hit is None:
            missing.append(key)
        else:
            matched[key] = hit
    used = {p.name for p in matched.values()}
    return {"matched": matched, "unmatched_conditions": missing,
            "unused_files": [p.name for p in files.values() if p.name not in used]}


def frame_interval_s(project, conditions) -> float:
    """The project's one frame interval: its `AcquisitionParams` (the app puts movies a few per cent apart on one,
    `APP_MIXED_TAU`), else the first condition's own median (a loader that kept mixed intervals)."""
    try:
        return float(project.params.tau_s)
    except Exception:                                       # noqa: BLE001 - mixed tau: the first condition's own
        table = project.condition(list(conditions)[0])
        return float(table["tau_s"].dropna().median())


def dwell_payload(parsed: dict, tau_s: float, *, labels: dict | None = None) -> dict:
    """What the dialog draws, from ``{key: read_dwell_file(...)}`` and the frame interval: per condition the events in
    ms (each row's interval x tau, as often as it counts), the histogram as a probability density over one-frame
    bins up to `MAX_FRAMES`, the cumulative distribution, and n / mean / median / SD. No test (the module's head)."""
    if not tau_s or tau_s <= 0:
        raise DwellImportError(f"the frame interval must be positive, got {tau_s}")
    labels = labels or {}
    frame_ms = 1000.0 * float(tau_s)
    edges = [frame_ms * k for k in range(MAX_FRAMES + 1)]
    out = {"conditions": [], "edges_ms": edges, "frame_ms": frame_ms, "rows": [],
           "columns": ["Condition", "Events", "Mean (ms)", "Median (ms)", "SD (ms)", "Beyond 30 frames"]}
    for key, got in parsed.items():
        events = sorted(frame_ms * iv for iv, n in zip(got["interval_frames"], got["counts"]) for _ in range(n))
        n = len(events)
        name = str(labels.get(key, key))
        if not n:
            out["conditions"].append({"key": key, "label": name, "n": 0, "density": [], "cdf": []})
            out["rows"].append([name, "0", "", "", "", ""])
            continue
        width = frame_ms
        density = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            inside = sum(1 for e in events if lo <= e < hi or (hi == edges[-1] and e == hi))
            density.append(inside / (n * width))
        # one step per DISTINCT time (the share after its last event), not one per event
        cdf = [(e, (i + 1) / n) for i, e in enumerate(events) if i + 1 == n or events[i + 1] != e]
        mean = sum(events) / n
        median = events[n // 2] if n % 2 else 0.5 * (events[n // 2 - 1] + events[n // 2])
        sd = math.sqrt(sum((e - mean) ** 2 for e in events) / (n - 1)) if n > 1 else math.nan
        beyond = sum(1 for e in events if e > edges[-1])
        out["conditions"].append({"key": key, "label": name, "n": n, "density": density, "cdf": cdf,
                                  "mean_ms": mean, "median_ms": median, "sd_ms": sd, "beyond": beyond,
                                  "file": Path(got["path"]).name})
        out["rows"].append([name, f"{n:,}", f"{mean:.1f}", f"{median:.1f}",
                            "" if math.isnan(sd) else f"{sd:.1f}", f"{beyond:,}"])
    out["note"] = ("Dwell times as SMMTrack counted them, one event per row's count. The events carry no cell or "
                   "replicate, so no test is offered: a test on pooled events treats every event as independent.")
    # THE AXES, decided here: both panels run 0 to 30 frames, ticked every 5; the density on a log axis over the
    # decades its non-empty bins reach
    out["x_range"] = (0.0, edges[-1])
    out["x_ticks"] = [frame_ms * k for k in range(0, MAX_FRAMES + 1, 5)]
    positive = [d for c in out["conditions"] for d in c["density"] if d > 0]
    if positive:
        lo, hi = math.floor(math.log10(min(positive))), math.ceil(math.log10(max(positive)))
        hi = max(hi, lo + 1)
        out["density_range"] = (10.0 ** lo, 10.0 ** hi)
        out["density_ticks"] = [10.0 ** k for k in range(lo, hi + 1)]
    else:
        out["density_range"], out["density_ticks"] = None, []
    out["available"] = any(c["n"] for c in out["conditions"])
    return out
