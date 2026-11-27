"""THE PER-SPOT CALIBRATION, MADE IN THE APP (6 Nov 2026). Headless: no Qt.

Asked: "i wonder where to do the calibraion ?", then "ja build it"; then "measure on current data set can be
tricky?" and "yes to all": a per-spot fit turns each spot's reported precision into its true error through a
calibration MEASURED on spots known to be immobile (`precision`'s ruling: never learned from the data being fitted),
so one is made ONLY from a SEPARATE stationary measurement -- a folder of tracked u-track movies (fixed cells
expressing the fusion, or mNG stuck to glass), every track read and no cell outlines needed
(`tracks_without_cells`), never the conditions being analysed. The workspace header menu's "Per-spot calibration..."
(`ui.workspace.calibration_dialog`) makes it, puts it in the library and chooses it (`smtracker.calibrations`).

`precision.measure_from_tracks` measures it on the tracks' own one-frame steps: it CHECKS that the spots do not move
(steps two frames apart against one-frame steps, `precision.STATIONARY_LAG_RATIO_MAX`) and refuses by name otherwise,
models mislinks rather than trimming them, and keeps a decile check (observed error against the curve, by reported
precision). It may also be SAVED where a Run finds it without a choice: `precision.PROJECT_FILE` in a condition's
folder or the folder above them (`precision.calibration_for`), an existing file replaced only when asked.

Interfaces: `measure_folder(root)` (and `measure(canonical, ...)`) -> a plain payload (picklable),
``{"ok": True, ...}`` or ``{"ok": False, "error": ...}``; `summary_lines(payload)`, `decile_table(payload)` -- the words
and the table the dialog shows; `existing(folder)`, `save(payload, folder, replace=)`.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

__all__ = ["measure", "measure_folder", "tracks_without_cells", "summary_lines", "decile_table",
           "existing", "save", "calibration_of"]


def measure(canonical, *, name: str, origin: dict) -> dict:
    """The calibration of ``canonical``'s tracks (all of them immobile), as a plain payload: the curve, its source text
    (what every caption that uses it says), the checks and the deciles; ``{"ok": False, "error"}`` where the spots move,
    are too few or carry no per-spot precision. ``origin`` records where the tracks came from."""
    from .precision import PrecisionError, carries_precision, measure_from_tracks

    if canonical is None or len(canonical) == 0:
        return {"ok": False, "error": "no tracks to measure", **origin}
    if not carries_precision(canonical):
        return {"ok": False, "error": "these tracks carry no per-spot precision: the tracker wrote none (u-track "
                                      "does; TrackMate does when its spots were fitted)", **origin}
    tracks = int(canonical["uid"].nunique())
    source = f"measured on {name}, immobile spots, {date.today().isoformat()}"
    try:
        cal = measure_from_tracks(canonical, source=source)
    except PrecisionError as exc:
        return {"ok": False, "error": str(exc), "tracks": tracks, **origin}
    from .precision import STATIONARY_LAG_RATIO_MAX

    record = dict(cal.record)
    return {"ok": True, "c_nm": float(cal.c_nm), "p": float(cal.p), "source": source, "name": name,
            "steps": int(record.get("steps", 0)), "tracks": tracks,
            "lag_ratio": float(record.get("lag2_over_lag1", float("nan"))), "lag_limit": STATIONARY_LAG_RATIO_MAX,
            "outlier_share": float(record.get("outlier_share", float("nan"))),
            "deciles": [dict(d) for d in record.get("deciles", ())], "record": record, **origin}


def tracks_without_cells(root):
    """Every u-track movie under ``root`` (`<movie>/TrackingPackage/tracks/*.mat`, `backups/` skipped) as one canonical
    table, EVERY TRACK KEPT: no cell outlines are read or needed (immobilised molecules on glass have none, and the
    project loader's cell filter refuses a movie without them). The acquisition is u-track's own record
    (`utrack_meta.acquisition_params_from_project`). Only u-track: a TrackMate spot's precision comes from its fit after
    the run, which the project loader reads -- outline the cells and use the conditions instead."""
    import pandas as pd

    from .canonical import to_canonical
    from .readers import read_tracks
    from .utrack_meta import acquisition_params_from_project

    root = Path(root)
    files = [f for f in sorted(root.glob("**/TrackingPackage/tracks/*.mat"))
             if "backups" not in {part.lower() for part in f.parts}]
    if not files:
        raise ValueError(f"no u-track tracks under {root} (<movie>/TrackingPackage/tracks/*.mat)")
    per_folder: dict = {}
    for f in files:
        per_folder.setdefault(f.parent, []).append(f)
    parts = []
    for folder, found in per_folder.items():
        movie = folder.parents[1].name
        for f in found:
            name = movie if len(found) == 1 else f"{movie} {f.stem}"
            parts.append(read_tracks(f, source="utrack", movie=name))
    raw = pd.concat(parts, ignore_index=True)
    raw["condition"] = root.name or "stationary"
    raw["day"] = 1
    raw["cell"] = pd.NA
    return to_canonical(raw, acquisition_params_from_project(root))


def measure_folder(root) -> dict:
    """The calibration of every tracked movie under ``root`` (immobilised molecules, cells not needed): read with every
    track kept (`tracks_without_cells`)."""
    root = Path(root)
    origin = {"from": "folder", "root": str(root)}
    try:
        canonical = tracks_without_cells(root)
    except Exception as exc:                        # noqa: BLE001 - said in the dialog, by name
        return {"ok": False, "error": f"{root} could not be read: {exc}", **origin}
    return measure(canonical, name=root.name or str(root), origin=origin)


def summary_lines(payload: dict) -> list:
    """The dialog's words for a measurement: the curve, what it was measured on, the two checks."""
    if not payload.get("ok"):
        return [f"Not measured: {payload.get('error', 'unknown reason')}"]
    lines = [f"σ true = {payload['c_nm']:.3g} × reported^{payload['p']:.3g} nm",
             f"{payload['steps']:,} one-frame steps of {payload['tracks']:,} tracks",
             f"Immobile: steps two frames apart are {payload['lag_ratio']:.2f} times the one-frame steps "
             f"(median; at most {payload['lag_limit']:.2f} accepted)",
             f"Mislinks and false detections, modelled: {100 * payload['outlier_share']:.1f} % of the steps"]
    return lines


def decile_table(payload: dict) -> tuple:
    """``(columns, rows)``: per tenth of the spots by reported precision, the reported error, the error observed in
    the steps and the curve's -- the check that one power law describes them."""
    columns = ["Reported precision", "Observed error", "Calibration", "Steps"]
    rows = [[f"{d['reported_nm']:.1f} nm", f"{d['observed_nm']:.1f} nm", f"{d['curve_nm']:.1f} nm",
             f"{d['steps']:,}"] for d in payload.get("deciles") or ()]
    return columns, rows


def calibration_of(payload: dict):
    """The payload as a `precision.Calibration` (its record kept)."""
    from .precision import Calibration

    return Calibration(c_nm=float(payload["c_nm"]), p=float(payload["p"]), source=str(payload["source"]),
                       record=dict(payload.get("record") or {}))


def existing(folder) -> str | None:
    """The source of the calibration already saved in ``folder``, or None where there is none (an unreadable one says
    so rather than passing for none)."""
    from .precision import PROJECT_FILE, PrecisionError, load

    path = Path(folder) / PROJECT_FILE
    if not path.is_file():
        return None
    try:
        return load(path).source
    except (PrecisionError, OSError, ValueError, TypeError) as exc:
        return f"a file that could not be read ({exc})"


def save(payload: dict, folder, *, replace: bool = False) -> Path:
    """Save the measured calibration as `precision.PROJECT_FILE` in ``folder``: every Per spot Run of a project whose
    conditions sit in it (or are it) uses it. RAISES `FileExistsError` where one is there and ``replace`` is False."""
    from .precision import PROJECT_FILE
    from .precision import save as save_file

    if not payload.get("ok"):
        raise ValueError("nothing measured to save")
    path = Path(folder) / PROJECT_FILE
    if path.exists() and not replace:
        raise FileExistsError(str(path))
    return save_file(path, calibration_of(payload))
