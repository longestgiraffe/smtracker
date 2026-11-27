"""Parallel execution for the bootstraps: one source of truth, deterministic.

Only the estimators that must refit per resample (SQD, HMM) run inside a
bootstrap loop, and each replicate is independent given its own seed. This farms
those refits across cores.

Two invariants make it safe to parallelize a statistical procedure:

1. **Determinism regardless of worker order.** The resample index sets are drawn
   from the parent RNG *in order, up front* (``resample_indices``), and each
   replicate carries its own derived fit seed. So the result is identical whether
   the workers finish in order, out of order, or run serially. Parallelism must
   never change a number.
2. **The dataset ships once per worker, not once per task.** A process pool is
   created with an ``initializer`` that stashes the (read-only) group list in a
   module global; the per-task payload is just an integer index. Pickling the
   whole table for every one of ``n_boot`` tasks would erase the speed-up.

Fallbacks, loudly benign: ``n_jobs=1`` (or a pool that fails to start, e.g. a
restricted sandbox) runs serially through the same code path, so behaviour is
identical, only slower. No new dependency: stdlib ``concurrent.futures``, which
the frozen app already supports (``multiprocessing.freeze_support`` in the
entry point).
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ProcessPoolExecutor

import numpy as np

__all__ = ["n_workers", "resample_indices", "bootstrap_map", "condition_map",
           "serial_only", "set_serial_only", "stop_pool", "lower_priority", "bulk_slots"]

#: NO-NESTED-POOLS policy (Item 7e). While a view computes on a GUI worker thread,
#: all process pools are disabled GLOBALLY so a bootstrap inside a per-condition
#: fit cannot spawn a pool from a worker thread (and Fit All, which owns the pool
#: at the batch level, is never nested inside a per-view pool). Set True around a
#: worker-thread compute; Fit All manages it explicitly.
#:
#: REFERENCE-COUNTED, not a bare bool -- `smtracker.ui.workspace.window` now
#: dispatches TWO independently-triggerable GUI workers (the SQD Compute
#: dialog's Run and Apparent-D Distribution's own Run), each wrapping its
#: own `compute()` in `set_serial_only(True)` / `finally: set_serial_only
#: (False)`. A bare bool broke exactly there: if the SHORT worker (Apparent-D,
#: no fitting) finished and cleared it to `False` while the LONG worker (a
#: real SQD bootstrap) was still mid-fit, that still-running fit could then
#: spawn a REAL `ProcessPoolExecutor` from a background QThreadPool thread --
#: the precise hazard this flag exists to forbid, and a plausible source of
#: the CPU-contention stutter reported once both sections' Runs became
#: independently pressable in quick succession. Counting means the guard
#: stays up as long as ANY caller still wants it, whichever else clears
#: first. `_LOCK` matters because two OS threads can call `set_serial_only`
#: at the same instant; `+= 1` / `-= 1` on a plain int is not atomic in
#: general (CPython's GIL makes THIS specific op close to safe today, but
#: relying on that is exactly the kind of implementation detail this file's
#: own determinism rules refuse to lean on elsewhere).
_LOCK = threading.Lock()
_SERIAL_ONLY_COUNT = 0


def serial_only() -> bool:
    with _LOCK:
        return _SERIAL_ONLY_COUNT > 0


def set_serial_only(value: bool) -> None:
    """`True` increments the guard, `False` decrements it -- see the count's
    own comment above for why. A `False` with nothing left to release is
    floor-clamped at 0 rather than going negative: that shape is an
    unbalanced `try`/`finally` in some CALLER, and going negative would
    silently need an extra `True` just to reach a real 0, masking that bug
    rather than surfacing it.
    """
    global _SERIAL_ONLY_COUNT
    with _LOCK:
        if value:
            _SERIAL_ONLY_COUNT += 1
        else:
            _SERIAL_ONLY_COUNT = max(0, _SERIAL_ONLY_COUNT - 1)

#: Below this many tasks, a process pool costs more than it saves. On Windows,
#: spawn re-imports the whole package per worker (~1–2 s each), so a handful of
#: cheap replicates is faster serial. Dev/test runs (small n_boot) therefore pay
#: no pool cost; the parallel path engages for the larger final runs.
_MIN_PARALLEL_TASKS = 24

#: Per-worker read-only state, set by the pool initializer (or directly, serial).
_SHARED: dict = {}


def n_workers(n_jobs: int | None) -> int:
    """Resolve a worker count. ``None``/``-1`` → all-but-one core (min 1)."""
    if n_jobs is None or n_jobs < 0:
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(n_jobs))


def resample_indices(n_units: int, n_boot: int, seed: int | None) -> list[np.ndarray]:
    """The resample index sets, drawn up front and in order for determinism."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n_units, size=n_units) for _ in range(n_boot)]


def _init(shared: dict) -> None:                       # pragma: no cover - worker
    _SHARED.clear()
    _SHARED.update(shared)


def bootstrap_map(worker, n_tasks: int, *, shared: dict, n_jobs: int | None):
    """Run ``worker(i, _SHARED)`` for ``i`` in ``range(n_tasks)``, results ordered.

    ``worker`` MUST be a module-level function (picklable). It reads the shared
    read-only payload from the second argument. Returns a list aligned to task
    index; a worker returning ``None`` is kept (the caller filters failures),
    so ordering and counts stay meaningful.
    """
    workers = n_workers(n_jobs)
    # SERIAL PASSES `shared` STRAIGHT TO THE WORKER, never through `_SHARED`. The
    # module global exists for pool WORKER processes, where the initializer fills it
    # once; in THIS process it was also overwritten on every serial call, so a map run
    # inside another map's serial loop replaced the outer map's payload mid-loop --
    # found when `select_n_pop` began pooling its starts inside the SQD bootstrap:
    # `KeyError: 'groups'` in 8 bootstrap tests.
    if serial_only() or workers == 1 or n_tasks < _MIN_PARALLEL_TASKS:
        return [worker(i, shared) for i in range(n_tasks)]

    try:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init, initargs=(shared,)
        ) as pool:
            return list(pool.map(_call, ((worker, i) for i in range(n_tasks))))
    except (OSError, ValueError, RuntimeError):
        # A sandbox that forbids process pools: fall back to serial, same result.
        return [worker(i, shared) for i in range(n_tasks)]


def _call(args):                                       # pragma: no cover - worker
    worker, i = args
    return worker(i, _SHARED)


def condition_map(worker, items, *, shared: dict, n_jobs: int | None):
    """Run ``worker(item, _SHARED)`` for each item, results ordered.

    For FEW but EXPENSIVE tasks (per-condition fits), unlike ``bootstrap_map``
    which is tuned for many cheap replicates, so there is no min-task threshold,
    a pool engages as soon as there are ≥2 items and >1 worker. Same determinism
    and same loud-benign serial fallback (``n_jobs=1`` or a pool that will not
    start). ``worker`` MUST be module-level (picklable); the shared payload ships
    once per worker via the initializer.
    """
    items = list(items)
    workers = min(n_workers(n_jobs), max(1, len(items)))
    if serial_only() or workers == 1 or len(items) < 2:
        return [worker(it, shared) for it in items]      # see bootstrap_map
    try:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init, initargs=(shared,)
        ) as pool:
            return list(pool.map(_call_item, ((worker, it) for it in items)))
    except (OSError, ValueError, RuntimeError):
        return [worker(it, shared) for it in items]


def _call_item(args):                                  # pragma: no cover - worker
    worker, item = args
    return worker(item, _SHARED)


def stop_pool(pool) -> None:
    """End a process pool NOW: what has not started is dropped, and every worker is stopped mid-job together with the
    processes it started itself (a per-condition fit's own bootstrap pool, `bootstrap_map`).

    `shutdown(wait=False, cancel_futures=True)` drops only what has not started: a worker in the middle of a fit runs
    it to the end with nobody left to read it (2 Nov 2026: three workers of a finished run were still computing,
    about two and a half of the laptop's eight threads, 40 minutes later, and the analysis felt slow). On Windows a
    worker and its children are ended as one tree (`taskkill /T`, no console window); elsewhere the worker alone. A
    worker that has already exited is not an error. The workers are read BEFORE the shutdown, which forgets them."""
    import subprocess
    import sys

    from .childproc import NO_WINDOW

    processes = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if not process.is_alive():
            continue
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True,
                           creationflags=NO_WINDOW, check=False)
        else:
            process.terminate()


def lower_priority() -> None:
    """A worker's initializer: run BELOW NORMAL priority (3 Nov 2026).

    Asked: "the software uses a lot of cpu when i do a few things". The workspace's workers take every core but one
    while a Run computes, which is what finishes it soonest; at normal priority they competed with the window
    itself, and with the figure composer drawing in it, for those cores. Below normal they still use every core
    nobody else wants, and the window comes first. The processes a worker starts itself (a bootstrap's own pool)
    inherit it on Windows. A failure is said, never fatal: a priority changes no result, and an initializer that
    raised would break the pool for every job."""
    import sys
    import warnings

    try:
        if sys.platform == "win32":
            import ctypes

            kernel32 = _kernel32()
            if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            os.nice(5)
    except OSError as exc:                              # pragma: no cover - a refused priority
        warnings.warn(f"a worker kept normal priority: {exc}", RuntimeWarning, stacklevel=1)


#: Windows' below-normal priority class (`SetPriorityClass`).
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def _kernel32():
    """kernel32 with the priority calls' types DECLARED: ctypes passes an undeclared argument as a 32-bit int, and
    `GetCurrentProcess`'s handle is pointer-sized -- on 64-bit Windows the call failed (measured 3 Nov 2026: the
    class read back as 0, the error). Its own instance, so the shared ``ctypes.windll`` keeps its prototypes."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.GetPriorityClass.restype = wintypes.DWORD
    kernel32.GetPriorityClass.argtypes = [wintypes.HANDLE]
    return kernel32


def bulk_slots(workers: int) -> int:
    """How many of ``workers`` the LONG jobs -- a section's Run, one job per condition -- may hold at once: all but
    one from three workers up, so one is always free for a job somebody is waiting on (the SQD figure, a preview, a
    redraw), which is submitted straight to the pool and so runs next (3 Nov 2026: the SQD figure waited minutes
    behind the Spatial Map's and Jump distances' jobs). Two workers or one keep every worker for the Run."""
    workers = int(workers)
    return workers - 1 if workers >= 3 else max(1, workers)
