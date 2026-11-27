"""How SMTracker starts another program: never in a window of its own.

WHY (27 Nov 2026, asked of the 3.0.0 build: "console always open and no way to closeit?"). The packaged
``smtracker.exe`` is a WINDOWS app since then (`packaging/build.py`, ``--windowed``), so it has no console. Windows gives
every console program such an app starts (MATLAB's launcher, Fiji, git, pip, taskkill, nvidia-smi) a NEW console window
of its own unless the start says otherwise. So every ``subprocess`` call in the package passes
``creationflags=NO_WINDOW``; what the program prints is still read through pipes, as before. Off Windows the flag is 0.

ONE SOURCE OF TRUTH: ``tests/test_childproc.py`` reads every ``subprocess.run`` / ``Popen`` / ``call`` /
``check_output`` / ``check_call`` in ``src/smtracker`` and fails one that does not pass ``creationflags``.
"""
from __future__ import annotations

import subprocess

__all__ = ["NO_WINDOW"]

#: ``subprocess.CREATE_NO_WINDOW`` on Windows, 0 elsewhere (where it means nothing).
NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0)
