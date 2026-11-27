"""Frozen entry point. Kept trivial on purpose — PyInstaller analyses this file,
so anything clever here becomes a packaging problem rather than a code problem."""

import multiprocessing
import os
import sys

# THE WINDOWS APP HAS NO CONSOLE (27 Nov 2026, asked: "console always open and no way to closeit?"). smtracker.exe is
# built --windowed, so Python starts it with sys.stdout and sys.stderr None, and a library that writes to them (a
# progress bar, a warning) would fail. They write to nothing instead: what went wrong is in the crash log
# (`smtracker.crashlog`), and the commands that print run through smtracker-console.exe, which has a console.
for _name in ("stdout", "stderr"):
    if getattr(sys, _name) is None:
        setattr(sys, _name, open(os.devnull, "w", encoding="utf-8"))

from smtracker.app import main  # noqa: E402  (after the streams exist)

if __name__ == "__main__":
    multiprocessing.freeze_support()      # Windows: without this a frozen app can respawn itself
    sys.exit(main())
