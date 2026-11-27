"""An optional package this install lacks, named BEFORE a run, and offered.

WHY THIS EXISTS, and it is one defect wearing two faces. `prep_model.write_projections`
reads frames, so it needs the reader for the acquisition's format; `nd2` is an optional
extra (`pip install .[raw]`). With it absent, every movie raised, every exception was
caught and appended to a note, and the `sum/` directory had already been created -- so a
Save that wrote NOTHING left behind a folder indistinguishable from one nobody had filled
yet. `channels.resolve_channels` then read that empty folder as a SUCCESSFUL `sum`
channel (`ok=True`, zero fields), and the classical segmenter, which reads `sum`, found
nothing and was reported broken. One missing package, two symptoms, neither naming it.

RULED (18 Oct 2026), for the two audiences this project has:

    a GIT CLONE gets a WARNING naming what to install, and an OFFER to run it;
    the INSTALLER built on a working machine must simply work, with no prompt.

The second half is already guaranteed elsewhere and is not this module's job: `nd2` is in
the `[raw]` extra, the bundle ships it, and `packaging/build.py`'s `--formats` check FAILS
a build whose frozen app cannot read the Nikon format. So this module deliberately
REFUSES TO OFFER AN INSTALL IN A FROZEN BUILD (`can_install`): pip into a bundled runtime
is not a repair, it is a second installation inside the first, and the honest answer
there is that the build is wrong and must be rebuilt.

NOT A SECOND REGISTRY. `rawmovie.readable_formats` already answers "which formats can
this install read, and what is missing", by `find_spec` so asking costs nothing. This
module adds only what that one does not: which packages a GIVEN SET of movies needs, the
command that would fix it, and running that command. Everything here is headless and
Qt-free; `ui.prep` paints it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .childproc import NO_WINDOW

__all__ = [
    "EXTRA",
    "PACKAGE_EXTRAS",
    "missing_for",
    "missing_for_movies",
    "install_args",
    "can_install",
    "install",
    "describe_missing",
]

#: The extra that carries the raw readers, for the message a user is shown.
EXTRA = "raw"

#: Which extra a package belongs to, so the message can name the supported install
#: (`pip install 'smtracker[raw]'`) rather than a bare package a user has to trust.
PACKAGE_EXTRAS = {"nd2": EXTRA, "tifffile": EXTRA, "saspt": "engines"}


def missing_for(suffixes) -> list[str]:
    """The packages needed to read ``suffixes`` that are NOT installed, deduplicated.

    Asked of the SUFFIXES ACTUALLY PRESENT rather than of every format this project
    knows: a MetaMorph folder must not be told to install the Nikon reader, and a
    warning that names something irrelevant teaches people to dismiss warnings.
    """
    from .rawmovie import readable_formats

    formats = readable_formats()
    out: list[str] = []
    for suffix in {str(s).lower() for s in suffixes}:
        info = formats.get(suffix)
        if info is None or info["available"]:
            continue
        if info["requires"] not in out:
            out.append(info["requires"])
    return sorted(out)


def missing_for_movies(paths) -> list[str]:
    """`missing_for` over the suffixes of the given acquisition paths."""
    return missing_for(Path(p).suffix for p in paths)


def install_args(packages) -> list[str]:
    """The argv that installs ``packages`` into THIS interpreter.

    `sys.executable -m pip`, never a bare `pip`: a machine with several environments has
    several pips, and installing into the wrong one leaves the symptom exactly as it was
    while reporting success. This project has already lost a day to two commands that
    looked identical and ran different code (`PYTHONPATH` redirecting an import).
    """
    return [sys.executable, "-m", "pip", "install", *sorted(packages)]


def can_install() -> tuple[bool, str]:
    """`(offer it, why not)`. A frozen build is never offered an install.

    See the module docstring: a bundle that cannot read a format was built wrong, and
    pip-ing into it would leave a working app that no rebuild reproduces.
    """
    if getattr(sys, "frozen", False):
        return False, ("This is a packaged build, so nothing can be installed into it: it was built without the "
                       "reader and has to be rebuilt. Tell whoever made it.")
    return True, ""


def describe_missing(packages, *, extra: str | None = None) -> str:
    """The sentence a window shows: what is missing, what it costs, how to fix it.

    It names the CONSEQUENCE first. "nd2 is not installed" reads as housekeeping; "the
    projections cannot be written, because reading .nd2 needs nd2" is why anyone cares.
    """
    if not packages:
        return ""
    names = ", ".join(sorted(packages))
    extras = {PACKAGE_EXTRAS.get(p) for p in packages} - {None}
    target = f"smtracker[{extra or (extras.pop() if len(extras) == 1 else EXTRA)}]"
    ok, why = can_install()
    # A PACKAGED BUILD IS NEVER SHOWN A pip COMMAND (tester B19, 3 Nov 2026): its user cannot run one, and the
    # sentence read "... cannot be read. this is a packaged build ... rebuilt (pip install 'smtracker[raw]')"
    if not ok:
        return f"{names} is not installed, so these acquisitions cannot be read. {why}"
    return (f"{names} is not installed, so these acquisitions cannot be read. "
            f"Install {names} to continue (pip install '{target}').")


def install(packages, *, on_line=None, timeout: float | None = 900.0) -> tuple[bool, str]:
    """Run the install, streaming each line to ``on_line``. Returns `(ok, last words)`.

    ON A THREAD, NEVER THE GUI THREAD -- it reaches the network and takes a minute. The
    output is streamed rather than collected, because a silent minute is the shape this
    project keeps having to fix: a window that looks hung is a window nobody trusts.

    A refusal is RETURNED, not raised: the caller is a button, and a traceback in a
    console nobody reads is how the last fallback failed silently.
    """
    ok, why = can_install()
    if not ok:
        return False, why
    if not packages:
        return True, "nothing to install"
    say = on_line or (lambda _line: None)
    args = install_args(packages)
    say(" ".join(args))
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
    except OSError as exc:
        return False, f"could not start pip: {exc}"
    last = ""
    try:
        for line in proc.stdout or ():
            line = line.rstrip()
            if line:
                last = line
                say(line)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return False, f"pip did not finish within {timeout:.0f} s"
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
    if proc.returncode != 0:
        return False, last or f"pip exited {proc.returncode}"
    # ASKED OF THE INSTALL, NOT OF PIP. pip can exit 0 having installed into a different
    # environment or satisfied a different version; the question is whether THIS
    # interpreter can now import it, which is what the caller is about to need.
    import importlib
    import importlib.util

    importlib.invalidate_caches()
    still = [p for p in packages if importlib.util.find_spec(p) is None]
    if still:
        return False, (f"pip reported success but {', '.join(still)} still cannot be "
                       f"imported here ({sys.executable})")
    return True, f"installed {', '.join(sorted(packages))}"
