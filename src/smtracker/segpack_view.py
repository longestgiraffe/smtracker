"""What the segmentation pack SAYS: the offer's wording, the choices' rows, the progress line and the command line.

Headless (no Qt): `ui.outlines.pack_dialog` only paints these, and ``smtracker --segmentation-pack`` prints them, so the
window and the command line cannot disagree about a size or a refusal. The rules live in `smtracker.segpack`.

Plain words: no dashes for pauses, no minus signs, sizes spelled in GB and MB.
"""
from __future__ import annotations

import time
from pathlib import Path

from . import segpack

__all__ = [
    "OFFER_TITLE", "OFFER_BODY", "WHY_NOT_NOW", "flavour_rows", "folder_check", "Pace", "progress_line",
    "runtime_offer_words", "done_words", "cli", "flavour_title",
]

OFFER_TITLE = "Segmentation pack: PyTorch and Cellpose"
OFFER_BODY = (
    "Segmenting with the Cellpose-SAM model, and training your own, need PyTorch and Cellpose. They are not part of this "
    "program, so most people never need to download them. If you want them, they are downloaded once from pytorch.org "
    "and pypi.org, checked against fixed checksums, and kept in a folder you choose. Without them the classical "
    "segmenter and Edit outlines work as before.")
WHY_NOT_NOW = "Press No to go on without them. You can open this again from Segmentation pack in the header."

_TITLES = {
    "cuda": "With GPU support (CUDA)",
    "cpu": "CPU only",
    "default": "For this Mac (uses its GPU)",
}


def flavour_title(flavour: str) -> str:
    return _TITLES.get(flavour, flavour)


def flavour_rows(gpu=None, platform: str | None = None, spec=None) -> list:
    """One dict per choice, the recommended one first: ``flavour``, ``title``, ``download``, ``installed`` (bytes),
    ``words`` (the line under the title), ``recommended`` (bool), ``reason`` (for the recommended one)."""
    chosen, reason = segpack.recommend(gpu)
    rows = []
    for flavour in segpack.flavours(platform, spec):
        download, installed = segpack.totals(flavour, platform, spec)
        rows.append({
            "flavour": flavour, "title": flavour_title(flavour), "download": download, "installed": installed,
            "words": f"Download {segpack.sizes_words(download)}, {segpack.sizes_words(installed)} on disk.",
            "recommended": flavour == chosen, "reason": reason if flavour == chosen else ""})
    rows.sort(key=lambda r: not r["recommended"])
    return rows


def folder_check(flavour: str, folder, platform: str | None = None, spec=None) -> tuple:
    """``(ok, sentence)`` about the chosen folder: the free space on its drive against what the install needs."""
    need = segpack.needed_bytes(flavour, platform, spec)
    try:
        free = segpack.free_bytes(folder)
    except OSError as exc:
        return False, f"{folder} cannot be checked ({type(exc).__name__}: {exc}). Pick another folder."
    drive = Path(folder).anchor or str(folder)
    if free < need:
        return False, (f"The drive {drive} has {segpack.sizes_words(free)} free and the install needs about "
                       f"{segpack.sizes_words(need)}. Pick a folder on a drive with more room.")
    return True, f"The drive {drive} has {segpack.sizes_words(free)} free; the install needs about {segpack.sizes_words(need)}."


class Pace:
    """Time left from the bytes that arrived since this run began (resumed bytes do not count as speed)."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.first = None

    def update(self, done: int) -> float | None:
        """Bytes per second so far, or None before there is a measure."""
        if self.first is None:
            self.first = done
            return None
        elapsed = self.clock() - self.started
        gained = done - self.first
        return gained / elapsed if elapsed > 1.0 and gained > 0 else None


def _minutes(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


def progress_line(p, pace: Pace | None = None) -> tuple:
    """``(fraction of the whole pack, text)`` for a `PackProgress`."""
    fraction = p.done / p.total if p.total else 0.0
    verb = {"download": "Downloading", "verify": "Checking", "unpack": "Unpacking", "finish": "Finishing"}[p.phase]
    if p.phase == "finish":
        return 1.0, "Finishing: the pack is put in place."
    text = (f"{verb} {p.filename} (file {p.index} of {p.count}): {segpack.sizes_words(p.file_done)} of "
            f"{segpack.sizes_words(p.file_total)}. In all {segpack.sizes_words(p.done)} of {segpack.sizes_words(p.total)}.")
    if pace is not None and p.phase == "download":
        rate = pace.update(p.done)
        if rate:
            text += f" {rate / 2**20:.1f} MB/s, about {_minutes((p.total - p.done) / rate)} left."
    return fraction, text


def runtime_offer_words() -> str:
    """What the Segment page says where no model can run in the packaged program: the pack is offered."""
    return ("Cellpose-SAM and training need PyTorch and Cellpose, which this program downloads on request. "
            "Press Get segmentation models to see what that takes.")


def done_words(st) -> str:
    return (f"Installed: {', '.join(st.packages[:3])} and the packages they need, in {st.folder}. Cellpose-SAM is now "
            f"offered. Its model file (about 1.2 GB) downloads the first time you segment with it.")


# ------------------------------------------------------------------- the command line
def cli(action: str, rest: list, *, flavour: str | None = None, folder=None, say=print) -> int:
    """``smtracker --segmentation-pack status|list|install|install-from DIR|remove``; returns the exit code.

    ``--cuda`` / ``--cpu`` pick the flavour (else the recommended one), ``--dir`` is the pack's own folder (else the
    default under the user's local app data). An install is remembered in the user's settings, so later runs activate it."""
    try:
        return _cli(action, rest, flavour, folder, say)
    except segpack.SegPackError as exc:
        say(f"segmentation pack: {exc}")
        return 1


def _flavour(flavour, say) -> str:
    if flavour:
        if flavour not in segpack.flavours():
            raise segpack.SegPackError(f"there is no {flavour!r} pack on this computer; the choices are "
                                       f"{', '.join(segpack.flavours())}.")
        return flavour
    chosen, reason = segpack.recommend()
    say(f"flavour: {chosen} ({reason})")
    return chosen


def _cli(action, rest, flavour, folder, say) -> int:
    if action == "status":
        key = segpack.platform_key()
        say(f"platform: {key or 'none pinned for this computer'}; pack version {segpack.PACK_VERSION}")
        gpu = segpack.detect_gpu()
        say("gpu: " + (f"{gpu.name}, {gpu.memory_mb} MB, driver {gpu.driver}" if gpu.present else f"none ({gpu.note})"))
        if key:
            chosen, reason = segpack.recommend(gpu)
            say(f"recommended: {chosen}. {reason}")
        remembered = segpack.remembered_pack()
        target = Path(folder) if folder else (Path(remembered["folder"]) if remembered else None)
        if target is None:
            say("no pack is remembered; nothing installed")
            return 0
        st = segpack.status(target, deep=True)
        say(f"{'remembered' if remembered and Path(remembered['folder']) == target else 'folder'}: {st.words()}")
        if st.packages:
            say("packages: " + ", ".join(st.packages))
        return 0 if st.state in ("installed", "absent") else 1
    if action == "list":
        chosen = _flavour(flavour, say)
        download, installed = segpack.totals(chosen)
        for w in segpack.wheels(chosen):
            say(f"{w['filename']}  {segpack.sizes_words(w['size'])}  sha256 {w['sha256']}  {w['url']}")
        say(f"total: download {segpack.sizes_words(download)}, on disk {segpack.sizes_words(installed)}")
        return 0
    if action in ("install", "install-from"):
        chosen = _flavour(flavour, say)
        if folder:
            target = Path(folder)
        else:
            target, why = segpack.propose_folder(chosen)
            if why:
                say(why)
        ok, line = folder_check(chosen, target)
        say(line)
        if not ok:
            return 1
        pace, last = Pace(), [0.0, ""]

        def tick(p):
            _fraction, text = progress_line(p, pace)
            now = time.monotonic()
            if p.phase != last[1] or now - last[0] > 5 or p.file_done >= p.file_total:
                last[0], last[1] = now, p.phase
                say(text)

        if action == "install":
            done = segpack.install(chosen, target, on_progress=tick)
        else:
            if len(rest) != 1:
                raise segpack.SegPackError("install-from needs the folder of wheels: --segmentation-pack install-from DIR")
            done = segpack.install_from_files(rest[0], chosen, target, on_progress=tick)
        segpack.remember_pack(done, chosen)
        segpack.activate(done)
        st = segpack.status(done)
        say(done_words(st))
        return 0
    if action == "remove":
        remembered = segpack.remembered_pack()
        target = Path(folder) if folder else (Path(remembered["folder"]) if remembered else None)
        if target is None:
            raise segpack.SegPackError("no pack is remembered; name one with --dir")
        gone = segpack.remove(target)
        say("removed: " + (", ".join(str(g) for g in gone) if gone else f"nothing was at {target}"))
        return 0
    raise segpack.SegPackError(f"unknown action {action!r}; use status, list, install, install-from DIR or remove")
