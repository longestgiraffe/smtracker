"""Updates: ask GitHub whether a newer SMTracker is published, and install it when the person presses the button.

ASKED (27 Nov 2026): "updater possible that checks the github repo?", then "yes build it, then use
longestgiraffe/smtracker". HEADLESS: this module holds every rule; ``ui.widgets.update_notice`` only paints it.

WHAT IT DOES, AND WHAT IT NEVER DOES
- At the start, at most ONCE A DAY, ONE anonymous GET of the newest release (`API_URL`): the headers are
  ``Accept: application/vnd.github+json`` and ``User-Agent: SMTracker/<version>``, nothing else. Only a FROZEN build
  does this by itself (`automatic_allowed`); a source run is updated with git. A 404 is "no release published yet",
  never an error. A failure at the start is silent on screen and written to the crash log; the Help entry
  "Check for updates" checks at once and always answers (`check_now`).
- A release is newer when its tag (``v3.0.1`` or ``3.0.1``) is strictly newer than ``smtracker.__version__``,
  compared NUMERICALLY (``3.0.10`` > ``3.0.9``; `parse_version`).
- NOTHING is installed without the person pressing "Download and install". On Windows that downloads the release's
  ``SMTracker-<version>-<commit>-setup.exe`` into ``%LOCALAPPDATA%/smtracker/updates`` (through the pack downloader,
  `segpack.fetch_file`: resumable, with progress and cancel), checks it against the line for it in the release's
  ``SHA256SUMS.txt`` and REFUSES to install when that file or line is missing or the hash differs
  (`UpdateRefused`). The installer has the pack's AppId, so it replaces the install in place and keeps the settings
  and the segmentation pack (`packaging/smtracker.iss`, ``CloseApplications=yes``). Elsewhere the release page opens in
  the browser.
- What is remembered is one ui.json key, ``updates``: ``{"checked_at": epoch seconds, "newest": version}``; the
  Options switch "Check for updates at start" (`options.updates_at_start`) turns the automatic check off.

INTEGRITY, SAID PLAINLY. The hash file comes from the same release as the installer, so it proves the download is
whole and is the file the publisher listed; it does not defend against a publisher whose account is taken over.

TESTS NEVER TOUCH A NETWORK. ``SMTRACKER_UPDATE_API`` replaces the API address, and only there ``file://`` is allowed
(for the API reply and for the files it names); the check treats that as a frozen build so a source run can exercise
the start page. Every request takes an ``opener`` that a test replaces.

PUBLISHING A RELEASE FOR THIS (docs/claude/qt_and_packaging.md, "THE UPDATER"): tag ``v<version>``, assets the setup.exe,
the zip and ``SHA256SUMS.txt`` (`packaging/build.py` writes it), a higher version every time.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .childproc import NO_WINDOW

__all__ = ["REPOSITORY", "API_URL", "API_ENV", "RELEASES_PAGE", "SETTINGS_KEY", "CHECK_EVERY_S", "TIMEOUT_S",
           "SUMS_NAME", "UpdateError", "UpdateRefused", "Asset", "Release", "Outcome", "parse_version", "is_newer",
           "parse_release", "fetch_latest", "check_now", "run_check", "due", "automatic_allowed", "runs_from_source",
           "choose_installer", "installs_here", "parse_sums", "format_sums", "download_installer", "start_installer",
           "updates_folder", "short_notes", "words_for", "FAILED_AT_START"]

#: THE ADDRESS, once: the GitHub repository whose releases are the updates.
REPOSITORY = "longestgiraffe/smtracker"
API_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPOSITORY}/releases"
#: Tests only: another API address, ``file://`` allowed (and only here).
API_ENV = "SMTRACKER_UPDATE_API"

SETTINGS_KEY = "updates"
CHECK_EVERY_S = 24 * 3600
TIMEOUT_S = 10.0
SUMS_NAME = "SHA256SUMS.txt"
#: Where an installer may be downloaded from (the release's own files redirect to GitHub's storage).
DOWNLOAD_HOSTS = ("github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com")
#: Room asked beyond the installer's own size (the file, its partial copy while it arrives, and a margin).
SPACE_MARGIN = 64 * 1024 * 1024
NOTES_LIMIT = 700
FAILED_AT_START = "update check failed"
_MAX_REPLY = 4 * 1024 * 1024


class UpdateError(Exception):
    """A check or a download that did not work; the message names what and why."""


class UpdateRefused(UpdateError):
    """The installer is not trusted (no hash to check it against, or the wrong one); nothing was installed."""


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int


@dataclass(frozen=True)
class Release:
    tag: str
    version: str                  # the tag without its leading "v", as written
    notes: str
    page_url: str
    assets: tuple = ()


@dataclass(frozen=True)
class Outcome:
    """What a check found. ``kind``: newer, current, no_release, failed or source."""
    kind: str
    words: str
    release: Release | None = None
    newest: str | None = None


# ------------------------------------------------------------------- versions
def parse_version(text) -> tuple:
    """``"v3.0.1"`` / ``"3.0.1"`` as ``(3, 0, 1)``. A tag that is not dotted numbers is refused, named."""
    match = re.fullmatch(r"[vV]?(\d+(?:\.\d+)*)", str(text).strip())
    if not match:
        raise UpdateError(f"the release tag {str(text)!r} is not a version number such as v3.0.1")
    return tuple(int(part) for part in match.group(1).split("."))


def _padded(a: tuple, b: tuple) -> tuple:
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)), b + (0,) * (n - len(b))


def is_newer(candidate, current) -> bool:
    """True when ``candidate`` is strictly newer than ``current`` (``3.0`` equals ``3.0.0``)."""
    a, b = _padded(parse_version(candidate), parse_version(current))
    return a > b


def current_version() -> str:
    from . import __version__

    return __version__


# ------------------------------------------------------------------- the request
def api_url() -> str:
    """GitHub's address, or the test override (``file://`` only; anything else there is refused)."""
    override = os.environ.get(API_ENV)
    if not override:
        return API_URL
    if urllib.parse.urlparse(override).scheme != "file":
        raise UpdateError(f"${API_ENV} takes a file:// address (it is for tests), not {override!r}")
    return override


def _file_urls_allowed() -> bool:
    return bool(os.environ.get(API_ENV))


def _headers() -> dict:
    return {"Accept": "application/vnd.github+json", "User-Agent": f"SMTracker/{current_version()}"}


def parse_release(data) -> Release:
    """A release from GitHub's JSON; a reply that is not one is refused, named."""
    if not isinstance(data, dict) or not isinstance(data.get("tag_name"), str):
        raise UpdateError("GitHub's answer is not a release (no tag_name)")
    tag = data["tag_name"].strip()
    parse_version(tag)                                                    # refuse a tag that is no version
    assets = []
    for raw in data.get("assets") or []:
        if isinstance(raw, dict) and isinstance(raw.get("name"), str) and isinstance(raw.get("browser_download_url"), str):
            assets.append(Asset(raw["name"], raw["browser_download_url"], int(raw.get("size") or 0)))
    return Release(tag=tag, version=tag.lstrip("vV"), notes=str(data.get("body") or ""),
                   page_url=str(data.get("html_url") or RELEASES_PAGE), assets=tuple(assets))


def fetch_latest(opener=None, timeout: float = TIMEOUT_S) -> Release | None:
    """The newest release, or None when none is published (a 404). Raises `UpdateError` on anything else."""
    opener = opener or urllib.request.urlopen
    url = api_url()
    try:
        with opener(urllib.request.Request(url, headers=_headers()), timeout=timeout) as reply:
            raw = reply.read(_MAX_REPLY)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise UpdateError(f"GitHub answered {exc.code} {exc.reason} for {url}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise UpdateError(f"{url} could not be reached: {type(exc).__name__}: {reason}") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise UpdateError(f"GitHub's answer from {url} is not JSON: {exc}") from exc
    return parse_release(data)


# ------------------------------------------------------------------- the rules
def runs_from_source(frozen: bool | None = None) -> bool:
    """A source run is updated with git. (The test override counts as a frozen build.)"""
    return not (is_frozen() if frozen is None else frozen) and not os.environ.get(API_ENV)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def automatic_allowed(frozen: bool | None = None) -> bool:
    """Only a frozen build checks by itself at the start."""
    return not runs_from_source(frozen)


def due(record, now: float | None = None) -> bool:
    """True when no check is recorded, the last was a day ago or more, or its time lies in the future."""
    now = time.time() if now is None else now
    last = record.get("checked_at") if isinstance(record, dict) else None
    if not isinstance(last, (int, float)) or isinstance(last, bool):
        return True
    return last > now or now - last >= CHECK_EVERY_S


def _settings_read():
    from .ui import theme

    return theme.remembered(SETTINGS_KEY)


def _settings_write(record: dict) -> None:
    from .ui import theme

    theme.remember(SETTINGS_KEY, record)


def _outcome(release: Release | None, current: str) -> Outcome:
    if release is None:
        return Outcome("no_release", words_for("no_release", current=current), None, None)
    if is_newer(release.version, current):
        return Outcome("newer", words_for("newer", version=release.version), release, release.version)
    return Outcome("current", words_for("current", current=current), release, release.version)


def check_now(*, opener=None, frozen: bool | None = None, current: str | None = None,
              timeout: float = TIMEOUT_S, read=None, write=None, now: float | None = None) -> Outcome:
    """The Help entry's check: at once, and it always answers (never raises). A source run says so, and asks nothing."""
    if runs_from_source(frozen):
        return Outcome("source", words_for("source"))
    return _run(opener, current or current_version(), timeout, read, write, now, at_start=False)


def run_check(*, enabled: bool = True, opener=None, frozen: bool | None = None, current: str | None = None,
              timeout: float = TIMEOUT_S, read=None, write=None, now: float | None = None) -> Outcome | None:
    """The check at the start: None (nothing asked) from a source run, with the option off, or when one was made in
    the last day. A failure is written to the crash log and returned as ``failed`` (the window shows nothing)."""
    if not automatic_allowed(frozen) or not enabled:
        return None
    if not due((read or _settings_read)(), now):
        return None
    return _run(opener, current or current_version(), timeout, read, write, now, at_start=True)


def _run(opener, current, timeout, read, write, now, *, at_start: bool) -> Outcome:
    stamp = time.time() if now is None else now
    previous = (read or _settings_read)()
    record = {"checked_at": stamp,
              "newest": previous.get("newest") if isinstance(previous, dict) else None}
    try:
        release = fetch_latest(opener, timeout)
    except UpdateError as exc:
        (write or _settings_write)(record)                    # the attempt counts: once a day, even offline
        if at_start:
            from . import crashlog

            crashlog.note(f"{FAILED_AT_START}: {exc}")
        return Outcome("failed", words_for("failed", reason=str(exc)))
    if release is not None:
        record["newest"] = release.version
    (write or _settings_write)(record)
    return _outcome(release, current)


# ------------------------------------------------------------------- the words
def words_for(kind: str, **kw) -> str:
    """The sentence for an outcome kind (one place for every word the person reads)."""
    if kind == "newer":
        return f"SMTracker {kw['version']} is available"
    if kind == "current":
        return f"SMTracker {kw['current']} is the newest version."
    if kind == "no_release":
        return f"SMTracker {kw['current']} is the newest version: no release has been published yet."
    if kind == "source":
        return "This copy runs from source, so it is updated with git, not here."
    if kind == "failed":
        return f"The check for updates did not work. {kw['reason']}"
    raise UpdateError(f"no words for {kind!r}")


def short_notes(body: str, limit: int = NOTES_LIMIT) -> str:
    """The release notes, plain and shortened to ``limit`` characters at a word."""
    text = re.sub(r"[#*`]+", "", str(body or ""))
    text = re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(None, 1)[0] + "…"


# ------------------------------------------------------------------- choosing and checking the installer
def installs_here(platform: str | None = None) -> bool:
    """Only Windows installs in place; elsewhere the release page opens."""
    return (platform or sys.platform).startswith("win")


def choose_installer(release: Release, platform: str | None = None) -> Asset | None:
    """The release's ``SMTracker-<version>-*-setup.exe`` on Windows, else None."""
    if not installs_here(platform):
        return None
    pattern = re.compile(rf"SMTracker-{re.escape(release.version)}-.+-setup\.exe", re.IGNORECASE)
    for asset in sorted(release.assets, key=lambda a: a.name):
        if pattern.fullmatch(asset.name):
            return asset
    return None


_SUMS_LINE = re.compile(r"([0-9a-fA-F]{64})[ \t]+\*?(.+?)\s*")


def parse_sums(text: str) -> dict:
    """``{file name: sha256}`` from a sha256sum file (``<64 hex>  <name>``; a ``*`` binary mark is allowed)."""
    found = {}
    for line in str(text).splitlines():
        match = _SUMS_LINE.fullmatch(line.strip())
        if match:
            found[match.group(2)] = match.group(1).lower()
    return found


def format_sums(pairs) -> str:
    """The text of a SHA256SUMS file from ``[(file name, sha256)]``, as sha256sum writes it."""
    return "".join(f"{digest.lower()}  {name}\n" for name, digest in pairs)


def updates_folder() -> Path:
    """Where installers are downloaded: ``%LOCALAPPDATA%/smtracker/updates`` (the user's data folder elsewhere)."""
    base = os.environ.get("LOCALAPPDATA") or os.path.join(Path.home(), ".local", "share")
    return Path(base) / "smtracker" / "updates"


def _check_asset_url(asset: Asset) -> None:
    parts = urllib.parse.urlparse(asset.url)
    if _file_urls_allowed() and parts.scheme == "file":
        return
    if parts.scheme != "https" or parts.netloc not in DOWNLOAD_HOSTS:
        raise UpdateRefused(f"{asset.name}: the address {asset.url} is not https on {', '.join(DOWNLOAD_HOSTS)}; refused")


def _expected_hash(release: Release, asset: Asset, opener, timeout: float) -> str:
    sums = next((a for a in release.assets if a.name.lower() == SUMS_NAME.lower()), None)
    if sums is None:
        raise UpdateRefused(f"Release {release.version} has no {SUMS_NAME}, so the installer cannot be checked and "
                            f"is not installed. Get it from {release.page_url}.")
    _check_asset_url(sums)
    request = urllib.request.Request(sums.url, headers={"User-Agent": f"SMTracker/{current_version()}"})
    try:
        with (opener or urllib.request.urlopen)(request, timeout=timeout) as reply:
            text = reply.read(_MAX_REPLY).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise UpdateError(f"{SUMS_NAME} could not be read from {sums.url}: {type(exc).__name__}: {exc}") from exc
    digest = parse_sums(text).get(asset.name)
    if digest is None:
        raise UpdateRefused(f"{SUMS_NAME} has no line for {asset.name}, so the installer cannot be checked and is "
                            f"not installed. Get it from {release.page_url}.")
    return digest


def download_installer(release: Release, folder=None, *, on_progress=None, cancel=None, opener=None,
                       platform: str | None = None, timeout: float = 60.0) -> Path:
    """Download and CHECK the installer of ``release``; the path of the verified file.

    ``on_progress(done, total)`` in bytes; ``cancel`` an Event or callable (what arrived is kept, and the next try
    continues it). Raises `UpdateRefused` (no installer, no hash, or a wrong hash; a wrong file is deleted) or
    `UpdateError` (space, network); `segpack.PackCancelled` on a cancel."""
    from . import segpack

    asset = choose_installer(release, platform)
    if asset is None:
        raise UpdateRefused(f"Release {release.version} has no installer for this computer "
                            f"(SMTracker-{release.version}-...-setup.exe). Get it from {release.page_url}.")
    _check_asset_url(asset)
    expected = _expected_hash(release, asset, opener, 30.0)
    folder = Path(folder) if folder is not None else updates_folder()
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise UpdateError(f"The folder {folder} for the download could not be made: {exc}") from exc
    target = folder / asset.name
    if target.is_file() and segpack.sha256_file(target) == expected:
        return target                                                    # an earlier try finished
    have = (folder / (asset.name + ".part")).stat().st_size if (folder / (asset.name + ".part")).exists() else 0
    need = max(asset.size - have, 0) + SPACE_MARGIN
    free = segpack.free_bytes(folder)
    if free < need:
        raise UpdateError(f"The installer needs about {segpack.sizes_words(need)} free in {folder}; "
                          f"{segpack.sizes_words(free)} is free there.")
    for old in folder.glob("SMTracker-*-setup.exe*"):                    # earlier versions' leftovers
        if not old.name.startswith(asset.name):
            old.unlink(missing_ok=True)
    try:
        segpack.fetch_file(asset.url, target, size=asset.size, name=asset.name, cancel=cancel, timeout=timeout,
                           user_agent=f"SMTracker/{current_version()}",
                           on_bytes=(lambda done: on_progress(done, asset.size)) if on_progress else None)
    except segpack.PackCancelled:
        raise
    except segpack.SegPackError as exc:
        raise UpdateError(str(exc)) from exc
    got = segpack.sha256_file(target)
    if got != expected:
        target.unlink(missing_ok=True)
        raise UpdateRefused(f"{asset.name} does not match its line in {SUMS_NAME} (expected {expected}, got {got}). "
                            f"The file was deleted and nothing was installed. Get it from {release.page_url}.")
    return target


def start_installer(path, popen=None) -> None:
    """Start the downloaded setup program so that it outlives this app (the caller then quits the application)."""
    if not installs_here():
        raise UpdateError("An installer is started only on Windows.")
    flags = NO_WINDOW | getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    quiet = dict(close_fds=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if popen is None:
            subprocess.Popen([str(path)], creationflags=flags, **quiet)
        else:
            popen([str(path)], creationflags=flags, **quiet)
    except OSError as exc:
        raise UpdateError(f"The installer {path} could not be started: {exc}") from exc
