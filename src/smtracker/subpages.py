"""SUBPAGES: the "+" under an analysis section -- a second, LIVE copy of the section ("SQD 2") with its own settings
and results. Headless; no Qt (approved 4 Nov 2026).

WHAT A SUBPAGE IS. The section's page (SQD's Populations, Jump distances and Signed jumps; the Spatial Map's grid) built
again, with its own Compute settings, its own Run, its own results and its own cache entries (the section cache's key
holds the settings, so two pages with the same settings share results, which is right). It STARTS AS A COPY OF THE
CURRENT PAGE'S SETTINGS and none of its results: it shows the section's empty state until Run. Everything on it is
re-runnable and independent of the first page.

THE KEYS. A page is addressed by a *page key*: ``"sqd"`` for the section's own page, ``"sqd~2"`` for its first added
copy. A view is ``<page key>.<page>``, ``"sqd~2.jumps"``, so the sidebar's ``select``, the breadcrumb and the window's
routing keep working on the one string they always read, and the section's own page keeps the plain keys every saved
session, test and cache entry already names. A *job key* (the window's running register) is built the same way:
``"jumps~2"``. Numbers are handed out once and NEVER REUSED within a project's life (the set remembers the next), so a
deleted page's number cannot come back as another page that a stale key, a saved map's source or an Undo would
mistake for it.

THE RULES (`SubpageSet`):

* ``add`` makes "<Section title> <n>" (n = the new number; "SQD 2", "SQD 3"), the names of one section unique;
* ``rename`` keeps names unique within a section (a clash gets " (2)"), never empty;
* ``delete`` returns what ``restore`` needs for the Undo (the page and its place);
* ``move`` shifts a page up or down among its section's copies; the section's own page is not in the list and is
  always first;
* ``state`` / ``from_state`` are plain dicts for the session file; a state from a build that had no subpages is
  simply an empty set.

Invariants: no Qt, nothing computed here; the window decides what a page holds (`ui.workspace.window`, the
``_PageSlot``), the sidebar draws the rows (`ui.workspace.sidebar`). ``ENABLED`` lists the sections that offer the "+"
(phase 1: SQD and Spatial Map); a section is added to the list when its window code builds a page per slot.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SEP", "ENABLED", "Subpage", "SubpageSet", "page_key", "split_view", "join_view", "base_view",
           "page_of_view", "job_key", "split_job", "page_number", "safe_name"]

#: Between a section and a page number in a key: ``sqd~2``.
SEP = "~"
#: The sections that offer the "+" now. Phase 2 appends the others one by one (MSD, Apparent Diffusion, HMM, Kinetics,
#: Clustering); every one is built from the same `SubpageSet` and the window's page slots.
ENABLED: tuple = ("sqd", "spatial", "msd", "appd", "hmm", "kinetics", "clusters")


@dataclass
class Subpage:
    """One added copy of a section: its number (the key's) and the name the user sees."""

    number: int
    name: str


def page_key(section: str, number: int | None) -> str:
    """``"sqd"`` for the section's own page (``number`` None), else ``"sqd~2"``."""
    return str(section) if number is None else f"{section}{SEP}{int(number)}"


def split_view(view: str) -> tuple:
    """``(section, number or None, page)`` of a view: ``"sqd~2.jumps"`` -> ``("sqd", 2, "jumps")``,
    ``"sqd.jumps"`` -> ``("sqd", None, "jumps")``, ``"sqd~2"`` -> ``("sqd", 2, "")``."""
    head, _, page = str(view).partition(".")
    section, sep, number = head.partition(SEP)
    if sep and number.isdigit():
        return section, int(number), page
    return head, None, page


def join_view(section: str, number: int | None, page: str = "") -> str:
    """The view of ``page`` on the page numbered ``number`` of ``section`` (``"sqd~2.jumps"``)."""
    head = page_key(section, number)
    return f"{head}.{page}" if page else head


def base_view(view: str) -> str:
    """``view`` as the section's own page would name it: ``"sqd~2.jumps"`` -> ``"sqd.jumps"``. A plain view is
    returned unchanged. Everything that reads a view for WHICH SECTION AND PAGE (the rail, the toolbar) reads this."""
    section, _number, page = split_view(view)
    return f"{section}.{page}" if page else section


def page_of_view(view: str) -> str:
    """The page key a view belongs to: ``"sqd~2.jumps"`` -> ``"sqd~2"``, ``"sqd.jumps"`` -> ``"sqd"``."""
    section, number, _page = split_view(view)
    return page_key(section, number)


def page_number(key: str):
    """The number of a page key (``"sqd~2"`` -> 2), None for the section's own."""
    return split_view(key)[1]


def job_key(base: str, number: int | None) -> str:
    """A job key of a page: ``("jumps", 2)`` -> ``"jumps~2"``; the section's own page keeps ``"jumps"``."""
    return page_key(base, number)


def split_job(key: str) -> tuple:
    """``(base, number or None)`` of a job key: ``"jumps~2"`` -> ``("jumps", 2)``."""
    section, number, _page = split_view(key)
    return section, number


def safe_name(text: str) -> str:
    """A page's name as a file name carries it: letters, digits, ``._-`` kept, anything else an underscore."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(text)).strip("_") or "page"


class SubpageSet:
    """Every section's added pages, in their order, with the rules above. ``titles``: section key -> its title."""

    def __init__(self, titles: dict):
        self.titles = dict(titles)
        self._pages: dict = {}
        self._next: dict = {}

    # -- reading ------------------------------------------------------------------------------------- #
    def pages(self, section: str) -> list:
        """The added pages of ``section``, in order (copies; the set is changed only through its methods)."""
        return [Subpage(p.number, p.name) for p in self._pages.get(section, ())]

    def get(self, section: str, number: int):
        return next((p for p in self._pages.get(section, ()) if p.number == number), None)

    def numbers(self, section: str) -> list:
        return [p.number for p in self._pages.get(section, ())]

    def name(self, section: str, number: int | None) -> str:
        """The name shown for a page: the section's title for its own page, else the added page's name."""
        if number is None:
            return self.titles.get(section, section)
        page = self.get(section, number)
        if page is None:
            raise KeyError(page_key(section, number))
        return page.name

    def names(self, section: str) -> list:
        """``[(page key, name)]`` of the section's own page and then every added one, the order the sidebar lists."""
        return [(section, self.titles.get(section, section))] + [
            (page_key(section, p.number), p.name) for p in self._pages.get(section, ())]

    def unique_name(self, section: str, wanted: str, *, exclude: int | None = None) -> str:
        """``wanted`` stripped, and made unique among the section's page names (the section's own title counts):
        a clash gets " (2)", " (3)" ..."""
        wanted = " ".join(str(wanted).split())
        if not wanted:
            raise ValueError("a page needs a name")
        taken = {self.titles.get(section, section)} | {p.name for p in self._pages.get(section, ())
                                                       if p.number != exclude}
        if wanted not in taken:
            return wanted
        k = 2
        while f"{wanted} ({k})" in taken:
            k += 1
        return f"{wanted} ({k})"

    # -- changing ------------------------------------------------------------------------------------ #
    def add(self, section: str) -> Subpage:
        """A new page at the end of the section's list, named "<title> <n>"."""
        if section not in ENABLED:
            raise ValueError(f"{section!r} does not offer subpages yet")
        number = self._next.get(section, 2)
        self._next[section] = number + 1
        page = Subpage(number, self.unique_name(section, f"{self.titles.get(section, section)} {number}"))
        self._pages.setdefault(section, []).append(page)
        return Subpage(page.number, page.name)

    def rename(self, section: str, number: int, name: str) -> str:
        """Rename a page; returns the name it got (made unique). An empty name is refused."""
        page = self.get(section, number)
        if page is None:
            raise KeyError(page_key(section, number))
        page.name = self.unique_name(section, name, exclude=number)
        return page.name

    def delete(self, section: str, number: int) -> tuple:
        """Remove a page. Returns ``(Subpage, index)`` to hand to `restore` (the Undo)."""
        pages = self._pages.get(section, [])
        for index, page in enumerate(pages):
            if page.number == number:
                del pages[index]
                return Subpage(page.number, page.name), index
        raise KeyError(page_key(section, number))

    def restore(self, section: str, page: Subpage, index: int) -> None:
        """Put a deleted page back where it was (its number and name unchanged)."""
        if self.get(section, page.number) is not None:
            return
        pages = self._pages.setdefault(section, [])
        pages.insert(max(0, min(int(index), len(pages))), Subpage(page.number, self.unique_name(section, page.name)))

    def move(self, section: str, number: int, step: int) -> bool:
        """Shift a page ``step`` places (-1 up, +1 down) among the section's copies. False when it cannot move."""
        pages = self._pages.get(section, [])
        index = next((i for i, p in enumerate(pages) if p.number == number), None)
        if index is None:
            raise KeyError(page_key(section, number))
        target = max(0, min(len(pages) - 1, index + int(step)))
        if target == index:
            return False
        pages.insert(target, pages.pop(index))
        return True

    def move_to(self, section: str, number: int, index: int) -> bool:
        """Put a page at ``index`` among the section's copies (a drag's drop)."""
        pages = self._pages.get(section, [])
        now = next((i for i, p in enumerate(pages) if p.number == number), None)
        if now is None:
            raise KeyError(page_key(section, number))
        index = max(0, min(len(pages) - 1, int(index)))
        if index == now:
            return False
        pages.insert(index, pages.pop(now))
        return True

    # -- the session --------------------------------------------------------------------------------- #
    def state(self) -> dict:
        """Plain data for the session file: ``{"pages": {section: [[number, name], ...]}, "next": {section: n}}``."""
        return {"pages": {s: [[p.number, p.name] for p in ps] for s, ps in self._pages.items() if ps},
                "next": dict(self._next)}

    @classmethod
    def from_state(cls, titles: dict, state: dict | None) -> "SubpageSet":
        """A set read back from `state`; unknown or malformed entries are dropped, a missing state is an empty set."""
        out = cls(titles)
        state = state or {}
        for section, entries in (state.get("pages") or {}).items():
            if section not in ENABLED:
                continue
            for number, name in entries:
                out._pages.setdefault(section, []).append(Subpage(int(number), str(name)))
        for section, n in (state.get("next") or {}).items():
            out._next[section] = int(n)
        for section, pages in out._pages.items():
            top = max((p.number for p in pages), default=1)
            out._next[section] = max(out._next.get(section, 2), top + 1)
        return out
