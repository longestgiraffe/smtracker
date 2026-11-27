"""A top-level window on the screen it opens on: the arithmetic, with no Qt anywhere in it.

`ui/widgets/screenfit.py` is the Qt shell every window's door goes through; this is what it decides with (the Thin Qt
rule, `tests/test_ui_thin_qt.py`), so each rule is a test with numbers rather than a screenshot.

WHY (30 Oct 2026, on a laptop: "the software sometimes isnt correctly scaled and some parts are cut off the monitor,
and when u open for example the meta data tab even in full screen mode, it goes out of monitor as well. can we make
this more stable and depending on monitor resolution?"). The screen measured: 1920 x 1080 at 125 % Windows scaling,
so 1536 x 864 LOGICAL pixels and 1536 x 816 available without the taskbar. Qt already scales by the device pixel
ratio; what left the screen were SIZES chosen for a desktop monitor -- a window opened at 1320 x 860 or 1360 x 900,
a window MINIMUM taller than the screen (the analysis door's start screen asked for 850 px), and the Metadata panel
adding its 335 px to the workspace's minimum width, which a maximised window obeys by growing past the screen.

THE RULES
- `fit_window`: a window opens at its preferred client size where that fits the available area with its frame, and
  centred in it; where it does not, the window opens MAXIMISED (its normal size, for when it is restored, is the
  preferred size cut to the screen). ``maximise=True`` always maximises (the analysis door's start screen always
  did), ``False`` never does.
- `capped_minimum`: a window's minimum never exceeds the available area less its frame. A layout that asks for more
  is held at the screen (its content then scrolls or squeezes; every page that can scroll does) -- never a window
  past the screen's edge. ``None`` when the layout's own minimum fits: the layout keeps control of it.
- `inside`: a normal window that reaches past a screen it moved to (a laptop undocked from a monitor) is cut to that
  screen and moved as little as it takes, never re-centred.
- `panel_docks`: a side panel (the Metadata panel) sits BESIDE the page when the window has its width to spare above
  what everything else needs AND the page's view, narrowed by it, still holds the page; OVER the page's right edge
  when not -- it never raises the window's minimum past what the window already is, and never cuts the page.

INVARIANTS
- `fit_window(...)` returns a client rectangle whose frame lies inside ``available`` (when not maximised).
- `capped_minimum(m, a, f)` is ``None`` or has each side <= the available side less the frame on that axis.
- Nothing here reads a screen: every input is a number the caller measured.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DEFAULT_FRAME", "WindowFit", "capped_minimum", "fit_window", "inside", "outside", "panel_docks"]

#: A window's frame (left, top, right, bottom) in logical pixels when it cannot be measured yet (before its native
#: window exists): the Windows 11 title bar is 31 px at 100 % and 125 %, the side and bottom borders are thin. Used
#: only to decide before the first show; afterwards the window's own frame is measured.
DEFAULT_FRAME = (1, 31, 1, 1)


@dataclass(frozen=True)
class WindowFit:
    """Where a window opens: its CLIENT rectangle (x, y, width, height, the frame outside it), and whether it is
    maximised instead (the rectangle is then its normal geometry, for when it is restored)."""

    x: int
    y: int
    width: int
    height: int
    maximised: bool


def _room(available, frame) -> tuple[int, int]:
    """The client size the available area holds once the frame is taken off."""
    _ax, _ay, aw, ah = available
    left, top, right, bottom = frame
    return max(0, aw - left - right), max(0, ah - top - bottom)


def fit_window(preferred, available, frame=DEFAULT_FRAME, *, minimum=(0, 0), maximise=None) -> WindowFit:
    """Where a window whose preferred client size is ``preferred`` (w, h) opens in ``available`` (x, y, w, h), its
    frame ``frame`` (left, top, right, bottom) outside the client area, its minimum ``minimum`` (w, h).

    The client size is the preferred one cut to the room (and never below the minimum, itself cut to the room),
    centred; maximised when the preferred size did not fit, or always / never by ``maximise``."""
    ax, ay, aw, ah = available
    room_w, room_h = _room(available, frame)
    pw, ph = preferred
    min_w, min_h = min(minimum[0], room_w), min(minimum[1], room_h)
    width = max(min(pw, room_w), min_w)
    height = max(min(ph, room_h), min_h)
    left, top, right, bottom = frame
    outer_w, outer_h = width + left + right, height + top + bottom
    x = ax + max(0, (aw - outer_w) // 2) + left
    y = ay + max(0, (ah - outer_h) // 2) + top
    too_big = pw > room_w or ph > room_h
    return WindowFit(x, y, width, height, too_big if maximise is None else bool(maximise))


def capped_minimum(natural, available, frame=DEFAULT_FRAME):
    """The minimum a window may hold on this screen, or ``None`` when its layout's own (``natural``, w h) fits.

    Each side is the natural minimum cut to the available area less the frame on that axis."""
    room_w, room_h = _room(available, frame)
    nw, nh = natural
    if nw <= room_w and nh <= room_h:
        return None
    return min(nw, room_w), min(nh, room_h)


def inside(frame_rect, client_size, available, frame=DEFAULT_FRAME, *, minimum=(0, 0)) -> tuple[int, int, int, int]:
    """The client rectangle (x, y, w, h) that brings a window whose frame is ``frame_rect`` (x, y, w, h) and whose
    client is ``client_size`` (w, h) inside ``available``: cut to the room (never below ``minimum``, itself cut to the
    room) and moved as little as that takes -- a window dragged to another screen stays where it was put."""
    fx, fy, _fw, _fh = frame_rect
    ax, ay, aw, ah = available
    left, top, right, bottom = frame
    room_w, room_h = _room(available, frame)
    width = max(min(client_size[0], room_w), min(minimum[0], room_w))
    height = max(min(client_size[1], room_h), min(minimum[1], room_h))
    outer_w, outer_h = width + left + right, height + top + bottom
    x = min(max(fx, ax), ax + aw - outer_w)
    y = min(max(fy, ay), ay + ah - outer_h)
    return x + left, y + top, width, height


def outside(frame_rect, available) -> bool:
    """Whether a window's frame (x, y, w, h) reaches past ``available`` (x, y, w, h) on any side."""
    fx, fy, fw, fh = frame_rect
    ax, ay, aw, ah = available
    return fx < ax or fy < ay or fx + fw > ax + aw or fy + fh > ay + ah


def panel_docks(window_width: int, others_need: int, panel_width: int, *, page_slack: float = float("inf"),
                slot: int = 0) -> bool:
    """Whether a side panel ``panel_width`` wide sits BESIDE the page in a window ``window_width`` wide whose other
    columns need ``others_need`` at least: only when the window has that width to spare. Otherwise it lies over the
    page's edge, so opening it never makes the window wider than it is (`ui.workspace.metadata.MetadataDock`).

    ``page_slack``: how much wider than its content's minimum the page's view is now, with the panel's slot ``slot``
    wide. A page that scrolls only up and down hides its content's minimum from the window's, and a view narrower
    than it CUTS the page at its right edge: the panel sits beside the page only where the view, having given up
    the panel's width, still holds it (Jump distances' cards asked for 868 px beside a panel that left 833)."""
    return window_width - others_need >= panel_width and page_slack + slot - panel_width >= 0
