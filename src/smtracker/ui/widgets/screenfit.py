"""Every top-level window fits the screen it is on: sized at its first show, its minimum held to the screen, and
both checked again when it moves to another screen. The arithmetic is `smtracker.screenfit_view` (headless); this is
the Qt shell, and every window's door goes through it.

Asked 30 Oct 2026, on a laptop at 125 % (1536 x 816 logical available): "some parts are cut off the monitor, and when
u open for example the meta data tab even in full screen mode, it goes out of monitor as well. can we make this more
stable and depending on monitor resolution?" Qt already scales by the device pixel ratio; what left the screen were
sizes: windows opened at a desktop's 1320 x 860 or 1360 x 900, and minimums taller or wider than the screen, which a
maximised window obeys by growing past it. MEASURED with `scripts/gui_sweep.py --windows` (the record is in
`docs/claude/qt_and_packaging.md`).

USE
- `show_fitted(window)` in place of ``window.show()`` at a door: the window opens at the size it set for itself
  (its PREFERRED size, ``resize`` in its constructor) where that fits the screen, centred, and MAXIMISED where it
  does not (``maximise=True`` always maximises, as the analysis door's start screen always did).
- `fit_to_screen(window)` for a window the caller shows itself (a dialog run with ``exec()``, a window placed over
  another's geometry): the same sizing and the same watch, without the show.

THE WATCH (`ScreenFit`, an event filter the window owns). After every layout change the window's minimum is held to
the available area less its frame (`screenfit_view.capped_minimum`): a layout asking for more gets the screen, never
a window past its edge -- a page that can scroll then scrolls, the rule every page follows. When the layout's own
minimum fits again the explicit minimum is dropped and the layout keeps control of it. When the window moves to
another screen (`QWindow.screenChanged`, a laptop docked to a monitor and back) or the screen's available area
changes (`availableGeometryChanged`, the taskbar, a new resolution), the minimum is held to the new screen and a
window now larger than it is brought back inside.

NOT DONE HERE, on purpose: no font or theme scale, no ``QT_SCALE_FACTOR`` -- Qt's device pixel ratio already maps a
125 % screen; the problem was never the scale.
"""
from __future__ import annotations

from qtpy.QtCore import QEvent, QObject, QTimer

from ... import screenfit_view as sv

__all__ = ["SETTLE_MS", "ScreenFit", "fit_to_screen", "show_fitted"]

#: How long a window must rest after a screen change before it is brought inside the new screen.
SETTLE_MS = 300


def _rect(r) -> tuple[int, int, int, int]:
    return r.x(), r.y(), r.width(), r.height()


def _frame(window) -> tuple[int, int, int, int]:
    """The window's frame (left, top, right, bottom): measured once its native window exists and is shown,
    `screenfit_view.DEFAULT_FRAME` before."""
    handle = window.windowHandle()
    if handle is not None and window.isVisible():
        m = handle.frameMargins()
        if m.left() or m.top() or m.right() or m.bottom():
            return m.left(), m.top(), m.right(), m.bottom()
    return sv.DEFAULT_FRAME


def _screen(window):
    from qtpy.QtWidgets import QApplication

    return window.screen() or QApplication.primaryScreen()


class ScreenFit(QObject):
    """The watch on one window (see the module docstring). Parented to the window, so it goes with it."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        #: Whether the window's minimum is currently held explicitly (its layout asked for more than the screen).
        self.capped = False
        self._watched_screen = None
        self._handle = None
        self._pending = False
        #: After a screen change, the window is brought inside once it has stopped moving: a window dragged to
        #: another screen is not resized under the pointer, and the platform's own placement lands first.
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(SETTLE_MS)
        self._settle.timeout.connect(self.refit)
        window.installEventFilter(self)

    @classmethod
    def install(cls, window) -> ScreenFit:
        """The window's watch, made once."""
        fit = getattr(window, "_screen_fit", None)
        if fit is None:
            fit = cls(window)
            window._screen_fit = fit
        return fit

    # -- the minimum ------------------------------------------------------------------------------------------ #
    def natural_minimum(self) -> tuple[int, int]:
        """What the window's layout asks for at least, whatever minimum is held on the window."""
        layout = self.window.layout()
        size = layout.totalMinimumSize() if layout is not None else self.window.minimumSizeHint()
        return size.width(), size.height()

    def cap(self, screen=None) -> None:
        """Hold the window's minimum to its screen (``screen``, else the one it is on), or give it back to the
        layout when the layout's own fits."""
        self._pending = False
        try:
            screen = screen or _screen(self.window)
        except RuntimeError:                     # the window is being torn down
            return
        if screen is None:
            return
        held = sv.capped_minimum(self.natural_minimum(), _rect(screen.availableGeometry()), _frame(self.window))
        if held is not None:
            if (self.window.minimumWidth(), self.window.minimumHeight()) != held:
                self.window.setMinimumSize(*held)
            self.capped = True
        elif self.capped:
            # zero clears the explicit minimum (`QWidget.setMinimumSize`), and the layout sets its own again
            self.capped = False
            self.window.setMinimumSize(0, 0)
            layout = self.window.layout()
            if layout is not None:
                layout.invalidate()
                layout.activate()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.window:
            kind = event.type()
            if kind == QEvent.LayoutRequest and not self._pending:
                # after the layout has answered it: the filter sees the request first. The context object drops
                # the call with this watch if the window goes first.
                self._pending = True
                QTimer.singleShot(0, self, self.cap)
            elif kind == QEvent.Show:
                self._watch_screen()
            elif kind == QEvent.Move and self._settle.isActive():
                self._settle.start()             # still moving: wait until it rests
        return False

    # -- the screen ------------------------------------------------------------------------------------------- #
    def _watch_screen(self) -> None:
        handle = self.window.windowHandle()
        if handle is not None and handle is not self._handle:
            self._handle = handle
            handle.screenChanged.connect(self._screen_changed)
        self._follow(_screen(self.window))

    def _follow(self, screen) -> None:
        if screen is self._watched_screen:
            return
        if self._watched_screen is not None:
            try:
                self._watched_screen.availableGeometryChanged.disconnect(self._available_changed)
            except (RuntimeError, TypeError):
                pass
        self._watched_screen = screen
        if screen is not None:
            screen.availableGeometryChanged.connect(self._available_changed)

    def _screen_changed(self, screen) -> None:
        self._follow(screen)
        self._changed()

    def _available_changed(self, _geometry) -> None:
        self._changed()

    def _changed(self) -> None:
        # The minimum at once (it moves nothing, and a maximised window must not outgrow the new screen), after the
        # platform has finished the move that changed the screen; the size and place once the window rests.
        QTimer.singleShot(0, self, self.cap)
        self._settle.start()

    def refit(self) -> None:
        """On another screen (or a changed one): the minimum held to it, and a normal window now reaching past it
        brought back inside -- cut to the screen's size, moved as little as that takes (`screenfit_view.inside`). A
        maximised or full-screen window is the platform's to place."""
        try:
            self.cap()
            window = self.window
            screen = _screen(window)
            if screen is None or window.isMaximized() or window.isFullScreen() or not window.isVisible():
                return
            frame = window.frameGeometry()
            available = _rect(screen.availableGeometry())
            if not sv.outside(_rect(frame), available):
                return
            x, y, w, h = sv.inside(_rect(frame), (window.width(), window.height()), available, _frame(window),
                                   minimum=(window.minimumWidth(), window.minimumHeight()))
            window.setGeometry(x, y, w, h)
        except RuntimeError:                    # the window went while the timer ran
            return


def preferred_size(window) -> tuple[int, int]:
    """The size ``window`` asked for: what its constructor resized it to, else its size hint (a window never resized
    still reads Qt's default 640 x 480, which is no one's preference)."""
    from qtpy.QtCore import Qt

    if window.testAttribute(Qt.WA_Resized):
        return window.width(), window.height()
    hint = window.sizeHint()
    return hint.width(), hint.height()


def fit_to_screen(window, *, maximise=None, screen=None):
    """Size and place ``window`` for its screen (``screen``, else the one it is on) without showing it, and watch it
    from now on. Returns the `screenfit_view.WindowFit` decided, so the caller knows whether to maximise."""
    watch = ScreenFit.install(window)
    screen = screen or _screen(window)
    if screen is None:
        return None
    preferred = preferred_size(window)
    watch.cap(screen)
    natural = watch.natural_minimum()
    fit = sv.fit_window(preferred, _rect(screen.availableGeometry()), _frame(window),
                        minimum=natural, maximise=maximise)
    window.setGeometry(fit.x, fit.y, fit.width, fit.height)
    return fit


def show_fitted(window, *, maximise=None, screen=None) -> None:
    """Show ``window`` fitted to its screen (`fit_to_screen`): normal and centred where its preferred size fits,
    maximised where it does not (or always, with ``maximise=True``)."""
    fit = fit_to_screen(window, maximise=maximise, screen=screen)
    if fit is not None and fit.maximised:
        window.showMaximized()
    else:
        window.show()
