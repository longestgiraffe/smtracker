"""The analysis flow, end to end: home → confirm → a loaded project.

Two screens in a stack and one worker between them. The window owns the
sequencing so neither screen has to know about the other: `HomePage` emits a
folder, `ConfirmPage` shows what is in it, `scan.Loader` fills the numbers, and
the window hands the finished project on.

THE FOLDER IS REMEMBERED WHEN IT IS OPENED, not when it is chosen. A folder that
turned out to be the wrong one — no conditions under it — has not been opened by
any useful definition, and putting it at the top of the recents list would push a
real project off the end for a mistake the user has already corrected.

THE THREAD IS JOINED ON CLOSE. A QThread destroyed while running calls `abort()`:
no exception, no traceback, nothing to catch. That is the same rule `open_gui`
carries, and it is why `closeEvent` is not left to Qt.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QApplication, QMessageBox, QStackedWidget, QVBoxLayout, QWidget

from ..theme import typography as ty
from ..widgets.primitives import breathe, discard, label, scrollbar_css
from . import scan
from .confirm import LoadingPage
from .home import HomePage
from .screens import ConfirmPage

__all__ = ["OnboardingWindow", "open_project"]

#: The size the design was drawn at. It is a FLOOR, not a size: the window opens
#: maximized, because the Confirm screen is a table of every series in the
#: project and the number of rows is the experiment's, not the layout's. An
#: earlier version pinned this at 980x640 -- a number I chose, not one the design
#: asked for -- and a 30-series project scrolled inside a window with empty
#: desktop around it.
WIDTH, HEIGHT = 1456, 940


class OnboardingWindow(QWidget):
    """Home and Confirm, and the load between them. Emits `opened(project)`."""

    opened = Signal(object)
    cancelled = Signal()

    def __init__(self, t, parent=None):
        super().__init__(parent)
        self.t = t
        self.loader: scan.Loader | None = None
        #: (found, day_maps, sources, leave_out) of the load in flight, for `_ask_leave_out`
        self._load_context = None
        self.scout: scan.Scout | None = None
        self.confirm: ConfirmPage | None = None
        #: Confirm Project's colour choice, set at hand-over (see `_hand_on`).
        self.colours = None
        from ...buildstamp import build_stamp

        self.setWindowTitle(f"SMTracker  ·  {build_stamp()}")
        # THE SCROLLBAR IS shadcn's, and it is set ONCE here so every scroll
        # area in the flow inherits it (Qt stylesheets cascade by type). Qt's
        # native bar is wide platform chrome that reads as a different
        # application inside this design -- the prototype set this rule on
        # its own shell, and dropping that class on the way in is what put
        # the native bars back. `scrollbar_css` is the ONE copy of it now
        # (`widgets.primitives`), shared with every other scroll area in this
        # UI stack rather than a second typed-out copy that could drift.
        self.setStyleSheet(
            f"QWidget {{ background:{t.background}; color:{t.foreground}; }}"
            + scrollbar_css(t))
        self.resize(WIDTH, HEIGHT)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.pages = QStackedWidget()
        lay.addWidget(self.pages)

        self.home = HomePage(t)
        self.home.chosen.connect(self.open_folder)
        self.home.help_requested.connect(self.open_help)
        self.home.options_requested.connect(self.open_options)
        self.home.menu_requested.connect(self.back_to_menu)
        self.pages.addWidget(self.home)
        #: Set by Back to menu: the door (`app.main`) opens the stage menu again instead of ending (`take_after_close`)
        self.return_to_menu = False

        # NO PRE-WARM HERE, AND THE REASON IT WENT IS WORTH KEEPING.
        #
        # A `_prewarm` used to build a throwaway `ConfirmPage` on a 150 ms timer
        # so the real one would not freeze on arrival. It was justified by a real
        # measurement -- the first `ConfirmPage` in a process cost 4.1 s and every
        # one after it 210 ms -- and the diagnosis attached to that number was
        # wrong: it read as "Qt initialising itself incrementally as widgets join
        # a tree", i.e. an unavoidable cost that could only be MOVED.
        #
        # It was none of that. It was a FONT FALLBACK SEARCH: the bundled face had
        # no U+2009, so the first string containing one sent Qt walking all 2 942
        # installed families, synchronously, inside whatever `addWidget` happened
        # to touch it first. Changing the face to Inter removed the search, and
        # with it the thing the pre-warm existed to amortise.
        #
        # MEASURED after the swap, on this exact sequence -- home screen up, then
        # a 30-series project: the real `ConfirmPage` costs 367 ms with no
        # pre-warm and 360 ms with one, for 318 ms of work on the home screen.
        # Seven milliseconds is not a reason. `HomePage` already absorbs the one
        # cost that is genuinely unavoidable (Qt populating the font database,
        # ~1.7 s on the first text this process draws), and it absorbs it in the
        # right place, where a window is appearing and nothing waits on a result.
        #
        # `ConditionBody.warm()` is NOT the same case and stays: it defers the
        # groups board, which is 826 ms of real widget building on 30 series and
        # does not shrink because a glyph was found. See `_show_confirm`.

    # -- the sequence -------------------------------------------------------- #

    def open_folder(self, root) -> None:
        """A folder was chosen. Put a screen up FIRST, then go and look.

        NOTHING SLOW HAPPENS HERE. Discovery globs recursively -- 1.8 s on a
        30-series project -- and it used to run on this line, before the screen
        that would have said so existed: the window simply stopped for two
        seconds between the folder picker closing and anything appearing. The
        screen goes up empty and fills from `Scout`.
        """
        self.discovery = None
        self.loading = LoadingPage(self.t, root)
        self.loading.cancelled.connect(self.back_to_home)
        self.pages.addWidget(self.loading)
        self.pages.setCurrentWidget(self.loading)
        self.loading.say("Looking in this folder…")

        self.scout = scan.Scout(root)
        self.scout.found.connect(self.loading.add_condition)
        self.scout.done.connect(self._scouted)
        self.scout.start()

    def _scouted(self, found) -> None:
        """Discovery is in. Ask what the folders mean when that is worth asking (`design_view`), else go on."""
        scout, self.scout = getattr(self, "scout", None), None
        if scout is not None:
            scout.stop()
        if getattr(found, "design", None) is not None:
            self._ask_design(found)
            return
        self._go_on(found)

    def _ask_design(self, found) -> None:
        """The folder-design page (`design.DesignPage`): a design saved with the project makes the door look at the
        folder again through it; "load without" goes on with what discovery found."""
        from .design import DesignPage

        page = DesignPage(self.t, found.design)
        self.design_page = page

        def drop() -> None:
            self.design_page = None
            self.pages.removeWidget(page)
            page.deleteLater()

        def designed() -> None:
            drop()
            self.open_folder(found.root)

        def skipped() -> None:
            drop()
            if found.design.get("saved_file"):
                # A SAVED DESIGN THAT DOES NOT FIT, loaded without: set aside by name, never deleted, so the
                # loader and the replicate folders stop reading it (`design_view.set_aside`)
                from ...design_view import set_aside

                set_aside(found.root)
                self.open_folder(found.root)
                return
            self.pages.setCurrentWidget(self.loading)
            self._go_on(found)

        page.designed.connect(designed)
        page.skipped.connect(skipped)
        page.cancelled.connect(lambda: (drop(), self.back_to_home()))
        self.pages.addWidget(page)
        self.pages.setCurrentWidget(page)

    def _go_on(self, found) -> None:
        """Either say what is wrong, or start reading."""
        if not found.ok:
            self.loading.on_failed(found.problem)
            return
        self.discovery = found
        self._remember(found)
        self.loading.set_summary(found.summary())

        # A CONDITION TRACKED TWICE IS A QUESTION, NOT AN ERROR. Reported: a condition
        # holding u-track and TrackMate output stopped this door with the loader's
        # refusal. The page asks, per such condition, with nothing preselected.
        from ...onboarding_view import needs_tracker_choice

        if needs_tracker_choice(found.trackers):
            self._ask_trackers(found)
            return
        self._start_load(found, None)

    def _ask_trackers(self, found) -> None:
        from .trackers import TrackerChoicePage

        page = TrackerChoicePage(self.t, found.root, found.trackers)
        self.tracker_page = page

        def chosen(sources) -> None:
            self.pages.setCurrentWidget(self.loading)
            self.tracker_page = None
            self.pages.removeWidget(page)
            page.deleteLater()
            self._start_load(found, sources)

        def back() -> None:
            self.tracker_page = None
            self.pages.removeWidget(page)
            page.deleteLater()
            self.back_to_home()

        page.chosen.connect(chosen)
        page.cancelled.connect(back)
        self.pages.addWidget(page)
        self.pages.setCurrentWidget(page)

    def _start_load(self, found, sources) -> None:
        """Day maps on this thread, then the loader on its own; ``sources`` is the
        tracker chosen per condition, or None where one tracker made everything."""
        # RESOLVED HERE, ON THE GUI THREAD, BEFORE THE WORKER STARTS -- it is a
        # lookup in `day_map.json`, and `resolve_day_maps` is the SAME function
        # `open_gui` and the command line use, so this path cannot settle a
        # condition's replicate days differently from either of them. Skipping
        # this and handing `Loader` only `series_per_day` was the bug: a
        # condition whose series blocks are non-contiguous under the literal
        # per-10 rule (e.g. 001-009 / 031-040 / 041-050) raised `GroupingError`
        # straight out of `load_project`, with no chance to derive or ask.
        #
        # `interactive=False`: this window has its OWN place to review and edit
        # this mapping -- the Confirm page's group board, at `_hand_on` below --
        # so popping `DayMapDialog` here as well asked the same question twice.
        # A remembered mapping still applies; a first-time project gets the
        # naming authority's derived answer, silently, and the group board is
        # where a wrong guess gets corrected.
        from ...app import resolve_day_maps

        self.loading.say("Reading the tracker's tables…")
        try:
            # THE DAYS ARE NUMBERED FROM THE TRACKER CHOSEN: a condition tracked twice
            # may hold different movies per tracker, and numbering the other one's
            # left the load refusing "day mapping does not cover these movies".
            day_maps = resolve_day_maps(found.root, report=self.loading.say,
                                        interactive=False, sources=sources)
        # reported to the user on the loading page
        except Exception as exc:                      # noqa: BLE001 - reported
            self.loading.on_failed(f"{type(exc).__name__}: {exc}")
            return

        self._start_loader(found, day_maps, sources, leave_out=None)

    def _start_loader(self, found, day_maps, sources, *, leave_out) -> None:
        """The loader on its own thread; ``leave_out`` is the answer to `_ask_leave_out`, kept for a second
        question (a movie of another condition found missing after the first are left out)."""
        self.loader = scan.Loader(found.root, day_maps=day_maps, sources=sources, leave_out=leave_out)
        self.loader.progress.connect(self.loading.on_progress)
        self.loader.loaded.connect(self._show_confirm)
        self.loader.failed.connect(self.loading.on_failed)
        # A BOUND METHOD, never a lambda: the signal comes from the loader's thread, and only a slot of this
        # window is queued to the GUI thread -- a lambda would run there and touch widgets from the worker
        self._load_context = (found, day_maps, sources, leave_out)
        self.loader.missing.connect(self._ask_leave_out)
        self.loader.start()

    def _ask_leave_out(self, by_condition) -> None:
        """Movies tracked without a pixel size or frame interval: ASK whether to leave them out
        (`onboarding_view.leave_out_question`), and on yes load again without them."""
        from ...onboarding_view import leave_out_question

        found, day_maps, sources, earlier = self._load_context
        self._stop()
        question = leave_out_question(by_condition)
        merged = {c: dict(m) for c, m in (earlier or {}).items()}
        for condition, movies in question["leave_out"].items():
            merged.setdefault(condition, {}).update(movies)
        self.loading.ask_leave_out(question["title"], question["body"])

        def chosen():
            self.loading.leave_out_chosen.disconnect(chosen)
            self.loading.resume()
            self._start_loader(found, day_maps, sources, leave_out=merged)

        self.loading.leave_out_chosen.connect(chosen)

    def _show_confirm(self, loaded, projects) -> None:
        """The load landed, and the adapter has already run on the worker.

        What is left is Qt, which can only happen here. The screen still costs
        about a second to build on a 30-series project -- `ConditionBody` defers
        the groups panel for exactly this reason -- so the line above the
        spinner says what is happening before the thread stops answering.
        """
        self.loading.say("Building the screen…")
        QApplication.processEvents()
        self._stop()
        self.loaded = loaded
        # `ConfirmPage` breathes between conditions and between lanes, so the
        # spinner on the screen behind it keeps turning while this runs.
        self.confirm = ConfirmPage(self.t, projects)

        self.confirm.cancelled.connect(self.back_to_home)
        self.confirm.opened.connect(lambda: self._hand_on(self.loaded))
        self.pages.addWidget(self.confirm)
        self.pages.setCurrentWidget(self.confirm)
        # ONLY NOW. The groups board is warmed once this page is actually up --
        # armed any earlier, its timer can come due inside the page's own build.
        self.confirm.warm()
        self._tell_left_out(loaded)

    def _tell_left_out(self, loaded) -> None:
        """A window naming the movies the load left out for lack of cell outlines (`onboarding_view.
        unsegmented_notice`), over the confirm page."""
        from ...onboarding_view import unsegmented_notice
        from ..widgets.dialogs import NoticeDialog

        canonical = getattr(loaded, "canonical", None)
        notice = unsegmented_notice(getattr(canonical, "attrs", {}).get("dropped_unsegmented"),
                                    list(getattr(loaded, "conditions", []) or []))
        if notice is None:
            return
        dialog = NoticeDialog(self.t, notice["title"], notice["body"], content=self.pages, parent=self)
        dialog.closed.connect(dialog.deleteLater)
        self._notice = dialog                     # held: a dialog collected while open takes its children with it
        dialog.open()

    def _rechoose(self) -> None:
        """Back to the picker, with the current load stopped first."""
        self.back_to_home()
        self.home.browse()

    def back_to_home(self) -> None:
        """Cancel: stop the load, drop the screen, and go back to choosing."""
        self._stop()
        page, self.confirm = self.confirm, None
        # The folder just tried was remembered when it was found: show it.
        self.home.refresh_recents()
        self.pages.setCurrentWidget(self.home)
        if page is not None:
            self.pages.removeWidget(page)
            page.deleteLater()

    def _hand_on(self, project) -> None:
        self._stop()
        # THE COLOUR CHOICE GOES WITH THE PROJECT. Confirm Project's palette and
        # per-condition picks were once dropped here, so the workspace opened on
        # Okabe-Ito whatever was chosen. Read before `_regrouped`, which may be
        # the last thing that sees `self.confirm`.
        self.colours = (self.confirm.colours.model.copy()
                        if self.confirm is not None else None)
        # THE ORDER DRAGGED ON THE TABS (6 Nov 2026): remembered for the project, which the workspace reads first
        # (`condition_order.recall` under `condition_order.project_key`, the same key both sides)
        if self.confirm is not None and getattr(self.confirm, "order_changed", False):
            from ... import condition_order

            key = condition_order.project_key(list((getattr(project, "roots", None) or {}).values()))
            if key:
                condition_order.remember(key, self.confirm.condition_order())
        self.opened.emit(self._regrouped(project))

    def _regrouped(self, project):
        """Apply the group board's confirmed arrangement -- ONLY HERE, on
        "Open workspace", never while a series is merely being dragged.

        `project.regroup` is cheap by design (see its own docstring: it
        relabels an already-loaded column, it does not re-read anything), so
        the cost this guards is not the relabel itself -- it is writing
        `day_map.json`. That write is the same "confirmed, don't ask again"
        commit this project treats as deliberate everywhere else (the day
        dialog this window no longer shows saved on confirm and nowhere
        earlier), and doing it on every drop would put a half-considered
        arrangement on disk as if it were the record.

        A SAVE THAT FAILS IS SAID, NOT SWALLOWED. The arrangement still applies
        to this session (the regroup above is already done); only the memory of
        it failed, so the next open would ask again with nothing having told the
        user why. The two failures a save can have are named: an `OSError` from
        the write and a `DayMapError` for a mapping it refuses. Anything else is
        a bug and is raised.
        """
        if self.confirm is None:
            return project
        from ...daymap import DayMapError, DayMapping, day_map_to_save, load_day_map, save_day_map
        from dataclasses import replace

        from ...project import regroup
        from .adapt import day_maps_from, left_out_from, pairs_from

        day_maps = day_maps_from(self.confirm.projects)
        if not day_maps:
            return project
        # WHAT THE BOARD HELD OUT OR LEFT WITHOUT A GROUP, remembered with the grouping (tester D11, ruled 4 Nov 2026)
        left_out = left_out_from(self.confirm.projects)
        # THE PAIRS THE BOARD DECLARED (5 Nov 2026, opt-in): numbered as `day_maps` numbers the groups, so they are the
        # workspace's replicates as they are the file's; nothing while Pair replicates is unticked
        pairs = pairs_from(self.confirm.projects, bool(getattr(getattr(self.confirm, "pairing", None), "on", False)))
        regrouped = regroup(project, day_maps)
        if pairs or getattr(regrouped, "replicate_pairs", None):
            regrouped = replace(regrouped, replicate_pairs={c: tags for c, tags in pairs.items() if c in day_maps})
        failed: list[str] = []
        for condition, mapping in day_maps.items():
            path = project.roots.get(condition)
            original = project.day_maps.get(condition)
            # PERSISTED, NOT THE SAME MAPPING `regroup` JUST USED. day_map.json
            # is also read by the RAW load on the next open (`assign_groups`
            # refuses a mapping with any gap), so a series held out or left
            # unplaced on the group board -- correctly ABSENT from `mapping`,
            # see `adapt.day_maps_from` -- must still be written with ITS OWN
            # day here, or the next load raises "does not cover these movies"
            # for a series nobody meant to remove from the FILE, only from the
            # fit. `original` is the exact mapping the loader used to get this
            # project open at all, so it is always a complete covering;
            # `mapping` only overrides the series the board actually placed.
            # Those series are written ALSO as held out / no group (tester D11,
            # ruled 4 Nov 2026), so the next open leaves them out again
            # (`adapt._saved_left_out`) instead of bringing them back.
            if path is None or original is None:
                continue
            complete = {**original, **mapping}
            # ONLY A CHANGED MAPPING IS WRITTEN (tester B10, 3 Nov 2026): every open rewrote every condition's
            # day_map.json as "manual", so a synthetic set's or an earlier session's provenance was lost and the door
            # wrote into the data folders on every open with nothing changed (`daymap.day_map_to_save`)
            try:
                on_disk = load_day_map(path)
            except DayMapError:                       # unreadable: written again, as a confirmation always was
                on_disk = None
            attrs = getattr(getattr(project, "canonical", None), "attrs", None) or {}
            provenance = ((attrs.get("day_mappings") or {}).get(condition) or {}).get("source")
            held = left_out.get(condition) or {}
            source = day_map_to_save(
                complete, original, None if on_disk is None else on_disk.movie_to_day, derived_source=provenance,
                left_out=held, on_disk_source=None if on_disk is None else on_disk.saved_source,
                on_disk_left_out=None if on_disk is None else {"held_out": on_disk.held_out,
                                                               "no_group": on_disk.no_group},
                pairs=pairs.get(condition), on_disk_pairs=None if on_disk is None else on_disk.pairs)
            if source is None:
                continue
            try:
                save_day_map(path, DayMapping(movie_to_day=complete, source=source,
                                              held_out=[m for m in held.get("held_out", ()) if m in complete],
                                              no_group=[m for m in held.get("no_group", ()) if m in complete],
                                              pairs=dict(pairs.get(condition) or {})))
            except (OSError, DayMapError) as exc:
                failed.append(f"{condition}: {type(exc).__name__}: {exc}")
        if failed:
            self._warn_not_saved(failed)
        return regrouped

    def _warn_not_saved(self, failed: list[str]) -> None:
        """The arrangement is applied for this session; only the memory of it
        failed, so the next open will ask again. Said once, listing each
        condition that could not be saved and why."""
        QMessageBox.warning(
            self, "Replicate grouping not saved",
            "Your grouping is applied for this session, but it could not be "
            "saved, so the next open will ask again:\n\n" + "\n".join(failed))

    def _refuse(self, found) -> None:
        """Say what is wrong WHERE the folder was chosen, and stay put.

        `discover` phrases the problem as the thing to do next, so this shows it
        under the drop zone rather than in a message box -- a modal dialog on a
        folder that simply needs re-picking is a click for nothing.
        """
        if getattr(self, "_problem", None) is not None:
            discard(self._problem)
        self._problem = label(found.problem.splitlines()[0], ty.ui("sm"),
                              self.t.foreground)
        self._problem.setWordWrap(True)
        self._problem.setAlignment(Qt.AlignHCenter)
        self.home.layout().addWidget(self._problem)

    def _remember(self, found) -> None:
        """Put the opened folder at the top of the recents list.

        NO CATCH HERE: `session` already handles the failures this list can
        have (`load_session` never raises on a bad file and `save_session`
        drops a failed write, by its own best-effort design), so a broad catch
        here guarded nothing real and could only hide a bug."""
        from ...session import remember_project

        remember_project(found.root, conditions=len(found.conditions),
                         series=found.series)

    # -- the start screen's header ------------------------------------------- #
    def open_help(self) -> None:
        """The start page's Help: the menu of the manual on the website or in SMTracker (analysis chapter)."""
        from ..widgets.help import show_help_menu

        show_help_menu("analysis", self, self.home.help_button)

    def open_options(self):
        """The analysis's Options screen (the standards a new analysis starts from), over this window."""
        from ..widgets.options_dialog import open_options

        return open_options(self, self.t, "analysis", content=self.pages)

    def back_to_menu(self) -> None:
        """Back to the stage menu: the door reads `return_to_menu` after this window closes."""
        global _AFTER_CLOSE
        self.return_to_menu = True
        _AFTER_CLOSE = "menu"
        self.close()

    # -- teardown ------------------------------------------------------------ #

    def _stop(self) -> None:
        loader, self.loader = self.loader, None
        if loader is not None:
            loader.stop()
        scout, self.scout = getattr(self, "scout", None), None
        if scout is not None:
            scout.stop()

    def closeEvent(self, e):
        self._stop()
        self.cancelled.emit()
        super().closeEvent(e)


#: What the start screen asked for as it closed ("menu"), for the door (`app.main`); read once by `take_after_close`.
_AFTER_CLOSE: str | None = None


def take_after_close() -> str | None:
    """"menu" when the start screen closed through Back to menu, else None; cleared as it is read."""
    global _AFTER_CLOSE
    after, _AFTER_CLOSE = _AFTER_CLOSE, None
    return after


def show_onboarding(window) -> None:
    """The start screen's door: MAXIMISED, as it always opened, its normal size `WIDTH` x `HEIGHT` cut to the screen,
    and its minimum held to the screen it is on (`ui.widgets.screenfit`; on a 1536 x 816 laptop the home page's 850 px
    minimum made the maximised window 38 px taller than the screen, 30 Oct 2026)."""
    from ..widgets.screenfit import show_fitted

    show_fitted(window, maximise=True)


def open_project(*, color_scheme=None, parent=None):
    """Run the flow modally and return ``(project, colours)``, or ``(None, None)``.

    ``colours`` is Confirm Project's `condition_colours.ConditionColours` (palette
    and per-condition slots), for `open_workspace(colours=)`.

    The blocking form, for `app.main`, which is a script: it needs the project
    in hand to open the browser with. A caller that already has an event loop
    should construct `OnboardingWindow` and connect to `opened` instead.
    """
    from qtpy.QtWidgets import QApplication

    # Constructing a widget before a QApplication exists calls abort() -- no
    # traceback, no message. Every door in this application carries this line.
    app = QApplication.instance() or QApplication([])
    from ...appicon import apply as _apply_icon
    _apply_icon()
    from ..theme import current

    # A TOP-LEVEL WINDOW AND THE APPLICATION'S OWN LOOP -- NOT `QDialog.exec()`.
    #
    # This window used to live inside a QDialog run modally, and that single
    # choice was the freeze. MEASURED on the reference project, same machine,
    # same real window, cold load, worst stall in the event loop:
    #
    #     QDialog.exec()      3 933 ms, 4 043 ms
    #     app.exec()             50 ms,    76 ms
    #
    # A modal nested loop starves the animation and paint timers while the
    # worker thread is busy, so the spinner stopped for four seconds on a screen
    # whose whole job is to say the application is alive. Every measurement that
    # said this flow was smooth was taken with a plain window, which is the
    # mistake: the harness was not running what the application runs.
    window = OnboardingWindow(current(color_scheme), parent=parent)
    show_onboarding(window)

    picked = {}

    def take(project) -> None:
        picked["project"] = project
        picked["colours"] = getattr(window, "colours", None)
        window.close()
        app.quit()

    window.opened.connect(take)
    window.cancelled.connect(app.quit)
    (app.exec_() if hasattr(app, "exec_") else app.exec())
    window._stop()
    return picked.get("project"), picked.get("colours")
