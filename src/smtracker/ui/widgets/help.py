"""The Help buttons: a menu of two ways to read the manual at a stage's chapter, and Check for updates (thin Qt).

Everything about the manual -- where it is, how it is rendered, which chapter a stage opens --
is `smtracker.manual`'s; this only hands the page it writes to the default browser, and says
so in a message box when it cannot (a manual missing from a build is a packaging mistake,
named, never an empty page).
"""
from __future__ import annotations


def open_website(stage: str, parent=None) -> bool:
    """Open ``stage``'s chapter on the website in the default browser. False, with a message naming the
    address, when the browser cannot be started."""
    from qtpy.QtCore import QUrl
    from qtpy.QtGui import QDesktopServices
    from qtpy.QtWidgets import QMessageBox

    from ...manual import website_url

    url = website_url(stage)
    if not QDesktopServices.openUrl(QUrl(url)):
        QMessageBox.information(parent, "Manual", f"Open this address in a browser:\n{url}")
        return False
    return True


def _action(key: str, t, stage: str, parent):
    if key == "website":
        return lambda: open_website(stage, parent)
    if key == "updates":
        def check():
            from .update_notice import check_for_updates

            check_for_updates(parent, t)
        return check
    return lambda: open_manual(stage, parent)


def help_menu(t, stage: str, parent=None):
    """The Help menu, in the app's dropdown style (`widgets.menus.Panel`): the two ways to read the manual
    (`manual.help_choices`), the website's in the browser and the copy in SMTracker, and Check for updates
    (`update_notice.check_for_updates`). Returned, not shown:
    `show_help_menu` places it. Each row's tip is its tooltip."""
    from ...manual import help_choices
    from .menus import Panel, menu_row

    menu = Panel(t, parent)
    for key, label, tip, _target in help_choices(stage):
        row = menu_row(t, menu, label, on_click=_action(key, t, stage, parent))
        row.setToolTip(tip)
        row.setObjectName(f"help_{key}")
    return menu


def show_help_menu(stage: str, parent, button, side: str = "below"):
    """Open the Help menu beside ``button`` (its `.t` is the theme): under it, its right edge on the button's
    (a header button), or to its left (the rail's). Returned, so the window can keep it alive."""
    from qtpy.QtCore import QPoint

    menu = help_menu(button.t, stage, parent)
    if side == "left":
        menu.adjustSize()
        menu.popup(button.mapToGlobal(QPoint(-menu.visible_width() - 4, 0)))
    else:
        menu.popup_under(button, align="end")
    parent._help_menu = menu  # held: a menu with no owner is freed at once
    return menu


def open_manual(stage: str, parent=None) -> bool:
    """Write the manual pages and open ``stage``'s in the default browser. False, with a message
    naming why, when it cannot."""
    from qtpy.QtCore import QUrl
    from qtpy.QtGui import QDesktopServices
    from qtpy.QtWidgets import QMessageBox

    from ...manual import ManualError, write_pages

    try:
        page = write_pages()[stage]
    except ManualError as exc:
        QMessageBox.warning(parent, "Manual", f"The manual could not be opened:\n{exc}")
        return False
    if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(page))):
        QMessageBox.information(parent, "Manual", f"Open this file in a browser:\n{page}")
        return False
    return True
