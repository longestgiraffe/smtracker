"""SMTracker 2.0's look, for the legacy window (`docs/legacy_window_plan.md`).

2.0 was one white MATLAB figure (`SMTracker.m:66-77`): uicontrols in bold-titled etched panels, tables striped white
and grey 0.94 (`'RowStriping','on'`), a light-grey Console strip, axes at font size 14 with the box on. This module
holds the style sheet for the window's widgets (the app's own theme must not reach in); the colours and the
matplotlib settings that make an axes read as MATLAB's are in `smtracker.legacy_style` (headless, so the drawing
modules can read them) and are re-exported here.
"""
from __future__ import annotations

from ...legacy_style import (APPD_POP_COLOURS, AXES_FONT_PT, CONSOLE_GREY, FONT_FAMILY, MATLAB_LINES,  # noqa: F401
                             MENU_DISABLED, MENU_HOVER, MENU_TEXT, PANEL_EDGE, POP_LINESTYLES, SLA_COLOURS, STRIPE_GREY, WHITE,
                             boxplot_style, matlab_rc)

WINDOW_QSS = f"""
QMainWindow, QWidget#legacyRoot, QTabWidget::pane, QScrollArea, QScrollArea > QWidget > QWidget {{
    background: {WHITE};
}}
QWidget {{
    color: #000000;
    font-family: "Helvetica", "Arial", "Inter", sans-serif;
    font-size: 9pt;
}}
QGroupBox {{
    background: {WHITE};
    border: 1px solid {PANEL_EDGE};
    border-radius: 0px;
    margin-top: 14px;
    padding: 6px 4px 4px 4px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 6px;
    padding: 0 3px;
    background: {WHITE};
}}
QGroupBox QWidget {{ font-weight: normal; }}
QTabWidget::pane {{ border: 1px solid {PANEL_EDGE}; top: -1px; }}
QTabBar::tab {{
    background: #ececec;
    border: 1px solid {PANEL_EDGE};
    border-bottom: none;
    padding: 3px 12px;
    margin-right: 1px;
}}
QTabBar::tab:selected {{ background: {WHITE}; font-weight: bold; }}
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fdfdfd, stop:1 #e4e4e4);
    border: 1px solid #9a9a9a;
    border-radius: 3px;
    padding: 3px 10px;
}}
QPushButton:hover {{ border-color: #3c7fb1; }}
QPushButton:pressed {{ background: #d6d6d6; }}
QPushButton:disabled {{ color: #9a9a9a; }}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {WHITE};
    border: 1px solid #9a9a9a;
    padding: 1px 4px;
    min-height: 18px;
}}
QTableWidget {{
    background: {WHITE};
    alternate-background-color: {STRIPE_GREY};
    gridline-color: #d8d8d8;
    border: 1px solid {PANEL_EDGE};
}}
QHeaderView::section {{
    background: #f7f7f7;
    border: 1px solid #d8d8d8;
    padding: 2px 4px;
}}
QPlainTextEdit#legacyConsole {{
    background: {CONSOLE_GREY};
    border: 1px solid {PANEL_EDGE};
    font-family: "Courier New", monospace;
}}
QPushButton#legacyBack {{ font-weight: bold; border-color: #3c7fb1; margin: 1px 4px; }}
QDialog, QMessageBox {{ background: {WHITE}; }}
QMenuBar {{ background: {WHITE}; color: {MENU_TEXT}; }}
QMenuBar::item {{ background: transparent; color: {MENU_TEXT}; padding: 3px 8px; }}
QMenuBar::item:selected {{ background: {MENU_HOVER}; color: {MENU_TEXT}; }}
QMenu {{ background: {WHITE}; color: {MENU_TEXT}; border: 1px solid {PANEL_EDGE}; }}
QMenu::item {{ background: transparent; color: {MENU_TEXT}; padding: 3px 24px 3px 20px; }}
QMenu::item:selected {{ background: {MENU_HOVER}; color: {MENU_TEXT}; }}
QMenu::item:disabled {{ background: transparent; color: {MENU_DISABLED}; }}
QMenu::separator {{ height: 1px; background: #d8d8d8; margin: 3px 0; }}
"""
