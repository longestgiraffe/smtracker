"""The day-mapping dialog, a THIN Qt shell over ``smtracker.daymap``.

Editable range/day table, pre-filled with the timestamp-derived proposal (or the
subfolder layout). Every computation (proposing, parsing ranges, validating
against timestamps) lives in ``daymap``; this file only moves values between the
table and those pure functions and shows their verdict. It computes nothing, so
(like the other widgets) an AST test keeps numpy/pandas/scipy out of it.

Honesty rules made visible:
- a red BANNER when acquisition timestamps are missing/unreadable (the proposal
  could not be verified: the user must enter it manually and knows it);
- warnings from the timestamp cross-check are shown and gate the Confirm button
  behind an explicit "I confirm despite the warnings" tick, never silent;
- hard errors (a movie unassigned / assigned twice) disable Confirm outright.
"""

from __future__ import annotations

import re

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import daymap as dm


def rows_from_mapping(mapping: "dm.DayMapping", tau_by_movie=None) -> list[dict]:
    """Table rows (one per day) from a DayMapping: pure, so it is unit-tested.

    Each row: the day number, the compact series RANGE spec, the date, the time
    span, the movie count, and the frame interval(s) in that day. The range spec is
    what the user edits.

    The tau column exists because a day that MIXES frame intervals is fatal: the
    day is the analysis unit and its movies get pooled into one number, while every
    D scales with 1/tau. Showing it per day makes the offending block visible while
    the user is assigning days, which is the only moment they can fix it.
    """
    tau_by_movie = tau_by_movie or {}
    rows = []
    for group in sorted(mapping.groups, key=lambda g: g.day):
        series = sorted(int(match.group()) for m in group.movies
                        if (match := re.search(r"\d+", m)))
        rates = sorted({round(float(tau_by_movie[m]), 6) for m in group.movies
                        if m in tau_by_movie})
        if not rates:
            tau_text = "–"
        elif len(rates) == 1:
            tau_text = f"{rates[0]:g}"
        else:
            tau_text = " / ".join(f"{r:g}" for r in rates) + "  MIXED"
        rows.append({
            "day": group.day,
            "spec": _compact_ranges(series),
            "date": group.date or "–",
            "span": ("–" if group.span_hours is None
                     else f"{group.span_hours:.1f} h"),
            "n": len(group.movies),
            "tau": tau_text,
        })
    return rows


def movie_rows(mapping: "dm.DayMapping", movies, stamps=None,
               tau_by_movie=None) -> list[dict]:
    """One row per movie: movie, series, its day, and its timestamp.

    The per-movie view is the general one: a range spec cannot express "series 7 is
    day 2" unless its neighbours agree, and sometimes they do not. Pure, so the
    dialog computes nothing.
    """
    times = {s.movie: s.timestamp for s in (stamps or [])}
    tau_by_movie = tau_by_movie or {}
    rows = []
    for movie in sorted(movies, key=dm.series_number):
        ts = times.get(movie)
        ms = tau_by_movie.get(movie)
        rows.append({
            "movie": movie,
            "series": dm.series_number(movie),
            "day": int(mapping.movie_to_day.get(movie, 0)) or "",
            "tau": "–" if ms is None else f"{float(ms):g}",
            "time": "–" if ts is None else ts.strftime("%Y-%m-%d %H:%M"),
        })
    return rows


def _compact_ranges(numbers: list[int]) -> str:
    """[1,2,3,5,6] → "1-3, 5-6". The inverse of ``daymap.parse_range_spec``."""
    if not numbers:
        return ""
    nums = sorted(set(numbers))
    parts, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        parts.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = n
    parts.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ", ".join(parts)


class DayMapDialog(QDialog):
    """Confirm/override the replicate-day mapping before it drives the analysis."""

    #: by-day table columns
    COLS = ["Replicate", "Series (editable)", "Date", "Time span", "N movies", "τ (ms)"]
    #: by-movie table columns, the general view; the Day cell is editable
    MOVIE_COLS = ["Movie", "Series", "Replicate (editable)", "τ (ms)", "Timestamp"]

    def __init__(self, mapping: "dm.DayMapping", *, stamps=None, movies=None,
                 tau_by_movie=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Replicate mapping")
        self._stamps = list(stamps or [])
        self._movies = list(movies or [s.movie for s in self._stamps])
        self._tau_by_movie = dict(tau_by_movie or {})
        self._result: dm.DayMapping | None = None

        layout = QVBoxLayout()

        self._banner = QLabel()
        self._banner.setWordWrap(True)
        self._banner.setStyleSheet(
            "background:#c53030; color:white; font-weight:bold; padding:6px;")
        layout.addWidget(self._banner)

        source_note = {
            "replicate_dir": "Source: the replicate folders (r1..rx). "
                             "Timestamps are only a cross-check. Confirm the folder→replicate "
                             "mapping below.",
            "movie_timestamp": "Source: movie timestamps (a proposal). Confirm the mapping below.",
            "manual": "Source: manual entry.",
        }.get(mapping.source, "")
        # When the folders are the source, SHOW the folder→day mapping explicitly so
        # the sign-off is on the authoritative structure, not just the movie table.
        if mapping.source == "replicate_dir" and mapping.groups:
            folders = "  ".join(f"r{g.day}→replicate {g.day} ({len(g.movies)} movies)"
                                for g in mapping.groups)
            source_note += "\n" + folders
        self._source_label = QLabel(source_note)
        self._source_label.setWordWrap(True)
        layout.addWidget(self._source_label)

        # ---- one-click rules: the lab's own convention, applied in one go ----
        # The ranges are otherwise retyped for every condition, which is where a
        # typo becomes a wrong replicate day that nothing downstream can catch.
        rules = QHBoxLayout()
        rules.addWidget(QLabel("Apply rule:"))
        self._rule_box = QComboBox()
        for key, (label, _fn) in dm.SERIES_RULES.items():
            self._rule_box.addItem(label, key)
        self._rule_box.addItem("Every N movies in series order = one replicate", "block")
        rules.addWidget(self._rule_box, 1)
        self._n_per_day = QSpinBox()
        self._n_per_day.setRange(1, 999)
        self._n_per_day.setValue(10)
        self._n_per_day.setPrefix("N = ")
        rules.addWidget(self._n_per_day)
        self._apply_rule_btn = QPushButton("Apply")
        self._apply_rule_btn.clicked.connect(self._apply_rule)
        rules.addWidget(self._apply_rule_btn)
        layout.addLayout(rules)

        # ---- two views of the same mapping: by day (ranges) and by movie ----
        self._tabs = QTabWidget()
        self._table = QTableWidget()
        self._table.setColumnCount(len(self.COLS))
        self._table.setHorizontalHeaderLabels(self.COLS)
        self._fill_table(rows_from_mapping(mapping, self._tau_by_movie))
        self._tabs.addTab(self._table, "By replicate (series ranges)")

        self._movie_table = QTableWidget()
        self._movie_table.setColumnCount(len(self.MOVIE_COLS))
        self._movie_table.setHorizontalHeaderLabels(self.MOVIE_COLS)
        self._fill_movie_table(movie_rows(mapping, self._movies, self._stamps,
                                          self._tau_by_movie))
        self._tabs.addTab(self._movie_table, "By movie (assign individually)")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self._tabs)

        self._status = QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._confirm_despite = QCheckBox("I confirm despite the warnings")
        self._confirm_despite.setVisible(False)
        self._confirm_despite.stateChanged.connect(self._refresh_confirm_enabled)
        layout.addWidget(self._confirm_despite)

        buttons = QHBoxLayout()
        self._validate_btn = QPushButton("Re-validate")
        self._validate_btn.clicked.connect(self.revalidate)
        self._confirm_btn = QPushButton("Confirm")
        self._confirm_btn.clicked.connect(self._on_confirm)
        buttons.addWidget(self._validate_btn)
        buttons.addStretch(1)
        buttons.addWidget(self._confirm_btn)
        layout.addLayout(buttons)
        self.setLayout(layout)

        # initial verdict (banner + status) from the proposal itself
        self._show_verdict(mapping.warnings, mapping.errors,
                           no_timestamps=mapping.timestamp_source in (None, "none"))

    # -- table <-> daymap (thin) ------------------------------------------- #

    def _fill_table(self, rows):
        self._table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, key in enumerate(("day", "spec", "date", "span", "n", "tau")):
                item = QTableWidgetItem(str(row[key]))
                if key != "spec":                       # only the range is editable
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(r, c, item)

    def _fill_movie_table(self, rows):
        self._movie_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, key in enumerate(("movie", "series", "day", "tau", "time")):
                item = QTableWidgetItem(str(row[key]))
                if key != "day":                        # only the day is editable
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self._movie_table.setItem(r, c, item)

    def _movie_assignments(self) -> dict:
        """``{movie: day}`` from the by-movie table. Blank/invalid cells are left
        OUT rather than defaulted, so an unassigned movie surfaces as the error it
        is instead of silently joining day 1."""
        out = {}
        for r in range(self._movie_table.rowCount()):
            movie_item = self._movie_table.item(r, 0)
            day_item = self._movie_table.item(r, 2)
            if movie_item is None or day_item is None:
                continue
            text = day_item.text().strip()
            if not text:
                continue
            try:
                out[movie_item.text()] = int(text)
            except ValueError:
                continue
        return out

    def _apply_rule(self):
        """Apply the selected series-number rule and refresh BOTH views."""
        rule = self._rule_box.currentData()
        mapping = dm.mapping_from_series_rule(
            self._movies, rule, n_per_day=self._n_per_day.value(),
            stamps=self._stamps)
        self._fill_table(rows_from_mapping(mapping, self._tau_by_movie))
        self._fill_movie_table(movie_rows(mapping, self._movies, self._stamps,
                                          self._tau_by_movie))
        self.revalidate()

    def _on_tab_changed(self, index):
        """Switching views carries the edits across, so the two never disagree."""
        try:
            mapping = self.current_mapping(from_tab=1 - index)
        except dm.DayMapError:
            return
        if not mapping.ok:
            return
        if index == 0:
            self._fill_table(rows_from_mapping(mapping, self._tau_by_movie))
        else:
            self._fill_movie_table(movie_rows(mapping, self._movies, self._stamps,
                                              self._tau_by_movie))

    def _table_specs(self) -> dict[int, str]:
        specs = {}
        for r in range(self._table.rowCount()):
            day_item = self._table.item(r, 0)
            spec_item = self._table.item(r, 1)
            if day_item is None or spec_item is None:
                continue
            specs[int(day_item.text())] = spec_item.text()
        return specs

    def current_mapping(self, from_tab: int | None = None) -> "dm.DayMapping":
        """The DayMapping for what is CURRENTLY shown (pure daymap call).

        The ACTIVE tab is the source of truth, because that is the one the user
        just edited. Both routes end in a daymap constructor; nothing is computed
        here.
        """
        tab = self._tabs.currentIndex() if from_tab is None else from_tab
        if tab == 1:
            return dm.mapping_from_assignments(self._movie_assignments(),
                                               stamps=self._stamps)
        return dm.mapping_from_ranges(self._table_specs(), self._movies)

    def revalidate(self) -> "dm.DayMapping":
        """Rebuild from the table and cross-check against the timestamps."""
        mapping = self.current_mapping()
        warnings, errors = ([], mapping.errors)
        if mapping.ok:
            warnings, extra = dm.validate_mapping(mapping.movie_to_day, self._stamps)
            errors = errors + extra
        # A day that MIXES frame intervals is fatal, not advisory: the day is the
        # analysis unit, its movies are pooled into one number, and every D scales
        # with 1/tau. So it joins the ERRORS, which disable Confirm outright,
        # unlike the timestamp warnings, this one cannot be confirmed despite.
        errors = errors + dm.tau_warnings(mapping.movie_to_day, self._tau_by_movie)
        mapping.warnings = list(warnings)
        mapping.errors = list(errors)
        self._pending = mapping
        self._show_verdict(warnings, errors,
                           no_timestamps=not any(s.timestamp for s in self._stamps))
        return mapping

    def _show_verdict(self, warnings, errors, *, no_timestamps):
        if no_timestamps:
            self._banner.setText(
                "No acquisition timestamps could be read, so the mapping cannot be "
                "checked against time. Enter it by hand and check it against your records.")
            self._banner.setVisible(True)
        else:
            self._banner.setVisible(False)
        lines = [f"ERROR: {e}" for e in errors] + [f"warning: {w}" for w in warnings]
        self._status.setText("\n".join(lines) if lines
                             else "Mapping is consistent with the timestamps.")
        self._confirm_despite.setVisible(bool(warnings) and not errors)
        if not warnings:
            self._confirm_despite.setChecked(False)
        self._has_errors = bool(errors)
        self._has_warnings = bool(warnings)
        self._refresh_confirm_enabled()

    def _refresh_confirm_enabled(self, *_):
        ok = not getattr(self, "_has_errors", False)
        if getattr(self, "_has_warnings", False):
            ok = ok and self._confirm_despite.isChecked()
        self._confirm_btn.setEnabled(ok)

    def _on_confirm(self):
        self._result = self.revalidate()
        if not self._result.errors and (
                not self._result.warnings or self._confirm_despite.isChecked()):
            self.accept()

    def result_mapping(self) -> "dm.DayMapping | None":
        """The confirmed mapping, or None if the dialog was cancelled."""
        return self._result
