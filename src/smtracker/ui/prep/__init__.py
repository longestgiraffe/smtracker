"""The preparation stage in the workspace's style: movies and crop, calibration and save.

Built beside the classic `tracking_gui` window, which keeps detection, linking and the
tracking run until those pages are ported. Every rule and every word comes from the
headless `smtracker.prep_model` / `smtracker.prep_view`; nothing here computes (thin Qt,
`tests/test_ui_thin_qt.py`).
"""
