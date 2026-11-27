"""Item 5: every legacy MATLAB file gets a row and a category, enforced by a test.

Why this is code and not a document. ``docs/legacy_conversion_status.md`` is the readable
account, and it has already drifted twice: it once claimed 159 files when a
non-recursing glob had missed 35, and its named-file count has moved as the document
grew. A prose inventory cannot tell you what it has left out. This module can, because
``tests/test_legacy_inventory.py`` walks the repository and asserts that EVERY ``.m``
file resolves to exactly one category. A file added, renamed or removed breaks the test
rather than quietly falling out of the count.

The categories are not a ranking and are never summed into a percentage complete. A tool
that is 60 per cent converted is a meaningless statement when a third of the files open
dialogs. Each category answers a different question:

``converted``
    a Python equivalent exists and is tested.
``converted_defect_fixed``
    the method is reproduced; a named defect in the original is not. Every one of these
    has a row in ``legacy_deviations.csv`` giving the file and line.
``legacy_mode``
    the old output is reproduced on purpose, in ``legacy_out/`` only, so a person can
    recognise their own figure. Not recommended for a claim.
``retired``
    deliberately not converted, with a reason recorded. An omission with an argument is
    not the same as an omission.
``planned``
    in scope, not built yet, and named here so it cannot be mistaken for retired.
``out_of_scope``
    GUI, dialogs, movie export, file browsing, parameter forms, the object model. The
    scope was never the whole tool.
``superseded_copy``
    a file the legacy itself no longer calls, usually an ``_old`` variant kept beside its
    replacement. Converting one would reproduce something the original had abandoned.
``third_party``
    vendored code with its own authorship and licence. Not ours to convert, and in most
    cases replaced by a Python library rather than by a translation.

The directory rules run last. An explicit per-file entry always wins, so a genuinely
interesting file inside a vendored directory can still be given its own row.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .guard import legacy_path

__all__ = ["CATEGORIES", "FILE_CATEGORY", "DIRECTORY_CATEGORY", "InventoryError",
           "categorise", "matlab_files", "build_inventory", "write_inventory"]


class InventoryError(RuntimeError):
    """Raised when a legacy file cannot be given a category."""


CATEGORIES = (
    "converted", "converted_defect_fixed", "legacy_mode", "retired", "planned",
    "out_of_scope", "superseded_copy", "third_party",
)

#: Whole directories, matched on any path component. Applied only when no per-file entry
#: exists. Every one of these is vendored code with its own authorship.
DIRECTORY_CATEGORY = {
    "bfmatlab": ("third_party", "Bio-Formats, OME. Replaced by tifffile and the native "
                 "MCOS decoder rather than translated"),
    "FACS": ("third_party", "flow-cytometry reader and a density scatter helper, "
             "unrelated to single-molecule tracking"),
    "SplashScreen": ("third_party", "a splash window"),
    "acf": ("third_party", "autocorrelation helper; scipy and numpy cover it"),
    "copyaxes": ("third_party", "a figure-handle utility"),
    "emprand": ("third_party", "sampling from an empirical distribution; "
                "numpy.random covers it"),
    "explore": ("third_party", "opens a path in the system file browser"),
    "tiffread2": ("third_party", "TIFF reader; tifffile covers it"),
    "ftest": ("third_party", "an F-test between nested models, by J. Conder. Reproduced "
              "with scipy in legacy.sla, which is why the original is not converted"),
    "kolm": ("retired", "Kolmogorov helpers for tests on POOLED steps. Never ported: a "
             "pooled-step test rejects at 58.3 per cent, 95 per cent [52.7, 63.8], "
             "against a nominal 5"),
    "histwc": ("third_party", "weighted histogram counts; numpy.histogram with weights "
               "covers it"),
    "bias": ("converted_defect_fixed", "smtracker.directionality. orientation_radians.m "
             "takes the absolute value of BOTH components, so every step lands in the "
             "first quadrant and what survives is AXIAL ANISOTROPY, not direction. Both "
             "are built and named apart (`orientation` signed, `axial_angle` the legacy "
             "fold), and `axial_null` supplies the chance rate a rod-shaped cell "
             "produces with no biology in it"),
    "private": ("third_party", "a Bio-Formats private helper"),
    "config": ("out_of_scope", "axis label strings for the GUI"),
}

#: Per-file entries. These WIN over the directory rules.
FILE_CATEGORY: dict[str, tuple[str, str]] = {}


def _add(category: str, reason: str, *names: str) -> None:
    """Assign a category, refusing to overwrite one silently.

    A second ``_add`` for the same file used to win by dict assignment, so a file could
    be listed under confinement and then quietly recategorised as a superseded copy two
    hundred lines later. Both readings looked right in isolation, which is the failure
    mode: the file is real either way and nothing reports the conflict.
    """
    for name in names:
        if name in FILE_CATEGORY and FILE_CATEGORY[name] != (category, reason):
            raise InventoryError(
                f"{name} is already categorised as {FILE_CATEGORY[name][0]!r} and is "
                f"being reassigned to {category!r}. Pick one and say why; a silent "
                f"overwrite makes the inventory depend on line order.")
        FILE_CATEGORY[name] = (category, reason)


# ------------------------------------------------------------------- reading the data
_add("converted", "smtracker.ingest, smtracker.readers",
     "importUTrack.m", "impSMTrack.m", "exp2SMTrack.m", "importTrackmate.m",
     "importTrackMateTracks.m", "nantrackConverter.m")
_add("converted", "smtracker.ingest and smtracker.cellmesh", "dataImport.m")
_add("converted", "smtracker.grouping discovery; the folders above a movie carry no "
     "meaning, so only the TrackingPackage tail is matched",
     "validateFolderStructure.m", "ListSubfolders.m", "mydir.m", "defDir.m")

# ------------------------------------------------------------------ units and geometry
_add("converted", "smtracker.canonical and smtracker.cellmesh",
     "cellTracks.m", "inCellContour.m")
_add("converted", "smtracker.cellcoords, including the pole extrapolation",
     "project2Mesh.m", "projCT.m")
_add("converted", "smtracker.spatial cell geometry", "getTracksByCellsize.m")
_add("converted", "smtracker.qc and smtracker.cellmesh.assignment_funnel",
     "getCellStats.m", "getTracksperCell.m", "corTracks.m")

# ------------------------------------------------------------------------- the SQD fit
_add("converted_defect_fixed",
     "smtracker.sqd, with the legacy seeds kept runnable; see legacy_deviations.csv",
     "sqdFit_.m", "sqdFit_Simultaneous.m", "calcCDF.m",
     "singlePopulation.m", "twoPopulation.m", "threePopulation.m")
_add("converted", "smtracker.sqd LEGACY_STARTS", "initializeVars.m")
_add("converted_defect_fixed", "smtracker.jd; six defects fixed, see the deviations",
     "fit_ECDF.m", "getDispl_gui.m")

# ---------------------------------------------------------------------------- appD/MSD
_add("converted", "smtracker.msd, fitting only the reliable early lags",
     "calcMSDTracks_.m", "getMeanMSD_gui.m", "getFitMSD_gui.m", "fitMSDTracks.m",
     "linWfit.m", "anomalousWfit.m", "getMSD.m", "delMSD.m")
_add("converted", "smtracker.appd, per-track apparent D with no offset removed",
     "calcDapp_mle.m", "calcAppD_gamma.m")
_add("converted_defect_fixed",
     "smtracker.appd_gamma. The Gamma mixture on per-track apparent D, shape = n_steps "
     "and mean = D, with a closed-form EM M-step. Defects fixed rather than ported: the "
     "legacy BIC multiplies densities with prod(), which UNDERFLOWS to exactly 0.0 in all "
     "24 fits on this data (mean log density is negative everywhere), so the sum of log "
     "likelihoods is used; the single hand-placed seed becomes a multistart with "
     "starts='legacy' still runnable; and n_pop = 1 no longer raises",
     "gammafit_one.m", "gammafit_two.m", "gammafit_three.m", "gammaCDF.m",
     "AppDFit_.m", "AppDFit_simul.m", "fit_appD_simultaneously.m",
     "update_appdParams.m", "calculatesBIC.m", "getAppD_gui.m",
     "sortParams_gamma.m", "sortp.m", "findStartingPoint.m")

# The weighted-histogram helpers sit UNDER bias/ but are vendored, so a per-file entry
# keeps them third-party where the directory rule would now call them converted.
_add("third_party", "weighted histogram counts; numpy.histogram with weights covers it",
     "histwc.m", "histwcv.m")
_add("retired", "the Gaussian appD variant places mass below zero for a positive "
     "quantity, and carries the complex-valued seed that the gamma variant does not",
     "calcAppD_gau.m", "gaussianCDF.m", "doubleGaussian.m", "doubleGaussianCDF.m",
     "doubleGau.m", "singleGau.m", "tripleGau.m")
_add("retired", "circular selection: it keeps tracks by how well they fit the model "
     "about to be fitted, dropping 526 tracks to 378, a 28 per cent loss",
     "appDFilter.m")
_add("legacy_mode",
     "smtracker.legacy.sla reproduces the F-test that classifies confined segments and "
     "prints the p value in the title. Its USE inside appDFilter, to select tracks before "
     "fitting them, is what is retired; the classification itself is reproduced",
     "ftestAnomalous.m")

# --------------------------------------------------------------------- confinement/SLA
_add("converted", "smtracker.confinement, with the chance rate the legacy never had",
     "dwellCount.m", "dwellCount2.m", "tauCounter2.m", "tauCounter3.m",
     "getConfinedSegments.m", "confinedMSDMotion.m", "isgreater.m", "nan2one.m")
_add("converted_defect_fixed", "smtracker.legacy.sla; the error bar divided by the lag "
     "count, the F-test on correlated lags, and the unlabelled exponent",
     "diffusionClassification.m")
_add("legacy_mode", "smtracker.legacy.sla, reproduced for recognition only",
     "fillSLATable_gui.m", "fillStatsTable_gui.m", "fillMSDtable.m")

# ----------------------------------------------------------------------- maps and plots
_add("legacy_mode", "smtracker.legacy.speedmap; the N label, the contour trim counts "
     "and the per-panel colour scale", "plotSpeedmap.m")
_add("legacy_mode", "smtracker.legacy.sda; the midpoint convention, the up-scale guard "
     "and the maximum fold",
     "plotHeatmap.m", "getDynamics.m", "getNormalizeTracks.m", "normalizeTracks.m",
     "mirrorPoints.m", "bin3d.m", "getAnomalousTracks.m")
_add("converted", "smtracker.maps, smtracker.render and smtracker.statemaps",
     "AppDPlot.m", "plotMSDTracks.m", "plotFitMSD_gui.m")
_add("converted", "smtracker.clustering, with the free-Brownian null it never had",
     "clusteringMSD.m")


# ------------------------------------------------------------------------- statistics
_add("retired", "a significance table built on POOLED steps. The pooled-step test "
     "rejects at 58.3 per cent, 95 per cent [52.7, 63.8], against a nominal 5; the "
     "per-cell control rejects at 5.7 per cent, [3.6, 8.9]",
     "significanceTable.m")
_add("converted", "smtracker.uncertainty and smtracker.compare, over cells or replicate "
     "days rather than pooled steps", "acf.m", "acf_bar.m")

# --------------------------------------------------------------- superseded by the file
_add("superseded_copy", "an older copy kept beside its replacement; the legacy itself no "
     "longer calls it",
     "calcMSD_gui_old.m", "getTracksByCellsize_old.m", "ginput_old.m", "tauCounter.m")

# ------------------------------------------------------------------------ out of scope
_add("out_of_scope", "GUI plumbing: dialogs, consoles, figure furniture, parameter forms",
     "about.m", "errorDialog.m", "printConsole.m", "printableFig.m", "expandAxes.m",
     "saveFigureAsSvg.m", "openpdf.m", "changeVersion.m", "checkToolbox.m",
     "checkUtrack.m", "saveParams.m", "setParams.m", "axisNames.m", "newLine.m",
     "giveFormatTab.m", "splash.m", "SplashScreen.m")
_add("out_of_scope", "movie and montage export; smtracker.viewer covers viewing from the "
     "canonical table instead",
     "getMontage.m", "getMovie.m", "moviePlayer.m", "moviePlotter.m", "readImage.m",
     "mountVideo.m", "getBox.m", "redrawDemo.m", "SMTracker_explorer.m")
_add("out_of_scope", "the MATLAB object model itself; v3 is a table plus pure functions",
     "SMT.m", "Blueprint_looping.m", "deleteTracks.m")
_add("out_of_scope", "small array utilities with a numpy one-liner equivalent",
     "cleanData.m", "getRidnan.m", "isEmptyCell.m", "reduce.m", "dist_.m", "dst.m",
     "AdvancedColormap.m", "is_octave.m")


# ---------------------------------------------------- the remainder, read one by one
_add("converted", "smtracker.confinement dwell events and smtracker.legacy.sla",
     "dwell_tracks.m", "dwellTime.m", "confTracks.m", "confTracksperCell.m")
_add("legacy_mode", "the free-to-confined transition count, reproduced in "
     "smtracker.legacy.sla beside the chance rate the legacy never printed",
     "nTrans.m")
_add("converted", "smtracker.msd and smtracker.observables",
     "calcMSD_gui.m", "cellMSD.m", "steps.m")
_add("converted", "smtracker.clustering on per-track MSD vectors",
     "calcMSD_clustering.m")
_add("converted", "smtracker.aggregate and smtracker.sqd", "sqdStruct.m", "getTracks.m")
_add("converted_defect_fixed",
     "smtracker.jd fits the per-unit fractions with shared D; the legacy simultaneous "
     "path carries the same n_pop = 1 crash as fit_ECDF", "fit_ECDF_simultaneously.m")
_add("retired", "the Gaussian appD variant, whose parameter sort belongs to a model that "
     "places mass below zero for a positive quantity",
     "sortParams_gau.m", "singleGaussian.m", "tripleGaussian.m")
_add("converted", "smtracker.msd per-track R squared", "r2.m")
_add("converted_defect_fixed",
     "smtracker.jd.remove_outliers; the legacy rule is one-sided, joined by AND across "
     "axes and computed on an uncleaned scale", "remOutlier.m")
_add("planned",
     "BLOCKED ON DATA, not on effort. spotIntensity.m integrates a 6-px inner against a "
     "12-px outer ROI FRAME BY FRAME out of the raw 1001-frame movie, with an "
     "instrumental background from frames 1-20 and a per-pixel illumination ratio. This "
     "project ships phase/ and sum/ only, so there is nothing to integrate. u-track's "
     "fitted Gaussian amplitude IS available and smtracker.intensity reads it, but it is "
     "a PEAK HEIGHT with no background subtracted and no illumination correction, so it "
     "is never presented as the legacy quantity",
     "spotIntensity.m", "trackIntensity.m", "bkgCell.m", "bkgSub.m", "firstFrames.m",
     "BleachingStep.m", "cropIntensity.m", "integratedIntensity.m", "bkgInstrumental.m",
     "backgroundCorrection.m", "autofluorescenceDecay.m", "unevenIllu.m",
     "unevenIllumination.m", "Roi.m", "bwExtend.m")
_add("out_of_scope", "per-cell QC plots that smtracker.qc reports from track_summary",
     "lengthTracks.m", "spotsPerFrame.m")
_add("out_of_scope", "movie and montage export; smtracker.viewer covers viewing from the "
     "canonical table instead",
     "allTracksonCell.m", "prepareToplotAllTracks.m", "track_movement.m",
     "trackDistance.m", "rotateTracks.m")
_add("out_of_scope", "the GUI itself, about 283 KB of it. The scope was never the whole "
     "tool; smtracker.app and smtracker.widgets are a thin Qt shell over pure functions",
     "SMTracker.m")
_add("out_of_scope", "not legacy code: the driver scripts/make_golden_appd.py runs in MATLAB R2016a (23 Nov 2026) "
     "to call SMTracker 2's own getAppD_gui.m and calcAppD_gamma.m, unchanged, on the reference's tracks for "
     "reference/golden_appd_gamma.json. It is the oracle's harness, so nothing in it is ported",
     "golden_appd_v2.m")


def matlab_files(root: str | Path = ".") -> list[Path]:
    """Every ``.m`` file in the repository, excluding version-control internals."""
    root = Path(root)
    return sorted(p for p in root.rglob("*.m") if ".git" not in p.parts)


def categorise(path: str | Path) -> tuple[str, str]:
    """The category and its reason for one legacy file.

    An explicit per-file entry wins over a directory rule. Raises when neither exists,
    because a file with no category is the state this module was written to make visible.
    """
    path = Path(path)
    if path.name in FILE_CATEGORY:
        return FILE_CATEGORY[path.name]
    for part in path.parts:
        if part in DIRECTORY_CATEGORY:
            return DIRECTORY_CATEGORY[part]
    raise InventoryError(
        f"{path.name} has no category. Every legacy file gets one, including a reason to "
        f"leave it alone: add it to FILE_CATEGORY, or add its directory to "
        f"DIRECTORY_CATEGORY. Known categories: {list(CATEGORIES)}")


def build_inventory(root: str | Path = ".") -> pd.DataFrame:
    """One row per legacy file: path, name, category, reason."""
    root = Path(root)
    rows = []
    for p in matlab_files(root):
        category, reason = categorise(p)
        p = p.relative_to(root) if p.is_absolute() else p
        if category not in CATEGORIES:
            raise InventoryError(f"{p.name}: unknown category {category!r}")
        rows.append({"file": p.as_posix(), "name": p.name, "directory": p.parent.as_posix(),
                     "category": category, "reason": reason})
    if not rows:
        raise InventoryError(f"no MATLAB files under {root}")
    return pd.DataFrame(rows)


def write_inventory(root: str | Path = ".", out_root=None) -> pd.DataFrame:
    """Deposit ``legacy_out/tables/legacy_file_inventory.csv``.

    The per-category counts are reported and are NEVER summed into a fraction converted.
    A tool that is 60 per cent converted is a meaningless statement when a third of its
    files open dialogs.
    """
    table = build_inventory(root)
    table.to_csv(legacy_path("tables", "legacy_file_inventory.csv", root=out_root),
                 index=False)
    return table
