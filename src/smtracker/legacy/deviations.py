"""The deviations table: every place legacy mode departs from the legacy code.

Why this exists. The PI asks why a legacy number differs from the old tool's number.
This table is the answer. One row per fix, with the file and line in the MATLAB source,
what the old code did, what this code does, and why.

A design choice is reproduced. A defect is fixed. A defect is code that does not do what
its own comment, label or variable name says. Pooled steps as the unit, the fixed seeds,
the nlparci intervals, the absence of a bootstrap and the auto colour scale are DESIGN.
They are reproduced and they are not in this table.

Every row was confirmed in the source before it was written. Four statuses, never SUMMED
AND never subtracted:

* ``defect_fixed`` is code that did not do what its own comment, label or variable name
  said.
* ``instruction_corrected_not_a_defect`` is an INSTRUCTION that was wrong for this
  repository, which is a different thing from a defect that was looked for and not found.
* ``checked_not_a_defect`` is code that was read because it looked wrong and turned out
  to be right. It exists so the next reader does not repeat the check, or worse, act on
  it: two of these describe source that invites a fix which would introduce the very
  defect it appears to remove.
* ``defect_reproduced`` is a v2 defect that legacy mode KEEPS, because it shapes the output
  a lab member is meant to recognise. Recorded so a reproduced defect is never read as an
  endorsed one; the modern path does not carry it.
"""
from __future__ import annotations

import pandas as pd

from .guard import legacy_path

__all__ = ["DEVIATIONS", "PROMPT_CORRECTIONS", "CHECKED_NOT_DEFECTS",
           "DEFECTS_REPRODUCED", "LEGACY_N_POP", "STATUSES", "write_deviations"]

#: Every status a row can carry, in the order they are written.
STATUSES = ("defect_fixed", "instruction_corrected_not_a_defect", "checked_not_a_defect",
            "defect_reproduced")

#: Confirmed defects, each read in the MATLAB source before it was fixed here.
DEVIATIONS = [
    {
        "file": "SMT_GUI/@SMT/sqdFit_Simultaneous.m",
        "line": 80,
        "legacy_behaviour": (
            "sd = [sd sd(end)] runs INSIDE the per-condition loop while j is advanced "
            "per condition (line 95, j = j + n_pop - 1). sd therefore grows by one "
            "element on every iteration, but it is never re-sliced per condition: "
            "sd_diff = sd(1:n_pop) always reads the first n_pop entries and "
            "sd_p = sd(1+n_pop:end) reads an ever-longer tail. Condition 2 receives a "
            "longer fraction error vector than condition 1."),
        "new_behaviour": (
            "the interval for each condition is sliced with the same per-condition "
            "offset the parameters use, so a fraction error belongs to the condition it "
            "is printed against. The final fraction is DERIVED (1 minus the others) and "
            "was never fitted, so its interval is reported as not available rather than "
            "copied from the previous fraction."),
        "reason": (
            "the variable is named sd_p and is printed as the fraction error of a named "
            "condition, so reading it from another condition's offset contradicts the "
            "name. Copying the last fitted interval onto a derived parameter invents an "
            "interval for a quantity that was not estimated."),
    },
    {
        "file": "SMT_GUI/@SMT/sqdFit_Simultaneous.m",
        "line": 112,
        "legacy_behaviour": (
            "the n_pop = 1 path sets p0 = .05, a scalar of length 1 (line 43), then "
            "chisqr calls singlePopulation([p(1) p(2) p(2+i)], ...). p(2) and p(2+i) are "
            "out of bounds, so MATLAB raises 'Index exceeds the number of array "
            "elements' and no single-population fit can be produced."),
        "new_behaviour": (
            "the single-population model takes one parameter, D. It has no fraction and "
            "no second D, so nothing is indexed past the end. The fit runs and returns "
            "D with its interval and a fraction of 100 percent by construction."),
        "reason": (
            "singlePopulation.m itself uses only x(1). The caller passes three "
            "arguments, so the call contradicts the function it calls."),
    },
    {
        "file": "SMT_GUI/Help_fcn/plotHeatmap.m",
        "line": 17,
        "legacy_behaviour": (
            "the location map bins into a 20-node grid and then calls "
            "interp2(v, 5, 'cubic'). Five refinements take 20 nodes to 609, so every "
            "displayed cell is one thirty-second of a data bin per axis and there are "
            "1024 displayed cells per bin. The short-axis data bin is ALREADY below one "
            "camera pixel before that: the grid spans 1.2 cell widths in 19 intervals, "
            "so for a cell about 0.9 um wide the bin is about 57 nm against a 106 nm "
            "pixel. Cubic convolution also overshoots at an edge, so the interpolated "
            "surface takes negative values where a count cannot, and the colour floor "
            "is pinned at zero (line 67) so the overshoot is clipped and reads as an "
            "empty bin."),
        "new_behaviour": (
            "the counts are drawn at the bin size they were measured at and nothing is "
            "interpolated. assert_bin_resolvable refuses a bin below one camera pixel "
            "and reports the shortfall; legacy_bin_report gives the verdict per AXIS, "
            "since the long axis clears a pixel before interpolation and the short axis "
            "does not."),
        "reason": (
            "a bin finer than the detector samples cannot carry information the "
            "detector did not record, so the added cells are interpolation drawn at the "
            "same weight as data, and a negative density contradicts the quantity being "
            "displayed."),
    },
    {
        "file": "SMT_GUI/Help_fcn/plotHeatmap.m",
        "line": 24,
        "legacy_behaviour": (
            "the location map folds by stacking the image with its three flips and "
            "taking max(vqm,[],3). The dynamics path in the same tab folds by "
            "replicating the reflected DATA fourfold (line 84 with mirrorPoints.m) and "
            "binning them together, which pools. One control labelled mirror therefore "
            "performs two different operations depending on which data source is "
            "selected, and the figure does not record which."),
        "new_behaviour": (
            "fold_grid takes the operation as an argument and defaults to pooling. "
            "fold_inflation measures, on the grid being drawn, how far the maximum fold "
            "sits above the pooling fold, split on the folded mean so the number "
            "belongs to the fold orbit rather than to whichever bin was read."),
        "reason": (
            "the maximum of four estimates of one quantity is biased upward, and biased "
            "most where counts are lowest, so the maximum fold raises exactly the "
            "sparse bins a reader is least entitled to trust. Pooling is unbiased."),
    },
    {
        "file": "SMT_GUI/Help_fcn/plotHeatmap.m",
        "line": 65,
        "legacy_behaviour": (
            "params.heatmapCaxis{typeMotion} is raised when a panel exceeds it and is "
            "never lowered, and panels already drawn are not redrawn. The colour scale "
            "a panel is displayed on therefore depends on the order the panels were "
            "clicked in that session, and two panels in one figure can carry different "
            "scales."),
        "new_behaviour": (
            "the colour limits are computed over the whole panel set before any panel "
            "is drawn, and the applied limits are printed on every panel."),
        "reason": (
            "a colour scale that depends on click order is not a property of the data, "
            "and two panels on different scales cannot be compared by eye, which is the "
            "only thing a map is read for."),
    },
]

#: Item 4, the SLA tab.
DEVIATIONS += [
    {
        "file": "SMT_GUI/Help_fcn/ftestAnomalous.m",
        "line": 18,
        "legacy_behaviour": (
            "numel(x) is passed as the sample size N, where x is the [dT M STD N] matrix "
            "returned by getMSD, so N is FOUR TIMES THE NUMBER OF LAGS rather than a "
            "sample size. N is then used only at diffusionClassification.m:50, "
            "errorbar(ax,t,msd,weights/sqrt(N)), under a legend reading 'MSD +- sem'. The "
            "weights are the STD column, so the intended quantity is STD/sqrt(n) with n "
            "the per-lag count, which is COLUMN 4 OF THE VERY MATRIX numel was called "
            "on. With thousands of tracks per lag against a few tens of lags the drawn "
            "bar is far too LARGE, so the curve looks less certain than it is."),
        "new_behaviour": (
            "the per-lag count is read from column 4 and both divisors are reported, "
            "never their difference, with the factor the drawn bar is wrong by."),
        "reason": (
            "the legend names the quantity as a standard error, and the number of lags "
            "is not the sample size of any of them."),
    },
    {
        "file": "SMT_GUI/Help_fcn/diffusionClassification.m",
        "line": 11,
        "legacy_behaviour": (
            "n = size(t,1), the number of MSD lags, is passed to ftest as the number of "
            "data. Successive lags of one MSD curve reuse the same displacements, so "
            "they are not independent and the F-test is computed on correlated points."),
        "new_behaviour": (
            "the lag count remains the default, because that is the reproduced design, "
            "and an honest independent count may be supplied so the gap between the two "
            "is MEASURED on the curve being classified rather than argued."),
        "reason": (
            "the same class as the nlparci intervals on correlated CDF points, which this "
            "project retired and reproduces only inside legacy mode."),
    },
    {
        "file": "SMT_GUI/Help_fcn/diffusionClassification.m",
        "line": 29,
        "legacy_behaviour": (
            "the branch on the fitted exponent tests > 1 and < 1 with no else, so an "
            "exponent of exactly 1 below the p threshold leaves typeMotion undefined and "
            "typeMotionStr{typeMotion} raises an index error."),
        "new_behaviour": (
            "the case is named: there is no honest label for a curve whose power law was "
            "accepted at an exponent of exactly 1, so it raises with that sentence "
            "instead of failing on a lookup."),
        "reason": (
            "vanishingly rare in floating point and still a hole, and an index error "
            "names neither the cause nor the fix."),
    },
    {
        "file": "SMT_GUI/@SMT/sqdFit_Simultaneous.m",
        "line": 12,
        "legacy_behaviour": (
            "with the Simultaneous box ticked (its default, SMTracker.m:800-804) and n_pop 2 or 3, "
            "sqdFit_Simultaneous runs after the 1-4-together fit and overwrites its D and fractions "
            "(SMTracker.m:5242-5247). It reads i_tlag = currentmTlag, which is 5 for the 1-4-together menu "
            "item, so it fits displacements FIVE frames apart at 5 x tau (lines 31, 66): a lag no menu offers."),
        "new_behaviour": (
            "lags 1-4 together is one fit to the four lags, D and fractions shared, one ECDF per lag stacked "
            "(sqdFit_.m:66-70), under every Shared D setting; nothing refits it at another lag."),
        "reason": (
            "the menu item says 1-4 together and the joint fit it names is computed first; overwriting it with "
            "a lag-5 fit reports a fit the user did not ask for under the label of the one they did."),
    },
    {
        "file": "SMT_GUI/@SMT/dwellTime.m",
        "line": 9,
        "legacy_behaviour": (
            "events = 0, then events = cat(1, events, a) for every row of SMMTrack's dwell-time file: each "
            "condition's event list starts with a zero-length event that no file holds. The histogram counts it "
            "(line 39), and the SEM std(values(values > 0)) / sqrt(length(values)) (line 54) divides by one event "
            "too many while the mean beside it leaves the zero out."),
        "new_behaviour": (
            "the events are the file's own, each row's interval repeated as often as it counts "
            "(smmtrack_view.dwell_payload); n, mean, median and SD over those events. v2's two-sample KS p is not "
            "offered: the events carry no cell or day, so a test could only treat every event as independent."),
        "reason": (
            "a zero-length dwell is not an event SMMTrack reported; adding one per condition moves the first bin "
            "and the SEM by an amount that depends on how few events the condition has."),
    },
]

#: READ, FOUND CLEAN, RECORDED. A check that came back negative is worth as much as one
#: that did not, and the next reader will otherwise make it again. These are NOT counted
#: with the defects and NOT counted with the corrected instructions.
CHECKED_NOT_DEFECTS = [
    {
        "file": "SMT_GUI/Help_fcn/confinedMSDMotion.m",
        "line": 15,
        "legacy_behaviour": (
            "x(2:minLength,:) is sliced with minLength = minSteps + 1, and tauCounter3.m:"
            "58 keeps a segment when tau >= minSteps where tau = k - 1 counts POINTS, so "
            "the shortest kept segment has exactly minSteps + 1 points. That reads as a "
            "slice one row past the end of every minimum-length segment."),
        "new_behaviour": (
            "reproduced unchanged. calcMSDTracks_.m:20 builds delays = (0:size(dr2,1)), "
            "so the matrix carries lag 0 at row 1 and the slice lands exactly on the last "
            "row. The arithmetic is correct."),
        "reason": (
            "a claim that was made and then refuted by reading one more file. Recorded so "
            "the next reader does not 'fix' a correct slice, which would drop the last "
            "lag from every classification."),
    },
    {
        "file": "SMT_GUI/Help_fcn/dwellCount.m",
        "line": 19,
        "legacy_behaviour": (
            "Lcut(i_gap-1:i_gap+1,:) would raise for a gap in the first or last row, and "
            "handles only the first gap when several exist, since MATLAB's colon takes "
            "the first element of each vector."),
        "new_behaviour": (
            "reproduced as written. u-track runs here with timeWindow = 1 and this "
            "project has measured gap loss at exactly zero in every group and band, so "
            "the gap index is empty and the line is a no-op on this data."),
        "reason": (
            "a reproduced no-op is NOT a fixed defect. Presenting it as one would claim a "
            "change that no figure on this data can show. It becomes reachable the moment "
            "gap closing is switched on, and is recorded for that reason."),
    },
    {
        "file": "SMT_GUI/Help_fcn/bin3d.m",
        "line": 24,
        "legacy_behaviour": (
            "bin3d builds Nx+1 by Ny+1 bins whose final row and column are degenerate, "
            "holding only the single maximum point, and pcolor then renders Nx by Ny "
            "faces from that matrix, discarding exactly those. The face colour is taken "
            "from the bin at its lower-left corner, which matches the binning above "
            "it."),
        "new_behaviour": (
            "reproduced as an Nx by Ny grid with no degenerate rim. The drawn result is "
            "the same."),
        "reason": (
            "two behaviours that look wrong separately agree with each other, so "
            "nothing wrong is drawn. Recorded because the degenerate rim is visible in "
            "the source and invites a fix that would introduce the defect it appears to "
            "remove."),
    },
    {
        "file": "SMT_GUI/Help_fcn/normalizeTracks.m",
        "line": 12,
        "legacy_behaviour": (
            "the location map divides by size(nTracks,1). normalizeTracks does NOT "
            "append a padding row per track, so that divisor is a genuine localisation "
            "count. The identical expression IS wrong at plotSpeedmap.m:182, where line "
            "100 pads first and the printed number is localisations plus tracks."),
        "new_behaviour": (
            "the localisation count and the track count are both reported and both "
            "named, on this map as on the speed map."),
        "reason": (
            "same expression, different verdict, because the two callers build their "
            "input differently. Recorded so the speed map's defect is not assumed to "
            "apply here by analogy."),
    },
    {
        "file": "SMT_GUI/Help_fcn/bin3d.m",
        "line": 18,
        "legacy_behaviour": (
            "a variable named sem is assigned nanstd with the division by numel "
            "commented out, so it holds a standard deviation. Even uncommented, "
            "dividing by n rather than by the square root of n would not give a "
            "standard error either."),
        "new_behaviour": (
            "not reproduced. Nothing in plotHeatmap reads it, so no legacy figure is "
            "affected and there is nothing to reproduce."),
        "reason": (
            "a name that contradicts its contents, in dead code. Recorded rather than "
            "fixed, because fixing unreached code would suggest a figure changed."),
    },
]

#: v2 DEFECTS THAT LEGACY MODE REPRODUCES ON PURPOSE (checked in the source 14 Oct 2026).
#: Legacy mode exists so a lab member recognises v2's output, so a defect that SHAPES that
#: output is kept, not fixed, and is written down here so a reproduced defect is never
#: read as an endorsed one. Status ``defect_reproduced``, counted on its own.
DEFECTS_REPRODUCED = [
    {
        "file": "SMT_GUI/Help_fcn/ftestAnomalous.m",
        "line": 17,
        "legacy_behaviour": (
            "w = x(:,3) takes the STD column of the getMSD matrix and hands it to linWfit "
            "and anomalousWfit as the MATLAB fit 'Weights', which multiply each squared "
            "residual. A lag's weight therefore GROWS with its scatter, so the noisiest "
            "lags count most in both fits and in the F-test that compares them. The "
            "per-track path in the same tab uses 1./std (fitMSDTracks.m:68)."),
        "new_behaviour": (
            "reproduced: legacy.sla.classify_motion weights by the standard deviation by "
            "default (weights='v2'), and weights='inverse_variance' gives 1/std^2 so the "
            "effect can be measured. On the seeded check (8 lags, 10 per cent scatter, 40 "
            "draws) a true exponent of 0.6 is called Brownian in 30 of 40 draws with v2's "
            "weights and 29 of 40 with 1/std^2."),
        "reason": (
            "a weight that rises with the scatter contradicts what a fit weight is for. "
            "Reproduced because it shapes the verdict v2 printed, and legacy mode shows "
            "v2's verdict."),
    },
    {
        "file": "SMT_GUI/Help_fcn/anomalousWfit.m",
        "line": 21,
        "legacy_behaviour": (
            "startingPoint = [exp(y) alphafit D_alpha] is handed to a fittype whose "
            "coefficients are declared in the order {'b','K','alpha'} (lines 25-27), so "
            "the offset b starts at the prefactor exp(y), the prefactor K at the exponent "
            "and the exponent at D_alpha. Bounds 'Upper',[1,Inf,2] (line 30): b <= 1, "
            "alpha <= 2, no lower bounds. This reading assumes coeffnames keeps the "
            "declared order; it was not run in MATLAB."),
        "new_behaviour": (
            "reproduced: legacy.sla.fit_anomalous_weighted seeds in v2's order with v2's "
            "bounds at MATLAB fit's default tolerances, and v2_start=False seeds "
            "(0, exp(y), alpha). On the seeded known-answer curves the converged fit does "
            "not depend on the order; on real curves its effect is UNVERIFIED."),
        "reason": (
            "the start values do not match the coefficients they are assigned to, which "
            "contradicts the variable names on line 19 to 21. A local optimum reached "
            "from a wrong start is still what v2 printed."),
    },
    {
        "file": "SMT_GUI/@SMT/getFitMSD_gui.m",
        "line": 11,
        "legacy_behaviour": (
            "the ensemble fit asks for rows 2:limit with limit = 'Linear Fit' + 1 inside a "
            "try, and the catch fits rows 2:end. In v2's default 'specific tau' mode the "
            "curve has min_length rows (calcMSD_gui.m:102-104) and validate() clamps "
            "'Linear Fit' to at most min_length (SMTracker.m:2410), so at the defaults "
            "(9 and 9) the try ALWAYS fails and the fit is lags 1-8, not the 9 the box "
            "shows, with no message."),
        "new_behaviour": (
            "reproduced: msd.legacy_ensemble_window gives lags 1-8 and fell_back = True "
            "at the defaults, and the MSD section's v2 block says that the box's 9 ran "
            "past the curve. The modern MSD fit refuses a window longer than the curve."),
        "reason": (
            "a box labelled 9 that fits 8 does not do what its label says, and the silent "
            "catch hides it. Reproduced because v2's printed D comes from those 8 lags."),
    },
    {
        "file": "SMT_GUI/Help_fcn/threePopulation.m",
        "line": 9,
        "legacy_behaviour": (
            "gamma = 1 - alpha - beta with alpha and beta each bounded to [1e-3, 1] "
            "independently (initializeVars.m:154-155) or [0, 1] (sqdFit_Simultaneous.m:"
            "55-56), so the third fraction can go NEGATIVE; the seeds alpha = beta = 0.5 "
            "start it at exactly 0 (initializeVars.m:153, sqdFit_Simultaneous.m:54)."),
        "new_behaviour": (
            "reproduced in legacy.sqd (the last fraction is 1 minus the others, "
            "unconstrained); the bounds are read from smtracker.sqd.V2_SQD_SEEDS. The "
            "modern fit keeps every fraction on a simplex."),
        "reason": (
            "a fraction below zero contradicts the quantity's name; it is where v2's "
            "1e-5 bubble clamp comes from. Reproduced because legacy mode shows v2's fit."),
    },
    {
        "file": "SMT_GUI/@SMT/sqdFit_Simultaneous.m",
        "line": 50,
        "legacy_behaviour": (
            "the simultaneous fit bounds every D at 1e-3 um^2/s and every fraction at 0, "
            "with seeds [.05 .5 | .5] and [.05 .2 .5 | .5 .5] (lines 44-56), while the "
            "per-condition fit it refits bounds D at 0.03^2/(4*tau), 0.011 at 20 ms, which "
            "initializeVars.m:146 names the 'lower bound' from localisation error, and "
            "the lower D at 3 at two populations. One SQD table holds both fits "
            "(SMTracker.m:5243-5245)."),
        "new_behaviour": (
            "reproduced: legacy.sqd.fit_legacy_sqd uses the per-condition set and "
            "fit_legacy_sqd_simultaneous the simultaneous one, each read from "
            "smtracker.sqd.V2_SQD_SEEDS, the one place v2's seeds are written down."),
        "reason": (
            "the simultaneous fit drops the bound its sibling's comment names, so two "
            "rows of one table were fitted under different limits. Reproduced, because "
            "each is what v2 ran."),
    },
]

#: The legacy SQD fit supports 1, 2 and 3 populations. That is not a limitation of this
#: implementation. It is what the source supports, and legacy mode reproduces the source.
LEGACY_N_POP = (1, 2, 3)

#: NOT an absent defect. A WRONG INSTRUCTION. Recorded so the instruction is corrected
#: once and cannot return as a request.
PROMPT_CORRECTIONS = [
    {
        "file": "BK prompt, section 4.2, item 1",
        "line": 0,
        "legacy_behaviour": (
            "the instruction states that fourPopulation contains a defect: D_b = x(4) "
            "should read D_d = x(4), and msd_d should use D_d, so that every "
            "four-population fit is degenerate. The instruction also asks legacy mode "
            "to support 4 populations."),
        "new_behaviour": (
            "NOT IMPLEMENTED, and the instruction is corrected. There is no "
            "fourPopulation function in this repository. SMT_GUI/Help_fcn holds "
            "singlePopulation.m, twoPopulation.m and threePopulation.m only. "
            "sqdFit_.m branches on n_pop 1, 2 and 3. sqdFit_Simultaneous.m switches on "
            "1, 2 and 3, with three case statements and no case 4. Legacy mode "
            "therefore supports 1, 2 and 3 populations, matching the source. A "
            "four-population fit written here would be a NEW METHOD carrying an OLD "
            "LABEL, which is the one thing legacy mode exists to prevent."),
        "reason": (
            "this row does not record a defect that was looked for and not found. It "
            "records an instruction that was wrong for this repository. A defect cannot "
            "be fixed in code that does not exist, and a method cannot be called legacy "
            "if no legacy version of it was ever written."),
    },
]


def write_deviations(root=None) -> "pd.DataFrame":
    """Write ``legacy_out/legacy_deviations.csv`` and return it.

    Two kinds of row, separated by ``status`` and never summed:

    * ``defect_fixed`` is code that did not do what its own comment, label or variable
      name said. Each one was read in the MATLAB source before it was fixed.
    * ``instruction_corrected_not_a_defect`` is an instruction that was wrong for this
      repository. It is not an absent defect and must not be counted as one.
    * ``checked_not_a_defect`` is code that was read because it looked wrong and turned
      out to be right. It is not a fix and must not be counted as one; it exists so the
      next reader does not repeat the check, or worse, act on it.
    * ``defect_reproduced`` is a v2 defect legacy mode keeps on purpose. It is not a fix
      and must not be counted as one.

    Every count is reported, never a difference between them.
    """
    rows = ([{**d, "status": "defect_fixed"} for d in DEVIATIONS]
            + [{**d, "status": "instruction_corrected_not_a_defect"}
               for d in PROMPT_CORRECTIONS]
            + [{**d, "status": "checked_not_a_defect"} for d in CHECKED_NOT_DEFECTS]
            + [{**d, "status": "defect_reproduced"} for d in DEFECTS_REPRODUCED])
    frame = pd.DataFrame(rows, columns=["file", "line", "legacy_behaviour",
                                        "new_behaviour", "reason", "status"])
    frame.to_csv(legacy_path("legacy_deviations.csv", root=root), index=False)
    return frame
