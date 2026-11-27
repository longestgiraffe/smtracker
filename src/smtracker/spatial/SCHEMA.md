# `smtracker.spatial` table schema (the contract)

Four tidy tables, exported once for each `source` (`observed`, `null_uniform`,
`null_kinetic`) and concatenated per file, so a downstream R script can `rbind` and
get its baseline by grouping on `source`. Column names, units and semantics are
FIXED. `io.validate_schema(df, kind)` checks a table has every required column;
extra columns (e.g. `null_rep` on the nulls) are allowed.

Cell frame: a spherocylinder (cylinder radius `R = width_um/2` + hemispherical
caps, total `length_um`). `u` along the long axis (`u=0` midcell), `v` across it
(`v=0` on the axis), µm. `apparent D` is `smtracker.appd.apparent_D(n_steps=4,
mode='split')` averaged per track — lag-1 `mean(Δx²+Δy²)/(4τ)` over 5-localisation
windows, **no 4σ² offset removed**, so `track_D_um2s` is APPARENT D = `D_true +
σ²/τ` and `edges` live on the apparent scale. `track_D_offset_removed_um2s =
track_D_um2s − σ²/τ` (NOT clipped; goes negative for genuinely-low-D tracks).

Populations are named by their D range (`D<0.15`, `D>=0.15`, …). Every track gets
one label; every localisation and step inherits it. A per-track threshold answers
"where are they", NOT "how many" — it does not recover the state occupancy.

## `localisations.csv` — one row per localisation

| column | units | definition |
|---|---|---|
| `source` | — | `observed` / `null_uniform` / `null_kinetic` |
| `dataset`,`enzyme`,`condition`,`replicate`,`session` | — | passed through (`session` = imaging day) |
| `cell_id`,`track_id`,`frame` | — | passed through |
| `x_um`,`y_um` | µm | raw camera frame, unchanged |
| `u_um`,`v_um` | µm | cell frame, signed |
| `l_signed`,`l_abs` | — | `u/(L/2)`; `abs` (0=midcell, 1=pole) |
| `halfwidth_um` | µm | `h(u)` |
| `r_norm` | — | `abs(v)/h`, clipped to 1 (1=membrane) |
| `r_norm_sigma` | — | `sigma_um/h` — uncertainty of `r_norm` here |
| `d_pole_um` | µm | `L/2 − abs(u)`, axial Cartesian (length-invariant) |
| `d_membrane_um` | µm | `h − abs(v)`, in-plane to the projected outline |
| `in_cap` | bool | `abs(u) > L/2 − R` |
| `cell_length_um`,`cell_width_um` | µm | that cell's geometry |
| `track_D_um2s` | µm²/s | per-track apparent D |
| `track_D_offset_removed_um2s` | µm²/s | `track_D_um2s − σ²/τ` (unclipped) |
| `population` | — | range label |
| `n_loc_track` | — | localisations in this track after filtering |
| `proj_weight` | — | `chord(u,v)/mean_chord`; expected 2-D density for a uniform 3-D filling (averages to 1) |

## `steps.csv` — one row per within-track consecutive pair

Identifiers + `source`, plus `frame_from`,`frame_to`,`dt_s`,`gap`,`du_um`,`dv_um`,
`abs_du_um`,`abs_dv_um`,`step_len_um`,`dr2_um2`,`u_mid`,`v_mid`,`l_abs_mid`,
`r_norm_mid`,`halfwidth_mid_um`,`proj_weight_mid`,`track_D_um2s`,
`track_D_offset_removed_um2s`,`population`. Adjacent steps only unless
`keep_gaps=True`. `abs_du_um`/`abs_dv_um` (axial vs radial displacement) have a
well-defined null — use them, NOT track centroids, to ask "is motion axial or
radial" (a fast track's centroid regresses to midcell).

## `tracks.csv` — one row per track

Identifiers + `source`, plus `n_loc`,`D_app_um2s`,`D_offset_removed_um2s`,
`population`,`u_centroid`,`v_centroid`,`l_abs_centroid`,`r_norm_centroid`,
`axial_span_um`,`radial_span_um`,`radius_gyration_um`,`n_gaps`. `D_app_um2s` here ==
`track_D_um2s` in the other tables (same value, different row meaning). Centroids are
DESCRIPTIVE DATA — never a per-track-centroid spatial statistic.

## `cells.csv` — one row per cell per source (the bootstrap unit)

`source`,`dataset`,`enzyme`,`condition`,`replicate`,`session`,`cell_id`,
`length_um`,`width_um`,`angle_rad`,`centroid_x`,`centroid_y`,`n_tracks`,`n_loc`,
`projected_area_um2`,`volume_um3`,`length_tertile`. COMPLETE even for cells with
`n_tracks = 0`.

## `map_bins.csv` — the map's own histogram (picture = numbers)

Grouping columns, `population`, `u_bin_centre`,`v_bin_centre`, `count`,
`weighted_count`, `pooled_count`,`pooled_weighted_count`,`summed_proj_weight`,
`enrichment`, `design_effect` (`Σnᵢ²/Σnᵢ` over tracks — the bin's precision cost),
`n_eff_track`,`n_eff_cell` (`(Σnᵢ)²/Σnᵢ²` with `i` a track, then a cell — how many equal
contributors the bin is worth; optional), `top_track_share`, `n_cells`,`n_tracks`, `reference`, `source`. `count` is raw
localisations, `weighted_count` is under `weight` (default `track`). Enrichment is
never bin-area normalised.

## `provenance.json`

package version, git commit, input paths + sha256, `edges`, `sigma_um`, `tau_s`,
`d_offset_um2s`, all null parameters + seeds, the D estimator's qualified name +
formula, and per-filter counts. No run's parameters need to be guessed.

## Where `sigma_um` comes from — and what the tracker actually stores

`sigma_um` is an INPUT to every table above (it sets `r_norm_sigma`, the offset
column and every null), and it is not measured by this pipeline. Since round 14 the
raw ingest table carries u-track's own per-localisation coordinate standard
deviations, `dx` and `dy` (pixels, from `tracksCoordAmpCG` columns 5:6, NaN when the
tracker records none — TrackMate exports no equivalent and writes NaN, never zero).

They are a **lower bound on σ, not a measurement of it**: they are the formal error
of the single-spot fit and exclude drift, sample motion and PSF-model error. Use
them to BRACKET σ from below (the round-13 bracket [16.5, 30] nm was set this way);
never write one into `sigma_um` as though it were a measured localisation precision.
The clean fix remains an immobile reference.

## Intervals that come from the SAME resamples are CORRELATED

`KineticsRow.occupancy_ci` and the rate intervals (`on_ci` / `off_ci`) are percentile
intervals over the **same unit resamples** — the same bootstrap frame, resample by
resample. They are therefore correlated, and **anything derived from both must not
combine them as independent**: no quadrature sum of an occupancy interval with a rate
interval, and no product/ratio interval built by multiplying their half-widths. A
quantity that depends on both must be recomputed PER RESAMPLE and its own percentile
taken — the bootstrap frame carries `occupancy_i`, `dwell_i_s` and `rate_i_to_j` on the
same rows precisely so that this is possible.

`occupancy_ci` is also **bounded on [0, 1]**. A percentile interval cannot leave that
range, so clipping is a no-op; an arm that lands ON a bound sets
`occupancy_ci_at_limit` and is marked in the panel, because a clipped arm ("not
resolved on that side") and a short arm ("precise") mean opposite things.

## σ configuration — one value, one provenance state, no silent default

`sigma_um` is an INPUT to every table here, and it now always travels with
`sigma_provenance` in `provenance.json`. `io.read_provenance` **refuses to load a run
that does not name both** — assuming `theoretical` for a file that does not say so would
silently reinterpret a run made under a lab-configured or measured σ.

Three states, and only three (`AcquisitionParams.SIGMA_SOURCES`):

| state | what it means | how it is set |
|---|---|---|
| `theoretical` | **the default** — computed from the optical parameters the file already carries. Usable with no action from the user. | automatic |
| `configured` | a lab-supplied constant. Instrument/lab-level config with a per-dataset override — never a per-run dialog. | `with_localization_sigma_nm(project, nm)` |
| `measured` | from a stationary reference sample. | `with_localization_sigma_nm(project, nm, "measured")`, by hand |

Older exports used `derived_pixelation` / `manual` / `msd_intercept` /
`immobile_control`; those are migrated (→ `theoretical` / `configured` / `configured` /
`measured`) so a stored analysis keeps its provenance instead of being re-defaulted.

### The two theoretical bases — both numbers, deliberately NOT reconciled

`sigma_basis` records which one a `theoretical` σ rests on:

| basis | value here | what it is |
|---|---|---|
| `psf` | **98.8 nm** (u-track `psfSigma` 0.931681 px × 106.000003 nm) | the width of a single emitter's image, from NA 1.49 / emission 517 nm. Used when the tracker recorded it — READ from `movieData.mat`, never recomputed, since reimplementing u-track's fitted high-NA approximation would be a guess. |
| `pixelation` | **30.6 nm** (pixel / √12) | the SD of a uniform distribution over one pixel — how finely the camera samples. The fallback when no PSF width is on file. |

They differ by **3.2×** and are left unreconciled, because they are different
quantities: achievable localization precision is smaller than the PSF width by roughly
√N photons, so `psf` is an upper-ish bound and `pixelation` a lower one. **The
consequence is not cosmetic**: the localization floor σ²/(4τ) is **0.122 µm²/s** on the
`psf` basis against **0.0117** on `pixelation` — the first sits at the `D<0.15`
classification threshold, i.e. it says the whole low-D class is at or below the
resolution limit. Quote the basis whenever the floor is quoted. The clean resolution is
a stationary reference sample (→ `measured`); until then neither number is a measurement
of localization precision.

`dx, dy` are **not** promoted into any of this (see above): they are a per-localization
LOWER bound, and a real stationary-reference measurement is entered by hand.

### Two σ→D conversions, and which is which

They differ by a factor of six and both are correct for their own job:

| quantity | formula | at σ = 98.8 nm, τ = 0.02 s | what it is |
|---|---|---|---|
| localisation floor | **σ²/(4τ)** | **0.122 µm²/s** | a BOUND, in the legacy convention: the D whose characteristic squared displacement `4Dτ` equals σ². It answers "below what D is a molecule indistinguishable from stationary". |
| static-error offset | **σ²/τ** | 0.488 µm²/s | the BIAS in a 2-D single-lag apparent D. Two independent localisations contribute `2σ²` per axis, so `4σ²` in 2-D, and `MSD = 4Dτ + 4σ²` ⇒ `D_app = D_true + σ²/τ`. This is what the correction subtracts. |
| corrected-D shift | **1.5·(σ₂²−σ₁²)/τ** | 0.732 µm²/s (from σ = 30 nm) | the offset above times the Berglund blur factor `1/(1−2R) = 1.5` at `R = 1/6`. |

So the 6× between the floor and the test's expression is **4 × 1.5**: a factor 4 from the
2-D two-endpoint noise variance (`4σ²` in the MSD versus the floor's bare `σ²`), and 1.5
from the blur factor. The floor is a threshold convention inherited from the legacy
bounds (`initializeVars.m`: `0.03²/(4·τ)`); the offset is the actual measurement bias.
Do not substitute one for the other.
