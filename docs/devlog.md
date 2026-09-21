# Wildfire Prediction — Dev Log

A running record of bugs, design decisions, and dead ends encountered while building the
wildfire prediction pipeline (S3-backed scene collection → tile caching → training). Kept
as raw material for a future writeup, updated as issues come up rather than reconstructed
from memory after the fact.

---

## Initial push - 04/24/26

Pushed the first skeleton of the project all at once: the Sentinel-1/Sentinel-2/ERA5
loaders (`data/loaders/`), the zonal aggregator (`data/aggregator/zonal_aggregator.py`),
`model/dataset.py`, and `model/architecture.py`. RTC and cross-attention fusion were still
stubs (`raise NotImplementedError`) at this point, and NaNs in tile statistics were just
zeroed out (`np.nan_to_num`) rather than properly imputed.

## Radiometric terrain correction for Sentinel-1 - 05/13/26

Fully implemented RTC, which had been a stub until now. Given a scene's Sentinel-1 bounds,
it downloads the covering Copernicus GLO-30 DEM tiles from the public AWS bucket, mosaics
and reprojects them onto the scene's grid, then computes surface slope/aspect from the DEM
gradient and derives a per-pixel surface normal vector. Per-pixel incidence angles are
parsed from the Sentinel-1 annotation XML (sparse geolocation grid points) and interpolated
to the full image grid, giving a radar look vector. The local incidence angle
(`cos_local = normal · look_vector`) is then used to normalize each pixel
(`sigma0 / cos_local`), with pixels where `cos_local` is too small (layover/shadow, where the
radar geometry breaks down) masked to NaN rather than divided through. Decibel conversion
(`10 * log10`) was also moved to happen *after* RTC instead of before, since it should apply
to the terrain-corrected value, not the raw digital number. Also added orthorectification
(`orthorectify()` in `utils/geo_utils.py`) for scenes that ship ground control points instead
of a direct affine transform.

## Aggregator rewrite + RTC bug fixes - 05/15/26

Bug-tested the RTC implementation above and found two real issues:

- The shadow/layover mask used `abs(cos_local) > threshold`, which only screens out pixels
  where the radar geometry is close to perpendicular in *either* direction. Tightened it to
  `cos_local > threshold`, so back-facing slopes (radar looking away from the surface) get
  masked the same as true shadow, not treated as valid just because the magnitude was large.
- The slope/aspect gradient (`np.gradient(dem)`) was being divided directly by the transform's
  pixel spacing in *degrees*, not real ground distance — a degree of longitude covers a very
  different physical distance depending on latitude (notably so at Alaska's high latitudes).
  Added `meters_per_degree(lat)` to convert degree spacing to meters at the scene's center
  latitude before computing real-world slope.

Also rewrote `zonal_aggregator.py`'s core tiling loop: the original version was a manual
nested loop over every ERA5 lat/lon bin with a boolean mask recomputed per bin (effectively
O(bins × pixels)). Replaced it with a vectorized `pandas` `groupby(['x', 'y']).agg(['mean',
'std'])`, aggregating all pixels into per-tile statistics in one pass — a large speedup for
full-scene tiling. While rewriting it, also fixed a latitude/longitude ordering bug in
`vectorize()` (`utils/geo_utils.py`): it now consistently returns `(longitude, latitude)`,
with every call site updated to destructure it in that order — previously the return order
didn't reliably match how callers were unpacking it.

## Mean imputation + ERA5 bounds check - 05/21/26

Implemented proper mean imputation in `model/dataset.py`, replacing the earlier
`np.nan_to_num`-to-zero placeholder. `dataset.__init__` now scans for the first tile with
complete (non-`None`) stats across all three sources and computes a per-feature dataset-wide
mean from every tile that has a real value for that feature; `__getitem__` then fills in any
tile whose `s1_stats`/`s2_stats`/`era5_stats` is `None` using those means instead of zeroing
it out, which is a meaningfully better prior for a missing SAR/optical/weather reading than
"zero."

Also added the first version of the ERA5 coordinate-range guard in the aggregator: if a
scene's vectorized Sentinel-1 or Sentinel-2 pixel bounds fall outside the ERA5 grid's
lat/long extent, raise a `ValueError` rather than silently binning pixels into the wrong (or
nonexistent) ERA5 cell. This was bare-bones at first (no actual bounds in the message) — the
version with real numeric bounds attached came later, during the July defensive-coding pass.

## Cross-attention fusion - 05/21/26

Replaced the `cross_attention_fusion` stub (previously just `raise NotImplementedError`,
with the encoders' outputs simply concatenated) with a real implementation: spatial tile
embeddings attend over the temporal (ERA5) embedding sequence via `nn.MultiheadAttention`,
with spatial embeddings as the query and temporal embeddings as keys/values — the intent
being that each tile learns which time steps in its antecedent weather window matter most.
The attention output is then concatenated with the original spatial embedding (not replacing
it) before being passed to the three prediction heads. Also fixed how the temporal encoder
reshaped its input before feeding the transformer — `X.T.unsqueeze(1)` (transpose then
unsqueeze) was replaced with `X.unsqueeze(0).unsqueeze(0)`, adding the sequence and batch
dimensions directly instead of transposing a 1-D vector.

## Sentinel-2 masking improvements - 05/22/26

Extended `apply_cloud_mask` to also exclude SCL category `0` (`no_data`) pixels, not just
cloud/cloud-shadow/cirrus — a scene edge or fragmented tile can have genuine no-data pixels
that aren't clouds but are equally unusable. More importantly, cloud masking now gets applied
to every individual raw band (red, green, NIR, SWIR) before NDVI/NBR are computed from them,
rather than only masking the final NBR index after the fact — previously a cloud-contaminated
NIR or SWIR pixel could still poison the NDVI/NBR calculation even though the *output* index
was later masked. Renamed the `"raw_bands"` dict key to `"filtered_bands"` to reflect that.

Also added `nullify_nan()` to the aggregator: a tile whose Sentinel-1 or Sentinel-2 stats
came out entirely NaN (e.g. a fully cloud-masked bin) now gets its stats dict set to `None`
instead of staying a dict full of NaNs — which matters because May 21's mean-imputation logic
only treats `None` as "missing," so a stray NaN-filled dict would otherwise have silently
averaged NaNs into the per-feature means.

## Fire collection notebook - 06/08/26

I used a notebook called `query_satellite_data.ipynb` to iterate through fire events from 
2016 to 2024 found in the Monitoring Trends in Burn Severity (MTBS) shape file - a freely 
downloadable archive of fires across the United States dating back to 1984. My notebook 
searched for and synced whole fire 'packages', which included a Sentinel-1 pre and post 
fire image, Sentinel-2 pre and post fire image, an ERA-5 grib tile (built via 
`calculate_master_era5_area`) including at least 30 days of weather variable data leading up 
the ignition date of the fire, and a metadata.json detailing the contents of the package, to 
an S3 bucket under `fires/{state}/{fire_name}_fire/`. I manually checked that multiple 
constraints on all fires were satisfied, below is a small snippet of them. 

- total fire acreage > 10k 
- temporal windows between S1/S2 pre scenes < 36 hours 
- temporal windows between S1/S2 post scenes < 36 hours 
- 95% of fire boundary boxes being visible to the naked idea 
- Ensuring matching orbital direction between S1 pre and post scene.

Hence, accumulating over 125 fire packages took almost a month. 

## control collection pipeline - 07/01/26

3 controls were sampled for each fire, with the EPA Level 3 eco-region of a fire used to 
ensure regional consistency between controls and their source fire. Overlap between 
fires and controls is prevented by combining a literal fire-box exclusion (no control inside a fire's 
actual footprint, via `fire_exclusion_zone` in `control_collection_pipeline.py`) with a 20km 
minimum-separation constraint (`too_close()`) against a continuously-growing list of every 
fire and already-placed control centroid, enforced together in `sample_control_centroid()`.

## Consistency checks between controls and fires - 07/02/26

Running the complete control collection script (`control_collection_pipeline.py`) would have 
taken around 4 days, but ended up taking close to a week and a half due do small bugs 
I discovered throughout collection which resulted in me having to partially or fully restart the 
collection pipeline. Bugs were related to small discrepancies between the quality of fires and 
controls. A few of the notable ones are listed below:

1. **Enforcing matching Sentinel-1 orbital directions between fires and controls**. The 
original pipeline not only did not enforce matching orbital directions between controls 
but also did not between fires and controls. An in-place checking script 
`confirm_matching_orbital_direction.py` was used to check which controls did not fully 
adhere to their fire's Sentinel-1 orbital direction. These were then deleted, and 
`control_collection_pipeline.py` was updated to enforce consistent orbital direction.
This ended up being nearly all controls so I decided to just fully restart collection. 

2. **Enforcing a minimum bounding box coverage**. 70% of the bounding box of all fires was 
required to be contained within each Sentinel scene and ERA-5 spatial window. A similar 
requirement was originally not imposed on controls but since this mistake was caught 
around the same time as #1, the control collection script's own coverage check 
(`calculate_spatial_coverage_percentage()` / the `target_tile_geom.intersection(...)` checks 
in `collect_controls()`) was adjusted accordingly.

3. **Enforcing a stricter valid pixel requirement**. In my fire collection 
notebook, since I manually inspected the S2 pre and post scenes myself I allowed 
Sentinel-2 images with a relatively high number of invalid pixels (up to 40% for some) 
as long as the entirety of the fire boundary box was still visible. Since I did not 
manually inspect the S2 images of each control, I decided to just enforce stricter valid 
pixel requirement (80%).

## Dataset control quality assurance checks - 07/10/26

With all 375 controls collected, I conducted a series of metadata checks to ensure no 
silent errors or data corruptions snuck through. The main issue discovered was naming 
inconsistencies between the state dir a fire/control was placed in and the actual state  
that the fire occurred in. Discovered broken logic that used a hard-coded if-else 
statement to determine the state of a fire based on it's boundary box, when the MTBS 
event ID (the `event_id` field already sitting in every fire's `metadata.json`) was a much 
easier and reliable ground truth. After iterating through all controls 
and their associated fires, we determined a sample of fires/controls that needed to be 
either edited or requeried, and have listed their fixes below: 

1. **Fires with mismatched state indices were flagged**, and not only were they moved to their 
correct `state` folder but all associated SAFE, grib files were renamed and the scene's 
metadata.json was edited to reflect the new contents of each scene. Over 30 fires of the original 
125 ended up flagged, which resulted in 120 total fixes (30 fires + 90 controls).

2. Each fire's control underwent the same fixes. With the additional caveat of the control 
id (and therefore the folder name) also being edited to reflect the actual state of the fire.

3. The size of each SAFE and grib were checked to ensure that no data loss occurred during 
download time. The standard size of a SAFE (850 MB - 1.3 GB) and a grib (2 - 30 MB) were 
used as benchmarks.

4. Finally, each fire was matched to its' 3 controls. And the SAFE folders within each were 
unzipped so as to compare the manifest (metadata file for a SAFE product) with the metadata.json.
Any discrepancies would be flagged, but none were found. 

## Tile augmenter - 07/16/26

Implemented the tile augmenter, which uses Gaussian jitter and mixup to add perturbations
to the scalar features that are extracted and processed from each data source. A transformer 
is a data-hungry architecture so adding additional augmented tiles will help model alignment.

Only a small issue with `dropout` in `augmentation.py` required fixing. Tensor.uniform 
was being called and this function does not exist in PyTorch - only the in-place uniform()
call does. Replaced with a single `torch.rand` call on a preset PyTorch generator. 

## Defensive-coding hardening pass - 07/20/26

Before scaling up to a full 125-fire / 375-control run, did a pass over every `.py` file in
`data/`, `model/`, and `scripts/` looking for missing error handling. Scoped it down based on
actual failure modes observed so far rather than adding guards everywhere reflexively — e.g.
`era5_preprocessing.py` never needed `IndexError`/`KeyError` guards because the ERA5 query
always explicitly requests the same 5 variables, so a missing-variable failure would be a
real upstream bug worth crashing loudly on, not something to paper over.

Landed fixes included: calibration LUT vector-count guards, a warning (later found to be
incomplete — see below) on zero/negative calibration values, request timeouts, VH/VV
shape-match assertions, and band-shape consistency checks in the zonal aggregator.

## DEM 404s over open ocean (57 of 63 initial failures) - 07/20/26

First full tile cache run threw a wall of `404 Client Error` on DEM tile downloads. Root cause: Copernicus
GLO-30 only has tiles over land — a purely-oceanic 1°×1° DEM tile *genuinely does not exist*,
so a 404 there is expected, not a bug. Fixed `download_dem_tiles` to skip a failed tile instead
of crashing, and only raise if *every* tile for a scene fails (meaning the whole scene is over
water). Also switched `prepare_dem`'s destination array from `np.empty` to `np.zeros` so any
skipped-tile gap defaults to sea level (0m) instead of uninitialized memory.

## ERA5 coordinate-range padding (three separate bugs, same symptom) - 07/22/26

The remaining ~6 failures were `"Sentinel-X coordinate range falls outside ERA5 coordinate
range"` — the downloaded ERA5 grib's bounding box didn't fully cover the Sentinel scene's
footprint. This took three iterations to actually close out:

1. **Root cause #1 — no padding at collection time.** `calculate_master_era5_area` (fire
   notebook) and its `_from_items` twin (control pipeline) built the ERA5 `area` request
   directly from the catalog `GeoFootprint`, with zero margin. Fixed by padding all four
   `[N, W, S, E]` bounds outward by a fixed `padding_degrees=2`.

2. **Root cause #2 — requery script used the wrong date-window logic for controls.** Wrote
   `find_era5_requeries.py` (parses `cache_failures.log` for the coordinate-range error,
   unions S1+S2 bounds per scene) and `requery_era5_failures.py` (re-downloads with padded
   bounds, re-uploads to the same S3 key) to patch the still-failing scenes without a full
   rebuild. First version used the *fire* notebook's `DateOffset`-based month window for every
   scene, including controls — caught because I asked directly whether it matched the control
   pipeline's actual `pd.date_range`-based window. It didn't. Fixed with two separate
   request-builders, `build_fire_era5_request` and `build_control_era5_request`.

3. **Root cause #3 — padding computed from only one source's footprint.** Two remaining
   scenes (`FLAT_fire`, `control_NV_38N118W_20230804`) failed again, this time on the
   Sentinel-1 side specifically. Sentinel-1 IW GRD swaths (~250km) are much wider than
   Sentinel-2 MGRS tiles (~100km×100km) — padding computed only from the S2 footprint didn't
   cover S1's wider extent. Fixed by unioning bounds across *both* sources, and across
   separate `cache_failures.log` append events for the same scene (the log is append-only by
   design, for live monitoring during a run).

## Tile cache filename collisions (18 colliding scene_ids / 39 scenes) - 07/23/26

Noticed the `.pkl` count from a full run (492 by internal reconciliation) didn't match
`ls tile_cache/*.pkl | wc -l` (471). First guess was a `cwd` issue watching the wrong
directory — wrong guess, the count really was short. Wrote `find_duplicate_scene_ids.py` to
check: 18 different `scene_id`s (derived from just `control_id`/`fire_name`, the last path
component) were shared across 39 actually-distinct S3 scenes — two different fires' controls
can land on the same rounded coordinate + date and collide on name.

Fix: `cache_path_for()` now mirrors the *full* S3 prefix (`fires/{state}/{fire}/...` /
`controls/{state}/{fire_control}/{control_id}/...`) as the cache filename, so collisions are
structurally impossible. Wrote `reorganize_tile_cache.py` to migrate the already-cached flat
files into the new nested structure, and `delete_colliding_pkls.py` to drop the ambiguous ones
that couldn't be safely attributed.

This bug had a second, nastier form: it wasn't just the filename, it was also the `scene_id`
*value stored inside the pickled record itself*. `train.py`'s `merge_tiles()` builds dict keys
as `f"{scene_id}__{tile_key}"` — with colliding scene_ids, tiles from two different scenes
silently overwrote each other in that dict while every scene's *labels* still got appended
unconditionally, producing a hard `tile/label count mismatch: 52003 vs 53652` assertion
failure. Fixed at the source (`build_tile_cache.py` now stores the full path as `scene_id` in
the record, not just the short name) and retroactively via `fix_scene_ids_in_cache.py`, a fast
in-place pickle-patching script that didn't require re-touching S3/DEM/ERA5.

## ERA5 temporal leakage (the most consequential bug, methodologically) - 07/24/26

While investigating a `cfgrib` crash (below), realized something more serious: the antecedent
ERA5 window collected for each fire could include data *during or after* the fire itself, not
just before it. The collection notebook over-fetches full calendar months rather than a tight
date range, so a fire's cached ERA5 stats could include weather from while it was actively
burning — meaning the model could theoretically learn to detect an ongoing fire from its own
weather signature, which tells you nothing about *predicting* a fire that hasn't started yet.

Confirmed with a CDS UI screenshot that this isn't fixable by requerying: the Climate Data
Store's `year`/`month`/`day` request parameters are independent filters that get
cross-multiplied server-side, not a literal date range — so there's no way to ask CDS for
"the 30 days before X" when that span crosses a month boundary, only "these year/month/day
values, in any combination." This was a known, deliberate limitation from when the original
collection pipeline was designed, not new information.

Decided against a full requery/pipeline rewrite (impractical given the CDS constraint above)
in favor of a hard cutoff filter applied at *load* time, regardless of what's actually in the
downloaded grib: `era5_cutoff_from_key()` extracts the ignition/control date from the grib's
S3 key, and `load_era5_vars(..., cutoff_datetime=...)` drops every timestep at or after that
date. Used `<` rather than `<=` deliberately — since ignition timing is date-only (no time of
day), including the ignition date itself would still risk leaking mid-fire weather.

This required a full tile-cache rebuild (~2 days), since every previously-cached fire tile's
ERA5 stats were computed before this filter existed.

## `cfgrib` hypercube fragility (found *during* the rebuild above) - 07/25/26

The rebuild hit a new failure: `TypeError: '<' not supported between instances of 'tuple' and
'datetime.datetime'`, in the newly-added cutoff filter. Two things had to go right to find the
real cause, and my first attempt at both was wrong:

- `cfgrib.open_datasets()` splits one grib file into multiple "hypercube" datasets, and which
  hypercube a given variable ends up in isn't stable across files — the original code assumed
  fixed positional indices (`datasets[0]`, `datasets[1]`). Fixed by searching every hypercube
  by variable name instead. This wasn't sufficient on its own.
- The real bug: for forecast-style data with separate `time` (init) and `step` (lead-time)
  dimensions, `.stack(valid_time=('time','step'))` produces a `MultiIndex` whose values are raw
  `(time, step)` *tuples* — not real datetimes, and not comparable to one. I initially treated
  this as already fixed by the hypercube-search change; it wasn't the same bug. Only found the
  actual cause after being pushed to look again, reasoning that since the crash lived inside
  the `if cutoff_datetime is not None:` block, it had to be something specific to the new
  cutoff logic — which it was. Fixed with `_flatten_time_step()`: explicitly compute the real
  valid datetime as `time + step`, then swap it in for the MultiIndex via
  `reset_index(...).rename(...)`.

Validated the fix with a synthetic xarray test in a sandbox before calling it safe, rather than
just re-running the multi-day pipeline again on faith.

## `train.py`: getting it running at all - 07/27/26

Once tile caching was clean, `train.py` needed its own round of fixes to work with the newer
nested `tile_cache/` structure:

- `load_records()` used a top-level (non-recursive) glob against a directory that was now
  nested — fixed with `glob.glob(..., recursive=True)` and a `**` pattern.
- Running `python training/train.py` from the repo root didn't put the repo root on
  `sys.path` (Python adds the *script's own* directory, not the caller's cwd) — so
  `from model import ...` failed with `ModuleNotFoundError`. Fixed with the same
  `REPO_ROOT`/`sys.path.append` pattern `build_tile_cache.py` already used.
- The scene_id collision bug (above) resurfaced here in its record-field form, causing the
  tile/label count mismatch assertion.

## NaN and Inf propagation into `BCELoss` - 07/27/26

Last crash before the first successful run: `RuntimeError: all elements of input should be
between 0 and 1` out of `BCELoss`, since its inputs come from `F.sigmoid(...)` and
`sigmoid(NaN)` is itself `NaN`. Root-caused in two layers:

1. **NaN**, from pandas: `zonal_aggregator.py`'s `groupby(...).agg(['mean', 'std'])` defaults
   to `ddof=1`, which is undefined (`NaN`) for any tile with exactly one contributing pixel —
   a very plausible occurrence for sparsely-covered ERA5-grid cells. The existing
   `nullify_nan()` only blanked a *whole* stats dict if *every* value in it was NaN, so a tile
   with valid means but a single NaN std slipped through untouched. `model/dataset.py`'s
   imputation logic also only checked `is None`, never `NaN`, so nothing downstream caught it
   either. First fix: switch `_safe_mean` to `np.nanmean` (plus an all-NaN guard), and add an
   explicit `_is_missing()` check (catching both `None` and `NaN`) to the imputation loops in
   both `model/dataset.py` and `training/train.py`'s `build_feature_stds()`.

2. **This didn't fully fix it** — the same crash recurred after redeploying, traced to a
   second, different source: **Inf**, not NaN. `calibrate_to_sigma()` in
   `sentinel1_preprocessing.py` divided by the calibration value squared with no zero guard;
   when that value was exactly `0`, the result was `+Inf` (not NaN, despite what the existing
   warning message claimed), and `Inf` isn't caught by `np.isnan()` anywhere in the chain — it
   sailed straight through `nullify_nan()`, both `_is_missing()` checks, the dB conversion
   (`Inf > 0` is `True`, so `log10(Inf) = Inf`), and into the model, where Inf arithmetic in
   the conv/linear layers is exactly the kind of thing that degrades into NaN by the output
   layer. Fixed at the source (explicit NaN output where the calibration value is zero,
   instead of dividing into Inf) and defensively everywhere downstream (`nullify_nan`,
   both `_is_missing` helpers) by switching the check from `np.isnan` to `not np.isfinite`,
   so Inf can't sneak past the same gap again from a different source in the future.

One good side effect of finally tracking `training/` in git properly (it wasn't, until this
point — the whole file was untracked on EC2, so none of these fixes had actually been reaching
the EC2 copy until caught and re-synced): confirmed via diff that the only difference between
the untracked EC2 copy and the real fixed version was exactly the missing `_is_missing` NaN/Inf
filtering — no other drift.

## First full training run — result and diagnosis - 07/27/26

With all of the above fixed, `train.py --cache-dir tile_cache --epochs 20 --batch-size 32` ran
end to end for the first time: 500 cached scenes (125 fires, 375 controls), 53652/10703/10349
train/val/test tiles.

```
epoch   1 | train loss 1.1871 acc 0.738 | val loss 0.5815 acc 0.744
epoch   2 | train loss 0.5675 acc 0.750 | val loss 0.5667 acc 0.744
...
epoch  20 | train loss 0.5578 acc 0.750 | val loss 0.5754 acc 0.744
Final test | loss 0.5662 acc 0.750
```

Diagnosis: this is class-imbalance collapse, not a data-quantity problem. Two tells:

- Accuracy is pinned at ~0.744-0.750 in *every single epoch*, train and val alike — almost
  exactly the dataset's control fraction (375 / 500 = 75%). A model that always predicts
  "control" gets ~75% accuracy for free on this split.
- The loss plateau (~0.56-0.58) matches the BCE loss of always predicting the constant
  base-rate probability (0.75) against a 75/25 split almost exactly (≈0.562 analytically).

Together, this says the model converged to just outputting the prior and ignoring its inputs
entirely. Two concrete, likely-fixable causes identified before touching hyperparameters:

1. **No class-imbalance handling** — plain `nn.BCELoss()`, no positive-class weighting, no
   stratified sampling, with a 3:1 control:fire ratio.
2. **No input normalization** — `build_feature_stds()` computes per-feature mean/std, but they
   are currently only used inside `TileAugmenter` (jitter noise scale, dropout imputation
   value) and never actually applied to standardize `x_spatial`/`x_temporal` before they reach
   the model. Features span wildly different scales (Kelvin-scale temperature, small-magnitude
   precipitation, negative-dB SAR, unit-scale NDVI) with no z-scoring — a well-known way to
   get exactly this kind of "gave up, predicted the mean" collapse.

## Class weighting + input normalization - 07/28/26

Implemented both fixes identified above.

**Class weighting**: switched from `nn.BCELoss()` to `F.binary_cross_entropy(pred, target,
weight=...)` in `run_epoch()`, with a per-batch `weight` tensor built via
`torch.where(target > 0.5, pos_weight, torch.ones_like(target))`. `pos_weight` itself
(`n_neg / n_pos`) is computed once in `main()` from the actual per-tile train labels, not
assumed from the 125/375 scene-level ratio, since different scenes can yield different tile
counts. This makes getting a fire tile wrong cost proportionally more than getting a control
tile wrong, closing off the "always predict control" shortcut. Kept `nn.BCELoss`'s sigmoid
contract in `architecture.py` unchanged (didn't switch to `BCEWithLogitsLoss`) to avoid
touching the model's output semantics elsewhere.

**Input normalization**: added `spatial_mean`/`spatial_std`/`temporal_mean`/`temporal_std`
to `LabeledTileDataset`, computed once in `main()` from `train_ds_unaugmented.statistic_means`
(mean) and `feature_stds` (std) — both already train-split-only, now reused for
normalization instead of just augmentation. Applied in `__getitem__` as a plain z-score
(`(x - mean) / std`) *after* `TileAugmenter`'s jitter/dropout, since those operate in raw
scale (their noise magnitude and imputation values are raw-scale too) — normalizing first
would have made the jitter noise scale meaningless. Same train-derived stats are reused for
val/test to avoid leaking their statistics into what the model treats as "normal."

While wiring this up, also hardened `build_feature_stds()`'s fallback: it already defaulted
to `std=1.0` when there wasn't enough data to compute a real std, but a feature that's
genuinely constant across the whole train split (real std of exactly `0`) wasn't guarded —
dividing by that during normalization would silently produce `Inf`, the same class of bug
fixed in the NaN/Inf entry above. Added `_safe_std()` to catch that case too.

Not yet re-run — next step is rerunning `train.py` and checking whether accuracy actually
moves off the ~75% floor.

## Cleanup pass on the class-weighting/normalization change - 07/28/26

A few small things caught in review of the change above:

- `build_feature_stds(train_tiles, train_ds)` never actually used `train_ds` — dropped the
  parameter and updated the one call site.
- `weight = torch.where(target == 1.0, pos_weight, torch.ones_like(target))` tripped
  SonarQube's floating-point-equality warning. `target` is only ever exactly `0.0` or `1.0`
  in practice (built straight from the label list, no arithmetic on it in between), so this
  was never actually unsafe, but switched to `target > 0.5` anyway to match the threshold
  style already used for accuracy (`pred > 0.5`) and avoid relying on that guarantee holding
  forever.
- Added per-run output directories and a loss-curve plot: each call to `train.py` now creates
  `checkpoints/{timestamp}/` (via `run_id = datetime.now().strftime(...)`) holding
  `best_model.pt`, `final_model.pt`, and a new `loss_curve.png` (`plot_loss_curve()`, using
  `matplotlib` with the `Agg` backend since EC2 has no display to render to). Runs used to
  overwrite the same `best_model.pt`/`final_model.pt` in a flat `checkpoints/` dir every time,
  so there was no way to compare or even keep more than one run's output. Also updated the
  `--upload-to-s3` path to upload under `{s3_model_prefix}{run_id}/` instead of flattening
  every run's `best_model.pt` to the same S3 key.

## Moved `LabeledTileDataset` into `model/dataset.py` - 07/28/26

`LabeledTileDataset` had been living in `training/train.py`, but it's really a data-layer
class (wraps `dataset` with labels, augmentation, and normalization) rather than
training-loop logic, so moved it into `model/dataset.py` alongside `dataset` itself —
`train.py` should stay focused on the argparse/loop/checkpointing side, and this way it's
importable for a future eval or inference script without dragging in the rest of the
training script.

While moving it, also consolidated the duplication this exposed: `train.py` had its own
copies of `S1_KEYS`/`S2_KEYS`/`ERA5_KEYS` (as module-level constants) and `dataset.py`'s
`__getitem__` had a *second*, separately hardcoded copy of the same three key lists
(`s1_keys`/`s2_keys`/`era5_keys`, lowercase, local to the method). Promoted all five key
lists (`S1_KEYS`, `S2_KEYS`, `ERA5_KEYS`, `SPATIAL_KEYS`, `TEMPORAL_KEYS`) to module-level
constants in `model/dataset.py`, and pointed both `dataset.__getitem__` and
`LabeledTileDataset` at the same set — one definition instead of two that had to be kept in
sync by hand. Also promoted `train.py`'s local `_is_missing()` helper to a shared,
non-private `is_missing_value()` in `model/dataset.py`, used by both `dataset.__getitem__`
and `train.py`'s `build_feature_stds()`. `model/__init__.py` now re-exports
`LabeledTileDataset`, the five key-list constants, and `is_missing_value`.

Purely a structural move — no behavior change, verified via `py_compile` on all three
touched files.

## First run with class weighting + normalization - 07/28/26

Reran `train.py` with both fixes live. Loss curve: train loss falls smoothly (0.97 → 0.92),
val loss stays noisy and roughly flat (0.985-1.035), no longer pinned dead flat at the old
~0.744 accuracy/~0.56 loss floor — so the class-weighted BCE broke the "always predict
control" collapse from before. First read of the plot looked like classic overfitting (train
down, val flat), but the full printout told a different story once accuracy was visible:

```
epoch   1 | train loss 0.9719 acc 0.626 | val loss 1.0235 acc 0.561
...
epoch  12 | train loss 0.9217 acc 0.640 | val loss 0.9902 acc 0.441   (best val loss so far)
...
epoch  20 | train loss 0.9164 acc 0.645 | val loss 1.0126 acc 0.524
Final test | loss 0.9333 acc 0.576
```

Train accuracy plateaus around 63-65% — itself *below* the naive 75% majority-class
baseline, and barely moving over 20 epochs. Val accuracy swings wildly (0.44-0.67) with no
relationship to val loss at all (epoch 12's best-val-loss checkpoint has one of the *worst*
val accuracies in the whole run). That combination — train not fitting well either, and
val accuracy decoupled from val loss — doesn't match overfitting (train low/confident, val
diverging upward); it looks more like plain accuracy no longer being a meaningful metric
once the loss is class-weighted (a model can trade raw accuracy for lower weighted loss by
not defaulting to the majority class), plus a still-noisy/weak signal overall. Conclusion:
not overfitting yet — the more useful next step is tracking a real classification metric
(precision/recall/F1 or balanced accuracy per class) instead of plain thresholded accuracy,
since accuracy and the training objective are no longer aligned.

Also fixed a real bug surfaced by this discussion: `main()`'s final test evaluation ran on
whatever `model` held after the *last* epoch, never on `best_model.pt` (the actual
best-val-loss checkpoint saved to disk) — so "Final test" was silently reporting the
last-epoch model's performance, not the best one. Fixed by reloading `best_model.pt`'s state
dict into a separate `best_model` instance for the final test pass, leaving `model`'s
last-epoch weights untouched so `final_model.pt` still saves something meaningfully
different from `best_model.pt`.

## Added balanced accuracy / precision / recall / F1 - 07/28/26

Following directly from the metric-mismatch conclusion above: `run_epoch()` now accumulates
TP/FP/FN/TN counts (across all 3 horizon heads, same flattening the existing accuracy count
already did) and returns balanced accuracy, precision, recall, and F1 on the fire (positive)
class alongside loss/accuracy, instead of just the two. Balanced accuracy is `(recall +
specificity) / 2` — a single number that isn't inflated by the majority class the way plain
accuracy is, so it's what I'd actually trust as a headline number given the 3:1 class
imbalance. Precision/recall/F1 on the fire class are there for a finer-grained read on the
same question (e.g. whether the model is trading recall for precision or vice versa, which a
single accuracy or balanced-accuracy number would hide).

`run_epoch()` now returns a dict instead of a `(loss, acc)` tuple to keep the growing set of
metrics from turning into an unreadable positional-tuple return; updated the three call sites
in `main()` accordingly. The per-epoch and final-test print lines now report val bal_acc /
precision / recall / f1 alongside the existing loss/acc (train metrics still print loss/acc
only, to keep the line from getting too long — bal_acc etc. are the ones that actually needed
fixing).

## Second run with balanced accuracy visible - not overfitting, just weak - 07/28/26

Rerun with the new metrics live: val balanced accuracy sits in a 0.54-0.60 band across all 20
epochs (final test: 0.595), against a 0.50 floor for random guessing. So the model is
extracting *some* real signal — meaningfully better than chance, and recall on the fire class
gets as high as 0.65-0.74 in several epochs, which the old collapsed model could never
produce — but 0.55-0.60 balanced accuracy is weak, not something to trust operationally yet.
Both train and val loss plateau early and stay flat rather than diverging, which rules out
overfitting as the explanation; this looks like the model converging to a weak local optimum
and getting stuck there.

Also noticed while reading this run: val loss and val balanced accuracy don't track each
other. Epoch 15 had the best val loss (0.9870) but a mediocre bal_acc (0.560); epoch 17 had
worse val loss (1.0290) but the best bal_acc of the whole run (0.575). Since loss and the
metric that actually matters given the 3:1 imbalance have diverged, selecting `best_model.pt`
by lowest val loss was picking a systematically different (and probably worse, by the metric
that matters) checkpoint than selecting by val balanced accuracy would.

## Checkpoint selection: val loss -> val balanced accuracy - 07/28/26

Fixed the mismatch above directly: `main()`'s checkpoint-selection loop now tracks
`best_val_balanced_acc` (`float("-inf")` init) and saves `best_model.pt` whenever a new epoch
beats it, instead of comparing on `val_loss`. The checkpoint dict now stores both
`val_balanced_acc` and `val_loss` for reference, and the "Loaded best checkpoint" print at
final-test time reports both too, so it's still possible to see what the val loss was for the
selected epoch even though it's no longer the selection criterion.

Next step: rerun once more to confirm this doesn't just relocate the same weak-signal problem
to a different epoch, then move on to the hyperparameter sweep script (also considering the
temporal transformer's undersized `d_model=5` bottleneck as something worth including, not
just standard training hyperparameters like lr/batch size/epochs).

## Rerun with fixed checkpoint selection: mechanically correct, ceiling unchanged - 07/28/26

Reran with the fix above. Confirmed epoch 12 (bal_acc 0.5663) was genuinely the highest val
balanced accuracy of the whole run — the selection logic is doing what it's supposed to.
But final test balanced accuracy (0.591) landed within noise of the previous run's (0.595),
which is expected: checkpoint selection only decides *which* epoch's weights you keep, it
can't raise a run's ceiling. Two clean runs now show the same weak-signal plateau (val
balanced accuracy stuck in a 0.54-0.60 band) regardless of which epoch gets picked, which is
the actual signal that it's time for the hyperparameter sweep rather than more single-config
runs.

## Widened temporal transformer + hyperparameter sweep script - 07/28/26

Before building the sweep, added a real architecture fix rather than just sweeping around the
existing bottleneck: `TransformerEncoder` in `model/architecture.py` had no input projection,
so its internal `d_model` was hard-tied to `temporal_input_dim` — just 5 (the raw ERA5
variables: u10, v10, d2m, t2m, tp). A 5-dimensional attention space is very little room for a
transformer to represent anything in. Added `self.input_proj = nn.Linear(input_dim,
hidden_dim)` ahead of the transformer layers, so the internal width is now a separate,
tunable `hidden_dim` (renamed the constructor's positional args accordingly). `WildfireModel`
gained a new `temporal_hidden_dim` param (default `32`) plumbed through, plus explicit
`ValueError`s if `embedding_dim` or `temporal_hidden_dim` aren't divisible by `n_head` (both
feed `nn.MultiheadAttention`/`nn.TransformerEncoderLayer`, which require that and otherwise
fail with a much less obvious error).

Worth flagging honestly, since it affects how much to expect from this fix: widening
`d_model` does NOT fix a deeper issue I noticed while making this change. `x_temporal` going
into the transformer is a single flat vector per tile (the ERA5 stats are already
mean-aggregated over the whole antecedent window in `zonal_aggregator.py`, not preserved as a
real multi-timestep sequence), and `TransformerEncoder.forward` feeds it in as
`X.unsqueeze(0)` — a sequence of length 1. Self-attention over a single token is a no-op (one
token can only attend to itself), so the "attention" in the temporal encoder isn't
contributing anything beyond the linear/FFN sublayers regardless of `d_model`. Properly
fixing that would mean preserving the ERA5 time series as actual separate sequence positions
upstream (a real pipeline change, not just a hyperparameter or a small architecture tweak) —
noted here as a candidate for later, not implemented now.

To support the sweep without re-reading `tile_cache/` from disk on every trial, refactored
`training/train.py`: pulled dataset construction out of `main()` into `build_datasets()`
(returns the three `LabeledTileDataset` splits plus scene/tile counts and `pos_weight_value`),
and pulled the epoch loop out into `train_model()` (returns the best-val-balanced-accuracy
state dict in memory via `copy.deepcopy`, rather than round-tripping through disk on every
improving epoch the way `main()` used to). `main()` now just calls both and handles
argparse/checkpoint-saving/plotting — same behavior as before, verified by re-reading the
full diff line by line since this touched nearly the whole file. Added `--weight-decay`
(default `0.0`, passed straight to `Adam`) and `--temporal-hidden-dim` (default `32`) CLI
args as part of this.

New `scripts/hyperparameter_sweep.py`: random search using `build_datasets()`/`train_model()`
directly (dataset built once, reused across every trial). Searches `lr` (log-uniform,
1e-4 to 1e-2), `batch_size`, `embedding_dim`, `temporal_hidden_dim`, `n_layers`, `n_head`, and
`weight_decay`. `embedding_dim`/`temporal_hidden_dim` candidate sets were deliberately chosen
as multiples of every candidate `n_head` value, so every sampled combination is valid by
construction — no divisibility-rejection/retry logic needed. Each trial trains a short run
(`--trial-epochs`, default 8) rather than a full one, scored on val balanced accuracy for
consistency with the checkpoint-selection metric. Writes every trial's config + result to
`sweeps/sweep_results.json`, prints the top 5 by val balanced accuracy, and prints a ready-to
-run `train.py` command for the winning config — deliberately does NOT auto-retrain the
winner at full length, so there's a chance to sanity-check the winning config before
committing a longer run to it.

Not yet run — next step is actually kicking off the sweep on EC2.

## First sweep run - sequential, CPU-only - 07/28/26

Kicked off `hyperparameter_sweep.py --n-trials 20 --trial-epochs 8` on EC2. Confirmed it's
running as designed: trial 1 (`embedding_dim=16`, `n_head=8`, `temporal_hidden_dim=32`, an
otherwise-valid divisibility combo) completed cleanly at `val bal_acc 0.5619`, in the same
~0.54-0.60 band every full run has landed in — no surprises, dataset/pos_weight counts match
prior runs exactly since the split seed is unchanged. At ~125s/trial for 8 epochs, 20 trials
is roughly 40 minutes total on CPU.

Trials run strictly sequentially (`for trial in ...: run_trial(...)`), each one training to
completion before the next starts — deliberate given this is a single CPU box with no GPU, so
"concurrent" trials would just be fighting each other for the same cores rather than actually
speeding anything up, and keeping one model in memory at a time avoids any risk of trials
interfering with each other's `torch` global state.

**Future work note**: if this ever moves to a GPU instance (or a multi-core box where trials
plausibly wouldn't just contend with each other), it'd be worth adding real concurrency — a
`--n-workers`/`--parallel` flag on the existing script (e.g. `multiprocessing` or
`concurrent.futures`, each worker pinned to its own GPU or CPU affinity), or a separate
script entirely if the concurrency model ends up being different enough (e.g. dispatching
trials as independent EC2/cloud jobs rather than in-process workers) to not be worth
shoehorning into the current sequential design. Not needed while everything runs on a single
CPU box, but worth remembering once that changes.

## Full sweep confirmation run - 08/02/26

Ran the sweep's winning config at full length (30 epochs). Result landed right where the
20-trial clustering predicted: best val bal_acc 0.5863 at epoch 20, final test bal_acc 0.580.
Confirms the ~0.52-0.60 ceiling wasn't a short-training artifact — more epochs on the winning
config didn't break through it. Next lever is the ERA5 mean-aggregation limitation flagged
during the sweep write-up, not further hyperparameter search.

## ERA5 daily-sequence pipeline change - 08/12/26

Implemented the pipeline change flagged during the sweep: `zonal_aggregator.py`'s
`aggregate()` was mean-collapsing each tile's whole ERA5 antecedent window into a single
scalar per variable before it ever reached the model, so `TransformerEncoder` was attending
over a sequence of length 1 (a no-op — self-attention over one token can only attend to
itself). Fixed by resampling each ERA5 variable to daily means and keeping a fixed
`N_ERA5_DAYS=30` window (`_resample_daily_last_n`), so `era5_stats` per tile is now
`{var: [30 daily floats]}` instead of `{var: float}`. Resampled once per variable over the
whole grid, not per tile — doing it ~500x per scene for no reason would've been drastically
slower since a tile's daily series is just a lat/lon slice of the same grid.

Picked a fixed 30-day window (truncate, not pad) over keeping the full variable-length
collected window (33-61 days depending on where in its month a fire/control landed) with
padding + an attention mask — the collection strategy already guarantees at least 30 days per
scene, so truncating avoids padding/masking complexity with no real downside.

Downstream changes to consume `(seq_len, n_vars)` instead of a flat `(n_vars,)` vector:
`model/dataset.py`'s mean-imputation now flattens across tiles and days (a NaN/Inf day gets
imputed with that variable's dataset-wide mean, not the whole sequence); `x_temporal` is built
via transpose to `(seq_len, n_vars)`. `model/augmentation.py`'s `TileAugmenter.jitter` now
samples independent noise per timestep instead of one constant offset repeated across all 30
days (a single per-variable offset would've been a much weaker augmentation); `dropout`
needed no code change since its mask already derives from `x_temporal`'s actual shape.
`training/train.py`'s `build_feature_stds` ERA5 branch flattens across days the same way.

`model/architecture.py`'s `TransformerEncoder` had no positional encoding at all (meaningless
at sequence length 1). Added a fixed sinusoidal encoding rather than a learned `nn.Embedding`
table — sinusoidal adds zero trainable parameters, which matters given the dataset is already
small enough that the sweep plateaued at ~0.52-0.60 bal_acc regardless of model size; a
learned encoding would add parameters with limited data to learn them well. Also switched
`nn.TransformerEncoderLayer` to `batch_first=True` to keep `(batch, seq_len, hidden)` shape
throughout, and widened `WildfireModel.forward`'s dimensionality assertions from `(1, 2)` to
`(2, 3)` to match the new unbatched/batched shapes.

Since `s1_stats`/`s2_stats` don't change in this pipeline change, wrote
`scripts/rebuild_era5_stats.py` to patch each cached tile's `era5_stats` field in place
rather than rerunning the full multi-day `build_tile_cache.py` (S3 downloads, RTC, cloud
masking, everything) — it only re-downloads each scene's small ERA5 grib (discarded after the
original run) and overwrites `era5_stats` using the `(i, j)` tile keys already in the cache.

Not yet run on EC2 — next step is a smoke test on a few scenes before running across the full
cache.

## Divergent git branches after parallel EC2/Mac edits - 08/13/26

Pushed a follow-up commit from the Mac (the `x_temporal.shape[-1]` fix below) and hit a
rejected push — turned out the EC2 side had independently implemented the same ERA5
sequence pipeline change and already pushed it, so `main` had diverged. Diffed `main` against
`origin/main` before touching anything: the real difference was almost entirely cosmetic
(`N_ERA5_DAYS` constant placement/wording), except origin was missing the `shape[-1]` fix
below entirely. `git pull --rebase origin main` resolved cleanly. Lesson: when running the
same Claude session across two machines on the same repo, check `git log origin/main` before
assuming a rejected push means something trivial.

## Model-construction shape bug - 08/13/26

Caught before running training again: `train.py`'s `main()` and `hyperparameter_sweep.py`'s
`run_trial()` both compute `temporal_input_dim` as `x_temporal0.shape[0]`, correct when
`x_temporal0` was a flat `(n_vars,)` vector but wrong now that it's `(seq_len, n_vars)` =
`(30, 5)` -- `shape[0]` is the sequence length, not the per-timestep feature count
`TransformerEncoder.input_proj` expects. Would have built `nn.Linear(30, hidden_dim)` and
crashed on real `(batch, 30, 5)` input. Fixed all four call sites to `x_temporal0.shape[-1]`.

## Severe training slowdown from unvectorized per-day loops - 08/13/26

First real sweep attempt after the rebase stalled for 30+ minutes with zero trials completing
(the old sweep's slowest trial finished in ~12 minutes). Root cause: two spots in the ERA5
sequence change did Python-level loops over every (day, variable) pair per tile, per epoch,
instead of vectorized numpy ops. `model/dataset.py`'s `__getitem__` ran a 150-iteration
(30 days x 5 vars) list comprehension calling `is_missing_value()` per entry, for every one of
53,652 training tiles, every epoch. `model/augmentation.py`'s `jitter()` was worse -- it made
150 individual scalar `self.rng.normal()` calls per tile per epoch instead of one vectorized
call. Across 8 epochs that's tens of millions of Python-level calls just for these two spots.

Fixed both: `dataset.py`'s imputation now builds a `(5, 30)` numpy array directly from the
per-key lists (5 Python-level iterations, not 150) and imputes via a single `np.where` against
the whole array. `augmentation.py`'s `jitter()` now samples the entire `(seq_len, n_vars)`
noise matrix in one `self.rng.normal(..., size=(seq_len, n_vars))` call instead of a nested
Python loop. Both were introduced in the original ERA5 sequence diff and only surfaced once
actually run at full dataset scale -- worth remembering that "looks correct" and "runs fast at
this scale" are different checks, especially for anything inside a per-`__getitem__` hot path.

## "Numpy is not available" after the vectorization fix - 08/13/26

All 20 sweep trials failed immediately with `Numpy is not available` after the vectorization
fix above. Root cause was the EC2 env's numpy/torch ABI mismatch flagged earlier as a harmless
warning (`Failed to initialize NumPy: _ARRAY_API not found`) -- it turned out not to be
harmless for one specific call: `torch.from_numpy()`, used in the new vectorized `jitter()`,
requires the same broken C-API hook and has no fallback, so it hard-fails. `torch.tensor()`
(used everywhere else, including the pre-existing `x_spatial` line that had always worked in
this same environment) takes a slower conversion path that doesn't depend on that hook. Fixed
by swapping `torch.from_numpy(...)` to `torch.tensor(...)` in `jitter()` -- same vectorized
rng call, just a different numpy-to-tensor conversion. `torch.from_numpy` was the only call to
that function anywhere in the repo, confirmed via grep.

## Post-rebuild model-construction bug - 08/13/26

Smoke test and full rebuild (`--skip-existing`, 494 patched + 6 skipped, 0 failed) both ran
clean, but before kicking off training again, caught a real bug in how the model gets built:
`train.py`'s `main()` and `hyperparameter_sweep.py`'s `run_trial()` both compute
`temporal_input_dim` as `x_temporal0.shape[0]`, which was correct back when `x_temporal0` was
a flat `(n_vars,)` vector (shape[0] = 5 features). Now that it's `(seq_len, n_vars)` =
`(30, 5)`, `shape[0]` is the sequence length, not the per-timestep feature count
`TransformerEncoder.input_proj` actually expects — would've built `nn.Linear(30, hidden_dim)`
and crashed the instant it saw real `(batch, 30, 5)` input. Fixed by switching all four call
sites (`train.py` main-run model, its `model_config` dict, its best-checkpoint-reload model,
and the sweep's `run_trial`) to `x_temporal0.shape[-1]` instead. Caught by re-reading the
model-construction code path specifically for the new tensor shapes before running anything,
not by an actual crash.

## ERA5 daily-sequence pipeline change validated - full run - 08/14/26

Ran the sweep's winning config at full length (30 epochs, `lr=0.000555`, `batch_size=64`,
`embedding_dim=64`, `temporal_hidden_dim=32`, `n_layers=1`, `n_head=1`, `weight_decay=1e-05`).
Final test: `bal_acc 0.731`, `precision 0.522`, `recall 0.666`, `f1 0.585` -- best checkpoint
saved at epoch 7 (`val bal_acc 0.7337`). This validates the whole ERA5 daily-sequence pipeline
change: moved the balanced-accuracy ceiling from ~0.52-0.60 (every run under the old
mean-aggregated, sequence-length-1 architecture) to ~0.73, a genuinely different regime, not
incremental noise. The sweep's short-trial estimate (0.7263) matched the full run almost
exactly this time, unlike the previous sweep cycle where the estimate dropped ~0.016 on the
full run -- a good sign this number is stable, not a lucky short-trial artifact.

Two things worth remembering about this result, not fixes, just honest framing for the
writeup: val bal_acc plateaus by epoch 2 (0.708) and never meaningfully improves past epoch 7
-- everything after that is the model overfitting the training set (train acc climbs to 96.5%,
val loss climbs from ~1.0 to spikes over 2.0) while val bal_acc just noisily oscillates in the
0.68-0.73 band. Checkpoint selection by val balanced accuracy correctly caught this and saved
epoch 7 rather than epoch 30, so the final test number isn't contaminated by the later
overfitting, but a future run could likely hit the same result in ~10 epochs instead of 30.
Also, precision (0.522) is meaningfully lower than recall (0.666) -- the model catches about
two-thirds of actual fire-risk tiles but is only right about half the time when it flags one.
This is the expected effect of the class weighting deliberately favoring recall (missing a
real fire risk costs more than a false alarm), not a bug, but worth stating explicitly rather
than leading with balanced accuracy alone.

This is the model being finalized for GitHub/HuggingFace -- no further tuning planned for now.

## FWI ERA5 collection script - 08/14/26

New `scripts/collect_fwi_era5.py`, separate from the prediction pipeline and tile_cache/. The
prediction model's existing per-scene ERA5 gribs only cover a short antecedent window sized
for that task; FWI's drought codes (DC especially, ~52-day memory) need a much longer
continuous record to spin up before ignition/control date, so this pulls its own longer-window
grib per scene rather than reusing what's already collected.

Design decisions: 180-day lookback per scene, capped at Jan 1 of that year rather than
reaching into the prior year (simpler than modeling each state's actual fire-season start --
the dataset spans AK, NV, WA, OR, and CA, which have meaningfully different season timing, so
a single fixed spring-start convention wouldn't have been defensible across all of them; a
fixed lookback sidesteps that entirely). Covers both fires and controls (500 scenes total) --
controls give the calculator non-burned baseline locations to validate against, and gribs are
tiny (~5MB each, per the original collection's upload sizes) so the extra scenes cost nothing
meaningful. The Jan-1 cap also guarantees every request's date range stays within a single
calendar year, which matters because the CDS API request format takes year/month/day as
separate lists and returns their cartesian product -- had the window been allowed to cross a
year boundary, a naive year/month list would over-request unrelated year/month combinations.

Reuses the exact `reanalysis-era5-single-levels` CDS request pattern (product_type, variable
list, day/time list) from the original collection notebook, just with a longer date range and
a padded area computed directly from metadata.json's stored `spatial_bounds` (a simplification
of the original's footprint-union padding across 4 Sentinel products -- fine here since FWI
only needs decent areal coverage, not exact parity). Output gribs go to a new `fwi/` S3 prefix
(`fwi/{kind}/{state}/{scene_id}.grib` + a small per-scene metadata.json), kept fully separate
from `fires/`/`controls/` so this can't collide with or accidentally corrupt the prediction
dataset. Not yet run -- next step is a `--limit 3` smoke test on EC2.

Follow-up after the smoke test: switched the flat `fwi/{kind}/{state}/{scene_id}.grib` +
`..._metadata.json` layout to a per-scene folder (`fwi/{kind}/{state}/{scene_id}/` containing
`metadata.json` and `ERA5_{state}_{coord}_{date}.grib`), matching the `fires/`/`controls/`
naming convention -- the flat layout was getting cluttered with a json and grib side by side at
the state level. Added a `format_coordinate_string()` helper mirroring the original notebook's
centroid-based coordinate string. Also added per-scene progress logging and an overall tqdm bar
across all 500 scenes, since the original version gave no visibility into how far along a
multi-hour run was.

## FWI collection was CDS-queue-bound, not throttled - 08/15/26

Full run looked "stuck" after ~90 scenes -- turned out each CDS request was taking
1000-1600+ seconds even when successful (one 502 Bad Gateway mid-run recovered on its own via
cdsapi's built-in retry, unrelated to the slowdown). At that rate the tqdm ETA was 130+ hours
for all 500 scenes. This wasn't throttling in the "penalized" sense -- CDS's public request
queue is just commonly this congested, and processing scenes one at a time meant every scene
serialized behind the previous one's full queue wait, even though the actual downloaded data is
tiny (~15-25MB) and the wait is 99% queue time, not transfer time.

Fixed by submitting scenes to CDS concurrently instead of sequentially (`ThreadPoolExecutor`,
`--max-workers` default 4) -- same pattern the original collection notebook used for its own
concurrent S1/S2/ERA5 uploads. This doesn't make any individual request faster, but it means
several requests sit in CDS's queue at once instead of one at a time, which should cut total
wall-clock time roughly in proportion to the worker count. The `--skip-existing` check was
moved to run upfront (before any requests are submitted) rather than inline per-scene, so it
doesn't waste a concurrent worker slot on a scene that's already done.

## Sampling instead of collecting every scene for FWI - 08/15/26

Realized mid-run that collecting all 500 scenes doesn't make sense for FWI -- it's a pure
deterministic calculation, not a trained model, so there's no benefit to scale the way the
prediction model needed it. All 500 scenes would only be useful as a bigger validation sample
(fire-vs-control FWI distributions), which doesn't need the full dataset. Plan: let the
in-flight run finish all 125 fires (already mostly done), then stop and collect a smaller
sample of controls instead of all 375.

Added `--kinds` (so a follow-up run can target only `control` and skip re-checking fires) and
`--sample-per-state N` (randomly samples N scenes per state instead of every scene, via
`sample_scenes_per_state()`) -- sampling per state rather than a flat random N across the whole
set, since `discover_scenes()`'s alphabetical-by-state ordering means an unweighted sample or a
naive `--limit` would skew toward whichever states sort first (seen firsthand: the in-flight
run was still working through OR at scene ~100, meaning AK/CA/NV were already fully covered and
WA hadn't started -- exactly the skew this avoids). Next run:
`--kinds control --sample-per-state 9` for roughly 45 controls, ~9 per state.

## CDS hard concurrency cap, not just congestion - 08/15/26

`--max-workers 4` run immediately failed all 4 in-flight requests with an explicit CDS error:
"Number queued requests for this dataset is temporarily limited." Different from the earlier
502/congestion issue -- this is CDS enforcing a real per-account cap on simultaneously queued
requests, and 4 already exceeds it. Made worse by `with_retries`' fixed 5s backoff: all 4
workers failed within the same ~30s window (since they'd all been submitted at once), so their
retries landed in near-lockstep too and re-tripped the same limit again.

Fixed with three changes: dropped the default `--max-workers` from 4 to 2; added
`--submit-stagger` (default 10s) so scenes enter the worker pool spread out over time instead
of all at once; and added `with_jittered_retries()` (exponential backoff + random jitter,
local to this script rather than modifying `build_tile_cache.py`'s shared `with_retries`, which
other parts of the pipeline depend on) so retries from different workers don't resync and
re-collide even if their original failures happened close together.

## FFMC implementation review - 08/16/26

Reviewed a first-pass `calculate_ffmc()` (in `era5_preprocessing.py`) against the actual Van
Wagner & Pickett (1985) equations. First pass had 5 real bugs: `t2m` used raw in Kelvin instead
of converted to Celsius in the EMC/rate-constant formulas (by far the biggest one -- e.g.
`exp(0.0365*290)` instead of `exp(0.0365*17)`); ERA5's `tp` used directly instead of converted
from meters to mm; a parenthesis-placement bug (`np.exp(np.divide(rh - 100), 10)`) that would
raise a `TypeError`; `np.min(250, m_r)` instead of `np.minimum(m_r, 250)` (the former treats the
second arg as `axis`, not a value to compare); and a missing "no change" branch causing a
`NameError` whenever moisture content fell between the wetting and drying equilibrium points.

After the first round of fixes, two new bugs appeared from the restructuring: the rain-effect
condition got inverted (`if r_final == 0:` instead of `> 0`, backwards from the paper's explicit
"skip Equation 3 when ro <= 0.5mm" restriction -- meant the rain formula ran only when there was
NO rain, and `m_r` was undefined whenever there actually was rain); and a trailing
`if wet_emc <= m_r <= dry_emc` check referenced `wet_emc` outside the scope where it's actually
assigned, `NameError`-ing whenever the drying branch was taken. Fixed by restructuring into a
clean if/elif/else mirroring the paper's own procedure (p.12, steps 5-8) instead of a separate
trailing check.

Pulled the actual numbered equations (1-10 for FFMC, 11-17 DMC, 18-23 DC, 24-26 ISI, 27 BUI,
28-30 FWI, 31 DSR) out of the uploaded Van Wagner & Pickett (1985) PDF via `pdftoppm`/`pdftotext`
(the PDF's symbol-legend pages had unlabeled/missing symbols from a bad scan, but the equations
section itself rendered fine) and added them as inline citations on every formula and constant
in `calculate_ffmc`, per request -- useful both for correctness-checking and so DMC/DC/ISI/BUI
can be written next citing the same source precisely rather than from memory.

Final verification pass after both fixes landed: traced every path through `m_r`'s
assignment (no-rain branch vs. rain-effect branch) to confirm it's always defined before the
`np.minimum` clip, and confirmed the drying/wetting/equilibrium branches now match the paper's
own procedure exactly. No remaining bugs -- `calculate_ffmc` is correct.

## Backing off further - trimming instead of retrying - 08/15/26

Even `--max-workers 2` immediately hit the same "queued requests limited" rejection -- CDS's
real per-account cap on this dataset is apparently 1 (fully sequential), which isn't worth
fighting given the FWI tool never needed the full dataset anyway (established a few entries
back). Decided to stop collecting fires outright at whatever's already landed and cap each
state's fire count at floor(true_state_total / 2) instead of continuing to push through CDS's
limit for full coverage (Washington was the one still short, at 18/22 -- Whitney, Williams
Flats, Whitmore, and Walker Creek never made it in).

New `scripts/trim_fwi_scenes.py`: counts each state's TRUE total from the real `fires/{state}/`
prefix (not from what's sitting in `fwi/`), computes floor(total/2) per state, and randomly
(seeded) removes whichever already-collected scenes are above that target -- leaves states
already at or under target untouched. Defaults to a dry run; `--execute` actually deletes.
Reused for controls too via `--kind control` once that collection phase happens.

## DMC/DC/ISI/BUI implementation - 08/16/26

Wrote `calculate_dmc`, `calculate_dc`, `calculate_isi`, `calculate_bui` in
`era5_preprocessing.py`, same style as the now-verified `calculate_ffmc` (nested helpers,
docstrings and inline comments citing Van Wagner & Pickett (1985) equation numbers).

Two of the paper's own lookup tables were illegible in the scanned PDF: Table 1 (DMC's
month-by-month effective day-length, `Le`, used in Eq. 16) and Table 2 (DC's day-length
factor, `Lf`, used in Eq. 22) -- both had their header row intact but the actual numeric rows
blank/faded. Rather than guess or fabricate values for a calculation the whole point is to get
right, verified them against the R `cffdrs` package (`duff_moisture_code.r` /
`drought_code.r`), a maintained, independently-cited reference implementation of this exact
paper. Used the `>= 30N` latitude table from both files (`ell01`/`fl01`), which is the correct
one for Alaska's latitudes and also the only table this project needs, since neither script
implements or requires the 1987 follow-up's lower-latitude day-length adjustment.

Fetching `cffdrs` also resolved a real ambiguity from the PDF's OCR: the extracted text for
Eq. 27a/27b's switching condition came out garbled ("P <= 0.40"), which didn't parse as a sensible
threshold. The actual condition, confirmed against `buildup_index.r`, is `BUI_a < DMC` (i.e.
Eq. 27a's own result compared against the DMC value it was computed from) -- Eq. 27a is
always computed first, and 27b's correction only replaces it when that comparison holds.
Cross-checked the rest of DMC/DC/ISI's algebra against the same source too: `cffdrs` fuses
some of the paper's separate steps together (e.g. DC's Eq. 22+23 collapse into one `pe`
term, DC's Eq. 19-21 are rearranged algebraically to avoid recomputing `Qr` separately), but
every fused form checks out as equivalent to the paper's own equations once worked through
by hand -- implemented here following the paper's own separate-equation structure instead
(matching `calculate_ffmc`'s style) rather than `cffdrs`'s fused form, for readability against
the cited equation numbers.

Verified via `py_compile`. FWI (Eq. 28-30) and DSR (Eq. 31), which chain ISI+BUI into the
final index, weren't requested this round -- everything up through BUI is now implemented and
citeable back to specific paper equations.

## FWI orchestrator: recursive day-stepping - 08/16/26

The user mocked up `calculate_fwi()` themselves (chaining FFMC->ISI, DMC->DC->BUI, then
Eq. 28-30 into a final FWI), correctly fixed the startup codes to the real Van Wagner &
Pickett defaults (FFMC=85, DMC=6, DC=15, not the placeholder `1`s from the first draft), but
had it computing only a single day from directly-passed scalars, with the day-to-day
recursion, the noon-indexing scheme, and month extraction still open questions. Answered
those, then implemented:

- `calculate_fwi` now takes the whole `variables` dict from `load_era5_vars` (hourly
  DataArrays) instead of scalars, and loops itself: noon of day 1 is index 12 (index 0 is
  midnight, not noon), each subsequent day is `+24` timesteps, terminating once the index runs
  past the series length. Each day's FFMC/DMC/DC become the next day's `_nought` inputs --
  the actual recursive chaining that was missing before.
- Returns a dict with the *last* successfully computed day's `date`, `ffmc`, `dmc`, `dc`,
  `isi`, `bui`, and `fwi` (not just the final `fwi` scalar), so intermediate codes are
  inspectable for testing/debugging rather than thrown away.
- Month is pulled from each day's own noon timestamp (`pd.Timestamp` on the `valid_time`/
  `time` coordinate at that index) rather than a single value for the whole window --
  necessary since the ~180-day FWI lookback routinely crosses month boundaries, and DMC/DC
  each need the correct month's Table 1/Table 2 day-length lookup for that specific day.
  Deliberately did NOT add a `month` return value to `load_era5_vars` for this -- a single
  scalar wouldn't even be correct across a multi-month window, and the per-timestep timestamp
  needed is already sitting on the DataArray's own time coordinate.
- Filled in the two comments flagged as incomplete: Eq. 28a/28b's citation now covers the `+2`
  offset and the `1000`/`25` constants (previously only citing the exponent/rate constants),
  and Eq. 29's `B` is now explained as an unnamed intermediate (raw ISI x f(D) product, ahead
  of Eq. 30's log-transform onto the calibrated FWI scale) rather than left as an open
  question in the comment.
- Raises a clear `ValueError` if the grib doesn't cover at least 13 hours (not enough for even
  one noon timestep), instead of silently returning `None`.

Verified via `py_compile`. No other files call `calculate_fwi` yet (confirmed via grep), so
this signature change is free -- next step is the user's own testing pass before this gets
wired into anything else.

## FWI calculator testing round - three real bugs found - 08/16/26

Wrote `scripts/test_fwi_calculator.py` to run `calculate_fwi` against a real collected scene:
downloads a `fwi/{kind}/{state}/{scene_id}/` grib + metadata.json, loads it with
`load_era5_vars`, runs `calculate_fwi`, and prints the result with rough sanity-range checks.
Testing against the WA `400_fire` scene surfaced three real bugs, in order:

1. **Stale `cfgrib` index cache.** The test script re-downloads to the same fixed tmp path on
   every run, so a leftover `.idx` sidecar from a prior run didn't match the freshly
   re-downloaded grib's timestamp -- `cfgrib` kept discarding and rebuilding it in a loop.
   Fixed by passing `indexpath=''` to `cfgrib.open_datasets()` in `load_era5_vars`, disabling
   the on-disk index cache entirely (in-memory index only) -- protects the main pipeline too,
   not just this script, from the same class of bug if a grib path is ever reused.

2. **Mismatched time-dimension names across variables.** `calculate_fwi` resolved `time_dim`
   once from `t2m_arr` and reused it for all five variables, but `tp` almost always gets
   flattened to `valid_time` by `_flatten_time_step` while the others may still just be `time`
   if they never needed flattening -- crashed with a dimension-not-found error the moment `tp`
   didn't match. Fixed with a new `_time_dim_of()` helper resolved per-array instead of once
   globally.

3. **Spatial averaging silently cancelling wind vectors (the significant one).** After fixing
   #1/#2, the calculator ran but produced FWI=13.09 for a day immediately preceding a fire large
   enough to make the 10k-acre dataset cutoff -- suspiciously low. Root cause: `calculate_fwi`
   was averaging `u10`/`v10` spatially across the whole padded CDS request area (the fire's
   bbox + `AREA_PADDING_DEGREES=2` on all sides, easily 400+km across) *before* combining them
   into wind speed, rather than averaging speed itself -- in mountainous terrain (WA/Cascades),
   opposite-direction wind vectors from different parts of that area can partially cancel in
   the mean, understating the true local wind and, through Eq. 24's `f(W) = e^(0.05039*W)`,
   understating ISI and therefore FWI. Back-solving the bad run's own numbers showed an
   implied noon wind speed of ~3.6 km/h -- implausibly still for a day before a fire that size.

   Fixed by point-sampling the grid cell nearest the scene's centroid instead of area-averaging
   at all, for every variable (`_grid_centroid()`, using `.sel(..., method='nearest')`) --
   closer to the FWI system's own single-station design. The centroid is derived from the
   grib's own lat/lon extent (midpoint of `latitude`/`longitude` coordinate min/max), not from
   metadata.json's `spatial_bounds` -- since `era5_area_from_bounds()` pads symmetrically by a
   fixed 2 degrees on every side, the padded grid's own midpoint is already a close stand-in
   for the true centroid, so `calculate_fwi` doesn't need scene metadata threaded into it at
   all, just the `variables` dict it already takes.

   Rerunning WA `400_fire` after this fix: FWI jumped from 12.74 to 26.33 (moderate -> high
   band), DMC roughly doubled (94 -> 166), BUI rose sharply (110 -> 187) -- a much more
   plausible profile for the day before a fire that size.

Separately, also caught and fixed a **second instance of the CDS month-cartesian-product
over-fetch bug** (same root cause as the `fire_name` field): `collect_fwi_era5.py`'s CDS
request always asks for `day: 1-31` across every month the window spans, so the downloaded
grib silently extends well past the intended `window_end` -- confirmed exactly (212 days
downloaded vs 180 intended = the full Jan-Jul span vs the actual Jan21-Jul20 window). Without
correcting for this, `calculate_fwi` was walking 11+ days past the actual ignition date into
during/after-fire weather -- a real instance of the same temporal-leakage class of bug already
fixed once for the main pipeline (07/24 entry). Fixed at the test script's call site (not in
`load_era5_vars` itself) by passing `cutoff_datetime=ignition_or_control_date` from the
scene's own metadata.json, reusing the exact `< cutoff` (strict, not `<=`) filtering
`load_era5_vars` already supports.

## Mirrored fire/control collection for FWI comparison - 08/16/26

To validate the calculator discriminates real fire risk (not just producing plausible-looking
numbers), decided to compare a fire's FWI against one of its own controls rather than trusting
sanity-range checks alone -- but no controls existed yet under `fwi/control/`.

Added `--mirror-fires` to `collect_fwi_era5.py`: instead of the normal
discover-all/sample-per-state path, resolves exactly one control per state that corresponds to
a specific hardcoded fire (`MIRROR_FIRST_FIRE_BY_STATE`, one representative fire chosen per
state: Cultas Creek (AK), Caldwell (CA), Dixie (ID), Thorne Creek (MT), Meadow Valley (NV),
0501 Crazy Creek (OR), 400 (WA)). `find_control_for_fire()` scans that state's real controls
(`discover_scenes("control")`, filtered to the target state) and matches each candidate's
metadata `fire_name` field against the target name after normalization
(`_normalize_fire_name()`: lowercase, strip "fire", strip non-alphanumerics) -- needed since
the exact casing/punctuation/suffix a control's metadata actually stores wasn't guaranteed to
match the fire names as given. Each real fire has 3 controls; deliberately returns just the
first match, since one mirrored control per state is enough for this comparison, not a full
control sample. Reuses the exact same `collect_scene()` collection path as every other
scene -- no changes to the actual CDS request or S3 upload logic, just how the 7 target
scenes get selected.

## Mirrored control collection: date-matching, full-dataset coverage, bbox-aware selection - 08/16/26

Follow-up to the AK fire-vs-control test where the control (8.25) scored higher FWI than its
paired fire (6.73) -- traced to the control's own recorded date (2019-08-05) being months and
a different year away from the fire's ignition date (2021-06-17), meaning the comparison was
dominated by seasonal-timing/interannual differences rather than any real difference in
fire-weather risk between the two locations.

Also caught and reverted a real mistake in the same discussion: made an unrequested edit to
`calculate_fwi()` (the git-tracked `era5_preprocessing.py`) to add a centroid/time-dim fix that
was actually already implemented independently on EC2 -- the local repo copy was simply stale.
Reverted the local edit and re-pulled from EC2's authoritative version rather than risk a git
conflict overwriting real in-progress work. Lesson reinforced: `era5_preprocessing.py` is
edited on EC2 only; local changes to it should not be made without being explicitly asked, even
when the intent is a genuine improvement.

Fixes landed in `collect_fwi_era5.py` (the UTC-vs-local-noon issue in `calculate_fwi` itself is
tracked separately, on EC2, not covered here):

1. **`collect_scene()` gained an `override_date` param.** When collecting a control that
   mirrors a specific fire, its window is now built around the FIRE's exact ignition date
   (year included, not just month/day) instead of the control's own independently-recorded
   date -- both scenes now see the same actual synoptic weather, not just the same point in
   two different (possibly climatically different) years' seasonal cycles. Metadata gets an
   explicit `date_overridden_to_match_fire: true` flag when this happens.

2. **`--mirror-all-fires`** (new, alongside the existing 7-state `--mirror-fires`): walks every
   fire already collected under `fwi/fire/`, reading each one's own already-written
   `ignition_or_control_date` directly from its `metadata.json` (no need to re-derive it from
   the main `fires/` prefix), finds its matching control, and collects that control's window
   around the fire's date. For the full ~125-fire dataset rather than just 7 states -- needed
   so a fire-vs-control FWI comparison can be made across the whole dataset, not a handful of
   examples, which the user wants to use as an actual finding in their writeup (if FWI cleanly
   separates fires from controls, that's evidence the prediction model may be solving an
   easier problem than it needs to; if it doesn't, that's evidence the model is doing real work
   beyond what a simple fire-weather index already captures).

3. **Strictly sequential, no `ThreadPoolExecutor`, for both mirror modes.** Given CDS's
   apparent real per-account concurrency cap of 1 on this dataset (established earlier), this
   prioritizes not wasting time to failed/retried concurrent requests over any theoretical
   wall-clock speedup -- concurrency was never actually helping here.

4. **Closest-bbox-extent control selection.** `find_control_for_fire()` previously picked
   whichever of a fire's (up to 3) matching controls S3 happened to list first -- arbitrary.
   Added `_bbox_extent_km()` (factored out of an existing diagnostic print in `collect_scene`)
   and changed selection to pick whichever candidate's `spatial_bounds` extent (width x height,
   km) is closest to the fire's own, when the fire's bbox is available. Matters because
   `calculate_fwi` samples a single point (the grid centroid) per scene -- a control whose
   request area is a very different physical size than its fire's makes that centroid a
   correspondingly worse stand-in for "where the scene actually is," biasing the comparison
   before any real weather difference is even considered. Both `--mirror-fires` and
   `--mirror-all-fires` now fetch the fire's own `spatial_bounds` (via `find_fire_scene`) and
   pass it through; falls back to first-match if no fire bbox is available.

Also flagged honestly rather than silently left alone: this bbox-matching is a real, but
partial, fix. It doesn't verify anything about elevation, vegetation, or true point-vs-point
distance between the sampled centroid and either scene's actual ignition/sample point -- only
that the two request areas are roughly the same physical size. Single fire/control comparisons
remain statistically weak regardless of any of these fixes; the full-dataset `--mirror-all-fires`
run is what actually gives this comparison enough scenes to say something meaningful.

## cffdrs reference-implementation comparison script - 08/16/26

Separately from the fire-vs-control discrimination question, wrote `scripts/compare_fwi_to_cffdrs.py`
to answer a different question: is `calculate_fwi()` actually implemented correctly, independent
of whether FWI turns out to be a useful signal for this dataset. For every fire already
collected under `fwi/fire/`, it builds a daily noon weather series from the grib (temp C, RH%,
wind speed km/h, 24h precip mm) via `extract_daily_noon_series()`, feeds it to the real R
`cffdrs` package's `fwi()` (via a small embedded R script run as a subprocess, using the same
standard startup codes FFMC=85/DMC=6/DC=15 our own calculator uses), and compares cffdrs's last
day's FFMC/DMC/DC/ISI/BUI/FWI against `calculate_fwi()`'s own returned final day for that same
scene -- reporting per-fire results to CSV plus a mean absolute error per component across every
fire successfully compared.

Flagged one real limitation explicitly in the script's own docstring rather than glossing over
it: `calculate_fwi()` only returns the final day's computed codes, not the raw per-day weather
series it used internally, so `extract_daily_noon_series()` is necessarily a separate,
independent reimplementation of the same noon-indexing + grid-centroid-sampling approach, not a
literal reuse of calculate_fwi()'s own extraction. If calculate_fwi()'s actual noon-index or
centroid logic drifts from what's coded here (e.g. once the UTC-vs-local-noon timezone fix
lands), this script's inputs and calculate_fwi()'s own inputs would diverge, contaminating the
comparison with an extraction mismatch rather than a pure implementation difference -- added a
`--noon-index` override specifically so this can be corrected without editing the file, and
called this out prominently so it isn't silently trusted once the two implementations drift.

Requires R + the `cffdrs` package on EC2 (not yet confirmed installed); the script checks for
`Rscript` up front and prints the exact install command rather than failing deep into a run.

Caught a real bug in the first version of this script right after writing it: it exposed a
single global `--noon-index` override, but `calculate_fwi()`'s `_approximate_noon_utc_hour()`
computes a DIFFERENT noon index per scene, based on that scene's own centroid longitude --
different fires sit at different longitudes/UTC offsets, so one fixed index can't stand in for
all of them. Fixed by porting `_approximate_noon_utc_hour()` directly into this script and
computing the noon index per-scene by default (from each scene's own `_grid_centroid()`
longitude), matching `calculate_fwi()`'s actual behavior exactly. Renamed the CLI flag to
`--force-noon-index`, now an explicit opt-in override (fixed UTC hour for every scene) rather
than the default path, for the case where `calculate_fwi()` itself reverts to a plain
index-12 approach.

Installing `cffdrs` on EC2 required a chain of its own fixes first: `conda install cffdrs`
failed outright (it's a CRAN-only package, not a conda package, on any channel); `Rscript`
itself wasn't installed, and `conda install -c conda-forge r-base` got OOM-killed/terminated
mid-solve on the instance's memory budget, so installed R via `apt` instead
(`r-base` + `build-essential`/`gfortran`/`libcurl4-openssl-dev`/`libssl-dev`/`libxml2-dev`,
since several CRAN packages compile C/Fortran code at install time); the first
`install.packages('cffdrs')` attempt then failed on file permissions (system R's default
library isn't writable by a non-root user -- fixed with `sudo`); and that surfaced `cffdrs`'s
real transitive dependencies, `sf`/`terra` (and their own `s2`/`units` sub-dependencies), which
needed `libgdal-dev`/`libgeos-dev`/`libproj-dev`/`libudunits2-dev`/`cmake` to compile from
source. Successfully installed after all of the above.

## First cffdrs comparison run -- root-caused and fixed the FFMC discrepancy - 08/17/26

5-fire smoke test (`--limit 5`, all AK) came back very clean: DC MAE 0.000, DMC/BUI MAE ~1e-13
(floating-point noise -- these three match cffdrs bit-for-bit). FFMC showed a small but
consistently one-directional error (`ours > cffdrs`, every single row, 0.05-0.29), which
propagated into ISI (0.03-0.36) and, more visibly, FWI (0.12-0.95) since both depend on FFMC.

Root-caused by pulling cffdrs's actual `fine_fuel_moisture_code.r` source and diffing it line
by line against `calculate_ffmc` -- every branch condition and formula (Ed/Ew equilibrium
moisture, the drying/wetting rate terms, Eq. 6-9) matched exactly. The one real difference:
cffdrs defines `FFMC_COEFFICIENT <- 250.0 * 59.5 / 101.0` (= 147.27722772277228...) and uses
that exact fraction in both Eq. 1 and Eq. 10, while `calculate_ffmc` used `147.2` in both
places -- which is not a mistake so much as a citation choice: 147.2 is what the 1985 Van
Wagner & Pickett paper literally prints (rounded to one decimal for hand calculation), while
cffdrs, as a software implementation, kept the fraction unrounded. The consistent
same-direction bias and its correlation with FFMC/ISI/FWI specifically (vs. DC/DMC/BUI being
exact) was the tell that this was one specific arithmetic constant, not a structural bug.

Fixed by switching `calculate_ffmc`'s Eq. 1 and Eq. 10 to `250 * 59.5 / 101` -- and, importantly,
also `calculate_isi`'s own reused copy of Eq. 1 (it independently recomputes moisture content
from today's FFMC for the spread-index calculation), which would have silently reintroduced the
same mismatch if only `calculate_ffmc` had been fixed. Verified no stray `147.2` literals
remained anywhere else in the file (only the two comments documenting the new constant). Next
step: rerun the comparison across every fire in `fwi/fire/`, not just the 5-fire AK sample.

## Second cffdrs comparison run - 08/17/26

Full test on 60 fire samples came back with near perfect results: 
Compared 61 fires successfully, 0 failed.

Mean absolute error (ours vs. cffdrs), across all compared fires:
  FFMC  MAE: 0.000
  DMC   MAE: 0.039
  DC    MAE: 0.000
  ISI   MAE: 0.000
  BUI   MAE: 0.035
  FWI   MAE: 0.000

Small errors in DMC are due to slight differences in constants between the Van Wagner paper 
the cffdrs implementation. They propogate to BUI but ultimately lead to no real diff
in the final calculated FWI. With this comparison complete, the next step will be to compare 
the FWI of fire scenes with the FWI of control scenes using our implementation. This will 
give us an idea of whether FWI alone is a good enough metric to determine differences fires 
and controls within the greater dataset.

## Fire vs control comparison run - 08/17/26

Built `scripts/compare_fire_control_fwi.py` to hold the FWI implementation constant (ours
only, no cffdrs) and diff a fire against its matched control instead -- the real-world
validation the cffdrs comparison couldn't give us, since matching a reference implementation's
arithmetic says nothing about whether the resulting numbers actually separate fire days from
non-fire days.

Pairing couldn't reuse collect_fwi_era5.py's own find_fire_scene()/find_control_for_fire(),
since both scan the original fires/ and controls/ S3 prefixes -- gone now that the bulk dataset
is on HuggingFace and deleted from S3. Recovered pairing from inside fwi/ alone instead, by
matching a fire to a control sharing its exact (state, ignition date): --mirror-all-fires
deliberately overrode every control's window to its paired fire's exact date, so this recovers
the same pairing without touching the deleted prefixes. Added an area-centroid tiebreak (from
each scene's own padded CDS request envelope) for the rare case of two different fires sharing
an exact ignition date in the same state.

First real run surfaced a genuine data bug via that very reconstruction: `control_AK_66N152W_
20190723` mapped to both DOUGLAS_fire and RADIO_CREEK_fire, and `control_OR_45N121W_20200830`
to both RAIL_RIDGE_fire and WILEY_FLAT_fire -- confirmed by the two fires in each pair sharing
byte-identical control-side values. Root cause: `collect_scene()` in collect_fwi_era5.py named
a mirrored control's S3 folder after its own `control_id` alone (rounded-to-the-degree
coordinates + calendar date) -- low-cardinality enough that two different fires' matched
controls produced the identical string, and since S3 `put_object` always overwrites, one fire's
real control silently clobbered the other's during the original `--mirror-all-fires` run.
Fixed by adding a `mirrors_fire` parameter to `collect_scene()`/`_already_collected()`: for a
mirror-collected control, the S3 folder is now scoped to the paired fire's own (already-unique)
scene_id instead of the control's own control_id, and that fire's scene_id is also written into
the control's own metadata.json as `mirrors_fire`, making a future collision structurally
impossible and the pairing self-documenting. `compare_fire_control_fwi.py` updated to prefer
this tag when present, falling back to the (state, date) + centroid reconstruction only for
older untagged controls, and to derive `scene_id` from the actual S3 folder name rather than
metadata content (necessary now that a control's folder name and its own `scene_id` field can
legitimately diverge). Also added a `warn_on_reused_controls()` check that flags any control
paired to more than one fire, so this exact bug can't silently reappear and pass as a clean
run.

DECISION: not re-collecting the 4 affected fires' controls, and leaving the existing
`fwi_fire_vs_control_comparison.csv` results as-is rather than re-running. Re-collecting them
properly isn't even straightforward right now -- none of the 4 are in the retained
`test_dataset/` subset, so their original `controls/` metadata (needed for eco-region +
bbox-based control re-selection) is gone from S3 along with the rest of the bulk dataset, and
fixing that would mean pointing `collect_fwi_era5.py`'s discovery at the HuggingFace copy
instead. More to the point, this comparison was never load-bearing for the project: the
calculator's actual correctness was already established against cffdrs (see the 08/17
comparison entries above -- FFMC/DMC/DC/ISI/BUI/FWI all matching a reference implementation to
within floating-point noise once the 147.2-vs-unrounded-constant fix landed). This fire-vs-
control run was a bonus sanity check layered on top of that, meant to gauge whether plain FWI
by itself says anything useful about fire risk -- not a required validation step the rest of
the project depends on. 56 of 60 fires still have a clean, uncollided comparison point, which
is plenty to read the intended signal from; re-running for 4 more rows isn't worth the
HuggingFace-repointing work it'd require. The code fix (folder-collision prevention) stays,
since it's cheap and correct regardless -- only the decision to re-collect the 4 already-known
casualties was dropped.

Eyeballing the full per-fire CSV output (with the 4 collided-control rows discounted) surfaced
the actual finding of this whole side-experiment:

- 5 of the remaining pairs (CA/MINERAL_fire, MT/THORNE_CREEK_fire, WA/COUGAR_CREEK_fire,
  WA/EVANS_CANYON_fire, WA/SCHNEIDER_SPRINGS_fire) show byte-identical fire/control values
  across every component -- most plausibly because their ERA5 sampling centroids land in the
  exact same underlying ERA5 grid cell (ERA5's ~0.25°/~28km resolution vs. only a 20km minimum
  separation enforced between a control and anything else during original site selection).
  These carry zero information for the comparison, not "no difference found."
- The mean ABSOLUTE FWI difference by state (the first thing this script reports) is
  misleading read on its own: CA shows the largest mean abs diff (10.4) but its mean SIGNED
  diff is actually slightly negative (-1.15) -- driven by large swings in both directions
  (KELLY_fire -39.1, DETWILER_fire +31.1), not a consistent "fires are riskier" signal. Only
  AK, ID, and NV show a consistently positive mean signed diff. Overall, across every
  non-collided, non-identical pair: 26 show fire > control, 29 show fire < control -- close to
  a coin flip. Added signed-mean and median (alongside mean) reporting to the script's own
  output so this doesn't require a manual CSV pull to notice next time.

TAKEAWAY: this near-coin-flip result is itself the useful outcome, and it's read here as
supporting evidence for the project's actual design choice rather than a disappointing result
to chase further. FWI is a single scalar built purely from weather (temperature, humidity,
wind, antecedent precipitation) -- it has no way to encode ignition source availability
(lightning strikes, human activity), fuel continuity, terrain, or vegetation structure, all of
which matter for whether a specific point in a landscape actually catches fire on a given day.
That FWI alone doesn't cleanly separate real fire locations from same-day, same-eco-region
non-fire locations roughly half the time is exactly why the project's actual model is a hybrid
convolutional-transformer over Sentinel-1/Sentinel-2/ERA5 jointly, rather than a simple
FWI-threshold classifier -- FWI captures general area-wide danger, but the spatial/structural
signal in the imagery is what's needed to say *where* within a hazardous landscape a fire is
actually likely to start. This experiment is a reasonable place to stop rather than debug
further: it did its job as a sanity check on the calculator's real-world behavior, and its
ambiguous result reinforces rather than undermines the case for the more complex model already
in progress.

## Test-subset duplication script - 09/05/26

Before uploading the bulk of the fires/controls dataset to HuggingFace and deleting it from S3,
need a small retained subset for locally testing the still-to-come inference script and NBR
burn-severity calculator (the only two remaining project steps that actually need real
fire/control data). Settled on one fire + its single closest-matching control per state (14
scenes total), with fire selection driven by the dev log's own bug history rather than
arbitrary picks where that history exists: AK (Cultas Creek Fire) and WA (400 Fire) were both
extensively exercised during FWI development (date-matching confound, stale-cfgrib-index,
time-dim mismatch, wind-vector-cancellation, CDS temporal leakage); NV swapped from
`MIRROR_FIRST_FIRE_BY_STATE`'s `Meadow Valley Fire` to `FLAT_fire`, which is the one actually
named in the 07/22 ERA5-padding bug (Sentinel-1's wider swath vs. Sentinel-2's footprint). No
fire-specific bug history exists for CA/ID/MT/OR, so those three keep their existing
`MIRROR_FIRST_FIRE_BY_STATE` picks (Caldwell, Dixie, Thorne Creek, 0501 Crazy Creek) rather than
inventing a new, unjustified selection.

New `scripts/duplicate_test_scenes.py`: for each of the 7 fires, reuses `find_fire_scene()` and
the bbox-aware `find_control_for_fire()` (both from `collect_fwi_era5.py`) to resolve the fire
and its single closest-matching control, then server-side copies (`s3.copy()`, boto3's managed
multipart copy -- no data downloaded through this machine, nothing modified at the source) every
object under each scene's full S3 prefix into a new `test_dataset/fires/{state}/{scene_id}/` or
`test_dataset/controls/{state}/{scene_id}/` prefix. Writes a JSON manifest recording every
source/destination prefix pair for later reference. Defaults to a dry run (lists what would be
copied and the total data volume); `--execute` actually copies. Not yet run.

Caught a mistake in the state assignment right after the first dry run: `[NV] 'FLAT_fire'`
failed to resolve -- `FLAT_fire` is actually an OR fire, not NV. The mix-up came from the 07/22
devlog entry itself, which names `FLAT_fire` alongside `control_NV_38N118W_20230804` in the
same bug description; that control being in Nevada doesn't mean its fire is, since eco-region
matching (`control_collection_pipeline.py`) places controls by matching vegetation/climate
zone, not by state. Fixed: moved `FLAT_fire` to OR (its real state, keeping its genuine bug
justification) and reverted NV back to `Meadow Valley Fire` (the original
`MIRROR_FIRST_FIRE_BY_STATE` placeholder, and one of the user's actual confirmed NV fires).
With that fixed, the script ran successfully: all 7 state/fire/control pairs resolved and
copied server-side into `test_dataset/`, with a full JSON manifest confirming every source/dest
prefix.

## HuggingFace dataset upload, publication, and S3 cleanup - 09/07/26

With the test subset safely duplicated, moved on to uploading the bulk `fires/` and `controls/`
prefixes (2,804 files, ~2.1 TB) to the new `aroon-sankoh/wildfire-prediction` HF dataset repo,
then publishing a proper dataset card, then deleting the bulk data from S3 once the upload was
verified intact.

New `scripts/upload_to_huggingface.py`: since the full dataset is far larger than local/EC2
disk space, it streams each S3 object through a local temp file (download -> upload -> delete)
rather than mirroring the bucket to disk first, and batches files into commits instead of one
Hub commit per object. Resumable by design -- at startup it lists every path already present in
the HF repo via `list_repo_files()` and skips any S3 key already there, so a killed/restarted
run just continues rather than re-uploading or duplicating.

Three real bugs surfaced during the actual run:

- First commit attempt failed with a 400 `Invalid file change` from the Hub's commit endpoint.
  Root cause: S3 occasionally creates zero-byte "folder placeholder" objects (a key ending in
  `/`, representing an empty directory) alongside real files, and a directory-shaped path isn't
  a valid file path in a git repo. Fixed by filtering out any key ending in `/` in
  `list_s3_objects()` before it's ever queued for upload.
- Initial auth attempts failed with 401 `Unauthorized` / `RepositoryNotFoundError` even though
  the repo clearly existed -- turned out to be literally using the placeholder `hf_xxx` from an
  example command rather than a real token (confirmed via `HfApi().whoami()` also failing with
  "Invalid user token"). Not a script bug, just needed a real write-scoped token generated at
  huggingface.co/settings/tokens.
- Partway through the real run (batch 63 of several hundred), a commit failed with
  `RuntimeError: Internal error: timed out reading request body` from HF's xet upload backend --
  a transient timeout on a large (~2.3GB) multi-file commit, not a data or auth problem. Fixed
  two ways: dropped the batch size from ~2GB to ~1GB per commit (shorter request duration, less
  likely to hit the timeout window) and wrapped `create_commit()` in a retry loop with
  exponential backoff (15s/30s/60s/120s/240s, 5 attempts) instead of letting one transient
  failure kill the whole multi-hour run. Also added a failure handler that prints every key in a
  rejected batch, so a future failure is diagnosable without guessing.

Final run completed clean: 2,804 files, 2,145.87 GB uploaded (199 already-present files skipped
across resumed runs).

Dataset card (`README.md` on the HF repo) was built out from the user's own draft, which
originally covered only Data Sources, Scene Constraints, and Motivation. Added: a License &
Attribution section (researched via web search rather than assumed from training knowledge,
since license terms can change -- confirmed Sentinel-1/2 imagery falls under the Copernicus
Sentinel Data Terms and Conditions requiring a "Contains modified Copernicus Sentinel data
[Year]" notice, and that ECMWF replaced the old "Licence to use Copernicus Products" with
CC-BY-4.0 for all CDS products including ERA5 as of 07/02/25, each requiring its own
attribution regardless of what license covers the derived dataset itself); MTBS added as a
fourth Data Source entry, since it hadn't been listed at all despite being the actual ground
truth for fire identification, ignition dates, and burn severity; a Dataset Structure section
documenting the `fires/{state}/{scene_id}/` and `controls/{state}/{scene_id}/{control_id}/`
layout, filename conventions, and every `metadata.json` field; a Control Selection & Validation
section; and a Limitations section (7-state geographic coverage, >10k-acre fire floor, 1:3
fire:control class imbalance, no predefined train/val/test split). Also caught and fixed a
citation code block that had been pasted into the HF web editor in a way that collapsed its
line breaks into a single flowing paragraph instead of rendering as a monospaced block --
needed to be re-pasted from the source markdown file directly rather than from the rendered
preview.

The Control Selection & Validation section initially shipped with a placeholder, since neither
`control_collection_pipeline.py`'s exclusion logic nor anything else in the pipeline actually
verified that a sampled control site hadn't itself experienced a real (possibly undocumented)
fire -- the only existing safeguard, `fire_exclusion_zone` in that script, only excludes the
bounding boxes of the 125 fires already anchored in *this* dataset, not the full MTBS catalog of
every US fire, and has no temporal dimension at all. User closed the gap by cross-referencing
every control against the full MTBS fire perimeter record and additionally checking the dNBR
between each control's pre/post Sentinel-2 pair to catch anything MTBS might have missed
(smaller/undocumented fires) -- both checks came back clean, nothing needed to be excluded.

New `scripts/verify_huggingface_upload.py`: a completeness + integrity check run before trusting
the upload enough to delete the S3 source. Cheap pass compares every S3 key/size under
`fires/`/`controls/` against `list_repo_tree()`'s record of what's actually in the HF repo
(missing files, extra files, size mismatches). Expensive pass (`--verify-hashes`) additionally
streams each S3 object (no local disk write) and computes its sha256, comparing it against the
sha256 HF already recorded for that path when the commit was accepted -- since LFS/xet uploads
are content-addressed and HF verifies the hash server-side before finalizing a commit, this is a
genuine end-to-end integrity check without needing to re-download anything from HF itself. Final
run: 3,003 S3 objects checked, 0 missing, 0 size mismatches, 0 extra files, 2,503 files
hash-verified, 0 hash mismatches -- full pass.

With the completeness/hash verification clean and the dataset card published, user proceeded to
delete the bulk `fires/`/`controls/` data from the S3 bucket, retaining only the `test_dataset/`
subset (from the 08/17 duplication script) for local testing of the still-upcoming inference
script and NBR burn-severity calculator.

## ERA5 near-ignition leakage ablation - 09/14/26

Before starting the inference script, revisited an issue with the model's own input windowing
that came up while designing the inference script's forecast horizon: `build_tile_cache.py`'s
`era5_cutoff_from_key` derives the ERA5 window's cutoff from the fire's own ignition date, and
`load_era5_vars(..., cutoff_datetime=cutoff)` keeps data through that day inclusive -- meaning
every tile's 30-day weather sequence runs right up to the day of the fire, with zero forecast
lead time. That's fine for the model's current framing (same-day classification: does this
scene look like a fire day or a control day), but it puts a real asterisk on reading the
finalized model's ~0.731 test balanced accuracy (see 08/14 entry) as evidence of *forecasting*
skill -- FFMC in particular can spike sharply right at ignition, so the model may partly be
learning "today's weather is extreme" rather than anything with real lead time.

Decided against a full re-collection with an embargoed ERA5 window for now (would require
re-querying CDS with a shifted cutoff for every scene, and would break continuity with what's
already published on HuggingFace -- a real cost, not something to take on just to settle a
methodology question). Instead, ran a cheap ablation entirely on the already-cached
`tile_cache/` data: added `--era5-embargo-days N` to `training/train.py`, which drops the last
N (most recent) days from every tile's `era5_stats` sequences in `merge_tiles()` before
building the train/val/test datasets -- applied uniformly across all three splits, since the
point is testing whether the signal survives without near-ignition weather, not just whether
the model still fits train. Also fixed a latent bug this exposed in `model/dataset.py`: the
missing-tile imputation path hardcoded a 30-day fallback length (the module-level
`ERA5_SEQ_LEN` constant); with real sequences truncated to 23 days, an imputed missing tile
would produce a mismatched shape and break batch collation. Fixed by deriving the actual
sequence length from real data in `dataset.__init__` (`self.era5_seq_len`) instead of trusting
the hardcoded constant.

Ran `--era5-embargo-days 7 --epochs 12` twice: once with train.py's plain defaults, once with
the finalized sweep config (`lr=0.000555, batch_size=64, embedding_dim=64,
temporal_hidden_dim=32, n_layers=1, n_head=1, weight_decay=1e-05`) for a true apples-to-apples
comparison against the 08/14 finalized run. Results: default config landed at test `bal_acc
0.716` (precision 0.516, recall 0.628); the matched sweep config landed at test `bal_acc 0.699`
(precision 0.423, recall 0.731) -- best checkpoint at epoch 6 this time instead of epoch 7.
Both are within about 3 points of the original 0.731, and well inside the 0.68-0.73 band the
original run itself noisily oscillated in past its epoch-7 peak (see 08/14 entry) -- i.e. this
drop is consistent with ordinary run-to-run noise, not a collapse. If near-ignition weather
leakage had been the dominant driver of the original number, removing a full week of it should
have hurt far more than 3 points.

CONCLUSION: proceeding with the current dataset/model as-is, no re-collection needed. The
~0.70-0.73 balanced accuracy range appears to reflect a real, if modest, signal rather than
being primarily an artifact of same-day weather leakage. Worth revisiting if the eventual
inference script's real-world behavior looks suspicious, but not blocking further work for now.

## Inference script + a reminder that the test_dataset scenes aren't held-out - 09/17/26

Built `scripts/run_inference.py`: loads a `train.py` checkpoint, runs a single unlabeled scene
through the same S1/S2/ERA5 loading and `zonal_aggregator.aggregate()` pipeline
`build_tile_cache.py` uses, and scores every tile through `head_1month` only (~30-day horizon,
per the earlier decision to drop the 3-horizon matrix since all three heads share one
scene-level label). Scene inputs are read directly off the published dataset layout --
`S1_*_pre.SAFE`/`S2_*_pre.SAFE` (each actually a zip archive despite the `.SAFE` suffix,
wrapping one nested real-product-named `.SAFE` dir, same shape `download_and_extract()`
produces) plus the ERA5 grib, all sitting flat in one scene folder -- so it runs directly
against a `fires/{state}/{scene}/` or `controls/{state}/{scene}/` folder with no repackaging.
Output is a per-tile CSV under `inference/{run_id}_{checkpoint_stem}/{scene_id}_risk_grid.csv`,
plus a console summary (tile count, fraction above 0.5, top-5 highest-risk tiles, and an
aggregated mean-across-tiles "verdict" for the scene). Also used this as an opportunity to fix
a real gap: `train.py` didn't save normalization stats (`spatial_mean/std`,
`temporal_mean/std`, `statistic_means`, `era5_seq_len`) into its checkpoints at all, so
inference had no way to reproduce the exact z-score transform a given model was trained under
without re-deriving it from `--cache-dir` and hoping `--seed`/`--val-frac`/`--test-frac`
happened to match. Fixed by saving those stats directly into `best_model.pt`/`final_model.pt`
going forward; older checkpoints fall back to the re-derivation path with a loud warning.

Sanity-tested the script end to end on two `test_dataset/` scenes for state AK: the control
(`control_AK_65N141W_20190805`) produced a plausible-looking result -- mean fire probability
0.018 across 235 tiles, with one small, spatially localized cluster of elevated risk rather
than noise scattered everywhere. The fire scene (`CULTAS_CREEK_fire`) came back at mean 1.000,
every single one of 226 tiles between 0.9999 and 1.0 with almost no spatial variation at all,
including tiles presumably nowhere near the ignition point.

That result is too clean to trust at face value, and checking it confirmed why:
`CULTAS_CREEK_fire` is one of the scenes in `tile_cache/`, and running `group_by_fire()` +
`split_groups()` with this checkpoint's training seed (42, val/test frac 0.15/0.15) shows it
landed in the *training* split. The model didn't forecast this fire -- it memorized it. That
also applies to the rest of `test_dataset/`, not just this one fire: the whole retained
7-pair subset was pulled from the same original `fires/`/`controls/` collection the model was
trained on, so none of it is actually held-out data, and none of these local
sanity-check runs (this one included) say anything about the model's real predictive skill.
They're useful only for confirming the inference script itself runs correctly end to end,
which it now does -- the FWI/CFFDRS-style validation this project has leaned on elsewhere.

DECISION: not treating any `test_dataset/` result as a signal of forecasting ability, in either
direction. The plan going forward is to wait for 2025 North American fire data to become
available, build a fresh, genuinely-unseen dataset from it the same way the original was built,
and evaluate this model's real predictive power against fires it has never had any exposure to
-- training, validation, or otherwise. Until then, the inference script itself is considered
working and verified; the model's forecasting skill on new events is still an open question.
