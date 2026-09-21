# Project State — Last updated on 09/21/26

A summary of where this project actually stands right now, written before a multi-month
pause (grad school program starting, waiting on 2025 fire data). The devlog has the full
chronological history of bugs/fixes/decisions, and this doc is meant to be readable on its
own without rereading all of that since it's super long.

## Architecture, end to end

A scene (fire or control) is a folder with `metadata.json`, Sentinel-1 pre/post SAFE
products, Sentinel-2 pre/post SAFE products, and one ERA5 grib. Only the *pre*-fire
S1/S2 products and the ERA5 window feed the model -- post-fire imagery is only used by
the burn severity calculator, never for prediction, since there's obviously no "post" for
a forecast that hasn't happened yet.

Pipeline: `data/loaders/` (RTC + calibration for S1, NBR/NDVI + cloud masking for S2,
FWI + windowed loading for ERA5) -> `data/aggregator/zonal_aggregator.py` bins every
pixel onto the ERA5 grid and produces per-tile stats -> `scripts/build_tile_cache.py`
runs that once per scene and caches the resulting `tiles` dict to disk, so
training/inference never repeat the load+aggregate work -> `model/dataset.py` handles
mean-imputation for missing/NaN tiles and z-score normalization -> `model/architecture.py`
(`WildfireModel`) is a CNN (spatial, per-tile S1/S2 stats) + Transformer (temporal, 30-day
ERA5 sequence) with cross-attention fusion, ending in three sigmoid heads
(`head_1month`/`head_3month`/`head_6month`) -> `training/train.py` trains and checkpoints
it (checkpoints now also carry normalization stats, so inference doesn't need to
re-derive them from `--cache-dir`) -> `scripts/run_inference.py` runs a trained
checkpoint on one new scene, producing a per-tile risk grid at a single ~30-day horizon
(`head_1month` only -- see caveats below for why).

## Other tools in scripts/

`hyperparameter_sweep.py` -- random search over model/training config, reusing one
fixed train/val split and scoring by validation balanced accuracy. Prints top configs
and a ready-to-run `train.py` command rather than auto-retraining.

`fwi_calculator.py` -- runs the from-scratch Canadian FWI implementation
(`data/loaders/era5_preprocessing.py`) on a single scene, with guardrails requiring at
least 120 days of hourly data and all 5 ERA5 variables. Originally built to validate the
FWI math against CFFDRS and to check whether a simple FWI-threshold classifier would
have been competitive with the full model (it wasn't clearly better or worse --
signed FWI diffs between fires and their matched controls were close to a coin flip,
which is part of why the more complex model design was worth pursuing).

`nbr_burn_severity_calculator.py` -- computes dNBR from a scene's pre/post Sentinel-2
pair and classifies severity using that specific fire's own MTBS-calibrated thresholds
(MTBS calibrates per fire, not globally), then validates computed burned acreage against
MTBS's officially reported acreage inside the real perimeter polygon. First real test
(`CULTAS_CREEK_fire`, AK) came back at a 0.81 computed-to-official ratio, plausible given
known sources of divergence (cloud masking, downsampling, and a pre-fire image that
likely isn't the exact same one MTBS used).

`build_tile_cache.py` also now supports `--source local`, processing scenes already on
disk (e.g. downloaded from the published HuggingFace dataset) with no AWS access needed
at all, in addition to the original S3-backed workflow.

## Open items

**Same-day classification, not genuine forecasting.** The ERA5 window for every tile
currently runs right up through the ignition/control date itself with zero forecast
lead time (`era5_cutoff_from_key` in `build_tile_cache.py`). The model is answering
"does this look like a fire day or a control day," not "will this area burn in the next
30 days." This is a framing problem, not necessarily a leakage problem, see below for more.

**Embargo ablation result.** Ran `--era5-embargo-days 7` (dropping the most recent
week from every tile's ERA5 sequence) as a cheap test of how much the model's ~0.731
test balanced accuracy depends on near-ignition weather specifically. Two runs came back
at 0.716 and 0.699 -- both within the ~0.68-0.73 noise range the original run itself
oscillated in, not a collapse. Conclusion: the signal doesn't appear to be primarily a
same-day-weather artifact, but the same-day *framing* caveat above still stands until
the model is actually retrained with a real embargoed window and forecast lead time.

**Need a genuinely held-out test set.** The retained `test_dataset/` S3 subset (used for
local sanity-testing `run_inference.py` and the burn severity calculator) is not
held-out data, it was used for training. Local inference runs against it (including the ~1.0
probability result on that fire) reflect memorization, not real predictive skill. The
plan is to wait for 2025 North American fire data, build a fresh dataset from it the
same way this one was built, and evaluate against fires the model has never touched:
training, validation, or otherwise.

**The 3-horizon heads aren't real yet.** `head_1month`/`head_3month`/`head_6month` all
currently get the exact same scene-level binary label (fire vs. control), so nothing
distinguishes them in a validated way, and therefore only `head_1month` is used by
`run_inference.py`. Making the other two heads meaningful requires collecting and
labeling data with actual per-horizon ground truth (e.g. distinguishing "this location
burned within 1 month of the observation window" from "within 3 months" from "within 6
months"), stratified so each horizon has enough positive examples to train on, not just
reusing the same fire/control label three times. This is a data collection problem, not
a modeling one, and hasn't been started due to time and data constraints. 
