"""
This script is a one-time preprocessing pass that turns raw S1/S2/ERA5 scenes into
cached per-scene `tiles` dicts on disk, so training and inference never have to redo 
any loading or aggregating logic.

Two sources are supported:
  --source s3 (default): walks the fires/ and controls/ prefixes in your @Aroon-Sankoh's 
    personal bucket, downloading and unzipping each scene. Requires AWS credentials and 
    access to @Aroon-Sankoh's bucket.
  --source local: processes scenes already sitting on disk, e.g. downloaded from the
    published HuggingFace dataset (via snapshot_download or the huggingface-cli) rather
    than needing any AWS access at all. --local-dir must contain fires/ and controls/
    subfolders laid out exactly like the published dataset (state/scene_id/ folders,
    each with metadata.json, S1_*_pre.SAFE, S2_*_pre.SAFE, and one *.grib file, which
    is the same layout run_inference.py and nbr_burn_severity_calculator.py already expect).

Usage:
    python build_tile_cache.py --limit 3                              # S3, test on 3 scenes
    python build_tile_cache.py                                        # S3, full run
    python build_tile_cache.py --skip-existing                        # S3, resume after interruption
    python build_tile_cache.py --source local --local-dir ~/hf_dataset  # local, no AWS needed
"""

import argparse
import functools
import glob
import json
import os
import pickle
import re
import shutil
import signal
import sys
import time
import traceback
import zipfile
import rasterio

import boto3
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
from data.loaders import load_era5_vars, load_sentinel1_bands, load_sentinel2_bands
from data.aggregator import aggregate

BUCKET_NAME = "wildfire-scenes-s3-202802195212-eu-central-1-an"
FIRES_PREFIX = "fires"
CONTROLS_PREFIX = "controls"
SCENE_TIMEOUT_SECONDS = 1800

s3 = boto3.client("s3")


class SceneTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise SceneTimeout("Scene exceeded max processing time")


def run_with_timeout(fn, *args, timeout_seconds=SCENE_TIMEOUT_SECONDS, **kwargs):
    # guards against a single hung download/read stalling the whole multiple day run silently
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def with_retries(fn, *args, retries=3, backoff=5, label="", **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == retries:
                raise
            print(f"    [retry] {label or fn.__name__} failed (attempt {attempt}/{retries}): {e} -- retrying in {backoff}s", flush=True)
            time.sleep(backoff)


def list_keys_under(prefix):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def discover_scenes(kind):
    top_prefix = FIRES_PREFIX if kind == "fire" else CONTROLS_PREFIX
    meta_keys = [k for k in list_keys_under(f"{top_prefix}/") if k.endswith("metadata.json")]
    return [(k, k.rsplit("/", 1)[0] + "/") for k in meta_keys]


def discover_scenes_local(local_dir, kind):
    """
    Local directory equivalent of discover_scenes(). It walks <local_dir>/fires/ or
    <local_dir>/controls/ for state/scene_id/metadata.json and matches the published
    dataset's exact layout.
    """
    top_prefix = FIRES_PREFIX if kind == "fire" else CONTROLS_PREFIX
    pattern = os.path.join(local_dir, top_prefix, "*", "*", "metadata.json")
    meta_paths = sorted(glob.glob(pattern))
    return [(p, os.path.dirname(p)) for p in meta_paths]


def resolve_zip_key(scene_prefix, contents_value):
    """
    Resolves a metadata.json contents value to the actual S3 zip key. The underlying zip structure of a
    zip file is not consistent, some scenes store the zip directly at the key and others store it one 
    level down inside a named folder.
    """
    # resolves the underlying zip structure of a SAFE file as contents are sometimes the flat zip object itself and sometimes a folder
    exact_key = scene_prefix + contents_value
    try:
        with_retries(s3.head_object, Bucket=BUCKET_NAME, Key=exact_key, label=f"head_object({exact_key})")
        return exact_key
    except s3.exceptions.ClientError:
        pass

    folder_prefix = exact_key + "/"
    resp = with_retries(s3.list_objects_v2, Bucket=BUCKET_NAME, Prefix=folder_prefix, label=f"list_objects({folder_prefix})")
    all_keys = [o["Key"] for o in resp.get("Contents", [])]
    zip_keys = [k for k in all_keys if k.lower().endswith(".zip")]
    if len(zip_keys) == 1:
        return zip_keys[0]

    raise ValueError(
        f"Could not resolve a single object for {exact_key}, not found as an "
        f"exact key, and folder search under {folder_prefix} found {len(all_keys)} "
        f"objects: {all_keys}."
    )


def download_and_extract(zip_key, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    local_zip = os.path.join(extract_dir, os.path.basename(zip_key))

    def _download_and_unzip():
        s3.download_file(BUCKET_NAME, zip_key, local_zip)
        with zipfile.ZipFile(local_zip, "r") as zf:
            zf.extractall(extract_dir)

    try:
        with_retries(_download_and_unzip, label=f"download+unzip({zip_key})")
    except zipfile.BadZipFile:
        # almost always a truncated transfer rather than a actual malformed file, so do one clean retry 
        print(f"    [retry] {zip_key} produced a BadZipFile, re-downloading once...", flush=True)
        if os.path.exists(local_zip):
            os.remove(local_zip)
        _download_and_unzip()
    finally:
        if os.path.exists(local_zip):
            os.remove(local_zip)

    safe_dirs = glob.glob(os.path.join(extract_dir, "*.SAFE"))
    if len(safe_dirs) != 1:
        raise ValueError(f"Expected exactly one SAFE dir after extracting {zip_key}, found {safe_dirs}")
    return safe_dirs[0]


def download_grib(grib_key, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    local_path = os.path.join(dest_dir, os.path.basename(grib_key))
    with_retries(s3.download_file, BUCKET_NAME, grib_key, local_path, label=f"download({grib_key})")
    return local_path


def ensure_extracted_safe_dir(path, extract_dir, label):
    """
    Local directory equivalent of download_and_extract() for a scene already on disk. 
    """
    if os.path.isdir(path):
        nested = glob.glob(os.path.join(path, "*.SAFE"))
        if len(nested) > 1:
            raise ValueError(f"Expected at most one nested .SAFE dir under {path}, found {nested}")
        return nested[0] if nested else path

    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(extract_dir)

    safe_dirs = glob.glob(os.path.join(extract_dir, "*.SAFE"))
    if len(safe_dirs) != 1:
        raise ValueError(f"Expected exactly one .SAFE dir after extracting {path}, found {safe_dirs}")
    return safe_dirs[0]


def glob_one(pattern, label):
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one match for {label} ({pattern}), found {len(matches)}: {matches}")
    return matches[0]


def load_s1_pre(safe_dir, dem_output_dir):
    vh_band = glob_one(os.path.join(safe_dir, "measurement", "*-vh-*.tiff"), "S1 VH band")
    vv_band = glob_one(os.path.join(safe_dir, "measurement", "*-vv-*.tiff"), "S1 VV band")
    vh_cal = glob_one(os.path.join(safe_dir, "annotation", "calibration", "calibration-*-vh-*.xml"), "S1 VH calibration")
    vv_cal = glob_one(os.path.join(safe_dir, "annotation", "calibration", "calibration-*-vv-*.xml"), "S1 VV calibration")
    annotation_xml = glob_one(os.path.join(safe_dir, "annotation", "*-vh-*.xml"), "S1 annotation")
    return load_sentinel1_bands(vh_band, vv_band, vh_cal, vv_cal, annotation_xml, dem_output_dir, downsample_factor=10)


def load_s2_pre(safe_dir, downsample_factor=10):
    granule_dir = glob_one(os.path.join(safe_dir, "GRANULE", "*"), "S2 GRANULE")
    red = glob_one(os.path.join(granule_dir, "IMG_DATA", "R10m", "*_B04_10m.jp2"), "S2 red (B04)")
    green = glob_one(os.path.join(granule_dir, "IMG_DATA", "R10m", "*_B03_10m.jp2"), "S2 green (B03)")
    nir = glob_one(os.path.join(granule_dir, "IMG_DATA", "R10m", "*_B08_10m.jp2"), "S2 nir (B08)")
    swir = glob_one(os.path.join(granule_dir, "IMG_DATA", "R20m", "*_B11_20m.jp2"), "S2 swir (B11)")
    scl = glob_one(os.path.join(granule_dir, "IMG_DATA", "R20m", "*_SCL_20m.jp2"), "S2 SCL")

    # peek the native resolution to compute the downsampled target shape
    with rasterio.open(nir) as ds:
        native_height, native_width = ds.height, ds.width
    target_shape = (native_height // downsample_factor, native_width // downsample_factor)

    s2_data = load_sentinel2_bands(red, green, nir, swir, scl, target_shape=target_shape)
    bands, nir_transform, nir_shape, s2_crs = s2_data
    return bands, nir_transform, nir_shape, s2_crs


def _log(scene_id, msg, t0):
    print(f"    [{scene_id}] {msg} ({time.time() - t0:.1f}s elapsed)", flush=True)


def era5_cutoff_from_key(era5_key):
    """
    Grabs the fire ignition date to use as the cut off date for the antecedent ERA5 grib data.
    Ensures that no time-series after the ignition date is accidentally included.
    """
    match = re.search(r"(\d{8})\.grib$", era5_key)
    if not match:
        raise ValueError(f"Could not find an 8-digit date in ERA5 key: {era5_key}")
    # midnight of the ignition/control date -- the whole day itself is excluded, not just
    # what comes after it, since we don't know the exact time of day the fire ignited
    return pd.to_datetime(match.group(1), format="%Y%m%d")


def process_scene(metadata_key, scene_prefix, kind, tmp_root):
    """
    Runs one fire or control through the full pipeline: Download and extract S1/S2/ERA5 -> Load Bands -> Aggregate into Tiles.
    Failures are logged (not raised) so execution can move onto the next scene rather than aborting an entire run.
    """
    t0 = time.time()
    resp = s3.get_object(Bucket=BUCKET_NAME, Key=metadata_key)
    meta = json.loads(resp["Body"].read())
    contents = meta["contents"]

    scene_id = meta.get("control_id") or meta["fire_name"]
    scene_tmp = os.path.join(tmp_root, scene_id.replace("/", "_"))
    os.makedirs(scene_tmp, exist_ok=True)

    try:
        _log(scene_id, "resolving S3 keys...", t0)
        s1_zip_key = resolve_zip_key(scene_prefix, contents["S1_pre"])
        s2_zip_key = resolve_zip_key(scene_prefix, contents["S2_pre"])
        era5_key = scene_prefix + contents["ERA5"]

        _log(scene_id, "downloading + extracting S1_pre...", t0)
        s1_safe_dir = download_and_extract(s1_zip_key, os.path.join(scene_tmp, "s1"))

        _log(scene_id, "downloading + extracting S2_pre...", t0)
        s2_safe_dir = download_and_extract(s2_zip_key, os.path.join(scene_tmp, "s2"))

        _log(scene_id, "downloading ERA5 grib...", t0)
        era5_local = download_grib(era5_key, os.path.join(scene_tmp, "era5"))

        dem_dir = os.path.join(scene_tmp, "dem")
        _log(scene_id, "loading S1 bands (calibration + RTC, includes DEM download)...", t0)
        s1_data = load_s1_pre(s1_safe_dir, dem_dir)

        _log(scene_id, "loading S2 bands...", t0)
        s2_data = load_s2_pre(s2_safe_dir)

        _log(scene_id, "loading ERA5 vars...", t0)
        cutoff_datetime = era5_cutoff_from_key(era5_key)
        era5_data = load_era5_vars(era5_local, cutoff_datetime=cutoff_datetime)

        _log(scene_id, "aggregating into tiles...", t0)
        tiles = aggregate(s1_data, s2_data, era5_data)
        _log(scene_id, f"done -- {len(tiles)} tiles", t0)

        label = 1.0 if kind == "fire" else 0.0
        record = {
            "scene_id": scene_prefix.rstrip("/"),
            "kind": kind,
            "label": label,
            "fire_name": meta.get("fire_name"),
            "event_id": meta.get("event_id"),
            "matched_fire_event_id": meta.get("matched_fire_event_id"),
            "tiles": tiles,
        }
        return record, None
    except Exception as e:
        return None, f"{scene_id}: {e}\n{traceback.format_exc()}"
    finally:
        shutil.rmtree(scene_tmp, ignore_errors=True)


def process_scene_local(metadata_path, scene_dir, kind, tmp_root):
    """
    Local-dir equivalent of process_scene(), for a scene already sitting on disk
    (e.g. downloaded from the published HuggingFace dataset) instead of S3.
    """
    t0 = time.time()
    with open(metadata_path) as f:
        meta = json.load(f)

    scene_id = meta.get("control_id") or meta["fire_name"]
    scene_tmp = os.path.join(tmp_root, scene_id.replace("/", "_"))
    os.makedirs(scene_tmp, exist_ok=True)

    try:
        _log(scene_id, "resolving local scene files...", t0)
        s1_outer = glob_one(os.path.join(scene_dir, "S1_*_pre.SAFE"), "S1 pre-scene")
        s2_outer = glob_one(os.path.join(scene_dir, "S2_*_pre.SAFE"), "S2 pre-scene")
        era5_local = glob_one(os.path.join(scene_dir, "*.grib"), "ERA5 grib")

        _log(scene_id, "extracting S1_pre (if zipped)...", t0)
        s1_safe_dir = ensure_extracted_safe_dir(s1_outer, os.path.join(scene_tmp, "s1"), "S1 pre-scene")

        _log(scene_id, "extracting S2_pre (if zipped)...", t0)
        s2_safe_dir = ensure_extracted_safe_dir(s2_outer, os.path.join(scene_tmp, "s2"), "S2 pre-scene")

        dem_dir = os.path.join(scene_tmp, "dem")
        _log(scene_id, "loading S1 bands (calibration + RTC, includes DEM download)...", t0)
        s1_data = load_s1_pre(s1_safe_dir, dem_dir)

        _log(scene_id, "loading S2 bands...", t0)
        s2_data = load_s2_pre(s2_safe_dir)

        _log(scene_id, "loading ERA5 vars...", t0)
        cutoff_datetime = era5_cutoff_from_key(era5_local)
        era5_data = load_era5_vars(era5_local, cutoff_datetime=cutoff_datetime)

        _log(scene_id, "aggregating into tiles...", t0)
        tiles = aggregate(s1_data, s2_data, era5_data)
        _log(scene_id, f"done -- {len(tiles)} tiles", t0)

        label = 1.0 if kind == "fire" else 0.0
        record = {
            "scene_id": scene_id,
            "kind": kind,
            "label": label,
            "fire_name": meta.get("fire_name"),
            "event_id": meta.get("event_id"),
            "matched_fire_event_id": meta.get("matched_fire_event_id"),
            "tiles": tiles,
        }
        return record, None
    except Exception as e:
        return None, f"{scene_id}: {e}\n{traceback.format_exc()}"
    finally:
        shutil.rmtree(scene_tmp, ignore_errors=True)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["s3", "local"], default="s3",
                         help="'s3' (default): pull scenes from an S3 bucket, requires AWS credentials. "
                              "'local': process scenes already on disk (e.g. downloaded from the "
                              "published HuggingFace dataset) -- no AWS access needed at all.")
    parser.add_argument("--local-dir", default=None,
                         help="Required with --source local. Root dir containing fires/ and controls/ "
                              "subfolders laid out exactly like the published dataset.")
    parser.add_argument("--bucket-name", default=BUCKET_NAME,
                         help="S3 bucket to pull from with --source s3 -- point this at your own bucket "
                              "if you've collected your own scenes (defaults to the author's own).")
    parser.add_argument("--fires-prefix", default=FIRES_PREFIX)
    parser.add_argument("--controls-prefix", default=CONTROLS_PREFIX)
    parser.add_argument("--cache-dir", default="tile_cache")
    parser.add_argument("--tmp-dir", default="/tmp/wildfire_extract")
    parser.add_argument("--s3-cache-prefix", default="cache/tiles/")
    parser.add_argument("--upload-to-s3", action="store_true", help="Also upload each cached pickle back to S3.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N scenes (per kind), for smoke-testing.")
    return parser


def _record_failure(error, failures_log_path):
    print(f"  [FAILED] {error.splitlines()[0]}")
    # flush immediately rather than batching to disk at the end, so failures are visible mid-run
    with open(failures_log_path, "a") as f:
        f.write(error + "\n\n")


def cache_path_for(cache_dir, cache_relpath):
    """
    Mirrors the scene's relative fires/{state}/{scene}/ or controls/{state}/{scene}/
    path under cache_dir, implicitly rebuilding the structure of the source dataset --
    source-agnostic, works the same for an S3 key prefix or a local folder path.
    """
    return os.path.join(cache_dir, cache_relpath.rstrip("/") + ".pkl")


def process_and_cache_scene(process_fn, cache_relpath, args, failures_log_path):
    """
    Runs process_fn() and either caches the result to disk (and optionally uploads to S3) or 
    logs the failure. cache_relpath (e.g. "fires/AK/CULTAS_CREEK_fire") does not depend on source
    and determines both the display name and the on-disk cache path. Returns True/False 
    for success/failure, or None if skipped because it was already cached.
    """
    scene_id = cache_relpath.rstrip("/").rsplit("/", 1)[-1]
    out_path = cache_path_for(args.cache_dir, cache_relpath)

    if args.skip_existing and os.path.exists(out_path):
        print(f"  [skip] {scene_id} (already cached)")
        return None

    print(f"  [processing] {scene_id} ...")
    try:
        record, error = run_with_timeout(process_fn)
    except SceneTimeout as e:
        record, error = None, f"{scene_id}: TIMEOUT -- {e}"

    if error:
        _record_failure(error, failures_log_path)
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(record, f)
    if args.upload_to_s3:
        s3.upload_file(out_path, BUCKET_NAME, args.s3_cache_prefix + cache_relpath.rstrip("/") + ".pkl")

    print(f"  [done] {scene_id} -> {len(record['tiles'])} tiles")
    return True


def main():
    """
    Processes every fire, then every control, and caches each to --cache-dir as full/path/name/{scene_id}.pkl
    Failures are appended to cache_failures.log as they happen for real-time monitoring.
    Safe to re-run with the --skip-existing flag to resume where it left off after an interruption.
    """
    args = build_arg_parser().parse_args()

    if args.source == "local" and not args.local_dir:
        raise ValueError("--local-dir is required when --source local is set.")

    global BUCKET_NAME, FIRES_PREFIX, CONTROLS_PREFIX
    BUCKET_NAME = args.bucket_name
    FIRES_PREFIX = args.fires_prefix
    CONTROLS_PREFIX = args.controls_prefix

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    failures_log_path = os.path.join(args.cache_dir, "cache_failures.log")
    n_failures = 0
    processed = 0

    for kind in ("fire", "control"):
        if args.source == "local":
            scenes = discover_scenes_local(args.local_dir, kind)
        else:
            scenes = discover_scenes(kind)
        if args.limit:
            scenes = scenes[: args.limit]
        print(f"\n=== {kind}: {len(scenes)} scenes (source: {args.source}) ===")

        for meta_ref, scene_ref in scenes:
            if args.source == "local":
                cache_relpath = os.path.relpath(scene_ref, args.local_dir).replace(os.sep, "/")
                process_fn = functools.partial(process_scene_local, meta_ref, scene_ref, kind, args.tmp_dir)
            else:
                cache_relpath = scene_ref.rstrip("/")
                process_fn = functools.partial(process_scene, meta_ref, scene_ref, kind, args.tmp_dir)

            result = process_and_cache_scene(process_fn, cache_relpath, args, failures_log_path)
            if result is True:
                processed += 1
            elif result is False:
                n_failures += 1

    print(f"\n{processed} scenes cached, {n_failures} failed.")
    if n_failures:
        print(f"Failure details in {failures_log_path}")


if __name__ == "__main__":
    main()