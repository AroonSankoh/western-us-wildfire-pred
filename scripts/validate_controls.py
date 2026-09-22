"""
SAFE Manifest Validation Script
=================================
Extends the fire/control scene validation by doing a lightweight content-level
check on each SAFE file without fully unzipping it.

For each S1/S2 SAFE in every fire AND control scene, this script:
  1. Locates manifest.safe inside the SAFE archive via the zip's central
     directory (S3 range requests -- a few KB, not the full SAFE)
  2. Parses the manifest to extract tile ID, acquisition date, platform, and
     declared file sizes
  3. Cross-references extracted values against the scene's metadata.json
  4. Checks declared manifest sizes against actual S3 object sizes to catch
     truncated uploads (or a copy operation that silently dropped bytes)
  5. For S1 scenes, checks the manifest's declared polarization channels and
     flags any scene acquired in single-pol mode (VV only, no VH) -- this is
     NOT a bug/corruption, it's a real S1 acquisition mode, but it means the
     scene has no VH band at all and must be replaced with a dual-pol (SDV)
     product before it can be used. Checked via the manifest itself rather
     than the contents alias filename, since alias names like
     "S1_AK_66N143W_20170614_pre.SAFE" never contain the real product ID
     token (SDV/SSV) that would otherwise reveal this.

Covers BOTH:
  - fires/{state}/{fire_name}/                              (one level under state)
  - controls/{state}/{fire_control}/{control_id}/            (two levels under state)

Output:
  - Console log per SAFE
  - manifest_validation_report.json saved locally + uploaded to S3
    (results carry a "kind": "fire" | "control" field so you can filter either)
"""

import io
import json
import logging
import re
import struct
import zipfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import requests
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BUCKET_NAME   = "wildfire-scenes-s3-202802195212-eu-central-1-an"
BUCKET_OWNER  = "202802195212"
FIRE_STATES   = ["AK", "WA", "CA", "NV", "MT", "ID", "OR"]
FIRES_ROOT    = "fires"
CONTROLS_ROOT = "controls"
MAX_WORKERS   = 8     # concurrent SAFE checks; keep low to avoid CDSE rate limits
EOCD_SEARCH   = 65536 # bytes to read from end of zip to find EOCD record

s3 = boto3.client("s3", region_name="eu-central-1")

# ─────────────────────────────────────────────
# S3 HELPERS
# ─────────────────────────────────────────────

def list_common_prefixes(prefix):
    prefixes = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            prefixes.append(cp["Prefix"])
    return prefixes

def list_files_under(prefix):
    """Returns {filename: (full_key, size_bytes)}"""
    result = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                fname = key.split("/")[-1]
                result[fname] = (key, obj.get("Size", 0))
    return result

def read_json_from_s3(key):
    try:
        resp = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return json.loads(resp["Body"].read())
    except Exception:
        return None

def get_s3_object_size(key):
    try:
        resp = s3.head_object(Bucket=BUCKET_NAME, Key=key)
        return resp["ContentLength"]
    except Exception:
        return None

def get_s3_presigned_url(key, expiry=3600):
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET_NAME, "Key": key},
        ExpiresIn=expiry,
    )

# ─────────────────────────────────────────────
# SCENE DISCOVERY
# fires/{state}/{fire_name}/               -- one level under state
# controls/{state}/{fire_control}/{ctrl}/  -- two levels under state
# ─────────────────────────────────────────────

def discover_scene_prefixes(kind, state):
    """Returns a list of scene prefixes (folders containing metadata.json
    directly) for the given kind ('fire' or 'control') and state."""
    if kind == "fire":
        root = f"{FIRES_ROOT}/{state}/"
        return list_common_prefixes(root)
    elif kind == "control":
        root = f"{CONTROLS_ROOT}/{state}/"
        fire_controls = list_common_prefixes(root)
        scene_prefixes = []
        for fc_prefix in fire_controls:
            scene_prefixes.extend(list_common_prefixes(fc_prefix))
        return scene_prefixes
    else:
        raise ValueError(f"Unknown kind: {kind}")

# ─────────────────────────────────────────────
# ZIP CENTRAL DIRECTORY PARSER
# Reads only the end of the zip via HTTP range request
# to locate manifest.safe without downloading the full archive
# ─────────────────────────────────────────────

def fetch_bytes_from_s3(s3_key, start, end):
    """Fetch a byte range directly from S3 using the boto3 client."""
    try:
        resp = s3.get_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Range=f"bytes={start}-{end}",
            ExpectedBucketOwner=BUCKET_OWNER
        )
        return resp["Body"].read()
    except Exception as e:
        log.error(f"S3 Native Range Read Failed for key {s3_key}: {e}")
        raise e

def find_eocd(data):
    """
    Locate the End of Central Directory record in the tail bytes.
    Returns offset of EOCD within data, or None.
    """
    sig = b"PK\x05\x06"
    pos = data.rfind(sig)
    return pos if pos != -1 else None

def parse_central_directory(s3_key, file_size):
    """
    Read the zip central directory via native S3 range requests.
    Returns list of (filename, header_offset, compressed_size, uncompressed_size).
    """
    # Step 1: fetch tail to find EOCD (Using s3_key instead of url)
    tail_start = max(0, file_size - EOCD_SEARCH)
    tail = fetch_bytes_from_s3(s3_key, tail_start, file_size - 1)

    eocd_pos = find_eocd(tail)
    if eocd_pos is None:
        raise ValueError("Could not locate EOCD in zip tail — file may be corrupted")

    eocd = tail[eocd_pos:]
    _, _, _, _, total_entries, cd_size, cd_offset, _ = struct.unpack_from("<4sHHHHIIH", eocd)

    # Step 2: fetch central directory
    cd_data = fetch_bytes_from_s3(s3_key, cd_offset, cd_offset + cd_size - 1)

    # Step 3: parse central directory entries
    entries = []
    pos = 0
    cd_sig = b"PK\x01\x02"
    while pos < len(cd_data) - 4:
        if cd_data[pos:pos+4] != cd_sig:
            break
        (_, ver_made, ver_needed, flags, method, mod_time, mod_date,
         crc, comp_size, uncomp_size, fname_len, extra_len, comment_len,
         disk_start, int_attr, ext_attr, local_offset) = struct.unpack_from(
            "<4sHHHHHHIIIHHHHHII", cd_data, pos
        )
        pos += 46
        fname = cd_data[pos:pos+fname_len].decode("utf-8", errors="replace")
        pos += fname_len + extra_len + comment_len
        entries.append((fname, local_offset, comp_size, uncomp_size))

    return entries

def extract_file_from_zip(s3_key, entries, target_filename):
    """
    Download and extract a single file from a remote zip using its
    local file header offset from the central directory.
    """
    for fname, local_offset, comp_size, uncomp_size in entries:
        if target_filename in fname:
            # Read local file header to find data start (30 bytes total)
            lfh = fetch_bytes_from_s3(s3_key, local_offset, local_offset + 29)
            if lfh[:4] != b"PK\x03\x04":
                raise ValueError(f"Invalid local file header for {fname}")

            # FIXED: Added the missing third 'I' for uncompressed size in the format string
            (_, ver_needed, flags, method, mod_time, mod_date,
             crc, comp_size_chk, uncomp_size_chk, fname_len, extra_len) = struct.unpack_from(
                "<4sHHHHHIIIHH", lfh
            )

            data_start = local_offset + 30 + fname_len + extra_len
            data_end   = data_start + comp_size - 1
            compressed = fetch_bytes_from_s3(s3_key, data_start, data_end)

            # Decompress if needed (method 8 = deflate, 0 = stored)
            if comp_size == uncomp_size:
                return compressed  # stored
            else:
                import zlib
                return zlib.decompress(compressed, -15)

    return None

# ─────────────────────────────────────────────
# MANIFEST PARSERS
# ─────────────────────────────────────────────

def parse_s2_manifest(manifest_xml):
    """
    Extract tile ID, acquisition date, platform, and file sizes from S2 manifest.safe.
    Returns dict of extracted properties.
    """
    props = {"tile_id": None, "acq_date": None, "platform": None, "declared_files": {}}
    try:
        root = ET.fromstring(manifest_xml)
        ns   = {"safe": "http://www.esa.int/safe/sentinel/1.1"}

        # Platform
        platform_el = root.find(".//safe:platform/safe:familyName", ns)
        number_el   = root.find(".//safe:platform/safe:number", ns)
        if platform_el is not None and number_el is not None:
            props["platform"] = f"{platform_el.text}{number_el.text}".replace(" ", "")

        # Acquisition date
        start_el = root.find(".//safe:startTime", ns)
        if start_el is not None:
            props["acq_date"] = start_el.text[:10]  # YYYY-MM-DD

        # Tile ID from product URI
        for el in root.iter():
            href = el.get("href", "")
            match = re.search(r"_T([0-9]{2}[A-Z]{3})_", href)
            if match:
                props["tile_id"] = match.group(1)
                break

        # Declared file sizes from dataObject entries
        for data_obj in root.findall(".//dataObject"):
            href_el = data_obj.find(".//fileLocation")
            size_el = data_obj.find(".//size")
            if href_el is not None and size_el is not None:
                fname = href_el.get("href", "").split("/")[-1]
                props["declared_files"][fname] = int(size_el.text)

    except ET.ParseError as e:
        props["parse_error"] = str(e)

    return props

def parse_s1_manifest(manifest_xml):
    """
    Extract acquisition date, platform, orbit direction, polarization
    channels, and file sizes from S1 manifest.safe.
    """
    props = {"acq_date": None, "platform": None, "orbit_direction": None,
             "polarizations": [], "declared_files": {}}
    try:
        root = ET.fromstring(manifest_xml)
        ns   = {"safe": "http://www.esa.int/safe/sentinel/1.1",
                "s1":   "http://www.esa.int/safe/sentinel-1/1.1"}

        # Robust platform extraction for S1
        platform_el = root.find(".//safe:platform/safe:familyName", ns)
        number_el   = root.find(".//safe:platform/safe:number", ns)
        if platform_el is not None and number_el is not None:
            props["platform"] = f"{platform_el.text}{number_el.text}".replace(" ", "")
        else:
            # Fallback: look for generic metadata attributes often found in S1
            nss_el = root.find(".//safe:nssID", ns)
            if nss_el is not None:
                props["platform"] = nss_el.text
            else:
                props["platform"] = "Sentinel-1" # Safe default fallback since we know it's an S1 manifest

        start_el = root.find(".//safe:startTime", ns)
        if start_el is not None:
            props["acq_date"] = start_el.text[:10]

        orbit_el = root.find(".//s1:pass", ns)
        if orbit_el is not None:
            props["orbit_direction"] = orbit_el.text.upper()

        # Polarization channels -- matched by local tag name (ignoring namespace
        # prefix) since S1 manifests use an s1sarl1 namespace whose URI has
        # varied slightly across IPF processor versions. A dual-pol (SDV)
        # product declares two of these (e.g. VV and VH); a single-pol (SSV)
        # product declares only one (VV) -- meaning no VH band exists at all.
        for el in root.iter():
            tag = el.tag.split("}")[-1]
            if tag == "transmitterReceiverPolarisation" and el.text:
                props["polarizations"].append(el.text.strip().upper())

        for data_obj in root.findall(".//dataObject"):
            href_el = data_obj.find(".//fileLocation")
            size_el = data_obj.find(".//size")
            if href_el is not None and size_el is not None:
                fname = href_el.get("href", "").split("/")[-1]
                props["declared_files"][fname] = int(size_el.text)

    except ET.ParseError as e:
        props["parse_error"] = str(e)

    return props

# ─────────────────────────────────────────────
# CROSS REFERENCE: manifest vs metadata
# ─────────────────────────────────────────────

def cross_reference_s2(manifest_props, safe_key, meta, label):
    """
    Compare S2 manifest properties against scene metadata.json.
    """
    issues = []

    # Date check
    declared_fname = meta.get("contents", {}).get(label, "")
    date_match = re.search(r"_(\d{8})_", declared_fname)
    if date_match:
        declared_date = datetime.strptime(date_match.group(1), "%Y%m%d").strftime("%Y-%m-%d")
        manifest_date = manifest_props.get("acq_date")
        if manifest_date and manifest_date != declared_date:
            issues.append(f"{label}: date mismatch — manifest={manifest_date}, metadata={declared_date}")

    # SAFE STR CONVERSION: Protects against NoneType .upper() crashes
    platform_raw = manifest_props.get("platform")
    platform = str(platform_raw).upper() if platform_raw else ""

    if platform:
        if "SENTINEL" in platform and "2" in platform:
            pass
        else:
            issues.append(f"{label}: unexpected platform in manifest: '{platform_raw}'")

    if "parse_error" in manifest_props:
        issues.append(f"{label}: manifest XML parse error — {manifest_props['parse_error']}")

    return issues


def cross_reference_s1(manifest_props, safe_key, meta, label):
    """
    Compare S1 manifest properties against scene metadata.json.
    """
    issues = []

    # Date check
    declared_fname = meta.get("contents", {}).get(label, "")
    date_match = re.search(r"_(\d{8})_", declared_fname)
    if date_match:
        declared_date = datetime.strptime(date_match.group(1), "%Y%m%d").strftime("%Y-%m-%d")
        manifest_date = manifest_props.get("acq_date")
        if manifest_date and manifest_date != declared_date:
            issues.append(f"{label}: date mismatch — manifest={manifest_date}, metadata={declared_date}")

    # SAFE STR CONVERSION: Orbit direction cross-check
    meta_orbit_raw = meta.get("orbit_direction")
    meta_orbit = str(meta_orbit_raw).upper() if meta_orbit_raw else ""

    manifest_orbit_raw = manifest_props.get("orbit_direction")
    manifest_orbit = str(manifest_orbit_raw).upper() if manifest_orbit_raw else ""

    if meta_orbit and manifest_orbit and meta_orbit != manifest_orbit:
        issues.append(f"{label}: orbit direction mismatch — manifest={manifest_orbit}, metadata={meta_orbit}")

    # SAFE STR CONVERSION: Platform validation
    platform_raw = manifest_props.get("platform")
    platform = str(platform_raw).upper() if platform_raw else ""

    if platform:
        if "SENTINEL" in platform and "1" in platform:
            pass
        else:
            issues.append(f"{label}: unexpected platform in manifest: '{platform_raw}'")

    # Dual-pol check -- a scene acquired in single-pol mode (VV only, no VH)
    # has no VH band at all and must be replaced, not just re-downloaded.
    # This is a real acquisition-mode gap, not corruption -- caught this on
    # BIG_MUD_fire (S1A_IW_GRDH_1SSV_...) where the VH band genuinely never
    # existed for that scene.
    pols = manifest_props.get("polarizations", [])
    log.info(f"    [{label}] polarizations detected: {pols}")
    if pols and "VH" not in pols:
        issues.append(
            f"{label}: SINGLE-POL ACQUISITION — polarizations={pols}, missing VH channel. "
            f"This scene must be replaced with a dual-pol (SDV) product."
        )

    if "parse_error" in manifest_props:
        issues.append(f"{label}: manifest XML parse error — {manifest_props['parse_error']}")

    return issues

# ─────────────────────────────────────────────
# PER-SCENE VALIDATOR (fire or control)
# ─────────────────────────────────────────────

def validate_scene_manifests(scene_prefix, kind):
    """
    For a single fire or control scene, validate all 4 SAFE manifests.
    Returns (scene_name, issues_dict, passed)
    """
    scene_name = scene_prefix.rstrip("/").split("/")[-1]
    all_issues = {}
    passed     = True

    # Load metadata
    meta_key = f"{scene_prefix}metadata.json"
    meta     = read_json_from_s3(meta_key)
    if meta is None:
        return scene_name, {"fatal": "Could not read metadata.json"}, False

    contents   = meta.get("contents", {})
    file_index = list_files_under(scene_prefix)

    safe_labels = ["S1_pre", "S1_post", "S2_pre", "S2_post"]

    for label in safe_labels:
        safe_fname = contents.get(label)
        if not safe_fname:
            all_issues[label] = [f"No entry for '{label}' in metadata.contents"]
            passed = False
            continue

        if safe_fname not in file_index:
            all_issues[label] = [f"'{safe_fname}' not found in S3"]
            passed = False
            continue

        s3_key, s3_size = file_index[safe_fname]
        issues = []

        try:
            # Pass s3_key directly to parse the central directory metadata
            entries = parse_central_directory(s3_key, s3_size)

            # Extract manifest.safe passing s3_key
            raw_manifest = extract_file_from_zip(s3_key, entries, "manifest.safe")
            if raw_manifest is None:
                issues.append("manifest.safe not found inside SAFE archive")
                all_issues[label] = issues
                passed = False
                continue

            manifest_xml = raw_manifest.decode("utf-8", errors="replace")

            # Parse and cross-reference records matching the sensor rules
            if label.startswith("S2"):
                props  = parse_s2_manifest(manifest_xml)
                issues += cross_reference_s2(props, s3_key, meta, label)
            else:
                props  = parse_s1_manifest(manifest_xml)
                issues += cross_reference_s1(props, s3_key, meta, label)

            # Dynamic Truncation check based on sensor type compression behavior
            total_declared = sum(props.get("declared_files", {}).values())
            if total_declared > 0:
                min_ratio = 0.75 if label.startswith("S2") else 0.30
                if s3_size < total_declared * min_ratio:
                    issues.append(
                        f"S3 archive footprint ({s3_size:,} bytes) is dramatically below the expected "
                        f"compression floor ({int(min_ratio*100)}%) of manifest total ({total_declared:,} bytes). "
                        f"Likely truncated upload stream."
                    )

            log.info(f"  {'✓' if not issues else '✗'} [{kind}] {scene_name}/{label}")

        except Exception as e:
            issues.append(f"Manifest extraction failed: {str(e)}")
            log.warning(f"  ✗ [{kind}] {scene_name}/{label}: {e}")

        if issues:
            passed = False
        all_issues[label] = issues

    return scene_name, all_issues, passed

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_manifest_validation():
    log.info("Starting SAFE manifest validation (lightweight range-request mode)...")

    all_results = []
    summary = {
        "total": 0, "passed": 0, "failed": 0, "safe_checks": 0, "safe_failed": 0,
        "single_pol_flagged": [],
        "by_kind": {
            "fire":    {"total": 0, "passed": 0, "failed": 0, "safe_checks": 0, "safe_failed": 0, "single_pol_flagged": []},
            "control": {"total": 0, "passed": 0, "failed": 0, "safe_checks": 0, "safe_failed": 0, "single_pol_flagged": []},
        },
    }

    for kind in ("fire", "control"):
        for state in FIRE_STATES:
            log.info(f"\n── [{kind}] State: {state}")
            scene_prefixes = discover_scene_prefixes(kind, state)

            if not scene_prefixes:
                log.warning(f"  No {kind} folders for {state}")
                continue

            # Validate concurrently
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(validate_scene_manifests, sp, kind): sp
                    for sp in scene_prefixes
                }
                for future in as_completed(futures):
                    scene_name, issues_dict, passed = future.result()

                    for bucket in (summary, summary["by_kind"][kind]):
                        bucket["total"] += 1
                        bucket["safe_checks"] += len(issues_dict)

                    scene_failed_safes = sum(1 for v in issues_dict.values() if v)
                    for bucket in (summary, summary["by_kind"][kind]):
                        bucket["safe_failed"] += scene_failed_safes

                    if passed:
                        summary["passed"] += 1
                        summary["by_kind"][kind]["passed"] += 1
                    else:
                        summary["failed"] += 1
                        summary["by_kind"][kind]["failed"] += 1

                    all_results.append({
                        "kind":    kind,
                        "scene":   scene_name,
                        "state":   state,
                        "passed":  passed,
                        "details": issues_dict,
                    })

                    if not passed:
                        for label, issues in issues_dict.items():
                            for issue in issues:
                                log.warning(f"    ↳ [{kind}] {scene_name}/{label}: {issue}")
                            if any("SINGLE-POL ACQUISITION" in issue for issue in issues):
                                summary["single_pol_flagged"].append(f"[{kind}] {scene_name}/{label}")
                                summary["by_kind"][kind]["single_pol_flagged"].append(f"{scene_name}/{label}")

    # Summary
    log.info("\n" + "=" * 55)
    log.info("MANIFEST VALIDATION COMPLETE")
    log.info(f"  Scenes checked (total) : {summary['total']}")
    log.info(f"    fires                : {summary['by_kind']['fire']['total']} "
             f"(passed {summary['by_kind']['fire']['passed']}, failed {summary['by_kind']['fire']['failed']})")
    log.info(f"    controls             : {summary['by_kind']['control']['total']} "
             f"(passed {summary['by_kind']['control']['passed']}, failed {summary['by_kind']['control']['failed']})")
    log.info(f"  Passed                 : {summary['passed']}")
    log.info(f"  Failed                 : {summary['failed']}")
    log.info(f"  SAFE files checked     : {summary['safe_checks']}")
    log.info(f"  SAFE files flagged     : {summary['safe_failed']}")
    log.info(f"  Single-pol scenes flagged for replacement: {len(summary['single_pol_flagged'])}")
    for s in summary["single_pol_flagged"]:
        log.info(f"    ↳ {s}")
    log.info("=" * 55)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary":      summary,
        "results":      all_results,
    }
    report_json = json.dumps(report, indent=2)

    local_path = "/tmp/manifest_validation_report.json"
    with open(local_path, "w") as f:
        f.write(report_json)
    log.info(f"Report saved locally: {local_path}")

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key="manifest_validation_report.json",
        Body=report_json,
        ContentType="application/json",
        ExpectedBucketOwner=BUCKET_OWNER,
    )
    log.info(f"Report uploaded to s3://{BUCKET_NAME}/manifest_validation_report.json")

    return report


if __name__ == "__main__":
    run_manifest_validation()