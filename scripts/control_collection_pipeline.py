"""
Control Scene Collection Pipeline
==================================
Collects 3 control (non-fire) scenes per fire scene from S3, matched by:
  - Same EPA Level III ecoregion
  - Random date within fire-risk season (Jun–Sep)
  - 20x20 km fixed tile centered on sampled centroid
  - Min 20 km separation between all centroids (controls + fires)

Output S3 structure:
  controls/
    AK/
      control_AK_65N153W_20180712/
        S1_pre, S2_pre, S1_post, S2_post, ERA5, metadata.json
"""

import os
import json
import random
import logging
import warnings
from datetime import timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from boto3.s3.transfer import TransferConfig
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import pystac_client
import planetary_computer
import requests
import cdsapi
import re
import time
import math
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

from pyproj import Transformer
from botocore.config import Config
from rasterio.windows import from_bounds
from shapely.geometry import Point, box
from shapely.ops import unary_union, shape

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Force botocore to completely forget about configuration profiles
os.environ.pop("AWS_PROFILE", None)
os.environ.pop("AWS_DEFAULT_PROFILE", None)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

US_ECO_SHP        = "/home/ubuntu/alaska-wildfire-pred/EPA/us_eco_l3_state_boundaries.shp"
AK_ECO_SHP        = "/home/ubuntu/alaska-wildfire-pred/EPA/ak_eco_l3.shp"
BUCKET_NAME       = "wildfire-scenes-s3-202802195212-eu-central-1-an"
BUCKET_OWNER      = "202802195212"
CONTROLS_PREFIX   = "controls"
FIRE_STATES       = ["AK", "WA", "CA", "NV", "MT", "ID", "OR"]

TILE_SIZE_KM      = 20
HALF_DEG          = (TILE_SIZE_KM / 2) / 111.0   # ~0.09 degrees
MIN_DIST_KM       = 20
CONTROLS_PER_FIRE = 3
MAX_ATTEMPTS_PER_FIRE = 100
CLOUD_THRESH      = 20.0
MIN_VALID_PIX     = 0.80
FIRE_SEASON       = (6, 9)
YEAR_RANGE        = (2016, 2024)
MAX_SAMPLE_TRIES  = 50
MAX_SENSOR_DELTA_HOURS = 36

S3_TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=1024 * 1024 * 16,
    multipart_chunksize=1024 * 1024 * 16,
    max_concurrency=4,
    use_threads=True,
    max_io_queue=100,
    io_chunksize=1024 * 1024 * 2,
)

# ─────────────────────────────────────────────
# AWS + CATALOG
# ─────────────────────────────────────────────

session = boto3.Session()
s3 = session.client("s3", region_name="eu-central-1")

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def format_coordinate_strings(stac):
    """Mirrors fire pipeline: centroid → rounded lat/lon string."""
    min_long, min_lat, max_long, max_lat = stac
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_long + max_long) / 2.0
    lat_suffix = "N" if center_lat >= 0 else "S"
    lon_suffix = "E" if center_lon >= 0 else "W"
    lat_val = abs(int(round(center_lat)))
    lon_val = abs(int(round(center_lon)))
    return f"{lat_val}{lat_suffix}{lon_val}{lon_suffix}"

def get_cdse_token():
    auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    payload = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": os.getenv("CDSE_EMAIL"),
        "password": os.getenv("CDSE_PASSWORD"),
    }
    r = requests.post(auth_url, data=payload)
    r.raise_for_status()
    return r.json()["access_token"]

class ProgressPercentage:
    def __init__(self, filename, total_size):
        self._pbar = tqdm(
            desc=f"Uploading {filename}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=True,
        )
    def __call__(self, bytes_amount):
        self._pbar.update(bytes_amount)
    def close(self):
        self._pbar.close()

# ─────────────────────────────────────────────
# SPATIAL DATA
# ─────────────────────────────────────────────

def load_spatial_data():

    log.info("Loading and aligning EPA Level III ecoregions...")
    
    # 1. Load conterminous US ecoregions
    eco_us = gpd.read_file(US_ECO_SHP).to_crs(epsg=4326)
    
    # 2. Append Alaska if the file exists
    if os.path.exists(AK_ECO_SHP):
        log.info("Alaska ecoregions file detected. Merging layers...")
        eco_ak = gpd.read_file(AK_ECO_SHP).to_crs(epsg=4326)
        
        # Ensure column alignment matches for the spatial concat
        eco = gpd.GeoDataFrame(pd.concat([eco_us, eco_ak], ignore_index=True), crs="EPSG:4326")
    else:
        log.warning("ERROR: Alaska ecoregions file not found! Only processing lower-48 points.")
        eco = eco_us

    # Keep only columns required for execution to minimize RAM footprint
    eco = eco[['geometry', 'US_L3CODE', 'US_L3NAME']]
    
    log.info(f"Successfully unified {len(eco)} total spatial ecoregion polygons.")
    return None, eco

# ─────────────────────────────────────────────
# FIRE METADATA
# ─────────────────────────────────────────────

def load_all_fire_metadata():
    fire_scenes = []
    for state in FIRE_STATES:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=f"fires/{state}/"):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if "controls/" in key: # safeguard
                    continue
                if key.endswith("metadata.json"):
                    resp = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                    fire_scenes.append(json.loads(resp["Body"].read()))
    log.info(f"Loaded {len(fire_scenes)} true anchor fire metadata files.")
    return fire_scenes

def load_existing_control_centroids():
    centroids = []
    for state in FIRE_STATES:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=f"{CONTROLS_PREFIX}/{state}/"):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("metadata.json"):
                    resp = s3.get_object(Bucket=BUCKET_NAME, Key=obj["Key"])
                    meta = json.loads(resp["Body"].read())
                    b = meta["spatial_bounds"]
                    centroids.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    return centroids

# ─────────────────────────────────────────────
# DISTANCE + SAMPLING
# ─────────────────────────────────────────────

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def too_close(lon, lat, existing_centroids):
    if not existing_centroids:
        return False
    arr = np.array(existing_centroids)
    lons, lats = arr[:, 0], arr[:, 1]
    R = 6371.0
    dlat = np.radians(lats - lat)
    dlon = np.radians(lons - lon)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat)) * np.cos(np.radians(lats)) * np.sin(dlon/2)**2
    return np.any(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) < MIN_DIST_KM)

def sample_control_centroid(eco_geom, fire_exclusion_zone, existing_centroids):
    minx, miny, maxx, maxy = eco_geom.bounds
    for _ in range(MAX_SAMPLE_TRIES):
        lon = random.uniform(minx, maxx)
        lat = random.uniform(miny, maxy)
        pt = Point(lon, lat)
        if not eco_geom.contains(pt):
            continue
        if fire_exclusion_zone.contains(pt):
            continue
        if too_close(lon, lat, existing_centroids):
            continue
        return lon, lat
    return None

def sample_control_date():
    year  = random.randint(*YEAR_RANGE)
    month = random.randint(*FIRE_SEASON)
    days_in_month = 31 if month in [7, 8] else 30
    return date(year, month, random.randint(1, days_in_month))

# ─────────────────────────────────────────────
# QC CHECKS
# ─────────────────────────────────────────────

def scl_cloud_fraction(s2_item, bbox):
    try:
        with rasterio.open(s2_item.assets["SCL"].href) as src:
            # 1. Project degrees to raster's native UTM meters
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            minx, miny = transformer.transform(bbox[0], bbox[1])
            maxx, maxy = transformer.transform(bbox[2], bbox[3])
            
            # 2. Build the correct pixel window from native bounds
            win = from_bounds(minx, miny, maxx, maxy, src.transform)
            scl = src.read(1, window=win)
            
        if scl.size == 0:
            return 1.0  # Safe fallback for out-of-bounds slices
            
        bad = np.isin(scl, [3, 8, 9])
        return float(bad.sum()) / scl.size
    except Exception as e:
        log.warning(f"SCL check failed due to spatial indexing: {e}")
        return 1.0

def calculate_spatial_coverage_percentage(stac_item, bbox):
    """
    Calculates what percentage of the micro-bbox is covered 
    by the STAC item's geographic footprint.
    """
    try:
        # 1. Convert your micro-bbox tuple into a Shapely polygon
        micro_box_geom = box(*bbox)
        
        # 2. Extract the satellite scene's footprint geometry
        scene_geom = shape(stac_item.geometry)
        
        # 3. Calculate the intersection area
        intersection_area = micro_box_geom.intersection(scene_geom).area
        bbox_area = micro_box_geom.area
        
        if bbox_area == 0:
            return 0.0
            
        # 4. Return the coverage ratio as a percentage
        return round((intersection_area / bbox_area) * 100.0, 2)
    except Exception as e:
        log.warning(f"Failed geometric coverage calculation: {e}")
        return 0.0

def valid_pixel_fraction(s2_item, bbox, band="B04"):
    try:
        with rasterio.open(s2_item.assets[band].href) as src:
            # 1. Project degrees to raster's native UTM meters
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            minx, miny = transformer.transform(bbox[0], bbox[1])
            maxx, maxy = transformer.transform(bbox[2], bbox[3])
            
            # 2. Build the correct pixel window from native bounds
            win = from_bounds(minx, miny, maxx, maxy, src.transform)
            data = src.read(1, window=win).astype(float)
            
        if data.size == 0:
            return 0.0
            
        return float(((data > 0) & np.isfinite(data)).sum()) / data.size
    except Exception as e:
        log.warning(f"Valid pixel check failed due to spatial indexing: {e}")
        return 0.0

def get_s2_shape(s2_item, bbox, band="B04"):
    try:
        with rasterio.open(s2_item.assets[band].href) as src:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            minx, miny = transformer.transform(bbox[0], bbox[1])
            maxx, maxy = transformer.transform(bbox[2], bbox[3])
            
            win = from_bounds(minx, miny, maxx, maxy, src.transform)
            return src.read(1, window=win).shape
    except Exception as e:
        return None

def validate_sensor_time_delta(s1_pre_name, s1_post_name, s2_pre_name, s2_post_name, max_hours=120.0):
    """
    Checks the string names of CDSE products before download.
    Returns True if the sensor time delta is within limits, False otherwise.
    """
    def extract_dt_from_name(prod_name):
        if not prod_name:
            return None
        match = re.search(r'_(20\d{6}T\d{6})_', prod_name)
        if match:
            return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        return None

    s1_pre_dt  = extract_dt_from_name(s1_pre_name)
    s1_post_dt = extract_dt_from_name(s1_post_name)
    s2_pre_dt  = extract_dt_from_name(s2_pre_name)
    s2_post_dt = extract_dt_from_name(s2_post_name)

    if all([s1_pre_dt, s1_post_dt, s2_pre_dt, s2_post_dt]):
        pre_sensor_diff_hours = round(abs((s1_pre_dt - s2_pre_dt).total_seconds()) / 3600.0, 2)
        post_sensor_diff_hours = round(abs((s1_post_dt - s2_post_dt).total_seconds()) / 3600.0, 2)
        
        if pre_sensor_diff_hours > max_hours or post_sensor_diff_hours > max_hours:
            return False, pre_sensor_diff_hours, post_sensor_diff_hours
            
        return True, pre_sensor_diff_hours, post_sensor_diff_hours
        
    return False, None, None  # Reject if we fail to parse timestamps entirely

# ─────────────────────────────────────────────
# SENTINEL QUERIES (PLANETARY COMPUTER)
# ─────────────────────────────────────────────

def query_s2(bbox, date_center):
    d = pd.Timestamp(date_center)
    time_range = f"{(d - timedelta(days=15)).date()}/{(d + timedelta(days=15)).date()}"
    
    # Defensive micro-sleep to pace API hits and prevent rate limiting
    time.sleep(0.5) 
    
    items = []
    max_retries = 3
    for retry in range(max_retries):
        try:
            items = list(catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=time_range,
                query={"eo:cloud_cover": {"lt": CLOUD_THRESH}}, # Dynamic query boundary
            ).items())
            break # Success, break out of retry loop
            
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                sleep_time = (retry + 1) * 5
                log.warning(f"⚠️ STAC Rate limit hit. Cooling down for {sleep_time}s before retry...")
                time.sleep(sleep_time)
                continue
            else:
                log.warning(f"S2 query API execution failed: {e}")
                return None
    
    if not items:
        log.debug(f"DEBUG STAC: Zero tiles found for date window around {date_center}")
        return None

    # Sort items by lowest metadata cloud cover first so we analyze the cleanest data
    items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))

    # Fallback/first sanity check on the top item's metadata
    top_cloud_cover = items[0].properties.get("eo:cloud_cover", 100)
    if top_cloud_cover > CLOUD_THRESH:
        log.debug(f"DEBUG QC: Top item rejected due to clouds ({top_cloud_cover:.1f}% > {CLOUD_THRESH}%)")
        return None

    # Exhaustive structural array QC via custom fractions
    for idx, item in enumerate(items):
        thresh_fraction = CLOUD_THRESH / 100.0 if CLOUD_THRESH > 1.0 else CLOUD_THRESH
        
        cloud_frac = scl_cloud_fraction(item, bbox)
        valid_frac = valid_pixel_fraction(item, bbox)
        
        if cloud_frac <= thresh_fraction and valid_frac >= MIN_VALID_PIX:
            return item
            
    log.debug(f"DEBUG QC: All {len(items)} scenes failed SCL array validation.")
    return None

def query_s1(bbox, date_center, orbit_direction="ASCENDING"):
    d = pd.Timestamp(date_center)
    time_range = f"{(d - timedelta(days=15)).date()}/{(d + timedelta(days=15)).date()}"
    try:
        items = list(catalog.search(
            collections=["sentinel-1-grd"],
            bbox=bbox,
            datetime=time_range,
            query={"sat:orbit_state": {"eq": orbit_direction.lower()}},
        ).items())
        dual_pol_items = [i for i in items if "VV" in i.properties.get("sar:polarizations", []) and "VH" in i.properties.get("sar:polarizations", [])]
        return dual_pol_items[0] if dual_pol_items else None
    except Exception as e:
        log.warning(f"S1 query failed: {e}")
        return None

# ─────────────────────────────────────────────
# CDSE PINPOINTING (mirrors fire pipeline)
# ─────────────────────────────────────────────

def pinpoint_s2_scene_on_cdse(stac_item):
    tile_id  = stac_item.properties.get("s2:mgrs_tile")
    date_str = stac_item.datetime.strftime("%Y-%m-%d")

    query_url = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter=Collection/Name eq 'SENTINEL-2' and "
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'tileId' and att/att/Value eq '{tile_id}') and "
        f"ContentDate/Start ge {date_str}T00:00:00.000Z and "
        f"ContentDate/Start le {date_str}T23:59:59.999Z"
    )
    
    response_data = None
    max_retries = 3
    for retry in range(max_retries):
        try:
            r = requests.get(query_url, timeout=15)
            r.raise_for_status()  # Catches raw HTML errors (like 429 or 502) early
            response_data = r.json()
            break
        except Exception as e:
            if retry == max_retries - 1:
                log.error(f"CDSE S2 API failed completely after {max_retries} attempts: {e}")
                return None
            sleep_time = (retry + 1) * 5
            log.warning(f"⚠️ CDSE S2 API hiccup ({e}). Retrying in {sleep_time}s...")
            time.sleep(sleep_time)

    if not response_data:
        return None

    raw_matches = response_data.get("value", [])
    if not raw_matches:
        log.warning(f"No CDSE matches for S2 tile {tile_id} on {date_str}")
        return None

    l2a_matches = [p for p in raw_matches if "_MSIL2A_" in p["Name"]]
    if not l2a_matches:
        log.warning(f"No L2A matches for S2 tile {tile_id} on {date_str}")
        return None

    log.info(f"S2 pinpointed: {l2a_matches[0]['Name']}")
    return l2a_matches[0]

def pinpoint_s1_scene_on_cdse(stac_item):
    core_fingerprint = "_".join(stac_item.id.split("_")[:8])
    query_url = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter=Collection/Name eq 'SENTINEL-1' and "
        f"contains(Name, '{core_fingerprint}') and "
        f"contains(Name, '_COG')"
    )
    
    response_data = None
    max_retries = 3
    for retry in range(max_retries):
        try:
            r = requests.get(query_url, timeout=15)
            r.raise_for_status()  # Catches raw HTML errors early
            response_data = r.json()
            break
        except Exception as e:
            if retry == max_retries - 1:
                log.error(f"CDSE S1 API failed completely after {max_retries} attempts: {e}")
                return None
            sleep_time = (retry + 1) * 5
            log.warning(f"⚠️ CDSE S1 API hiccup ({e}). Retrying in {sleep_time}s...")
            time.sleep(sleep_time)

    if not response_data:
        return None

    matches  = response_data.get("value", [])
    if not matches:
        log.warning(f"No CDSE match for S1 fingerprint {core_fingerprint}")
        return None

    log.info(f"S1 pinpointed: {matches[0]['Name']}")
    return matches[0]

# ─────────────────────────────────────────────
# STREAMING UPLOAD (mirrors fire pipeline)
# ─────────────────────────────────────────────

def stream_product_to_s3(key, product, s3_folder_prefix, state_code, bbox):
    source, phase = key.split("_")
    raw_time      = product["ContentDate"]["Start"].replace("Z", "")
    timestamp_str = pd.to_datetime(raw_time).strftime("%Y%m%d")
    coord_str     = format_coordinate_strings(bbox)
    s3_filename   = f"{source}_{state_code.upper()}_{coord_str}_{timestamp_str}_{phase}.SAFE"
    s3_key        = f"{s3_folder_prefix}{s3_filename}"

    s3_path      = product.get("S3Path", "")
    download_url = (
        f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product['Id']})/$value"
        if not s3_path or s3_path.startswith("/eodata/")
        else s3_path
    )

    token   = get_cdse_token()
    headers = {"Authorization": f"Bearer {token}"}

    with requests.get(download_url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        progress   = ProgressPercentage(s3_filename, total_size)
        s3.upload_fileobj(
            Fileobj=r.raw,
            Bucket=BUCKET_NAME,
            Key=s3_key,
            ExtraArgs={"ExpectedBucketOwner": BUCKET_OWNER},
            Callback=progress,
            Config=S3_TRANSFER_CONFIG,
        )
        progress.close()

    log.info(f"✓ {key}: {s3_key}")
    return key, s3_filename

# ─────────────────────────────────────────────
# ERA5 DOWNLOAD (mirrors fire pipeline)
# ─────────────────────────────────────────────

def calculate_master_era5_area_from_items(cdse_products, padding_degrees=2):
    """
    Takes a list of Pystac Items (or STAC item dicts), extracts their 
    geographic footprints, and returns a single unified floor/ceil 
    ERA5 area envelope [North, West, South, East]. Pads outward by
    padding_degrees -- the catalog GeoFootprint isn't always a tight
    superset of the actual downloaded raster's footprint, and ERA5 is
    cheap enough that a generous margin costs nothing.
    """
    all_min_lon = []
    all_max_lon = []
    all_min_lat = []
    all_max_lat = []
    
    for product in cdse_products:
        if isinstance(product, dict) and 'GeoFootprint' in product:
            # Parse the CDSE GeoJSON footprint directly into a Shapely geometry
            geom_bounds = shape(product['GeoFootprint']).bounds
            
            all_min_lon.append(geom_bounds[0])
            all_min_lat.append(geom_bounds[1])
            all_max_lon.append(geom_bounds[2])
            all_max_lat.append(geom_bounds[3])
        else:
            raise ValueError("Invalid input: Expected a CDSE product dictionary containing 'GeoFootprint'.")
            
    # Calculate absolute outer integer thresholds matching native ERA5 spatial resolution, then pad outward
    master_era5_area = [
        math.ceil(max(all_max_lat)) + padding_degrees,   # North
        math.floor(min(all_min_lon)) - padding_degrees,  # West
        math.floor(min(all_min_lat)) - padding_degrees,  # South
        math.ceil(max(all_max_lon)) + padding_degrees    # East
    ]
    
    log.info(f"Spatial Envelope Rounded + Padded Coordinates [N, W, S, E]: {master_era5_area}")
    return master_era5_area


def download_control_era5(era5_area, control_date, s3_client, bucket_name, s3_folder_prefix, state_code, coord_str):
    """
    Downloads antecedent ERA5 GRIB data for a control scene window to /tmp then streams directly to S3.
    Matches the master aggregation structure of your primary fire script.
    """
    control_datetime = pd.to_datetime(control_date)
    start_date = control_datetime - timedelta(days=30)
    
    log.info(f"Control Target Date: {control_datetime.strftime('%Y-%m-%d')}")
    log.info(f"30-Day Control Window: {start_date.strftime('%Y-%m-%d')} to {control_datetime.strftime('%Y-%m-%d')}")
    
    # Isolate years and months spanned by this 30-day index window
    date_range = pd.date_range(start=start_date, end=control_datetime)
    years_list = sorted(list(set(date_range.strftime("%Y"))))
    months_list = sorted(list(set(date_range.strftime("%m"))))
    days_list = sorted(list(set(date_range.strftime("%d"))))
    
    c = cdsapi.Client(quiet=True)
    
    era5_timestamp = control_datetime.strftime('%Y%m%d')
    era5_filename = f"ERA5_{state_code.upper()}_{coord_str}_{era5_timestamp}.grib"
    era5_s3_key = f"{s3_folder_prefix}{era5_filename}"
    tmp_path = f"/tmp/{era5_filename}"
    
    request_params = {
        'product_type': 'reanalysis',
        'data_format': 'grib',  # Standard explicit parameter for current CDS engine
        'variable': [
            '10m_u_component_of_wind', 
            '10m_v_component_of_wind', 
            '2m_dewpoint_temperature',
            '2m_temperature', 
            'total_precipitation'
        ],
        'year': list(years_list),
        'month': list(months_list),
        'day': list(days_list),
        'time': [f'{h:02d}:00' for h in range(24)],
        'area': era5_area,
    }
    
    log.info("Transmitting download request token to Copernicus Climate Data Store...")
    try:
        c.retrieve('reanalysis-era5-single-levels', request_params, tmp_path)
        log.info(f"ERA5 GRIB downloaded to {tmp_path}, streaming to S3...")
        
        file_size = os.path.getsize(tmp_path)
        progress = ProgressPercentage(era5_filename, file_size)
        
        s3_client.upload_file(
            Filename=tmp_path,
            Bucket=bucket_name,
            Key=era5_s3_key,
            ExtraArgs={'ExpectedBucketOwner': '202802195212'},
            Callback=progress
        )
        progress.close()
        return era5_filename
        
    except Exception as e:
        log.error(f"Failed processing ERA5 download pipeline segment: {e}")
        return None
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
# ─────────────────────────────────────────────
# CONCURRENT UPLOAD ORCHESTRATOR
# ─────────────────────────────────────────────

def upload_control_scene(cdse_products, control_date, era5_area, state_code, s3_folder_prefix, coord_str, bbox):
    """
    Upload all 4 SAFE files + ERA5 concurrently, mirroring fire pipeline.
    Returns dict of uploaded filenames or raises on any failure.
    """
    uploaded_assets = {}

    # Crucial: Ensure your s3 client initialization is accessible or pass it here
    # Assuming 's3' client instance is a global or initialized up top
    global s3 

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit the 4 CDSE satellite assets using your regular bbox parameters
        # (Assuming your satellite streaming function needs bbox for something)
        # Fix: We pass bbox back here if it's required by your cdse wrapper, 
        # or handle it according to your existing parameters.
        futures = {
            executor.submit(
                stream_product_to_s3, key, product, s3_folder_prefix, state_code, bbox
            ): key
            for key, product in cdse_products.items()
        }
        
        # Submit the ERA5 worker with its master grid coordinates and coordinate identification string
        era5_future = executor.submit(
            download_control_era5, 
            era5_area,            # Pass the master grid coordinate list [N, W, S, E]
            control_date, 
            s3,                   # Pass your verified boto3 s3 client
            BUCKET_NAME,          # Your destination bucket name constant
            s3_folder_prefix, 
            state_code, 
            coord_str
        )
        futures[era5_future] = "ERA5"

        for future in as_completed(futures):
            key = futures[future]
            try:
                if key == "ERA5":
                    uploaded_assets["ERA5"] = future.result()
                else:
                    k, fname = future.result()
                    uploaded_assets[k] = fname
            except Exception as e:
                log.error(f"Upload failed for {key}: {e}")
                raise

    return uploaded_assets

# ─────────────────────────────────────────────
# TRACKER FOR ALREADY EXISTING FIRE CONTROLS
# ─────────────────────────────────────────────

def count_existing_s3_controls(state, fire_name_control):
    """
    Counts the number of unique control scene subdirectories already uploaded to S3
    for a specific anchor fire.
    """
    s3_client = boto3.client('s3')
    
    # Target the exact parent prefix folder we just designed
    prefix = f"{CONTROLS_PREFIX}/{state}/{fire_name_control}/"
    
    response = s3_client.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=prefix,
        Delimiter='/'  # This groups results by subfolder instead of listing every file
    )
    
    # 'CommonPrefixes' contains the subdirectories (e.g., your control_id folders)
    if 'CommonPrefixes' in response:
        return len(response['CommonPrefixes'])
    
    return 0

# ─────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────

def write_metadata(control_id, state, control_date, bbox,
                   uploaded_assets, s1_pre, s1_post, s2_pre, s2_post,
                   cdse_products, fire_name, fire_event_id, s3_folder_prefix, orbit_direction):
    
    # Extract precise high-fidelity timestamps from the actual CDSE filenames
    # (e.g. extracts from internal product names or uploaded keys)
    def extract_dt_from_name(prod_name):
        match = re.search(r'_(20\d{6}T\d{6})_', prod_name)
        if match:
            return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        return None

    # Parse datetimes to compute temporal gap features matching the fire pipeline
    s1_pre_dt  = extract_dt_from_name(cdse_products["S1_pre"]["Name"])
    s1_post_dt = extract_dt_from_name(cdse_products["S1_post"]["Name"])
    s2_pre_dt  = extract_dt_from_name(cdse_products["S2_pre"]["Name"])
    s2_post_dt = extract_dt_from_name(cdse_products["S2_post"]["Name"])

    if all([s1_pre_dt, s1_post_dt, s2_pre_dt, s2_post_dt]):
        s1_gap_days = round(abs((s1_post_dt - s1_pre_dt).total_seconds()) / 86400.0, 2)
        s2_gap_days = round(abs((s2_post_dt - s2_pre_dt).total_seconds()) / 86400.0, 2)
        pre_sensor_diff_hours = round(abs((s1_pre_dt - s2_pre_dt).total_seconds()) / 3600.0, 2)
        post_sensor_diff_hours = round(abs((s1_post_dt - s2_post_dt).total_seconds()) / 3600.0, 2)
    else:
        s1_gap_days, s2_gap_days, pre_sensor_diff_hours, post_sensor_diff_hours = None, None, None, None

    meta = {
        "control_id":             control_id,
        "matched_fire_event_id":  fire_event_id,
        "fire_name":              fire_name,
        "state":                  state,
        "control_date":           str(control_date),
        "total_acres":            0.0,  
        "orbit_direction":        orbit_direction,
        "temporal_gaps": {
            "S1_days": s1_gap_days,
            "S2_days": s2_gap_days
        },
        "sensor_time_difference_hours": {
            "pre_fire": pre_sensor_diff_hours,
            "post_fire": post_sensor_diff_hours
        },
        "spatial_bounds":         list(bbox),
        "spatial_coverage": {
            "S1_pre":  calculate_spatial_coverage_percentage(s1_pre, bbox), # Or map your coverage proxies here
            "S2_pre":  calculate_spatial_coverage_percentage(s2_pre, bbox),
            "S1_post": calculate_spatial_coverage_percentage(s1_post, bbox),
            "S2_post": calculate_spatial_coverage_percentage(s2_post, bbox),
            "ERA5":    100.0
        },
        "contents":               uploaded_assets
    }
    
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=f"{s3_folder_prefix}metadata.json",
        Body=json.dumps(meta, indent=4),
        ExpectedBucketOwner=BUCKET_OWNER,
        ContentType='application/json'
    )
    log.info(f"Metadata written for {control_id}")

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def collect_controls():
    _, eco = load_spatial_data()
    fire_scenes     = load_all_fire_metadata()
    existing_centroids = load_existing_control_centroids()

    # seed with fire centroids so controls stay >= MIN_DIST_KM away
    for fire in fire_scenes:
        b = fire["spatial_bounds"]
        existing_centroids.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))

    log.info(f"Seeded {len(existing_centroids)} existing centroids (fires + controls).")

    # ────────────────────────────────────────────────────────
    # Build an instant local exclusion zone from your true fires
    # ────────────────────────────────────────────────────────
    true_fire_boxes = []
    for fire in fire_scenes:
        b = fire["spatial_bounds"]
        true_fire_boxes.append(box(b[0], b[1], b[2], b[3]))
    fire_exclusion_zone = unary_union(true_fire_boxes)
    # ────────────────────────────────────────────────────────

    for fire in tqdm(fire_scenes, desc="Total Fire Scene Progress", leave=True):
        state    = fire["state"]
        fire_id  = fire["event_id"]
        orbit    = fire.get("orbit_direction", "ASCENDING")

        fire_name     = fire["fire_name"]
        clean_name = fire_name
        if clean_name.lower().endswith("_fire"):
            clean_name = clean_name[:-len("_fire")]
        fire_control = f"{clean_name.upper()}_control"

        existing_controls = count_existing_s3_controls(state, fire_control)
        if existing_controls >= CONTROLS_PER_FIRE:
            log.info(f" ⏭ Skipping {fire_id} ({fire_control}) - 3 controls already exist.")
            continue

        fire_pt   = Point((fire["spatial_bounds"][0] + fire["spatial_bounds"][2]) / 2,
                          (fire["spatial_bounds"][1] + fire["spatial_bounds"][3]) / 2)
        eco_match = eco[eco.geometry.contains(fire_pt)]
        if eco_match.empty:
            log.warning(f"No ecoregion for fire {fire_id}, skipping.")
            continue
        eco_geom = unary_union(eco_match.geometry.values)

        controls_collected = existing_controls
        attempts           = 0
        max_loop_limit     = MAX_ATTEMPTS_PER_FIRE * 10

        while controls_collected < CONTROLS_PER_FIRE and attempts < max_loop_limit:
            attempts += 1
            
            if attempts % 10 == 1:
                log.info(f"[{fire_id}] Searching for valid control scenes (Attempt {attempts}, Found {controls_collected}/3)...")

            # 1. Sample centroid location
            result = sample_control_centroid(eco_geom, fire_exclusion_zone, existing_centroids)
            if result is None:
                continue
            lon, lat = result
            bbox = (lon - HALF_DEG, lat - HALF_DEG, lon + HALF_DEG, lat + HALF_DEG)

            # ────────────────────────────────────────────────────────
            # 🚀 LOCAL PAIR-MATCHING OPTIMIZATION
            # ────────────────────────────────────────────────────────
            # Fetch the entire historical metadata catalog for THIS 20x20km tile ONCE
            full_time_range = f"{YEAR_RANGE[0]}-06-01/{YEAR_RANGE[1]}-09-30"
            try:
                all_s2_items = list(catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox,
                    datetime=full_time_range,
                    query={"eo:cloud_cover": {"lt": CLOUD_THRESH}}
                ).items())
                
                all_s1_items = list(catalog.search(
                    collections=["sentinel-1-grd"],
                    bbox=bbox,
                    datetime=full_time_range,
                    query={"sat:orbit_state": {"eq": orbit.lower()}}
                ).items())

                def is_dual_pol(item):
                    pols=item.properties.get("sar:polarizations", [])
                    return "VV" in pols and "VH" in pols
                all_s1_items = [item for item in all_s1_items if is_dual_pol(item)]
            except Exception as e:
                log.warning(f"Failed pulling historical catalog for tile: {e}")
                continue

            if not all_s2_items or not all_s1_items:
                continue

            # Group Sentinel-2 items by their literal date string so we can find pre/post pairs
            s2_by_date = {}
            for item in all_s2_items:
                dt = item.datetime
                if not (FIRE_SEASON[0] <= dt.month <= FIRE_SEASON[1]):
                    continue
                s2_by_date[dt.date()] = item

            # Shuffle the dates to ensure random sampling across the 2016-2024 window
            available_control_dates = list(s2_by_date.keys())
            random.shuffle(available_control_dates)

            matched_pair_found = False
            s2_pre, s2_post = None, None
            s1_pre, s1_post = None, None
            chosen_control_date = None

            # Define spatial footprint shapes for coverage verification 
            target_tile_geom = box(*bbox)
            target_tile_area = target_tile_geom.area

            # Spin through our local date options to find a perfectly synchronized pre/post pair
            for candidate_date in available_control_dates:
                target_pre_date  = candidate_date - timedelta(days=30)
                target_post_date = candidate_date + timedelta(days=30)

                # Look for matching S2 items in our local cache for pre and post windows
                s2_pre_cand = next((s2_by_date[d] for d in s2_by_date if abs((d - target_pre_date).days) <= 3), None)
                s2_post_cand = next((s2_by_date[d] for d in s2_by_date if abs((d - target_post_date).days) <= 3), None)

                if not s2_pre_cand or not s2_post_cand:
                    continue

                # Check that spatial coverage for each scene is >70% 
                try:
                    s2_pre_geom = shape(s2_pre_cand.geometry)
                    s2_post_geom = shape(s2_post_cand.geometry)
                    
                    s2_pre_cov = (target_tile_geom.intersection(s2_pre_geom).area / target_tile_area) * 100.0
                    s2_post_cov = (target_tile_geom.intersection(s2_post_geom).area / target_tile_area) * 100.0
                    
                    if s2_pre_cov < 70.0 or s2_post_cov < 70.0:
                        continue # Drop candidate date if optical swath edges clip too much area
                except Exception as e:
                    log.warning(f"Error computing S2 footprint geometry areas: {e}")
                    continue

                # Run your custom cloud and valid pixel validations on these candidates locally
                s2_pre_signed = planetary_computer.sign(s2_pre_cand)
                s2_post_signed = planetary_computer.sign(s2_post_cand)
                
                thresh_fraction = CLOUD_THRESH / 100.0
                if scl_cloud_fraction(s2_pre_signed, bbox) > thresh_fraction or scl_cloud_fraction(s2_post_signed, bbox) > thresh_fraction:
                    continue
                if valid_pixel_fraction(s2_pre_signed, bbox) < MIN_VALID_PIX or valid_pixel_fraction(s2_post_signed, bbox) < MIN_VALID_PIX:
                    continue

                # Ensure shape consistency
                if get_s2_shape(s2_pre_signed, bbox) != get_s2_shape(s2_post_signed, bbox):
                    continue

                # Now parse local S1 items to see if any fall within your strict 36-hour window
                s1_pre_cand = next((s1 for s1 in all_s1_items if abs((s1.datetime - s2_pre_cand.datetime).total_seconds()) / 3600.0 <= MAX_SENSOR_DELTA_HOURS), None)
                s1_post_cand = next((s1 for s1 in all_s1_items if abs((s1.datetime - s2_post_cand.datetime).total_seconds()) / 3600.0 <= MAX_SENSOR_DELTA_HOURS), None)

                if s1_pre_cand and s1_post_cand:
                    # Check that spatial coverage for each scene is >70% 
                    try:
                        s1_pre_geom = shape(s1_pre_cand.geometry)
                        s1_post_geom = shape(s1_post_cand.geometry)
                        
                        s1_pre_cov = (target_tile_geom.intersection(s1_pre_geom).area / target_tile_area) * 100.0
                        s1_post_cov = (target_tile_geom.intersection(s1_post_geom).area / target_tile_area) * 100.0
                        
                        if s1_pre_cov < 70.0 or s1_post_cov < 70.0:
                            continue # Drop candidate date if SAR swath edges clip too much area
                    except Exception as e:
                        log.warning(f"Error computing S1 footprint geometry areas: {e}")
                        continue

                if s1_pre_cand and s1_post_cand:
                    # Clear cut match found entirely in local memory!
                    s2_pre, s2_post = s2_pre_signed, s2_post_signed
                    s1_pre, s1_post = s1_pre_cand, s1_post_cand
                    chosen_control_date = candidate_date
                    matched_pair_found = True
                    break

            if not matched_pair_found:
                log.debug("No synchronized local pairs found for this coordinate block. Resampling centroid...")
                continue
            # ────────────────────────────────────────────────────────

            # 5. Pinpoint on CDSE
            cdse_products = {
                "S1_pre":  pinpoint_s1_scene_on_cdse(s1_pre),
                "S1_post": pinpoint_s1_scene_on_cdse(s1_post),
                "S2_pre":  pinpoint_s2_scene_on_cdse(s2_pre),
                "S2_post": pinpoint_s2_scene_on_cdse(s2_post),
            }
            if any(v is None for v in cdse_products.values()):
                log.info("CDSE pinpoint failed for one or more scenes, resampling.")
                continue

            # SAFETY CHECK: Ensure CDSE products themselves explicitly match your expectations 
            try:
                cdse_insufficient = False
                for label, product in cdse_products.items():
                    # Double check that the targeted CDSE footprints aren't returning empty/malformed structures
                    if "geometry" in product and product["geometry"] is not None:
                        cdse_geom = shape(product["geometry"])
                        cdse_cov = (target_tile_geom.intersection(cdse_geom).area / target_tile_area) * 100.0
                        if cdse_cov < 70.0:
                            log.warning(f"CDSE sourced footprint for {label} dropped to {cdse_cov:.2f}%! Aborting upload.")
                            cdse_insufficient = True
                            break
                if cdse_insufficient:
                    continue # Drop and resample without wasting S3 upload bandwidth
            except Exception as e:
                log.debug(f"Skipping strict CDSE geometry double-check: {e}")

            # Convert your CDSE footprints into the exact unified ERA5 grid boundaries
            try:
                era5_area = calculate_master_era5_area_from_items(cdse_products.values())
            except Exception as e:
                log.error(f"Failed parsing spatial footprint boundary coordinates: {e}")
                continue

            # 6. Upload concurrently
            coord_str  = format_coordinate_strings(bbox)
            date_str   = chosen_control_date.strftime("%Y%m%d")
            control_id = f"control_{state}_{coord_str}_{date_str}"
            s3_prefix  = f"{CONTROLS_PREFIX}/{state}/{fire_control}/{control_id}/"

            try:
                uploaded_assets = upload_control_scene(
                    cdse_products=cdse_products, 
                    control_date=chosen_control_date, 
                    era5_area=era5_area,        
                    state_code=state, 
                    s3_folder_prefix=s3_prefix,
                    coord_str=coord_str,        
                    bbox=bbox
                )
            except Exception:
                log.error("Upload failed, resampling.")
                continue

            # 7. Write metadata
            try: 
                write_metadata(
                    control_id, state, chosen_control_date, bbox,
                    uploaded_assets, s1_pre, s1_post, s2_pre, s2_post, cdse_products,
                    fire_name, fire_id, s3_prefix, orbit
                )
            except Exception as e:
                log.error(f"Metadata write failed for {control_id}, resampling: {e}")
                continue

            # 8. Register centroid
            existing_centroids.append((lon, lat))
            controls_collected += 1
            log.info(f"✓ SUCCESS: {control_id} ({controls_collected}/{CONTROLS_PER_FIRE})")

        if controls_collected < CONTROLS_PER_FIRE:
            log.error(f" CRITICAL TIMEOUT: Could only gather {controls_collected}/{CONTROLS_PER_FIRE} valid controls for fire {fire_id} after hitting the ceiling of {attempts} loop iterations.")

    log.info("Pipeline complete.")


if __name__ == "__main__":
    collect_controls()