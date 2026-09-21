"""
Tests for data/aggregator/zonal_aggregator.py (bin_data() and the full aggregate()
pipeline). Uses tiny synthetic in-memory rasters, no real Sentinel/ERA5 data or
S3/AWS access needed.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.transform import from_origin

from data.aggregator import aggregate, bin_data


def test_bin_data_assigns_expected_grid_indices():
    # ERA5-style ascending 0.25-degree grid
    era5_lats = np.array([10.0, 10.25, 10.5, 10.75])
    era5_lons = np.array([-100.0, -99.75, -99.5, -99.25])

    lats = np.array([10.1, 10.6, 10.9])
    lons = np.array([-99.9, -99.6, -99.3])

    x_inds, y_inds = bin_data(lats, lons, era5_lats, era5_lons)

    np.testing.assert_array_equal(x_inds, [0, 2, 3])
    np.testing.assert_array_equal(y_inds, [0, 1, 2])


def test_bin_data_point_outside_grid_produces_out_of_range_index():
    # aggregate() is expected to filter these out itself 
    era5_lats = np.array([10.0, 10.25, 10.5])
    era5_lons = np.array([-100.0, -99.75, -99.5])

    x_inds, y_inds = bin_data(np.array([9.0]), np.array([-101.0]), era5_lats, era5_lons)
    assert x_inds[0] < 0
    assert y_inds[0] < 0


def _synthetic_s1_data(shape, transform, crs, seed=0):
    rng = np.random.default_rng(seed)
    bands = {"vh_band": rng.normal(-15, 2, shape), "vv_band": rng.normal(-10, 2, shape)}
    return bands, transform, shape, crs


def _synthetic_s2_data(shape, transform, crs, seed=1):
    rng = np.random.default_rng(seed)
    nir = rng.normal(3000, 100, shape)
    swir = rng.normal(1000, 100, shape)
    red = rng.normal(500, 50, shape)
    green = rng.normal(700, 50, shape)
    ndvi = (nir - red) / (nir + red)
    nbr = (nir - swir) / (nir + swir)
    bands = {
        "indices": {"ndvi": ndvi, "nbr": nbr},
        "filtered_bands": {"red": red, "green": green, "nir": nir, "swir": swir},
    }
    return bands, transform, shape, crs


def _synthetic_era5_data(era5_lats, era5_lons, n_days=30, seed=2):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2021-06-01", periods=n_days, freq="D")

    def mkda(offset):
        data = rng.normal(280 + offset, 3, (n_days, len(era5_lats), len(era5_lons)))
        return xr.DataArray(data, dims=("valid_time", "latitude", "longitude"),
                             coords={"valid_time": times, "latitude": era5_lats, "longitude": era5_lons})

    return {k: mkda(i) for i, k in enumerate(["u10", "v10", "d2m", "t2m", "tp"])}


@pytest.fixture
def small_utm_scene():
    # a tiny 4x4-pixel, 100m-resolution scene in UTM zone 10N near ~(-123, 48.75), which
    # is fully contained within the era5 grid cell bounds defined alongside it below
    transform = from_origin(500000, 5400100, 100, 100)
    shape = (4, 4)
    crs = "EPSG:32610"
    era5_lats = np.array([48.5, 48.75, 49.0, 49.25])
    era5_lons = np.array([-123.25, -123.0, -122.75, -122.5])
    return transform, shape, crs, era5_lats, era5_lons


def test_aggregate_produces_tiles_with_expected_stat_keys(small_utm_scene):
    transform, shape, crs, era5_lats, era5_lons = small_utm_scene

    s1_data = _synthetic_s1_data(shape, transform, crs)
    s2_data = _synthetic_s2_data(shape, transform, crs)
    era5_data = _synthetic_era5_data(era5_lats, era5_lons)

    tiles = aggregate(s1_data, s2_data, era5_data)

    assert len(tiles) >= 1
    for tile in tiles.values():
        assert set(tile["s1_stats"].keys()) == {"vh_band_mean", "vh_band_std", "vv_band_mean", "vv_band_std"}
        assert set(tile["s2_stats"].keys()) == {
            "red_mean", "red_std", "green_mean", "green_std", "nir_mean", "nir_std",
            "swir_mean", "swir_std", "ndvi_mean", "ndvi_std", "nbr_mean", "nbr_std",
        }
        assert set(tile["era5_stats"].keys()) == {"u10", "v10", "d2m", "t2m", "tp"}
        # catches any accidental truncation/padding in the daily resample step (_resample_daily_last_n)
        assert len(tile["era5_stats"]["u10"]) == 30


def test_aggregate_raises_when_sentinel_falls_outside_era5_grid(small_utm_scene):
    transform, shape, crs, _, _ = small_utm_scene
    s1_data = _synthetic_s1_data(shape, transform, crs)
    s2_data = _synthetic_s2_data(shape, transform, crs)

    # ERA5 grid nowhere near the scene's actual location, should be caught by
    # aggregate()'s own bounding-box assertion rather than silently producing empty tiles
    far_away_lats = np.array([0.0, 0.25, 0.5, 0.75])
    far_away_lons = np.array([0.0, 0.25, 0.5, 0.75])
    era5_data = _synthetic_era5_data(far_away_lats, far_away_lons)

    with pytest.raises(ValueError):
        aggregate(s1_data, s2_data, era5_data)
