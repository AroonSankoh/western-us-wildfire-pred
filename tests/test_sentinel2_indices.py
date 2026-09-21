"""
Tests for the NBR/NDVI index math and cloud masking in
data/loaders/sentinel2_preprocessing.py. Uses tiny synthetic GeoTIFFs written to a
temporary dir instead of real Sentinel-2 imagery.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from data.loaders.sentinel2_preprocessing import calculate_nbr, calculate_ndvi, apply_cloud_mask


def _write_band(path, array, dtype="float32"):
    transform = from_origin(0, 10, 1, 1)
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
                        count=1, dtype=dtype, crs="EPSG:4326", transform=transform) as dst:
        dst.write(array.astype(dtype), 1)


@pytest.fixture
def bands(tmp_path):
    nir = np.array([[3000, 3200], [2800, 3100]], dtype="float32")
    swir = np.array([[1000, 1100], [900, 1050]], dtype="float32")
    red = np.array([[500, 520], [480, 510]], dtype="float32")

    paths = {}
    for name, arr in (("nir", nir), ("swir", swir), ("red", red)):
        path = tmp_path / f"{name}.tif"
        _write_band(path, arr)
        paths[name] = str(path)
    return paths, nir, swir, red


def _write_scl(tmp_path, array):
    path = tmp_path / "scl.tif"
    _write_band(path, array, dtype="uint8")
    return str(path)


def test_calculate_nbr_matches_hand_computed_formula(bands, tmp_path):
    paths, nir, swir, _ = bands
    scl = np.full(nir.shape, 4, dtype="uint8")  # 4 = vegetation, not in any risk category
    scl_path = _write_scl(tmp_path, scl)

    nbr, _, _, _ = calculate_nbr(paths["nir"], paths["swir"], scl_path)
    expected = (nir - swir) / (nir + swir)
    np.testing.assert_allclose(nbr, expected, rtol=1e-6)


def test_calculate_ndvi_matches_hand_computed_formula(bands, tmp_path):
    paths, nir, _, red = bands
    scl = np.full(nir.shape, 4, dtype="uint8")
    scl_path = _write_scl(tmp_path, scl)

    ndvi, _, _, _ = calculate_ndvi(paths["nir"], paths["red"], scl_path)
    expected = (nir - red) / (nir + red)
    np.testing.assert_allclose(ndvi, expected, rtol=1e-6)


def test_cloud_masked_pixels_become_nan(bands, tmp_path):
    paths, nir, swir, _ = bands
    # top-left pixel flagged high-probability cloud (SCL class 9), rest clear vegetation (4)
    scl = np.array([[9, 4], [4, 4]], dtype="uint8")
    scl_path = _write_scl(tmp_path, scl)

    nbr, _, _, _ = calculate_nbr(paths["nir"], paths["swir"], scl_path)
    assert np.isnan(nbr[0, 0])
    assert not np.any(np.isnan(nbr[np.array([[False, True], [True, True]])]))


def test_apply_cloud_mask_masks_every_documented_risk_category():
    band = np.ones((1, 5))
    # no_data=0, cloud_shadows=3, med_prob_cloud=8, high_prob_cloud=9, thin_cirrus=10, plus one clear pixel (4)
    scl = np.array([[0, 3, 8, 9, 10]], dtype="uint8")

    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        scl_path = os.path.join(d, "scl.tif")
        _write_band(scl_path, scl, dtype="uint8")
        masked = apply_cloud_mask(band, scl_path, band.shape)

    assert np.all(np.isnan(masked))  # every category here is a masked one, none is the clear "4"


def test_nbr_denominator_zero_produces_nan_not_a_crash(tmp_path):
    # nir == -swir makes nir+swir == 0, must safely produce NaN, not divide-by-zero garbage
    nir = np.array([[100.0, 0.0]], dtype="float32")
    swir = np.array([[-100.0, 0.0]], dtype="float32")
    scl = np.zeros(nir.shape, dtype="uint8")
    scl[:] = 4  # clear

    nir_path, swir_path, scl_path = str(tmp_path / "nir.tif"), str(tmp_path / "swir.tif"), str(tmp_path / "scl.tif")
    _write_band(nir_path, nir)
    _write_band(swir_path, swir)
    _write_band(scl_path, scl, dtype="uint8")

    nbr, _, _, _ = calculate_nbr(nir_path, swir_path, scl_path)
    assert np.isnan(nbr[0, 0])  # 100 + -100 == 0
    assert np.isnan(nbr[0, 1])  # 0 + 0 == 0
