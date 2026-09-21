"""
Tests for model/dataset.py's `dataset` class, specifically the mean-imputation path
for missing/NaN tiles and covers the era5_seq_len bug from the leakage ablation
(docs/devlog.md, 09/14/26). Requires torch, per the env.yml file.
"""

import numpy as np
import pytest

from model.dataset import dataset as TileDataset, S1_KEYS, S2_KEYS, ERA5_KEYS


def _full_s1_stats(vh_mean=-15.0):
    return {"vh_band_mean": vh_mean, "vh_band_std": 2.0, "vv_band_mean": -10.0, "vv_band_std": 2.0}


def _full_s2_stats():
    return {"red_mean": 500.0, "red_std": 50.0, "green_mean": 700.0, "green_std": 50.0,
            "nir_mean": 3000.0, "nir_std": 100.0, "swir_mean": 1000.0, "swir_std": 100.0,
            "ndvi_mean": 0.7, "ndvi_std": 0.05, "nbr_mean": 0.5, "nbr_std": 0.05}


def _full_era5_stats(seq_len, base=280.0):
    return {k: [base + i for i in range(seq_len)] for k in ERA5_KEYS}


def test_era5_seq_len_is_derived_from_real_data_not_hardcoded():
    """
    Regression test that checks if era5_seq_len reflects whatever length the real (non-None) 
    tiles actually have (23 in this case), not the module's ERA5_SEQ_LEN=30 constant. 
    so a later missing-tile imputation will match it.
    """
    tiles = {
        (0, 0): {"s1_stats": _full_s1_stats(), "s2_stats": _full_s2_stats(),
                  "era5_stats": _full_era5_stats(seq_len=23)},
    }
    ds = TileDataset(tiles)
    assert ds.era5_seq_len == 23


def test_missing_tile_is_imputed_at_the_same_length_as_real_tiles():
    """
    The actual regression scenario: one real tile with an embargo-truncated (23-day)
    era5_stats sequence, and one fully-missing tile (era5_stats=None). Both must come
    back with the same x_temporal shape.
    """
    tiles = {
        (0, 0): {"s1_stats": _full_s1_stats(), "s2_stats": _full_s2_stats(),
                  "era5_stats": _full_era5_stats(seq_len=23)},
        (0, 1): {"s1_stats": None, "s2_stats": None, "era5_stats": None},
    }
    ds = TileDataset(tiles)

    _, x_temporal_real = ds[0]
    _, x_temporal_missing = ds[1]

    assert x_temporal_real.shape == x_temporal_missing.shape == (23, len(ERA5_KEYS))


def test_missing_tile_imputed_with_dataset_wide_mean():
    """
    A missing tile's imputed era5 values should equal the dataset's own computed mean
    for each variable, not zero or some other placeholder.
    """
    tiles = {
        (0, 0): {"s1_stats": _full_s1_stats(), "s2_stats": _full_s2_stats(),
                  "era5_stats": {k: [10.0, 20.0, 30.0] for k in ERA5_KEYS}},
        (0, 1): {"s1_stats": _full_s1_stats(), "s2_stats": _full_s2_stats(),
                  "era5_stats": None},
    }
    ds = TileDataset(tiles)
    expected_mean = np.mean([10.0, 20.0, 30.0])  # same for every ERA5_KEYS var in this fixture

    _, x_temporal_missing = ds[1]
    np.testing.assert_allclose(x_temporal_missing.numpy(), expected_mean, rtol=1e-6)


def test_partial_nan_within_a_real_tile_is_imputed_per_variable():
    """
    A tile that has era5_stats but with some NaN days inside a variable's sequence
    should get those specific NaN entries replaced by that variable's own mean, not
    have the whole tile treated as fully missing.
    """
    era5_stats = _full_era5_stats(seq_len=5)
    era5_stats["t2m"] = [280.0, np.nan, 282.0, 283.0, 284.0]

    tiles = {
        (0, 0): {"s1_stats": _full_s1_stats(), "s2_stats": _full_s2_stats(), "era5_stats": era5_stats},
    }
    ds = TileDataset(tiles)
    _, x_temporal = ds[0]

    t2m_index = ERA5_KEYS.index("t2m")
    values = x_temporal[:, t2m_index].numpy()
    assert not np.any(np.isnan(values))
    expected_mean = np.mean([280.0, 282.0, 283.0, 284.0])  # NaN excluded from the mean itself
    assert values[1] == pytest.approx(expected_mean, rel=1e-6)


def test_raises_when_every_tile_is_missing():
    tiles = {(0, 0): {"s1_stats": None, "s2_stats": None, "era5_stats": None}}
    with pytest.raises(ValueError):
        TileDataset(tiles)
