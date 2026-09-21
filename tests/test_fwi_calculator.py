"""
Regression tests for the FWI system in data/loaders/era5_preprocessing.py.
Baselines were computed from the current FWI implementation, not derived by 
hand or any additional resources.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.loaders.era5_preprocessing import (
    calculate_ffmc, calculate_dmc, calculate_dc, calculate_isi, calculate_bui, calculate_fwi,
)


def test_ffmc_no_rain_matches_baseline():
    ffmc = calculate_ffmc(85.0, 290.15, 280.0, 5.0, 0.0, 0.0)
    assert ffmc == pytest.approx(86.26012916262106)


def test_ffmc_rain_at_the_0_5mm_skip_threshold_matches_no_rain():
    # the rain routine is only applied when ro > 0.5mm (p.12 restriction), exactly 0.5mm
    # (0.0005m) should behave identically to zero rain
    no_rain = calculate_ffmc(85.0, 290.15, 280.0, 5.0, 0.0, 0.0)
    at_threshold = calculate_ffmc(85.0, 290.15, 280.0, 5.0, 0.0, 0.0005)
    assert no_rain == at_threshold


def test_ffmc_heavy_rain_drops_the_code_substantially():
    # heavy rain should push FFMC down a lot relative to a dry day
    dry = calculate_ffmc(85.0, 290.15, 280.0, 5.0, 0.0, 0.0)
    wet = calculate_ffmc(85.0, 290.15, 280.0, 5.0, 0.0, 0.01)  # 10mm
    assert wet < dry
    assert wet == pytest.approx(62.13705158245683)


def test_dmc_no_rain_matches_baseline():
    dmc = calculate_dmc(6.0, 290.15, 280.0, 0.0, month=6)
    assert dmc == pytest.approx(8.324542236031364)


def test_dc_no_rain_matches_baseline():
    dc = calculate_dc(15.0, 290.15, 0.0, month=6)
    assert dc == pytest.approx(21.464)


def test_isi_increases_with_wind_speed():
    ffmc = 86.26012916262106
    isi_calm = calculate_isi(ffmc, 0.0, 0.0)
    isi_windy = calculate_isi(ffmc, 15.0, 0.0)
    assert isi_windy > isi_calm


def test_bui_matches_baseline():
    bui = calculate_bui(8.324542236031364, 21.464)
    assert bui == pytest.approx(8.453056020946212)


def test_bui_is_zero_when_dmc_and_dc_are_both_zero():
    # guards the explicit divide-by-zero special case in calculate_bui
    assert calculate_bui(0.0, 0.0) == 0.0


def _make_era5_variables(n_hours=48, t2m=290.15, d2m=280.0, u10=5.0, v10=0.0, tp=0.0):
    """
    Builds a synthetic single-gridpoint variables dict shaped like load_era5_vars's output.
    """
    times = pd.date_range("2021-06-01", periods=n_hours, freq="h")
    lat, lon = np.array([45.0]), np.array([0.0])

    def mkda(value):
        data = np.full((n_hours, 1, 1), value)
        return xr.DataArray(data, dims=("valid_time", "latitude", "longitude"),
                             coords={"valid_time": times, "latitude": lat, "longitude": lon})

    return {"t2m": mkda(t2m), "d2m": mkda(d2m), "u10": mkda(u10), "v10": mkda(v10), "tp": mkda(tp)}


def test_calculate_fwi_end_to_end_matches_baseline():
    variables = _make_era5_variables()
    result = calculate_fwi(variables)

    assert result["date"] == pd.Timestamp("2021-06-02 12:00:00")
    assert float(result["ffmc"]) == pytest.approx(86.56015602290451)
    assert float(result["dmc"]) == pytest.approx(10.649084472062729)
    assert float(result["dc"]) == pytest.approx(27.927999999999997)
    assert float(result["isi"]) == pytest.approx(6.484786720303954)
    assert float(result["bui"]) == pytest.approx(10.903895648713444)
    assert float(result["fwi"]) == pytest.approx(7.248604798739564)


def test_calculate_fwi_raises_with_too_few_hours():
    # need at least 13 hourly timesteps (through the first noon) to compute one day
    variables = _make_era5_variables(n_hours=6)
    try:
        calculate_fwi(variables)
        assert False, "expected a ValueError for too few timesteps"
    except ValueError:
        pass
