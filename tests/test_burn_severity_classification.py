"""
Tests for classify_severity() in scripts/nbr_burn_severity_calculator.py. Covers the
dNBR-to-severity-class mapping only, not S2 loading, MTBS matching, or perimeter
rasterization, which need shapefile data from a real scene.
"""

import numpy as np

from nbr_burn_severity_calculator import classify_severity, GENERIC_THRESHOLDS, SEVERITY_CLASSES


def test_classify_severity_matches_expected_bins_with_generic_thresholds():
    dnbr = np.array([-1200, -500, 0, 150, 300, 700], dtype=float)
    severity = classify_severity(dnbr, GENERIC_THRESHOLDS)

    expected = ["no_data", "high_regrowth", "unburned", "low", "moderate", "high"]
    assert list(severity) == expected


def test_classify_severity_is_monotonic_at_each_threshold_boundary():
    t = GENERIC_THRESHOLDS
    boundaries = [t["NODATA_THR"], t["GREENNESS_"], t["LOW_THRESH"], t["MODERATE_T"], t["HIGH_THRES"]]
    severity = classify_severity(np.array(boundaries, dtype=float), t)
    assert list(severity) == SEVERITY_CLASSES[:5]


def test_classify_severity_respects_per_fire_thresholds_not_just_generic_ones():
    tight_thresholds = {"NODATA_THR": -970, "GREENNESS_": -150, "LOW_THRESH": 50,
                         "MODERATE_T": 90, "HIGH_THRES": 150, "DNBR_OFFST": 0}
    value = np.array([150.0])

    generic_result = classify_severity(value, GENERIC_THRESHOLDS)[0]
    tight_result = classify_severity(value, tight_thresholds)[0]

    assert generic_result == "low"       # 150 > GENERIC's LOW_THRESH (100), <= MODERATE_T (270)
    assert tight_result == "moderate"    # 150 > tight's MODERATE_T (90), <= HIGH_THRES (150)
    assert generic_result != tight_result
