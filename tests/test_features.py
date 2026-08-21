import numpy as np
import pytest

from har.features.statistical import N_BINS, extract_statistical, feature_names, flatten_raw


def test_constant_signal_std_is_zero_mean_matches_bins_collapse():
    value = 2.5
    x = np.full((40, 3), value, dtype=np.float32)
    feats = extract_statistical(x)
    names = feature_names(3)
    by_name = dict(zip(names, feats, strict=True))
    assert feats.shape == (len(names),)
    assert np.isfinite(feats).all()
    for c in range(3):
        assert by_name[f"ch{c}_mean"] == pytest.approx(value)
        assert by_name[f"ch{c}_std"] == pytest.approx(0.0, abs=1e-6)
        assert by_name[f"ch{c}_bin_0"] == pytest.approx(1.0)
        bins = np.array([by_name[f"ch{c}_bin_{i}"] for i in range(N_BINS)])
        assert bins.sum() == pytest.approx(1.0)
        assert bins.max() == pytest.approx(1.0)
    assert by_name["accel_corr_xy"] == pytest.approx(0.0)
    assert by_name["accel_corr_xz"] == pytest.approx(0.0)
    assert by_name["accel_corr_yz"] == pytest.approx(0.0)


def test_flatten_raw_length_equals_t_times_c():
    t, c = 8, 6
    x = np.arange(t * c, dtype=np.float32).reshape(t, c)
    flat = flatten_raw(x)
    assert flat.shape == (t * c,)
    np.testing.assert_array_equal(flat, x.reshape(-1))


def _named(x: np.ndarray) -> dict[str, float]:
    feats = extract_statistical(x)
    names = feature_names(x.shape[1])
    assert feats.shape == (len(names),)
    return dict(zip(names, feats.tolist(), strict=True))


def test_range_normalized_histogram_puts_mass_at_range_edges():
    col = np.array([0.0] * 20 + [10.0] * 20, dtype=np.float64)
    by_name = _named(np.column_stack([col, col, col]))
    bins = np.array([by_name[f"ch0_bin_{i}"] for i in range(N_BINS)])
    assert bins[0] == pytest.approx(0.5)
    assert bins[-1] == pytest.approx(0.5)
    assert bins[1:-1] == pytest.approx(0.0)


def test_six_channel_vector_includes_gyro_trio():
    names6 = feature_names(6)
    names3 = feature_names(3)
    assert "accel_resultant_mean" in names3
    assert "gyro_resultant_mean" not in names3
    assert "gyro_resultant_mean" in names6
    t = np.linspace(0.0, 1.0, 50)
    x = np.zeros((50, 6), dtype=np.float64)
    x[:, 3] = t
    x[:, 4] = t
    x[:, 5] = 2.0 * t
    by_name = _named(x)
    assert by_name["accel_corr_xy"] == pytest.approx(0.0)
    assert by_name["accel_corr_xz"] == pytest.approx(0.0)
    assert by_name["accel_corr_yz"] == pytest.approx(0.0)
    assert by_name["accel_resultant_mean"] == pytest.approx(0.0)
    assert by_name["gyro_corr_xy"] == pytest.approx(1.0)
    assert by_name["gyro_corr_xz"] == pytest.approx(1.0)
    assert by_name["gyro_corr_yz"] == pytest.approx(1.0)
    expected = float(np.mean(np.sqrt(t**2 + t**2 + (2.0 * t) ** 2)))
    assert by_name["gyro_resultant_mean"] == pytest.approx(expected)


def test_identical_varying_channels_have_unit_corr():
    t = np.linspace(0.0, 1.0, 50)
    by_name = _named(np.column_stack([t, t, 2.0 * t]))
    assert by_name["accel_corr_xy"] == pytest.approx(1.0)
    assert by_name["accel_corr_xz"] == pytest.approx(1.0)
    assert by_name["accel_corr_yz"] == pytest.approx(1.0)


def test_accel_resultant_mean_is_vector_magnitude():
    x = np.zeros((8, 3), dtype=np.float64)
    x[:, 0] = 3.0
    x[:, 2] = 4.0
    by_name = _named(x)
    assert by_name["accel_resultant_mean"] == pytest.approx(5.0)


def test_mad_is_mean_abs_dev_not_median():
    col = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
    by_name = _named(col.reshape(-1, 1))
    assert by_name["ch0_mean"] == pytest.approx(2.0)
    assert by_name["ch0_mad"] == pytest.approx(3.2)


def test_nan_rows_are_skipped_in_statistical_features():
    x = np.ones((100, 6), dtype=np.float64)
    x[:5, :] = np.nan
    feats = extract_statistical(x)
    by_name = _named(x)
    assert np.isfinite(feats).all()
    assert by_name["ch0_mean"] == pytest.approx(1.0)
    bins = np.array([by_name[f"ch0_bin_{i}"] for i in range(N_BINS)])
    assert bins.sum() == pytest.approx(1.0)
    assert bins[0] == pytest.approx(1.0)


def test_empty_window_raises():
    empty = np.empty((0, 6))
    with pytest.raises(ValueError, match="at least one sample"):
        extract_statistical(empty)
