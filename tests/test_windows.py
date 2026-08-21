import math

import numpy as np
import pandas as pd
import pytest

from har.constants import CHANNEL_NAMES, LABEL_ORDER, TARGET_HZ
from har.data.windows import aligned_session_from_dataframe, make_windows, stack_windows
from har.types import AlignedSession, Device

NS_PER_S = 1_000_000_000


def _session(
    *,
    duration_s: float,
    hz: float = TARGET_HZ,
    subject_id: int = 1600,
    activity: str = "A",
    device: Device = "phone",
) -> AlignedSession:
    # Exclusive sample count: 10 s at 20 Hz is 200 samples, not inclusive 201.
    n = int(round(duration_s * hz))
    dt_ns = int(round(NS_PER_S / hz))
    timestamps = np.arange(n, dtype=np.int64) * dt_ns
    channels = np.zeros((n, 6), dtype=np.float32)
    channels[:, 0] = np.arange(n, dtype=np.float32)
    return AlignedSession(
        subject_id=subject_id,
        activity=activity,
        device=device,
        timestamps_ns=timestamps,
        channels=channels,
        hz=hz,
    )


def test_windows_from_one_activity_never_include_the_other_label():
    walking = _session(duration_s=10.0, activity="A")
    jogging = _session(duration_s=10.0, activity="B")
    concat = np.concatenate([walking.channels, jogging.channels], axis=0)
    assert concat.shape[0] == walking.channels.shape[0] + jogging.channels.shape[0]
    # Plan step 1: keep the concat, but do not pass it to make_windows.

    windows = make_windows(walking, length_s=5.0, hop_s=1.0, min_coverage=0.95)

    walking_y = LABEL_ORDER.index("A")
    jogging_y = LABEL_ORDER.index("B")
    assert windows
    assert all(w.y == walking_y for w in windows)
    assert all(w.y != jogging_y for w in windows)
    assert all(w.activity == "A" for w in windows)
    assert all(w.subject_id == walking.subject_id for w in windows)


def test_session_shorter_than_length_yields_empty():
    session = _session(duration_s=4.0)
    assert make_windows(session, length_s=5.0, hop_s=1.0, min_coverage=0.95) == []


def test_ten_second_session_has_six_windows():
    session = _session(duration_s=10.0)
    windows = make_windows(session, length_s=5.0, hop_s=1.0, min_coverage=0.95)
    expected = 1 + math.floor((10 - 5) / 1)
    assert expected == 6
    assert len(windows) == expected
    n_length = int(round(5.0 * TARGET_HZ))
    assert all(w.x.shape == (n_length, 6) for w in windows)


def _dataframe_from_session(session: AlignedSession) -> pd.DataFrame:
    n = int(session.timestamps_ns.shape[0])
    data: dict[str, np.ndarray] = {
        "subject_id": np.full(n, session.subject_id, dtype=np.int64),
        "activity": np.full(n, session.activity),
        "device": np.full(n, session.device),
        "timestamps_ns": np.asarray(session.timestamps_ns, dtype=np.int64),
        "hz": np.full(n, session.hz, dtype=np.float64),
    }
    for i, name in enumerate(CHANNEL_NAMES):
        data[name] = np.asarray(session.channels[:, i], dtype=np.float32)
    return pd.DataFrame(data)


def test_mixed_subject_dataframe_is_rejected():
    df = _dataframe_from_session(_session(duration_s=2.0))
    df.loc[len(df) // 2 :, "subject_id"] = 1601
    with pytest.raises(ValueError, match="subject_id"):
        aligned_session_from_dataframe(df)


def test_mixed_activity_dataframe_is_rejected():
    df = _dataframe_from_session(_session(duration_s=2.0))
    df.loc[len(df) // 2 :, "activity"] = "B"
    with pytest.raises(ValueError, match="activity"):
        aligned_session_from_dataframe(df)


def test_mixed_device_dataframe_is_rejected():
    df = _dataframe_from_session(_session(duration_s=2.0))
    df.loc[len(df) // 2 :, "device"] = "watch"
    with pytest.raises(ValueError, match="device"):
        aligned_session_from_dataframe(df)


def test_mixed_hz_dataframe_is_rejected():
    df = _dataframe_from_session(_session(duration_s=2.0))
    df.loc[len(df) // 2 :, "hz"] = 50.0
    with pytest.raises(ValueError, match="hz"):
        aligned_session_from_dataframe(df)


def test_unknown_device_dataframe_is_rejected():
    df = _dataframe_from_session(_session(duration_s=2.0))
    df["device"] = "tablet"
    with pytest.raises(ValueError, match="device"):
        aligned_session_from_dataframe(df)


def test_homogeneous_dataframe_round_trips_to_aligned_session():
    src = _session(duration_s=1.0, activity="B", subject_id=1603, device="watch")
    out = aligned_session_from_dataframe(_dataframe_from_session(src))
    assert out.subject_id == 1603
    assert out.activity == "B"
    assert out.device == "watch"
    assert out.hz == pytest.approx(TARGET_HZ)
    assert out.channels.shape == src.channels.shape
    assert out.channels.shape[1] == 6
    np.testing.assert_array_equal(out.timestamps_ns, src.timestamps_ns)
    np.testing.assert_allclose(out.channels, src.channels)


def test_min_coverage_drops_nan_heavy_windows():
    session = _session(duration_s=10.0)
    session.channels[:100, :] = np.nan
    windows = make_windows(session, length_s=5.0, hop_s=1.0, min_coverage=0.95)
    assert len(windows) == 1
    assert windows[0].start_ns == int(session.timestamps_ns[100])


def test_stack_windows_returns_x_y_groups():
    walking = make_windows(_session(duration_s=10.0, activity="A"), 5.0, 1.0, 0.95)
    jogging = make_windows(
        _session(duration_s=10.0, activity="B", subject_id=1601), 5.0, 1.0, 0.95
    )
    X, y, groups = stack_windows(walking + jogging)
    n_length = int(round(5.0 * TARGET_HZ))
    assert X.shape == (12, n_length, 6)
    assert y.shape == (12,)
    assert groups.shape == (12,)
    assert set(y) == {LABEL_ORDER.index("A"), LABEL_ORDER.index("B")}
    assert set(groups.tolist()) == {1600, 1601}


def test_stack_windows_empty_has_six_channels():
    X, y, groups = stack_windows([])
    assert X.shape == (0, 0, 6)
    assert y.shape == (0,)
    assert groups.shape == (0,)


def test_make_windows_rejects_mismatched_session_lengths():
    session = _session(duration_s=10.0)
    session.channels = session.channels[:-1]
    with pytest.raises(ValueError, match="length must match"):
        make_windows(session, length_s=5.0, hop_s=1.0, min_coverage=0.95)
