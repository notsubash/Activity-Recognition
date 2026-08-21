from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from har.data.parse import load_subject_sensor_file, parse_raw_line, split_activity_runs
from har.types import SessionKey

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_raw.txt"
OFFICIAL_LINE = "1600,A,252207666810782,-0.36476135,8.793503,1.0550842;"
NS_PER_S = 1_000_000_000


def test_parse_raw_line_strips_semicolon_and_types():
    subject_id, activity, timestamp, x, y, z = parse_raw_line(OFFICIAL_LINE)
    assert (subject_id, activity, timestamp) == (1600, "A", 252207666810782)
    assert type(subject_id) is int
    assert type(timestamp) is int
    assert type(x) is float and type(y) is float and type(z) is float
    assert z == pytest.approx(1.0550842)
    assert not isinstance(z, str)


def test_parse_raw_line_rejects_wrong_arity():
    with pytest.raises(ValueError, match="6 fields"):
        parse_raw_line("1600,A,1,0.1,0.2")
    with pytest.raises(ValueError, match="6 fields"):
        parse_raw_line("1600,A,1,0.1,0.2,0.3,9;")


def test_consecutive_same_activity_stays_together_then_b_starts_another():
    frames = load_subject_sensor_file(FIXTURE, "phone", "accel")
    assert len(frames) == 2

    first, second = frames
    assert first.key == SessionKey(1600, "A", "phone", "accel")
    assert second.key == SessionKey(1600, "B", "phone", "accel")
    np.testing.assert_array_equal(
        first.timestamps_ns,
        np.array([252207666810782, 252207717164786], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        second.timestamps_ns,
        np.array([252207767518790], dtype=np.int64),
    )
    assert first.xyz.shape == (2, 3)
    assert first.xyz.dtype == np.float32
    assert second.xyz.shape == (1, 3)


def _run_df(timestamps, activity="A"):
    n = len(timestamps)
    df = pd.DataFrame(
        {
            "subject_id": [1600] * n,
            "activity": [activity] * n,
            "timestamp": timestamps,
            "x": [0.0] * n,
            "y": [0.0] * n,
            "z": [0.0] * n,
            "device": ["phone"] * n,
            "sensor": ["accel"] * n,
        }
    )
    return df


def test_backward_timestamp_starts_new_session():
    frames = split_activity_runs(_run_df([100, 50]))
    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0].timestamps_ns, np.array([100], dtype=np.int64))
    np.testing.assert_array_equal(frames[1].timestamps_ns, np.array([50], dtype=np.int64))


def test_gap_over_two_seconds_starts_new_session():
    t0 = 0
    t1 = 2 * NS_PER_S + 1
    frames = split_activity_runs(_run_df([t0, t1]))
    assert len(frames) == 2


def test_two_second_gap_does_not_split():
    frames = split_activity_runs(_run_df([0, 2 * NS_PER_S]))
    assert len(frames) == 1
    assert len(frames[0].timestamps_ns) == 2
