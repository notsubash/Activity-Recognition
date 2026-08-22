from pathlib import Path

import pandas as pd

from har.data.audit import audit_dataset
from har.data.repair import prepare_dataset

TINY_ROOT = Path(__file__).resolve().parent / "fixtures" / "tiny_wisdm"


def test_tiny_wisdm_has_two_subjects_two_activities_at_20hz():
    sessions = audit_dataset(TINY_ROOT)
    subjects = set(sessions["subject_id"].astype(int))
    activities = set(sessions["activity"].astype(str))
    sensors = set(sessions["sensor"].astype(str))
    devices = set(sessions["device"].astype(str))

    assert subjects == {1600, 1601}
    assert activities == {"A", "B"}
    assert sensors == {"accel", "gyro"}
    assert devices == {"phone"}
    assert len(sessions) == 8
    assert sessions["implied_hz"].between(19.0, 21.0).all()
    assert (sessions["duration_s"] >= 2.0).all()
    assert (sessions["n_samples"] >= 40).all()


def test_tiny_wisdm_prepare_writes_four_aligned_sessions(tmp_path: Path):
    processed = tmp_path / "processed"
    prepare_dataset(TINY_ROOT, processed, {"repair": {"target_hz": 20.0}})
    parquets = sorted(processed.rglob("*.parquet"))
    names = {path.stem for path in parquets}
    assert names == {"1600_A_0", "1600_B_0", "1601_A_0", "1601_B_0"}
    for path in parquets:
        table = pd.read_parquet(path)
        assert len(table) > 0
        assert list(table.columns) == [
            "timestamps_ns",
            "ax",
            "ay",
            "az",
            "gx",
            "gy",
            "gz",
        ]
