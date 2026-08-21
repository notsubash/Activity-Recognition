import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from har.constants import TARGET_HZ
from har.data.repair import (
    align_device,
    prepare_dataset,
    reorient_phone_accel,
    resample_session,
    trim_start,
)
from har.types import Device, Sensor, SessionFrame, SessionKey

NS_PER_S = 1_000_000_000


def _frame(
    hz: float,
    duration_s: float,
    *,
    subject_id: int = 1600,
    activity: str = "A",
    device: Device = "phone",
    sensor: Sensor = "accel",
    t0_ns: int = 0,
    xyz: np.ndarray | None = None,
) -> SessionFrame:
    n = int(round(duration_s * hz)) + 1
    dt_ns = int(round(NS_PER_S / hz))
    timestamps = t0_ns + np.arange(n, dtype=np.int64) * dt_ns
    if xyz is None:
        t = np.arange(n, dtype=np.float64) / hz
        xyz = np.zeros((n, 3), dtype=np.float32)
        xyz[:, 0] = np.sin(2.0 * np.pi * t)
    return SessionFrame(
        key=SessionKey(subject_id, activity, device, sensor),
        timestamps_ns=timestamps,
        xyz=xyz,
    )


def test_resample_50hz_sine_3s_to_20hz_length():
    frame = _frame(50.0, 3.0)
    assert (frame.timestamps_ns[-1] - frame.timestamps_ns[0]) / NS_PER_S == pytest.approx(3.0)

    out = resample_session(frame, TARGET_HZ)

    assert len(out.timestamps_ns) == pytest.approx(60, abs=2)
    assert out.xyz.shape == (len(out.timestamps_ns), 3)
    assert out.key == frame.key
    dt_s = np.diff(out.timestamps_ns.astype(np.float64)) / NS_PER_S
    np.testing.assert_allclose(dt_s, 1.0 / TARGET_HZ, rtol=1e-6)
    t_out = (out.timestamps_ns - out.timestamps_ns[0]).astype(np.float64) / NS_PER_S
    expected = np.sin(2.0 * np.pi * t_out)
    assert np.corrcoef(out.xyz[:, 0], expected)[0, 1] > 0.99


def test_align_offset_clocks_six_channels_without_inner_join():
    accel = _frame(20.0, 2.0, sensor="accel", t0_ns=0)
    gyro = _frame(20.0, 2.0, sensor="gyro", t0_ns=25_000_000)
    assert set(accel.timestamps_ns).isdisjoint(set(gyro.timestamps_ns))

    aligned = align_device(accel, gyro, TARGET_HZ)

    assert aligned.channels.shape[1] == 6
    assert aligned.channels.shape[0] > 0
    assert aligned.device == "phone"
    assert aligned.subject_id == 1600
    assert aligned.activity == "A"
    assert aligned.hz == TARGET_HZ
    assert aligned.timestamps_ns[0] >= max(accel.timestamps_ns[0], gyro.timestamps_ns[0])
    assert aligned.timestamps_ns[-1] <= min(accel.timestamps_ns[-1], gyro.timestamps_ns[-1])


def test_reorient_neg_y_gravity_keeps_x_sine_phase():
    n = 101
    t = np.arange(n, dtype=np.float64) / 20.0
    xyz = np.zeros((n, 3), dtype=np.float32)
    xyz[:, 0] = 0.3 * np.sin(2.0 * np.pi * 2.0 * t)
    xyz[:, 1] = np.float32(-9.8) + np.float32(0.2 * np.sin(2.0 * np.pi * t))
    frame = SessionFrame(
        key=SessionKey(1600, "A", "phone", "accel"),
        timestamps_ns=(t * NS_PER_S).astype(np.int64),
        xyz=xyz,
    )

    out = reorient_phone_accel(frame)

    assert out.xyz[:, 1].mean() > 0
    assert np.corrcoef(out.xyz[:, 0], xyz[:, 0])[0, 1] > 0
    orig_ac = xyz[:, 1] - xyz[:, 1].mean()
    new_ac = out.xyz[:, 1] - out.xyz[:, 1].mean()
    assert np.corrcoef(orig_ac, new_ac)[0, 1] > 0


def test_reorient_leaves_watch_and_gyro_unchanged():
    watch = _frame(20.0, 1.0, device="watch")
    gyro = _frame(20.0, 1.0, sensor="gyro")
    assert reorient_phone_accel(watch) is watch
    assert reorient_phone_accel(gyro) is gyro


def test_resample_rejects_non_monotonic_timestamps():
    frame = _frame(20.0, 1.0)
    frame.timestamps_ns[3] = int(frame.timestamps_ns[2]) - 1
    with pytest.raises(ValueError, match="strictly increasing"):
        resample_session(frame, TARGET_HZ)


def test_align_empty_frame_raises_value_error():
    accel = SessionFrame(
        key=SessionKey(1600, "A", "phone", "accel"),
        timestamps_ns=np.array([], dtype=np.int64),
        xyz=np.zeros((0, 3), dtype=np.float32),
    )
    gyro = _frame(20.0, 1.0, sensor="gyro")
    with pytest.raises(ValueError, match="do not overlap"):
        align_device(accel, gyro, TARGET_HZ)


def test_trim_start_drops_leading_seconds():
    accel = _frame(20.0, 5.0, sensor="accel")
    gyro = _frame(20.0, 5.0, sensor="gyro")
    aligned = align_device(accel, gyro, TARGET_HZ)
    trimmed = trim_start(aligned, 1.0)
    elapsed = (trimmed.timestamps_ns[0] - aligned.timestamps_ns[0]) / NS_PER_S
    assert elapsed == pytest.approx(1.0, abs=1.0 / TARGET_HZ)
    assert len(trimmed.timestamps_ns) < len(aligned.timestamps_ns)


def _write_raw(
    raw_root: Path,
    subject_id: int,
    device: Device,
    sensor: Sensor,
    rows: list[tuple[str, int, float, float, float]],
) -> Path:
    path = raw_root / "raw" / device / sensor / f"data_{subject_id}_{sensor}_{device}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{subject_id},{activity},{timestamp},{x},{y},{z};" for activity, timestamp, x, y, z in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_prepare_dataset_writes_parquet_and_manifest(tmp_path: Path):
    raw_root = tmp_path / "wisdm-dataset"
    n = 61
    dt_ns = int(round(NS_PER_S / 20.0))
    rows = [("A", i * dt_ns, 0.1, -9.8, 0.2) for i in range(n)]
    gyro_rows = [("A", 10_000_000 + i * dt_ns, 0.01, 0.02, 0.03) for i in range(n)]
    accel_path = _write_raw(raw_root, 1600, "phone", "accel", rows)
    _write_raw(raw_root, 1600, "phone", "gyro", gyro_rows)
    processed = tmp_path / "processed"
    config = {
        "repair": {
            "target_hz": 20.0,
            "reorient": False,
            "trim_start_s": 0.0,
            "align_accel_gyro": True,
        }
    }

    prepare_dataset(raw_root, processed, config)

    manifest_path = processed / "manifest.jsonl"
    assert manifest_path.is_file()
    records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    assert len(records) == 1
    rec = records[0]
    for key in ("input_path", "n_in", "n_out", "hz_in", "hz_out", "reorient", "trim"):
        assert key in rec
    assert rec["n_in"] == n
    assert rec["n_out"] > 0
    assert rec["hz_out"] == pytest.approx(20.0)
    assert rec["reorient"] is False
    assert rec["trim"] == pytest.approx(0.0)
    assert Path(rec["input_path"]).name == accel_path.name
    parquets = list(processed.rglob("*.parquet"))
    assert len(parquets) == 1
    table = pd.read_parquet(parquets[0])
    assert list(table.columns) == [
        "timestamps_ns",
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
    ]
    assert len(table) == rec["n_out"]


def test_prepare_rejects_align_disabled(tmp_path: Path):
    raw_root = tmp_path / "wisdm-dataset"
    n = 11
    dt_ns = int(round(NS_PER_S / 20.0))
    rows = [("A", i * dt_ns, 0.1, 9.8, 0.2) for i in range(n)]
    _write_raw(raw_root, 1600, "phone", "accel", rows)
    _write_raw(raw_root, 1600, "phone", "gyro", rows)
    with pytest.raises(ValueError, match="align_accel_gyro"):
        prepare_dataset(
            raw_root,
            tmp_path / "processed",
            {"repair": {"target_hz": 20.0, "align_accel_gyro": False}},
        )


def test_prepare_keeps_unrelated_processed_files(tmp_path: Path):
    raw_root = tmp_path / "wisdm-dataset"
    n = 11
    dt_ns = int(round(NS_PER_S / 20.0))
    rows = [("A", i * dt_ns, 0.1, 9.8, 0.2) for i in range(n)]
    _write_raw(raw_root, 1600, "phone", "accel", rows)
    _write_raw(raw_root, 1600, "phone", "gyro", rows)
    processed = tmp_path / "processed"
    processed.mkdir()
    keep = processed / "raw.csv"
    keep.write_text("subject,x\n", encoding="utf-8")
    (processed / "stale.parquet").write_bytes(b"stale")

    prepare_dataset(raw_root, processed, {"repair": {"target_hz": 20.0}})

    assert keep.read_text(encoding="utf-8") == "subject,x\n"
    assert not (processed / "stale.parquet").exists()


def test_prepare_skips_duplicate_time_coverage(tmp_path: Path):
    raw_root = tmp_path / "wisdm-dataset"
    dt_ns = int(round(NS_PER_S / 20.0))
    accel_rows = [("A", i * dt_ns, 0.1, 9.8, 0.2) for i in range(21)]
    accel_rows += [("A", i * dt_ns, 0.2, 9.7, 0.1) for i in range(21)]
    gyro_rows = [("A", i * dt_ns, 0.01, 0.02, 0.03) for i in range(21)]
    _write_raw(raw_root, 1600, "phone", "accel", accel_rows)
    _write_raw(raw_root, 1600, "phone", "gyro", gyro_rows)

    prepare_dataset(raw_root, tmp_path / "processed", {"repair": {"target_hz": 20.0}})

    manifest = [
        json.loads(line)
        for line in (tmp_path / "processed" / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(manifest) == 1
