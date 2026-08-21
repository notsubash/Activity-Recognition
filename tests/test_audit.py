import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from har.constants import ACTIVITY_CODES, SUBJECT_ID_MAX, SUBJECT_ID_MIN
from har.data.audit import (
    DEVICES,
    SENSORS,
    STUDENT_TOTAL,
    WEISS_ROWS,
    WEISS_TOTAL,
    _warn_weiss_mismatch,
    audit_dataset,
    audit_session,
    coverage_grid,
    hz_mode,
    main,
    missing_cells,
    stream_totals,
    write_audit_tables,
    write_data_card,
)
from har.types import Device, Sensor, SessionFrame, SessionKey

NS_PER_S = 1_000_000_000


def _frame(
    hz: float,
    n: int = 41,
    *,
    subject_id: int = 1600,
    activity: str = "A",
    device: Device = "phone",
    sensor: Sensor = "accel",
    xyz: np.ndarray | None = None,
) -> SessionFrame:
    dt_ns = int(round(NS_PER_S / hz))
    timestamps = np.arange(n, dtype=np.int64) * dt_ns
    if xyz is None:
        xyz = np.zeros((n, 3), dtype=np.float32)
        xyz[:, 1] = 9.8
    return SessionFrame(
        key=SessionKey(subject_id, activity, device, sensor),
        timestamps_ns=timestamps,
        xyz=xyz,
    )


def test_implied_hz_20_vs_50_within_one_hz():
    a20 = audit_session(_frame(20.0), source_path="synth")
    a50 = audit_session(_frame(50.0), source_path="synth")
    assert a20.implied_hz == pytest.approx(20.0, abs=1.0)
    assert a50.implied_hz == pytest.approx(50.0, abs=1.0)
    assert a20.n_samples == 41
    assert a20.source_path == "synth"
    assert a20.duration_s == pytest.approx(2.0)
    assert a20.n_nan == 0
    assert a20.n_non_monotonic == 0
    assert a20.median_dt_ns == pytest.approx(a20.p05_dt_ns)
    assert a20.median_dt_ns == pytest.approx(a20.p95_dt_ns)


def test_n_nan_counts_xyz_nans():
    xyz = np.zeros((5, 3), dtype=np.float32)
    xyz[1, 0] = np.nan
    xyz[3, 2] = np.nan
    result = audit_session(_frame(20.0, n=5, xyz=xyz))
    assert result.n_nan == 2


def test_hz_mode_keeps_25_and_100_out_of_the_20_bin():
    assert hz_mode(20.0) == "20"
    assert hz_mode(25.0) == "25"
    assert hz_mode(50.0) == "50"
    assert hz_mode(51.0) == "50"
    assert hz_mode(100.0) == "100"


def test_implied_hz_nan_when_fewer_than_two_samples():
    frame = SessionFrame(
        key=SessionKey(1600, "A", "phone", "accel"),
        timestamps_ns=np.array([0], dtype=np.int64),
        xyz=np.zeros((1, 3), dtype=np.float32),
    )
    result = audit_session(frame)
    assert result.n_samples == 1
    assert math.isnan(result.implied_hz)
    assert math.isnan(result.median_dt_ns)


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


def _hz_rows(
    activity: str, hz: float, n: int, t0: int = 0
) -> list[tuple[str, int, float, float, float]]:
    dt_ns = int(round(NS_PER_S / hz))
    return [(activity, t0 + i * dt_ns, 0.1, 9.8, 0.2) for i in range(n)]


def test_missing_cells_lists_absent_activity(tmp_path: Path):
    raw_root = tmp_path / "wisdm-dataset"
    _write_raw(raw_root, 1609, "phone", "accel", _hz_rows("A", 20.0, 10))

    sessions = audit_dataset(raw_root)
    coverage = coverage_grid(sessions, subjects=[1609])
    missing = missing_cells(coverage)

    keys = set(
        zip(
            missing["subject_id"].tolist(),
            missing["activity"].tolist(),
            missing["device"].tolist(),
            missing["sensor"].tolist(),
        )
    )
    assert len(coverage) == 1 * len(ACTIVITY_CODES) * len(DEVICES) * len(SENSORS)
    assert (1609, "B", "phone", "accel") in keys
    assert (1609, "B", "phone", "gyro") in keys
    assert (1609, "B", "watch", "accel") in keys
    assert (1609, "B", "watch", "gyro") in keys
    assert (1609, "A", "phone", "accel") not in keys
    present = coverage[
        (coverage["subject_id"] == 1609)
        & (coverage["activity"] == "A")
        & (coverage["device"] == "phone")
        & (coverage["sensor"] == "accel")
    ]
    assert len(present) == 1
    assert int(np.asarray(present["n_samples"])[0]) == 10


def test_default_coverage_grid_is_51_by_18_by_4():
    coverage = coverage_grid(pd.DataFrame())
    n_subjects = SUBJECT_ID_MAX - SUBJECT_ID_MIN + 1
    assert n_subjects == 51
    assert len(coverage) == n_subjects * len(ACTIVITY_CODES) * len(DEVICES) * len(SENSORS)
    assert len(missing_cells(coverage)) == len(coverage)


def test_audit_dataset_warns_when_totals_differ_from_weiss(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    raw_root = tmp_path / "wisdm-dataset"
    _write_raw(raw_root, 1600, "phone", "accel", _hz_rows("A", 20.0, 8))

    with caplog.at_level(logging.WARNING, logger="har.data.audit"):
        sessions = audit_dataset(raw_root)

    totals = stream_totals(sessions)
    assert totals[("phone", "accel")] == 8
    assert "Weiss" in caplog.text
    assert str(WEISS_ROWS[("phone", "accel")]) in caplog.text
    assert WEISS_TOTAL == 15_630_426
    assert STUDENT_TOTAL == 15_649_253


def test_weiss_exact_totals_do_not_warn(caplog: pytest.LogCaptureFixture):
    rows = [{"device": d, "sensor": s, "n_samples": n} for (d, s), n in WEISS_ROWS.items()]
    with caplog.at_level(logging.WARNING, logger="har.data.audit"):
        _warn_weiss_mismatch(pd.DataFrame(rows))
    assert caplog.records == []


def test_write_data_card_includes_missing_and_hz_tables(tmp_path: Path):
    raw_root = tmp_path / "wisdm-dataset"
    _write_raw(raw_root, 1609, "phone", "accel", _hz_rows("A", 20.0, 21))
    sessions = audit_dataset(raw_root)
    audit_dir = tmp_path / "audit"
    write_audit_tables(sessions, audit_dir, subjects=[1609])

    card = tmp_path / "data_card.md"
    write_data_card(sessions, card, subjects=[1609])
    text = card.read_text(encoding="utf-8")
    assert "| 1609 | B | phone | accel |" in text
    assert "| 20 |" in text
    assert "NaN values in xyz" in text

    for name in ("sessions.csv", "coverage.csv", "missing_cells.csv", "hz_by_session.csv"):
        assert (audit_dir / name).is_file()

    missing = pd.read_csv(audit_dir / "missing_cells.csv")
    assert not missing.empty
    assert ((missing["subject_id"] == 1609) & (missing["activity"] == "B")).any()
    assert set(ACTIVITY_CODES) == set(coverage_grid(sessions, subjects=[1609])["activity"].unique())


def test_audit_dataset_raises_when_no_raw_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no raw sensor"):
        audit_dataset(tmp_path / "wisdm-dataset")


def test_main_does_not_write_card_when_raw_missing(tmp_path: Path):
    card = tmp_path / "data_card.md"
    card.write_text("KEEP\n", encoding="utf-8")
    audit_dir = tmp_path / "audit"
    with pytest.raises(FileNotFoundError, match="no raw sensor"):
        main(
            [
                "--raw-root",
                str(tmp_path / "missing"),
                "--audit-dir",
                str(audit_dir),
                "--data-card",
                str(card),
            ]
        )
    assert card.read_text(encoding="utf-8") == "KEEP\n"
    assert not (audit_dir / "sessions.csv").exists()


def test_main_resolves_nested_wisdm_layout(tmp_path: Path):
    nested = tmp_path / "outer" / "wisdm-dataset"
    _write_raw(nested, 1600, "phone", "accel", _hz_rows("A", 20.0, 4))
    card = tmp_path / "card.md"
    audit_dir = tmp_path / "audit"
    main(
        [
            "--raw-root",
            str(tmp_path / "outer"),
            "--audit-dir",
            str(audit_dir),
            "--data-card",
            str(card),
        ]
    )
    sessions = pd.read_csv(audit_dir / "sessions.csv")
    assert len(sessions) == 1
    assert int(np.asarray(sessions["n_samples"])[0]) == 4
    assert card.is_file()
