from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from har.types import Device, Sensor, SessionFrame, SessionKey

RAW_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "activity",
    "timestamp",
    "x",
    "y",
    "z",
)
MAX_GAP_S = 2.0
_MAX_GAP_NS = int(MAX_GAP_S * 1_000_000_000)


def parse_raw_line(line: str) -> tuple[int, str, int, float, float, float]:
    """Parse one official WISDM row: `subject-id, activity-code, timestamp, x, y, z;`."""
    stripped = line.strip().rstrip(";").strip()
    parts = [part.strip() for part in stripped.split(",")]
    if len(parts) != 6:
        raise ValueError(f"expected 6 fields, got {len(parts)}: {line!r}")
    subject_id, activity, timestamp, x, y, z = parts
    return int(subject_id), activity, int(timestamp), float(x), float(y), float(z)


def parse_raw_file(path: Path) -> pd.DataFrame:
    """Load one subject-sensor txt file as a DataFrame (all activities concatenated)."""
    rows: list[tuple[int, str, int, float, float, float]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(parse_raw_line(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    if not rows:
        return pd.DataFrame({name: pd.Series(dtype=object) for name in RAW_COLUMNS})
    subject_id, activity, timestamp, x, y, z = zip(*rows, strict=True)
    return pd.DataFrame(
        {
            "subject_id": subject_id,
            "activity": activity,
            "timestamp": timestamp,
            "x": x,
            "y": y,
            "z": z,
        }
    )


def split_activity_runs(df: pd.DataFrame) -> list[SessionFrame]:
    """Split a concatenated file into runs on activity change, time reversal, or gap > 2 s."""
    if df.empty:
        return []
    timestamps = df["timestamp"].to_numpy(dtype=np.int64, copy=False)
    activities = df["activity"].to_numpy(copy=False)
    subjects = df["subject_id"].to_numpy(copy=False)
    dt = np.diff(timestamps)
    split_before = np.concatenate(
        [
            [False],
            (activities[1:] != activities[:-1])
            | (subjects[1:] != subjects[:-1])
            | (dt < 0)
            | (dt > _MAX_GAP_NS),
        ]
    )
    starts = np.flatnonzero(split_before)
    bounds = np.concatenate([[0], starts, [len(df)]])
    return [_slice_to_frame(df.iloc[lo:hi]) for lo, hi in zip(bounds[:-1], bounds[1:])]


def load_subject_sensor_file(path: Path, device: Device, sensor: Sensor) -> list[SessionFrame]:
    """Parse one raw file and split it into SessionFrames."""
    df = parse_raw_file(path)
    df["device"] = device
    df["sensor"] = sensor
    return split_activity_runs(df)


def _slice_to_frame(chunk: pd.DataFrame) -> SessionFrame:
    first = chunk.iloc[0]
    return SessionFrame(
        key=SessionKey(
            subject_id=int(first["subject_id"]),
            activity=str(first["activity"]),
            device=first["device"],
            sensor=first["sensor"],
        ),
        timestamps_ns=chunk["timestamp"].to_numpy(dtype=np.int64),
        xyz=chunk[["x", "y", "z"]].to_numpy(dtype=np.float32),
    )
