from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Device = Literal["phone", "watch"]
Sensor = Literal["accel", "gyro"]
ActivityCode = str  # one of ACTIVITY_CODES


@dataclass(frozen=True)
class SessionKey:
    subject_id: int
    activity: ActivityCode
    device: Device
    sensor: Sensor


@dataclass(frozen=True)
class SessionAudit:
    key: SessionKey
    n_samples: int
    duration_s: float
    median_dt_ns: float
    p05_dt_ns: float
    p95_dt_ns: float
    implied_hz: float
    n_non_monotonic: int
    n_nan: int
    mean_x: float
    mean_y: float
    mean_z: float
    source_path: str


@dataclass
class SessionFrame:
    key: SessionKey
    timestamps_ns: np.ndarray  # shape (T,) int64
    xyz: np.ndarray  # shape (T, 3) float32


@dataclass
class AlignedSession:
    """One device, accel+gyro on a shared 20 Hz grid (T, 6)."""

    subject_id: int
    activity: ActivityCode
    device: Device
    timestamps_ns: np.ndarray  # (T,)
    channels: np.ndarray  # (T, 6) = ax,ay,az,gx,gy,gz
    hz: float  # 20.0 after repair


@dataclass
class WindowRecord:
    subject_id: int
    activity: ActivityCode
    device: Device
    start_ns: int
    end_ns: int
    x: np.ndarray  # (T, C) float32
    y: int  # label index into LABEL_ORDER
