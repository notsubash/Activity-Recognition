from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from har.constants import CHANNEL_NAMES, LABEL_ORDER
from har.types import AlignedSession, Device, WindowRecord


def aligned_session_from_dataframe(df: pd.DataFrame) -> AlignedSession:
    """Build one AlignedSession. Raises if subject_id, activity, device, or hz is mixed."""
    _require_unique_identity(df)
    missing = [name for name in ("timestamps_ns", *CHANNEL_NAMES) if name not in df.columns]
    if missing:
        raise ValueError(f"dataframe missing columns: {missing}")
    timestamps = df["timestamps_ns"].to_numpy(dtype=np.int64, copy=True)
    channels = np.column_stack(
        [df[name].to_numpy(dtype=np.float32, copy=True) for name in CHANNEL_NAMES]
    )
    if "hz" in df.columns:
        hz = float(df["hz"].iloc[0])
    else:
        hz = _implied_hz(timestamps)
    return AlignedSession(
        subject_id=int(df["subject_id"].iloc[0]),
        activity=str(df["activity"].iloc[0]),
        device=_device_of(df),
        timestamps_ns=timestamps,
        channels=channels,
        hz=hz,
    )


def make_windows(
    session: AlignedSession,
    length_s: float,
    hop_s: float,
    min_coverage: float,
) -> list[WindowRecord]:
    """Slide windows inside one (subject, activity, device) session. Never across sessions."""
    timestamps = np.asarray(session.timestamps_ns)
    channels = np.asarray(session.channels)
    if timestamps.ndim != 1 or channels.ndim != 2:
        raise ValueError("timestamps_ns must be (T,) and channels (T, C)")
    if timestamps.shape[0] != channels.shape[0]:
        raise ValueError("timestamps_ns and channels length must match")
    if channels.shape[1] != len(CHANNEL_NAMES):
        raise ValueError(f"channels must have {len(CHANNEL_NAMES)} columns")

    n_length = int(round(length_s * float(session.hz)))
    n_hop = int(round(hop_s * float(session.hz)))
    if n_length < 1 or n_hop < 1:
        raise ValueError("length_s and hop_s must map to at least 1 sample")
    n = int(timestamps.shape[0])
    if n < n_length:
        return []
    try:
        y = LABEL_ORDER.index(session.activity)
    except ValueError as exc:
        raise ValueError(f"unknown activity {session.activity!r}") from exc

    windows: list[WindowRecord] = []
    for start in range(0, n - n_length + 1, n_hop):
        end = start + n_length
        x = np.asarray(channels[start:end], dtype=np.float32).copy()
        if _coverage(x) < min_coverage:
            continue
        windows.append(
            WindowRecord(
                subject_id=session.subject_id,
                activity=session.activity,
                device=session.device,
                start_ns=int(timestamps[start]),
                end_ns=int(timestamps[end - 1]),
                x=x,
                y=y,
            )
        )
    return windows


def stack_windows(
    windows: Sequence[WindowRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack windows into X (N, T, C), y (N,), groups (N,) subject IDs.

    Empty input is ``X (0, 0, 6)`` so C stays the aligned accel+gyro width.
    """
    if not windows:
        return (
            np.empty((0, 0, len(CHANNEL_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    X = np.stack([w.x for w in windows], axis=0).astype(np.float32, copy=False)
    y = np.fromiter((w.y for w in windows), dtype=np.int64, count=len(windows))
    groups = np.fromiter((w.subject_id for w in windows), dtype=np.int64, count=len(windows))
    return X, y, groups


def _require_unique_identity(df: pd.DataFrame) -> None:
    if "subject_id" not in df.columns or "activity" not in df.columns:
        raise ValueError("dataframe must include subject_id and activity")
    if df["subject_id"].nunique(dropna=False) != 1:
        raise ValueError("mixed subject_id: an AlignedSession must have one subject")
    if df["activity"].nunique(dropna=False) != 1:
        raise ValueError("mixed activity: an AlignedSession must have one activity")
    if "device" in df.columns and df["device"].nunique(dropna=False) != 1:
        raise ValueError("mixed device: an AlignedSession must have one device")
    if "hz" in df.columns and df["hz"].nunique(dropna=False) != 1:
        raise ValueError("mixed hz: an AlignedSession must have one sampling rate")


def _device_of(df: pd.DataFrame) -> Device:
    if "device" not in df.columns:
        return "phone"
    raw = str(df["device"].iloc[0])
    if raw == "watch":
        return "watch"
    if raw == "phone":
        return "phone"
    raise ValueError(f"unknown device {raw!r}: expected phone or watch")


def _coverage(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.isfinite(x).all(axis=1).mean())


def _implied_hz(timestamps_ns: np.ndarray) -> float:
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    if ts.size < 2:
        return float("nan")
    med = float(np.median(np.diff(ts)))
    return (1e9 / med) if med > 0 else float("nan")
