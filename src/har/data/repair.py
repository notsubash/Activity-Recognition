from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from har.constants import CHANNEL_NAMES, DEFAULT_TRIM_START_S, TARGET_HZ
from har.data.audit import iter_raw_sensor_files
from har.data.download import resolve_raw_root
from har.data.parse import load_subject_sensor_file
from har.types import AlignedSession, SessionFrame

log = logging.getLogger(__name__)

NS_PER_S = 1_000_000_000


def resample_session(frame: SessionFrame, target_hz: float) -> SessionFrame:
    """Interpolate xyz onto an inclusive t0, t0+1/hz, ... t1 grid. Not every-k-th-row."""
    ts = np.asarray(frame.timestamps_ns, dtype=np.int64)
    xyz = np.asarray(frame.xyz, dtype=np.float32)
    if ts.size < 2:
        return SessionFrame(key=frame.key, timestamps_ns=ts.copy(), xyz=xyz.copy())
    grid = _time_grid_ns(int(ts[0]), int(ts[-1]), target_hz)
    out = _interp_xyz(ts, xyz, grid)
    return SessionFrame(key=frame.key, timestamps_ns=grid, xyz=out)


def reorient_phone_accel(frame: SessionFrame) -> SessionFrame:
    """rWISDM-style upright repair for phone accelerometer sessions.

    Gravity-ish axis is the one with largest |mean|. If that mean is negative,
    add ``2 * abs(mean)`` instead of multiplying by -1, so the AC waveform is
    not mirrored. Then swap X and Y when |mean_x| > |mean_y| so +Y is upright.
    Watch and gyro frames are returned unchanged.
    """
    if frame.key.device != "phone" or frame.key.sensor != "accel":
        return frame
    xyz = np.asarray(frame.xyz, dtype=np.float32).copy()
    if xyz.size == 0:
        return SessionFrame(key=frame.key, timestamps_ns=frame.timestamps_ns, xyz=xyz)
    means = np.asarray(np.nanmean(xyz, axis=0), dtype=np.float64)
    if not np.all(np.isfinite(means)):
        return SessionFrame(key=frame.key, timestamps_ns=frame.timestamps_ns, xyz=xyz)
    grav = int(np.argmax(np.abs(means)))
    grav_mean = float(means[grav])
    if grav_mean < 0:
        xyz[:, grav] = xyz[:, grav] + np.float32(2.0 * abs(grav_mean))
        means = np.asarray(np.nanmean(xyz, axis=0), dtype=np.float64)
    if abs(float(means[0])) > abs(float(means[1])):
        swapped = xyz[:, 0].copy()
        xyz[:, 0] = xyz[:, 1]
        xyz[:, 1] = swapped
    return SessionFrame(key=frame.key, timestamps_ns=frame.timestamps_ns, xyz=xyz)


def align_device(accel: SessionFrame, gyro: SessionFrame, target_hz: float) -> AlignedSession:
    """Resample accel and gyro onto one intersection grid. Not an exact-timestamp join."""
    if accel.key.sensor != "accel" or gyro.key.sensor != "gyro":
        raise ValueError("align_device expects accel then gyro SessionFrames")
    if (
        accel.key.subject_id != gyro.key.subject_id
        or accel.key.activity != gyro.key.activity
        or accel.key.device != gyro.key.device
    ):
        raise ValueError("accel and gyro session keys do not match")
    if accel.timestamps_ns.size == 0 or gyro.timestamps_ns.size == 0:
        raise ValueError("accel and gyro coverage do not overlap")
    t0 = max(int(accel.timestamps_ns[0]), int(gyro.timestamps_ns[0]))
    t1 = min(int(accel.timestamps_ns[-1]), int(gyro.timestamps_ns[-1]))
    if t1 < t0:
        raise ValueError("accel and gyro coverage do not overlap")
    grid = _time_grid_ns(t0, t1, target_hz)
    if grid.size == 0:
        raise ValueError("accel and gyro coverage do not overlap")
    a_xyz = _interp_xyz(accel.timestamps_ns, accel.xyz, grid)
    g_xyz = _interp_xyz(gyro.timestamps_ns, gyro.xyz, grid)
    channels = np.concatenate([a_xyz, g_xyz], axis=1)
    return AlignedSession(
        subject_id=accel.key.subject_id,
        activity=accel.key.activity,
        device=accel.key.device,
        timestamps_ns=grid,
        channels=channels,
        hz=float(target_hz),
    )


def trim_start(session: AlignedSession, trim_s: float) -> AlignedSession:
    if trim_s <= 0 or session.timestamps_ns.size == 0:
        return session
    cutoff = int(session.timestamps_ns[0]) + int(round(trim_s * NS_PER_S))
    mask = session.timestamps_ns >= cutoff
    return AlignedSession(
        subject_id=session.subject_id,
        activity=session.activity,
        device=session.device,
        timestamps_ns=session.timestamps_ns[mask],
        channels=session.channels[mask],
        hz=session.hz,
    )


def prepare_dataset(raw_root: Path, processed_dir: Path, config: Mapping[str, Any]) -> None:
    """Align each overlapping accel/gyro run and write parquet plus manifest.jsonl."""
    repair = _repair_section(config)
    target_hz = float(repair.get("target_hz", TARGET_HZ))
    do_reorient = bool(repair.get("reorient", False))
    trim_s = float(repair.get("trim_start_s", DEFAULT_TRIM_START_S))
    if not bool(repair.get("align_accel_gyro", True)):
        raise ValueError(
            "align_accel_gyro: false is not implemented; prepare writes 6-channel aligned sessions"
        )

    raw_root = Path(raw_root)
    processed_dir = Path(processed_dir)
    files = list(iter_raw_sensor_files(raw_root))
    if not files:
        raise FileNotFoundError(f"no raw sensor txt files under {raw_root}")

    grouped: dict[tuple[int, str, str], dict[str, list[tuple[SessionFrame, Path]]]] = defaultdict(
        lambda: {"accel": [], "gyro": []}
    )
    for path, device, sensor in files:
        for frame in load_subject_sensor_file(path, device, sensor):
            key = (frame.key.subject_id, frame.key.activity, frame.key.device)
            grouped[key][sensor].append((frame, path))

    _reset_processed_dir(processed_dir)
    manifest_path = processed_dir / "manifest.jsonl"
    n_written = 0
    with manifest_path.open("w", encoding="utf-8") as handle:
        for (subject_id, activity, device), streams in sorted(grouped.items()):
            accels = streams["accel"]
            gyros = streams["gyro"]
            if not accels or not gyros:
                log.warning(
                    "skip unpaired subject=%s activity=%s device=%s accel_runs=%s gyro_runs=%s",
                    subject_id,
                    activity,
                    device,
                    len(accels),
                    len(gyros),
                )
                continue
            run = 0
            written_ranges: list[tuple[int, int]] = []
            for accel, accel_path in accels:
                for gyro, gyro_path in gyros:
                    if not _overlaps(accel, gyro):
                        continue
                    work = reorient_phone_accel(accel) if do_reorient else accel
                    aligned = align_device(work, gyro, target_hz)
                    aligned = trim_start(aligned, trim_s)
                    if aligned.timestamps_ns.size == 0:
                        continue
                    span = (int(aligned.timestamps_ns[0]), int(aligned.timestamps_ns[-1]))
                    if _coverage_overlaps(span, written_ranges):
                        log.warning(
                            "skip overlapping coverage subject=%s activity=%s device=%s",
                            subject_id,
                            activity,
                            device,
                        )
                        continue
                    out_path = processed_dir / device / f"{subject_id}_{activity}_{run}.parquet"
                    _write_parquet(aligned, out_path)
                    record = {
                        "input_path": _relpath(accel_path, raw_root),
                        "gyro_path": _relpath(gyro_path, raw_root),
                        "n_in": int(accel.timestamps_ns.size),
                        "n_out": int(aligned.timestamps_ns.size),
                        "hz_in": _json_float(_implied_hz(accel.timestamps_ns)),
                        "hz_out": float(aligned.hz),
                        "reorient": do_reorient,
                        "trim": trim_s,
                        "subject_id": subject_id,
                        "activity": activity,
                        "device": device,
                        "output_path": _relpath(out_path, processed_dir),
                    }
                    handle.write(json.dumps(record) + "\n")
                    written_ranges.append(span)
                    run += 1
                    n_written += 1
    log.info("wrote %s aligned sessions to %s", n_written, processed_dir)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Repair WISDM sessions: resample, align accel/gyro, optional reorient, trim."
    )
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repo = _repo_root()
    cfg = _load_config(args.config or repo / "configs" / "default.yaml")
    data_raw = cfg.get("data")
    data_cfg: dict[str, Any] = data_raw if isinstance(data_raw, dict) else {}

    raw_root = args.raw_root or Path(
        data_cfg.get("raw_root") or repo / "data" / "external" / "wisdm-dataset"
    )
    if not raw_root.is_absolute():
        raw_root = repo / raw_root
    processed_dir = args.processed_dir or Path(
        data_cfg.get("processed_dir") or repo / "data" / "processed"
    )
    if not processed_dir.is_absolute():
        processed_dir = repo / processed_dir

    resolved = resolve_raw_root(raw_root)
    if resolved is not None:
        raw_root = resolved
    log.info("preparing %s -> %s", raw_root, processed_dir)
    prepare_dataset(raw_root, processed_dir, cfg)
    print(processed_dir)
    return processed_dir


def _time_grid_ns(t0: int, t1: int, hz: float) -> np.ndarray:
    if t1 < t0:
        return np.array([], dtype=np.int64)
    dt_ns = NS_PER_S / hz
    n = int(np.floor((t1 - t0) / dt_ns + 1e-9)) + 1
    grid = t0 + np.round(np.arange(n, dtype=np.float64) * dt_ns).astype(np.int64)
    if grid.size and int(grid[-1]) > t1:
        grid = grid[:-1]
    return grid


def _interp_xyz(timestamps_ns: np.ndarray, xyz: np.ndarray, grid_ns: np.ndarray) -> np.ndarray:
    ts = _require_strictly_increasing(timestamps_ns)
    t_s = ts.astype(np.float64) / NS_PER_S
    g_s = np.asarray(grid_ns, dtype=np.float64) / NS_PER_S
    xyz = np.asarray(xyz, dtype=np.float32)
    cols = [np.interp(g_s, t_s, xyz[:, i]) for i in range(xyz.shape[1])]
    return np.column_stack(cols).astype(np.float32)


def _require_strictly_increasing(timestamps_ns: np.ndarray) -> np.ndarray:
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    if ts.size >= 2 and bool(np.any(np.diff(ts) <= 0)):
        raise ValueError("timestamps_ns must be strictly increasing")
    return ts


def _coverage_overlaps(span: tuple[int, int], ranges: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(
        start <= existing_end and existing_start <= end for existing_start, existing_end in ranges
    )


def _overlaps(accel: SessionFrame, gyro: SessionFrame) -> bool:
    if accel.timestamps_ns.size == 0 or gyro.timestamps_ns.size == 0:
        return False
    return int(accel.timestamps_ns[0]) <= int(gyro.timestamps_ns[-1]) and int(
        gyro.timestamps_ns[0]
    ) <= int(accel.timestamps_ns[-1])


def _implied_hz(timestamps_ns: np.ndarray) -> float:
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    if ts.size < 2:
        return float("nan")
    med = float(np.median(np.diff(ts)))
    return (1e9 / med) if med > 0 else float("nan")


def _json_float(value: float) -> float | None:
    x = float(value)
    return x if math.isfinite(x) else None


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_parquet(session: AlignedSession, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, np.ndarray] = {
        "timestamps_ns": np.asarray(session.timestamps_ns, dtype=np.int64)
    }
    for i, name in enumerate(CHANNEL_NAMES):
        data[name] = np.asarray(session.channels[:, i], dtype=np.float32)
    pd.DataFrame(data).to_parquet(path, index=False)


def _reset_processed_dir(processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest = processed_dir / "manifest.jsonl"
    if manifest.is_file():
        manifest.unlink()
    for parquet in processed_dir.glob("*.parquet"):
        parquet.unlink()
    for name in ("phone", "watch"):
        out_dir = processed_dir / name
        if out_dir.is_dir():
            shutil.rmtree(out_dir)


def _repair_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    repair = config.get("repair")
    return repair if isinstance(repair, dict) else config


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    main()
