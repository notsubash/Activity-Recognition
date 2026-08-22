"""Statistical window features: per-channel summaries, range-normalized bins, trio corr.

``extract_statistical`` returns float64. ``flatten_raw`` returns float32.
"""

from __future__ import annotations

import numpy as np

N_BINS = 10
_CHANNEL_STAT_NAMES = ("mean", "std", "mad", "min", "max", "range")
_BIN_NAMES = tuple(f"bin_{i}" for i in range(N_BINS))
PER_CHANNEL_NAMES = _CHANNEL_STAT_NAMES + _BIN_NAMES
_TRIO_NAMES = ("resultant_mean", "corr_xy", "corr_xz", "corr_yz")


def feature_names(n_channels: int) -> tuple[str, ...]:
    names = [f"ch{c}_{name}" for c in range(n_channels) for name in PER_CHANNEL_NAMES]
    if n_channels >= 3:
        names.extend(f"accel_{name}" for name in _TRIO_NAMES)
    if n_channels >= 6:
        names.extend(f"gyro_{name}" for name in _TRIO_NAMES)
    return tuple(names)


def to_magnitude(x: np.ndarray) -> np.ndarray:
    """XYZ trios ``(T, 3k)`` to Euclidean magnitudes ``(T, k)``."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("x must be (T, C)")
    n_ch = int(arr.shape[1])
    if n_ch == 0 or n_ch % 3 != 0:
        raise ValueError("x channels must be groups of 3 (XYZ trios)")
    n_trios = n_ch // 3
    mags = np.empty((arr.shape[0], n_trios), dtype=np.float64)
    for i in range(n_trios):
        trio = arr[:, i * 3 : (i + 1) * 3]
        mags[:, i] = np.sqrt(np.sum(np.square(trio), axis=1))
    return mags


def flatten_raw(x: np.ndarray) -> np.ndarray:
    """Row-major ``(T, C)`` to ``(T * C,)``."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("x must be (T, C)")
    return arr.reshape(-1).copy()


def extract_statistical(x: np.ndarray) -> np.ndarray:
    """Features for one window ``x`` of shape ``(T, C)``."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("x must be (T, C)")
    if arr.shape[0] == 0:
        raise ValueError("x must have at least one sample")
    n_channels = int(arr.shape[1])
    parts: list[np.ndarray] = [_channel_features(arr[:, c]) for c in range(n_channels)]
    if n_channels >= 3:
        parts.append(_trio_features(arr[:, :3]))
    if n_channels >= 6:
        parts.append(_trio_features(arr[:, 3:6]))
    return np.concatenate(parts)


def _channel_features(col: np.ndarray) -> np.ndarray:
    mean = float(np.nanmean(col))
    std = float(np.nanstd(col))
    mad = float(np.nanmean(np.abs(col - mean)))
    lo = float(np.nanmin(col))
    hi = float(np.nanmax(col))
    span = hi - lo
    bins = _bin_fractions(col, lo, hi)
    return np.asarray((mean, std, mad, lo, hi, span, *bins), dtype=np.float64)


def _bin_fractions(col: np.ndarray, lo: float, hi: float) -> tuple[float, ...]:
    vals = col[np.isfinite(col)]
    if vals.size == 0:
        return tuple(0.0 for _ in range(N_BINS))
    if hi <= lo:
        return (1.0,) + tuple(0.0 for _ in range(N_BINS - 1))
    counts, _ = np.histogram(vals, bins=N_BINS, range=(lo, hi))
    return tuple((counts.astype(np.float64) / float(counts.sum())).tolist())


def _trio_features(xyz: np.ndarray) -> np.ndarray:
    resultant = np.sqrt(np.sum(np.square(xyz), axis=1))
    return np.asarray(
        (
            float(np.nanmean(resultant)),
            _corr(xyz[:, 0], xyz[:, 1]),
            _corr(xyz[:, 0], xyz[:, 2]),
            _corr(xyz[:, 1], xyz[:, 2]),
        ),
        dtype=np.float64,
    )


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return 0.0
    left = a[mask]
    right = b[mask]
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(value):
        return 0.0
    return value
