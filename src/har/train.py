"""Train CLI: load config, window, featurize, split, fit, log, write metrics json."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
import mlflow

import numpy as np
import pandas as pd
import yaml

from har.constants import CHANNEL_NAMES, LABEL_ORDER, TARGET_HZ
from har.data.windows import make_windows, stack_windows
from har.eval.metrics import MetricsDict, compute_metrics
from har.eval.splits import Split, group_kfold, grouped_holdout, leaky_split, loso
from har.features.statistical import extract_statistical
from har.models.baselines import fit_dummy, fit_logreg, fit_rf
from har.models.xgboost import fit_xgboost
from har.types import AlignedSession, Device

log = logging.getLogger(__name__)


def run_experiment(config_path: Path) -> Path:
    """Run one config. Returns the metrics JSON path."""
    repo = _repo_root()
    config_path = _require_config_path(Path(config_path), repo)
    cfg = _load_config(config_path, repo)
    data_cfg = _section(cfg, "data")
    processed_dir = _resolve_path(data_cfg.get("processed_dir") or "data/processed", repo)
    device_cfg = data_cfg.get("device")
    sessions = load_aligned_sessions(
        processed_dir,
        device=_device_filter(device_cfg),
        subjects=data_cfg.get("subjects"),
    )
    if not sessions:
        raise FileNotFoundError(f"no aligned sessions under {processed_dir}")
    log.info("loaded %d sessions from %s", len(sessions), processed_dir)

    window_cfg = _section(cfg, "window")
    log.info("windowing %d sessions", len(sessions))
    windows = []
    for session in sessions:
        length_s, hop_s = _window_seconds(session, window_cfg)
        windows.extend(
            make_windows(
                session,
                length_s=length_s,
                hop_s=hop_s,
                min_coverage=float(window_cfg.get("min_coverage", 0.95)),
            )
        )
    X, y, groups = stack_windows(windows)
    if X.shape[0] == 0:
        raise ValueError("no windows produced")
    log.info("stacked %d windows shape=%s", X.shape[0], tuple(int(d) for d in X.shape))
    feature_kind = str(_section(cfg, "features").get("kind") or "statistical")
    log.info("featurizing kind=%s", feature_kind)
    X_feat = _featurize(X, feature_kind)
    log.info("features shape=%s", tuple(int(d) for d in X_feat.shape))

    split_cfg = _section(cfg, "split")
    protocol = str(split_cfg.get("protocol") or "leaky")
    seed = int(cfg.get("seed", 42))
    splits = list(_iter_splits(X_feat, y, groups, split_cfg, seed))
    log.info("split protocol=%s folds=%d", protocol, len(splits))

    tracking = _section(cfg, "tracking")
    output_dir = _resolve_path(tracking.get("output_dir") or "docs/reports", repo)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_name = str(tracking.get("protocol_name") or protocol)
    model_cfg = _section(cfg, "model")
    model_name = str(model_cfg.get("name") or "xgboost")
    device_label = "both" if device_cfg in (None, "both", "all") else str(device_cfg)
    tracking_uri = _tracking_uri(tracking.get("tracking_uri"), repo)
    out = output_dir / f"{config_path.stem}.json"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(str(tracking.get("experiment") or "wisdm-har"))

    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    fold_rows: list[dict[str, object]] = []
    fold_metric_rows: list[MetricsDict] = []
    train_subj_union: set[int] = set()
    test_subj_union: set[int] = set()
    labels = list(range(len(LABEL_ORDER)))

    with mlflow.start_run() as run:
        mlflow.log_param("protocol", protocol)
        mlflow.log_param("protocol_name", protocol_name)
        mlflow.log_param("git_sha", _git_sha(repo))
        mlflow.log_param("model", model_name)
        mlflow.log_param("features", feature_kind)
        mlflow.log_param("device", device_label)
        mlflow.log_dict(_jsonable(cfg), "config.json")
        log.info(
            "mlflow run_id=%s experiment=%s",
            run.info.run_id,
            tracking.get("experiment") or "wisdm-har",
        )

        n_folds = len(splits)
        for fold_i, split in enumerate(splits, start=1):
            fold_train = _unique_ints(split.groups_train)
            fold_test = _unique_ints(split.groups_test)
            train_subj_union.update(fold_train)
            test_subj_union.update(fold_test)
            log.info(
                "fold %d/%d fit n_train=%d n_test=%d subjects_test=%s",
                fold_i,
                n_folds,
                split.X_train.shape[0],
                split.X_test.shape[0],
                ",".join(str(s) for s in fold_test),
            )
            model = _fit_model(split, model_cfg, protocol, seed)
            y_pred = np.asarray(model.predict(split.X_test))
            y_true_parts.append(np.asarray(split.y_test))
            y_pred_parts.append(y_pred)
            fold_metrics = compute_metrics(split.y_test, y_pred, labels)
            fold_metric_rows.append(fold_metrics)
            log.info("fold %d/%d macro_f1=%.4f", fold_i, n_folds, fold_metrics["macro_f1"])
            fold_rows.append(
                {
                    "subjects_train": fold_train,
                    "subjects_test": fold_test,
                    "macro_f1": fold_metrics["macro_f1"],
                    "accuracy": fold_metrics["accuracy"],
                }
            )

        y_true = np.concatenate(y_true_parts)
        y_pred = np.concatenate(y_pred_parts)
        pooled_metrics = compute_metrics(y_true, y_pred, labels)
        fold_f1 = [row["macro_f1"] for row in fold_metric_rows]
        mean_fold_macro_f1 = float(np.mean(fold_f1))
        std_fold_macro_f1 = float(np.std(fold_f1, ddof=1)) if len(fold_f1) > 1 else 0.0
        # GroupKFold is a partition, so pooled OOF is valid. grouped_holdout
        # repeats can re-use test subjects; headline metrics are the fold mean.
        metrics = (
            _mean_metrics(fold_metric_rows) if protocol == "grouped_holdout" else pooled_metrics
        )
        subjects_train = sorted(train_subj_union)
        subjects_test = sorted(test_subj_union)
        pooled_oof = len(splits) > 1
        if pooled_oof:
            mlflow.log_param("subjects_train", "pooled_oof")
            mlflow.log_param("subjects_test", "pooled_oof")
        else:
            mlflow.log_param("subjects_train", ",".join(str(s) for s in subjects_train))
            mlflow.log_param("subjects_test", ",".join(str(s) for s in subjects_test))

        mlflow.log_metric("macro_f1", metrics["macro_f1"])
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("balanced_accuracy", metrics["balanced_accuracy"])
        mlflow.log_metric("mean_fold_macro_f1", mean_fold_macro_f1)
        mlflow.log_metric("std_fold_macro_f1", std_fold_macro_f1)

        payload = {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class_f1": {str(k): v for k, v in metrics["per_class_f1"].items()},
            "per_group_f1": dict(metrics["per_group_f1"]),
            "mean_fold_macro_f1": mean_fold_macro_f1,
            "std_fold_macro_f1": std_fold_macro_f1,
            "pooled_macro_f1": pooled_metrics["macro_f1"],
            "protocol": protocol,
            "protocol_name": protocol_name,
            "model": model_name,
            "device": device_label,
            "features": feature_kind,
            "subjects_train": subjects_train,
            "subjects_test": subjects_test,
            "pooled_oof": pooled_oof,
            "folds": fold_rows,
            "mlflow_run_id": run.info.run_id,
            "mlflow_tracking_uri": tracking_uri,
            "n_windows": int(X.shape[0]),
            "n_timesteps": int(X.shape[1]),
            "n_channels": int(X.shape[2]),
            "n_features": int(X_feat.shape[1]),
            "n_folds": len(splits),
            "session_hz": sorted({float(s.hz) for s in sessions}),
        }
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(out))
        log.info("wrote %s macro_f1=%.4f protocol=%s", out, metrics["macro_f1"], protocol_name)
    return out


def load_aligned_sessions(
    processed_dir: Path,
    device: Device | None = None,
    subjects: list[int] | None = None,
) -> list[AlignedSession]:
    processed_dir = Path(processed_dir)
    subject_set = None if subjects is None else {int(s) for s in subjects}
    sessions: list[AlignedSession] = []
    for rec in _session_records(processed_dir, device):
        if device is not None and rec["device"] != device:
            continue
        if subject_set is not None and int(rec["subject_id"]) not in subject_set:
            continue
        path = processed_dir / rec["output_path"]
        if path.is_file():
            sessions.append(_session_from_parquet(path, rec))
    return sessions


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Train a HAR experiment from a YAML config.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = run_experiment(args.config)
    print(out)
    return out


def _session_records(processed_dir: Path, device: Device | None) -> list[dict[str, Any]]:
    manifest = processed_dir / "manifest.jsonl"
    if manifest.is_file():
        records: list[dict[str, Any]] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records
    pattern = f"{device}/*.parquet" if device else "*/*.parquet"
    records = []
    for path in sorted(processed_dir.glob(pattern)):
        subject_id, activity, _run = path.stem.split("_", 2)
        records.append(
            {
                "output_path": path.relative_to(processed_dir).as_posix(),
                "subject_id": int(subject_id),
                "activity": activity,
                "device": path.parent.name,
                "hz_out": TARGET_HZ,
            }
        )
    return records


def _session_from_parquet(path: Path, rec: Mapping[str, Any]) -> AlignedSession:
    df = pd.read_parquet(path)
    channels = np.column_stack(
        [df[name].to_numpy(dtype=np.float32, copy=True) for name in CHANNEL_NAMES]
    )
    return AlignedSession(
        subject_id=int(rec["subject_id"]),
        activity=str(rec["activity"]),
        device=_as_device(rec["device"]),
        timestamps_ns=df["timestamps_ns"].to_numpy(dtype=np.int64, copy=True),
        channels=channels,
        hz=float(rec.get("hz_out") or TARGET_HZ),
    )


def _window_seconds(session: AlignedSession, window_cfg: Mapping[str, Any]) -> tuple[float, float]:
    if "length_samples" in window_cfg:
        if "hop_samples" not in window_cfg:
            raise ValueError("window.hop_samples is required with length_samples")
        hz = float(session.hz)
        return float(window_cfg["length_samples"]) / hz, float(window_cfg["hop_samples"]) / hz
    return float(window_cfg.get("length_s", 5.0)), float(window_cfg.get("hop_s", 1.0))


def _featurize(X: np.ndarray, kind: str) -> np.ndarray:
    if kind == "raw_flat":
        return np.ascontiguousarray(X).reshape(X.shape[0], -1)
    if kind == "statistical":
        return np.stack([extract_statistical(X[i]) for i in range(X.shape[0])])
    raise ValueError(f"unknown features.kind {kind!r}")


def _iter_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    split_cfg: Mapping[str, Any],
    seed: int,
) -> Iterator[Split]:
    protocol = str(split_cfg.get("protocol") or "leaky")
    if protocol == "leaky":
        yield leaky_split(
            X,
            y,
            test_size=float(split_cfg.get("leaky_test_size", 0.2)),
            seed=seed,
            groups=groups,
        )
        return
    if protocol == "groupkfold":
        yield from group_kfold(X, y, groups, n_splits=int(split_cfg.get("n_splits", 5)), seed=seed)
        return
    if protocol == "loso":
        yield from loso(X, y, groups)
        return
    if protocol == "grouped_holdout":
        yield from grouped_holdout(
            X,
            y,
            groups,
            n_test=int(split_cfg.get("n_test", 5)),
            n_repeats=int(split_cfg.get("n_repeats", 3)),
            seed=seed,
        )
        return
    raise ValueError(f"unknown protocol {protocol!r}")


def _fit_model(split: Split, model_cfg: Mapping[str, Any], protocol: str, seed: int) -> Any:
    name = str(model_cfg.get("name") or "xgboost")
    if name == "dummy":
        strategy = str(model_cfg.get("strategy") or "most_frequent")
        return fit_dummy(split.X_train, split.y_train, strategy=strategy, seed=seed)
    if name == "xgboost":
        params = dict(model_cfg.get("params") or {})
        params.setdefault("random_state", seed)
        x_train, y_train, x_val, y_val = _train_val(split, protocol, seed)
        if x_val is not None:
            params.setdefault("early_stopping_rounds", 10)
        return fit_xgboost(x_train, y_train, x_val, y_val, params)
    if name == "logreg":
        params = dict(model_cfg.get("params") or {})
        params.setdefault("random_state", seed)
        return fit_logreg(split.X_train, split.y_train, params)
    if name == "rf":
        params = dict(model_cfg.get("params") or {})
        params.setdefault("random_state", seed)
        return fit_rf(split.X_train, split.y_train, params)
    raise ValueError(f"unknown model {name!r}")


def _train_val(
    split: Split, protocol: str, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Protocol A early-stops on test. B/C hold out a train subject, never test subjects."""
    if protocol == "leaky":
        return split.X_train, split.y_train, split.X_test, split.y_test
    subjects = np.unique(split.groups_train)
    if subjects.size < 2:
        return split.X_train, split.y_train, None, None
    val_subj = int(np.random.default_rng(seed).choice(subjects))
    val_mask = split.groups_train == val_subj
    train_mask = ~val_mask
    if not np.any(train_mask) or not np.any(val_mask):
        return split.X_train, split.y_train, None, None
    return (
        split.X_train[train_mask],
        split.y_train[train_mask],
        split.X_train[val_mask],
        split.y_train[val_mask],
    )


def _device_filter(raw: object) -> Device | None:
    if raw in (None, "both", "all"):
        return None
    return _as_device(raw)


def _as_device(raw: object) -> Device:
    value = str(raw)
    if value in ("phone", "watch"):
        return value
    raise ValueError(f"unknown device {value!r}")


def _unique_ints(values: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(values).tolist()]


def _mean_metrics(rows: list[MetricsDict]) -> MetricsDict:
    """Unweighted mean of per-fold metrics. Used when folds are not a partition."""
    if not rows:
        raise ValueError("no fold metrics to average")
    class_keys = list(rows[0]["per_class_f1"])
    group_keys = list(rows[0]["per_group_f1"])
    return {
        "accuracy": float(np.mean([row["accuracy"] for row in rows])),
        "balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in rows])),
        "macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
        "per_class_f1": {
            key: float(np.mean([row["per_class_f1"][key] for row in rows])) for key in class_keys
        },
        "per_group_f1": {
            key: float(np.mean([row["per_group_f1"][key] for row in rows])) for key in group_keys
        },
    }


def _section(cfg: Mapping[str, Any], name: str) -> dict[str, Any]:
    raw = cfg.get(name)
    return dict(raw) if isinstance(raw, dict) else {}


def _load_config(path: Path, repo: Path) -> dict[str, Any]:
    cfg = _deep_merge(_read_yaml(repo / "configs" / "default.yaml"), _read_yaml(path))
    window = cfg.get("window")
    if isinstance(window, dict) and "length_samples" in window:
        window.pop("length_s", None)
        window.pop("hop_s", None)
    return cfg


def _require_config_path(path: Path, repo: Path) -> Path:
    candidates = [path]
    if not path.is_absolute():
        candidates.append(repo / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"config not found: {path}")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_path(path: object, repo: Path) -> Path:
    resolved = Path(str(path))
    return resolved if resolved.is_absolute() else repo / resolved


def _tracking_uri(raw: object, repo: Path) -> str:
    if not raw:
        return (repo / "mlruns").resolve().as_uri()
    text = str(raw)
    if "://" in text or text.startswith("file:"):
        return text
    return _resolve_path(text, repo).resolve().as_uri()


def _git_sha(repo: Path) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _jsonable(value: object) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    main()
