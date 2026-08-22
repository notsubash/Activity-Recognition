"""Persist a fitted estimator for the API. XGBoost goes to ONNX; joblib is the fallback."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from har.constants import CHANNEL_NAMES, CODE_TO_NAME, GROUP_OF, LABEL_ORDER, TARGET_HZ
from har.features.statistical import extract_statistical, flatten_raw, to_magnitude
from har.types import Device

BUNDLE_FORMAT = "har.bundle.v1"
ONNX_META_FORMAT = "har.onnx.v1"


@dataclass
class ModelBundle:
    """Serving payload: estimator plus the window contract it was trained on."""

    estimator: Any
    n_timesteps: int
    n_channels: int
    hz: float
    device: Device
    features: str
    model_id: str
    abstain_threshold: float = 0.0
    channel_names: tuple[str, ...] = CHANNEL_NAMES
    format: str = BUNDLE_FORMAT


class OnnxEstimator:
    """onnxruntime wrapper with sklearn-style predict_proba."""

    def __init__(self, session: Any, *, classes: np.ndarray, n_features: int) -> None:
        self._session = session
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = int(n_features)
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if not inputs or not outputs:
            raise ValueError("ONNX model has no inputs or outputs")
        self._input_name = inputs[0].name
        names = [out.name.lower() for out in outputs]
        if "probabilities" in names:
            self._proba_index = names.index("probabilities")
        else:
            self._proba_index = len(outputs) - 1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        outputs = self._session.run(None, {self._input_name: arr})
        return np.asarray(outputs[self._proba_index], dtype=np.float64)


def save_bundle(path: Path, bundle: ModelBundle) -> Path:
    path = Path(path)
    if path.suffix == ".onnx":
        return save_onnx_bundle(path, bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    _force_cpu(bundle.estimator)
    import joblib

    joblib.dump(bundle, path)
    return path


def save_onnx_bundle(path: Path, bundle: ModelBundle) -> Path:
    path = Path(path)
    n_features = int(getattr(bundle.estimator, "n_features_in_"))
    export_onnx(bundle.estimator, path, n_features)
    classes = [int(c) for c in np.asarray(bundle.estimator.classes_).tolist()]
    meta = {
        "format": ONNX_META_FORMAT,
        "n_timesteps": int(bundle.n_timesteps),
        "n_channels": int(bundle.n_channels),
        "n_features": n_features,
        "hz": float(bundle.hz),
        "device": bundle.device,
        "features": bundle.features,
        "model_id": bundle.model_id,
        "abstain_threshold": float(bundle.abstain_threshold),
        "channel_names": list(bundle.channel_names[: bundle.n_channels]),
        "classes": classes,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def load_bundle(path: Path) -> ModelBundle:
    path = Path(path)
    if path.suffix == ".onnx":
        return load_onnx_bundle(path)
    import joblib

    loaded = joblib.load(path)
    if not isinstance(loaded, ModelBundle):
        raise TypeError(f"expected ModelBundle, got {type(loaded).__name__}")
    if loaded.format != BUNDLE_FORMAT:
        raise ValueError(f"unsupported bundle format {loaded.format!r}")
    _force_cpu(loaded.estimator)
    return loaded


def load_onnx_bundle(path: Path) -> ModelBundle:
    import onnxruntime as ort

    path = Path(path)
    meta_path = path.with_suffix(".json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing ONNX sidecar {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("format") != ONNX_META_FORMAT:
        raise ValueError(f"unsupported ONNX sidecar format {meta.get('format')!r}")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    estimator = OnnxEstimator(
        session,
        classes=np.asarray(meta["classes"], dtype=np.int64),
        n_features=int(meta["n_features"]),
    )
    names = meta.get("channel_names") or CHANNEL_NAMES[: int(meta["n_channels"])]
    return ModelBundle(
        estimator=estimator,
        n_timesteps=int(meta["n_timesteps"]),
        n_channels=int(meta["n_channels"]),
        hz=float(meta["hz"]),
        device=meta["device"],
        features=str(meta["features"]),
        model_id=str(meta["model_id"]),
        abstain_threshold=float(meta.get("abstain_threshold") or 0.0),
        channel_names=tuple(names),
        format=ONNX_META_FORMAT,
    )


def export_onnx(estimator: Any, path: Path, n_features: int) -> Path:
    """Convert an XGBoost classifier to ONNX. Feature extraction stays in Python."""
    if "XGB" not in type(estimator).__name__:
        raise TypeError(f"ONNX export supports XGBoost, got {type(estimator).__name__}")
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    _force_cpu(estimator)
    onnx_model = convert_xgboost(
        estimator,
        initial_types=[("input", FloatTensorType([None, int(n_features)]))],
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onnx_model.SerializeToString())
    return path


def predict_window(bundle: ModelBundle, samples: np.ndarray) -> dict[str, Any]:
    """Featurize one (T, C) window and return the /predict payload fields."""
    arr = np.asarray(samples, dtype=np.float32)
    expected = (int(bundle.n_timesteps), int(bundle.n_channels))
    if arr.ndim != 2 or arr.shape != expected:
        raise ValueError(f"expected samples shape {expected}, got {arr.shape}")
    feat = _featurize_one(arr, bundle.features)
    n_in = getattr(bundle.estimator, "n_features_in_", None)
    if n_in is not None and int(feat.shape[1]) != int(n_in):
        raise ValueError(f"expected {int(n_in)} features, got {int(feat.shape[1])}")
    if not hasattr(bundle.estimator, "predict_proba"):
        raise TypeError(f"{type(bundle.estimator).__name__} has no predict_proba")
    raw = np.asarray(bundle.estimator.predict_proba(feat)[0], dtype=np.float64)
    classes = np.asarray(getattr(bundle.estimator, "classes_", np.arange(raw.size)))
    proba = {code: 0.0 for code in LABEL_ORDER}
    for p, cls in zip(raw, classes, strict=True):
        idx = int(cls)
        if 0 <= idx < len(LABEL_ORDER):
            proba[LABEL_ORDER[idx]] = float(p)
    code = max(proba.items(), key=lambda item: item[1])[0]
    confidence = float(proba[code])
    return {
        "activity_code": code,
        "activity_name": CODE_TO_NAME[code],
        "group": GROUP_OF[code],
        "proba": proba,
        "confidence": confidence,
        "abstained": confidence < float(bundle.abstain_threshold),
    }


def export_from_config(
    config_path: Path,
    out_path: Path,
    *,
    model_id: str | None = None,
    abstain_threshold: float = 0.0,
) -> Path:
    """Fit on all windows from a train YAML (one device) and write ONNX or joblib."""
    from har.data.windows import make_windows, stack_windows
    from har.eval.splits import Split
    from har.train import (
        _apply_session_repair,
        _as_device,
        _device_filter,
        _featurize,
        _fit_model,
        _load_config,
        _repo_root,
        _require_config_path,
        _resolve_path,
        _section,
        _window_seconds,
        load_aligned_sessions,
    )

    repo = _repo_root()
    config_path = _require_config_path(Path(config_path), repo)
    cfg = _load_config(config_path, repo)
    model_cfg = _serve_model_cfg(_section(cfg, "model"))
    data_cfg = _section(cfg, "data")
    device_cfg = data_cfg.get("device")
    if device_cfg in (None, "both", "all"):
        raise ValueError("export needs data.device of phone or watch, not both")
    device = _as_device(device_cfg)
    processed_dir = _resolve_path(data_cfg.get("processed_dir") or "data/processed", repo)
    sessions = load_aligned_sessions(
        processed_dir,
        device=_device_filter(device_cfg),
        subjects=data_cfg.get("subjects"),
    )
    if not sessions:
        raise FileNotFoundError(f"no aligned sessions under {processed_dir}")
    sessions = _apply_session_repair(sessions, _section(cfg, "repair"))
    window_cfg = _section(cfg, "window")
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
    feature_kind = str(_section(cfg, "features").get("kind") or "statistical")
    X_feat = _featurize(X, feature_kind)
    seed = int(cfg.get("seed", 42))
    split = Split(
        X_train=X_feat,
        y_train=y,
        X_test=X_feat[:1],
        y_test=y[:1],
        groups_train=groups,
        groups_test=groups[:1],
    )
    model = _fit_model(split, model_cfg, protocol="groupkfold", seed=seed)
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} has no predict_proba; cannot export for serving")
    bundle = ModelBundle(
        estimator=model,
        n_timesteps=int(X.shape[1]),
        n_channels=int(X.shape[2]),
        hz=float(sessions[0].hz) if sessions[0].hz else TARGET_HZ,
        device=device,
        features=feature_kind,
        model_id=model_id or config_path.stem,
        abstain_threshold=float(abstain_threshold),
        channel_names=CHANNEL_NAMES[: int(X.shape[2])],
    )
    return save_bundle(out_path, bundle)


def bundle_from_env() -> ModelBundle:
    raw = os.environ.get("HAR_MODEL_PATH")
    if not raw:
        raise RuntimeError("HAR_MODEL_PATH is not set")
    bundle = load_bundle(Path(raw))
    threshold = os.environ.get("HAR_ABSTAIN_THRESHOLD")
    if threshold is not None and threshold != "":
        bundle.abstain_threshold = float(threshold)
    model_id = os.environ.get("HAR_MODEL_ID")
    if model_id:
        bundle.model_id = model_id
    return bundle


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Fit a one-device config and write an ONNX or joblib bundle."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Ends with .onnx (model + sidecar json) or .joblib (pickle).",
    )
    parser.add_argument("--model-id")
    parser.add_argument("--abstain-threshold", type=float, default=0.0)
    args = parser.parse_args(argv)
    out = export_from_config(
        args.config,
        args.out,
        model_id=args.model_id,
        abstain_threshold=args.abstain_threshold,
    )
    print(out)
    return out


def _serve_model_cfg(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """CPU-only export. Training YAML may say cuda; Docker has no GPU."""
    out = dict(model_cfg)
    if str(out.get("name") or "xgboost") == "hierarchical":
        raise TypeError("hierarchical has no predict_proba; cannot export for serving")
    params = dict(out.get("params") or {})
    params["device"] = "cpu"
    out["params"] = params
    return out


def _featurize_one(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "raw_flat":
        return flatten_raw(x).reshape(1, -1)
    if kind == "statistical":
        return extract_statistical(x).reshape(1, -1)
    if kind == "magnitude":
        return extract_statistical(to_magnitude(x)).reshape(1, -1)
    raise ValueError(f"unknown features.kind {kind!r}")


def _force_cpu(estimator: Any) -> None:
    setter = getattr(estimator, "set_params", None)
    if setter is not None:
        try:
            setter(device="cpu")
        except (TypeError, ValueError):
            pass
    get_booster = getattr(estimator, "get_booster", None)
    if get_booster is None:
        return
    try:
        get_booster().set_param({"device": "cpu"})
    except (TypeError, ValueError, AttributeError):
        return


if __name__ == "__main__":
    main()
