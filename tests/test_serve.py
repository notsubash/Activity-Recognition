from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from har.constants import ACTIVITY_CODES, CHANNEL_NAMES, CODE_TO_NAME, GROUP_OF, TARGET_HZ
from har.features.statistical import feature_names
from har.models.export import (
    ModelBundle,
    export_from_config,
    export_onnx,
    load_bundle,
    predict_window,
    save_bundle,
    save_onnx_bundle,
)
from har.models.xgboost import fit_xgboost
from har.serve.app import create_app
from har.serve.schema import MAX_BODY_BYTES

NS_PER_S = 1_000_000_000
N_T = 100
N_C = 6


class _StubEstimator:
    classes_ = np.arange(len(ACTIVITY_CODES))
    n_features_in_ = len(feature_names(N_C))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(np.asarray(X))
        p = np.zeros((n, len(ACTIVITY_CODES)), dtype=np.float64)
        p[:, 0] = 0.81
        p[:, 1:] = 0.19 / (len(ACTIVITY_CODES) - 1)
        return p

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(np.asarray(X)), dtype=np.int64)


def _stub_bundle(*, abstain_threshold: float = 0.0) -> ModelBundle:
    return ModelBundle(
        estimator=_StubEstimator(),
        n_timesteps=N_T,
        n_channels=N_C,
        hz=TARGET_HZ,
        device="phone",
        features="statistical",
        model_id="stub",
        abstain_threshold=abstain_threshold,
    )


def _payload(
    *,
    t: int = N_T,
    c: int = N_C,
    device: str = "phone",
    hz: float = TARGET_HZ,
    channels: list[str] | None = None,
) -> dict:
    return {
        "device": device,
        "hz": hz,
        "channels": list(CHANNEL_NAMES[:c] if channels is None else channels),
        "samples": [[0.0] * c for _ in range(t)],
    }


def _client(bundle: ModelBundle | None = None) -> TestClient:
    return TestClient(create_app(bundle or _stub_bundle()))


def test_health_and_labels_happy_path():
    with _client() as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "model_id": "stub"}
        labels = client.get("/labels")
        assert labels.status_code == 200
        body = labels.json()
        assert body["codes"] == list(ACTIVITY_CODES)
        assert body["names"] == dict(CODE_TO_NAME)
        assert body["groups"] == dict(GROUP_OF)


def test_predict_happy_path_with_stub():
    with _client() as client:
        resp = client.post("/predict", json=_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["activity_code"] == "A"
        assert body["activity_name"] == "walking"
        assert body["group"] == "locomotion"
        assert body["abstained"] is False
        assert body["confidence"] == pytest.approx(0.81)
        assert set(body["proba"]) == set(ACTIVITY_CODES)
        assert body["proba"]["A"] == pytest.approx(0.81)


def test_predict_wrong_t_returns_422():
    with _client() as client:
        resp = client.post("/predict", json=_payload(t=80))
        assert resp.status_code == 422
        assert resp.json()["detail"] == "expected T=100, got 80"


def test_predict_wrong_c_returns_422():
    with _client() as client:
        resp = client.post("/predict", json=_payload(c=3))
        assert resp.status_code == 422
        assert resp.json()["detail"] == "expected C=6, got 3"


def test_predict_wrong_device_returns_422():
    with _client() as client:
        resp = client.post("/predict", json=_payload(device="watch"))
        assert resp.status_code == 422
        assert resp.json()["detail"] == "expected device=phone, got watch"


def test_predict_wrong_hz_returns_422():
    with _client() as client:
        resp = client.post("/predict", json=_payload(hz=50.0))
        assert resp.status_code == 422
        assert resp.json()["detail"] == "expected hz=20.0, got 50.0"


def test_predict_wrong_channel_names_returns_422():
    swapped = ["ay", "ax", "az", "gx", "gy", "gz"]
    with _client() as client:
        resp = client.post("/predict", json=_payload(channels=swapped))
        assert resp.status_code == 422
        assert resp.json()["detail"] == (
            "channel names must be ['ax', 'ay', 'az', 'gx', 'gy', 'gz'] in that order"
        )


def test_predict_oversized_body_returns_413():
    with _client() as client:
        resp = client.post(
            "/predict",
            content=b"{}",
            headers={"Content-Length": str(MAX_BODY_BYTES + 1), "Content-Type": "application/json"},
        )
        assert resp.status_code == 413
        assert resp.json()["detail"] == "request too large"


def test_predict_abstains_below_threshold():
    with _client(_stub_bundle(abstain_threshold=0.9)) as client:
        body = client.post("/predict", json=_payload()).json()
        assert body["abstained"] is True
        assert body["activity_code"] == "A"


def test_joblib_roundtrip_matches_in_memory_predict(tmp_path: Path):
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 16, dtype=np.int64)
    x = rng.normal(size=(y.size, 8)).astype(np.float32)
    x[y == 1] += 4.0
    model = fit_xgboost(
        x[:24],
        y[:24],
        x[24:],
        y[24:],
        {"n_estimators": 2, "max_depth": 2, "n_jobs": 1, "device": "cpu"},
    )
    bundle = ModelBundle(
        estimator=model,
        n_timesteps=4,
        n_channels=2,
        hz=TARGET_HZ,
        device="watch",
        features="raw_flat",
        model_id="fixture-xgb",
    )
    path = save_bundle(tmp_path / "model.joblib", bundle)
    loaded = load_bundle(path)
    window = rng.normal(size=(4, 2)).astype(np.float32)
    assert predict_window(loaded, window)["activity_code"] == predict_window(bundle, window)[
        "activity_code"
    ]


@pytest.mark.parametrize("suffix", [".joblib", ".onnx"])
def test_export_from_config_serves_fixture_windows(tmp_path: Path, suffix: str):
    processed = tmp_path / "processed"
    for subject_id, ax in ((1600, 1.0), (1601, 8.0)):
        for activity in ("A", "B"):
            _write_session(processed, subject_id, activity, ax=ax)
    cfg = tmp_path / "phone.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "data": {"processed_dir": str(processed), "device": "phone"},
                "window": {"length_s": 5.0, "hop_s": 1.0, "min_coverage": 0.95},
                "features": {"kind": "statistical"},
                "model": {
                    "name": "xgboost",
                    "params": {
                        "n_estimators": 2,
                        "max_depth": 2,
                        "n_jobs": 1,
                        "device": "cuda",
                    },
                },
                "split": {"protocol": "groupkfold", "n_splits": 2},
            }
        ),
        encoding="utf-8",
    )
    out = export_from_config(cfg, tmp_path / f"phone{suffix}", model_id="fixture-phone")
    bundle = load_bundle(out)
    assert bundle.model_id == "fixture-phone"
    assert bundle.device == "phone"
    assert bundle.n_timesteps == N_T
    assert bundle.n_channels == N_C
    if suffix == ".joblib":
        assert bundle.estimator.get_params()["device"] == "cpu"
    else:
        assert out.with_suffix(".json").is_file()
    with TestClient(create_app(bundle)) as client:
        resp = client.post("/predict", json=_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["activity_code"] in ACTIVITY_CODES
        assert body["abstained"] is False


def test_export_onnx_rejects_non_xgboost(tmp_path: Path):
    with pytest.raises(TypeError, match="XGBoost"):
        export_onnx(object(), tmp_path / "model.onnx", n_features=104)


def test_onnx_roundtrip_matches_sklearn_proba(tmp_path: Path):
    rng = np.random.default_rng(0)
    y = np.array([0, 1, 2] * 12, dtype=np.int64)
    x = rng.normal(size=(y.size, 8)).astype(np.float32)
    for i in range(3):
        x[y == i, 0] += float(i) * 3.0
    model = fit_xgboost(
        x[:24],
        y[:24],
        x[24:],
        y[24:],
        {"n_estimators": 8, "max_depth": 3, "n_jobs": 1, "device": "cpu"},
    )
    bundle = ModelBundle(
        estimator=model,
        n_timesteps=4,
        n_channels=2,
        hz=TARGET_HZ,
        device="watch",
        features="raw_flat",
        model_id="onnx-xgb",
    )
    path = save_onnx_bundle(tmp_path / "model.onnx", bundle)
    loaded = load_bundle(path)
    x_te = x[24:28]
    np.testing.assert_allclose(
        loaded.estimator.predict_proba(x_te),
        model.predict_proba(x_te),
        rtol=1e-5,
        atol=1e-6,
    )
    window = rng.normal(size=(4, 2)).astype(np.float32)
    assert predict_window(loaded, window)["activity_code"] in ACTIVITY_CODES[:3]


def test_create_app_loads_onnx_har_model_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rng = np.random.default_rng(1)
    y = np.array([0, 1] * 16, dtype=np.int64)
    x = rng.normal(size=(y.size, 8)).astype(np.float32)
    x[y == 1] += 4.0
    model = fit_xgboost(
        x[:24],
        y[:24],
        x[24:],
        y[24:],
        {"n_estimators": 4, "max_depth": 2, "n_jobs": 1, "device": "cpu"},
    )
    path = save_onnx_bundle(
        tmp_path / "env.onnx",
        ModelBundle(
            estimator=model,
            n_timesteps=4,
            n_channels=2,
            hz=TARGET_HZ,
            device="watch",
            features="raw_flat",
            model_id="onnx-env",
        ),
    )
    monkeypatch.setenv("HAR_MODEL_PATH", str(path))
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok", "model_id": "onnx-env"}
        resp = client.post("/predict", json=_payload(t=4, c=2, device="watch"))
        assert resp.status_code == 200
        assert resp.json()["activity_code"] in ("A", "B")


def test_export_from_config_rejects_hierarchical(tmp_path: Path):
    cfg = tmp_path / "hier.yaml"
    cfg.write_text(
        yaml.safe_dump({"data": {"device": "phone"}, "model": {"name": "hierarchical"}}),
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="predict_proba"):
        export_from_config(cfg, tmp_path / "hier.joblib")


def test_create_app_loads_har_model_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = save_bundle(tmp_path / "env.joblib", _stub_bundle())
    monkeypatch.setenv("HAR_MODEL_PATH", str(path))
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok", "model_id": "stub"}
        resp = client.post("/predict", json=_payload())
        assert resp.status_code == 200
        assert resp.json()["activity_code"] == "A"


def test_predict_window_uses_argmax_over_padded_proba():
    class SubsetEstimator:
        classes_ = np.array([5, 10, 12])
        n_features_in_ = len(feature_names(N_C))

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            p = np.zeros((len(np.asarray(X)), 3), dtype=np.float64)
            p[:, 1] = 1.0
            return p

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(len(np.asarray(X)), dtype=np.int64)

    bundle = ModelBundle(
        estimator=SubsetEstimator(),
        n_timesteps=N_T,
        n_channels=N_C,
        hz=TARGET_HZ,
        device="phone",
        features="statistical",
        model_id="subset",
    )
    out = predict_window(bundle, np.zeros((N_T, N_C), dtype=np.float32))
    assert out["activity_code"] == "K"
    assert out["activity_name"] == "drinking"
    assert out["proba"]["K"] == pytest.approx(1.0)
    assert out["proba"]["A"] == pytest.approx(0.0)


def test_predict_p95_under_500ms():
    with _client() as client:
        payload = _payload()
        assert client.post("/predict", json=payload).status_code == 200
        times_ms: list[float] = []
        for _ in range(100):
            t0 = time.perf_counter()
            resp = client.post("/predict", json=payload)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            assert resp.status_code == 200
        p95 = float(np.percentile(times_ms, 95))
        assert p95 < 500.0


def _write_session(processed: Path, subject_id: int, activity: str, *, ax: float) -> None:
    n = int(round(6.0 * TARGET_HZ))
    dt_ns = int(round(NS_PER_S / TARGET_HZ))
    data: dict[str, np.ndarray] = {"timestamps_ns": np.arange(n, dtype=np.int64) * dt_ns}
    for name in CHANNEL_NAMES:
        col = np.zeros(n, dtype=np.float32)
        if name == "ax":
            col[:] = ax
        data[name] = col
    path = processed / "phone" / f"{subject_id}_{activity}_0.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(path, index=False)
