import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from har.constants import CHANNEL_NAMES, TARGET_HZ
from har.models.xgboost import STUDENT_XGB_PARAMS
from har.train import _load_config, _reorient_aligned, run_experiment
from har.types import AlignedSession

NS_PER_S = 1_000_000_000
REPO = Path(__file__).resolve().parents[1]


def _write_session(
    processed: Path,
    subject_id: int,
    activity: str,
    *,
    ax: float,
    device: str = "phone",
    duration_s: float = 6.0,
) -> None:
    hz = TARGET_HZ
    n = int(round(duration_s * hz))
    dt_ns = int(round(NS_PER_S / hz))
    data: dict[str, np.ndarray] = {
        "timestamps_ns": np.arange(n, dtype=np.int64) * dt_ns,
    }
    for name in CHANNEL_NAMES:
        col = np.zeros(n, dtype=np.float32)
        if name == "ax":
            col[:] = ax
        data[name] = col
    path = processed / device / f"{subject_id}_{activity}_0.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(path, index=False)


def _write_config(
    tmp: Path,
    processed: Path,
    model_name: str,
    *,
    window: dict | None = None,
    split: dict | None = None,
    protocol_name: str = "A2",
    device: str = "phone",
    features: str = "raw_flat",
    model_params: dict | None = None,
    filename: str = "exp.yaml",
    repair: dict | None = None,
) -> Path:
    model: dict = {"name": model_name}
    tiny = {"n_estimators": 2, "max_depth": 2, "n_jobs": 1, "device": "cpu"}
    if model_name in ("xgboost", "hierarchical"):
        model["params"] = model_params or tiny
    elif model_params:
        model["params"] = model_params
    cfg = {
        "seed": 0,
        "data": {"processed_dir": str(processed).replace("\\", "/"), "device": device},
        "window": window or {"length_s": 2.0, "hop_s": 1.0, "min_coverage": 0.0},
        "features": {"kind": features},
        "model": model,
        "split": split or {"protocol": "leaky", "leaky_test_size": 0.25},
        "tracking": {
            "experiment": "test-har",
            "tracking_uri": str((tmp / "mlruns").resolve()).replace("\\", "/"),
            "output_dir": str((tmp / "reports").resolve()).replace("\\", "/"),
            "protocol_name": protocol_name,
        },
    }
    if repair:
        cfg["repair"] = repair
    path = tmp / filename
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_run_experiment_logs_macro_f1_protocol_and_subjects(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for subject_id in (1600, 1601):
        _write_session(processed, subject_id, "A", ax=1.0)
        _write_session(processed, subject_id, "B", ax=-1.0)

    metrics_path = run_experiment(_write_config(tmp_path, processed, "xgboost"))
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert isinstance(payload["macro_f1"], float)
    assert payload["protocol"] == "leaky"
    assert payload["protocol_name"] == "A2"
    assert payload["subjects_train"]
    assert payload["subjects_test"]

    import mlflow

    client = mlflow.MlflowClient(tracking_uri=payload["mlflow_tracking_uri"])
    run = client.get_run(payload["mlflow_run_id"])
    assert "macro_f1" in run.data.metrics
    assert run.data.params["protocol"] == "leaky"
    assert run.data.params["protocol_name"] == "A2"
    assert run.data.params["subjects_train"]
    assert run.data.params["subjects_test"]


def test_protocol_a1_config_is_student_clone() -> None:
    cfg = yaml.safe_load(
        (REPO / "configs" / "protocol_a1_phone_raw_flat_xgb.yaml").read_text(encoding="utf-8")
    )
    assert cfg["split"]["protocol"] == "leaky"
    assert cfg["window"]["length_samples"] == 80
    assert cfg["window"]["hop_samples"] == 40
    assert cfg["features"]["kind"] == "raw_flat"
    assert cfg["data"]["device"] == "phone"
    params = cfg["model"]["params"]
    for key, value in STUDENT_XGB_PARAMS.items():
        assert params[key] == value
    assert params["early_stopping_rounds"] == 10
    assert params["device"] == "cuda"


def test_protocol_a2_config_is_session_safe_leaky() -> None:
    cfg = yaml.safe_load(
        (REPO / "configs" / "protocol_a2_phone_raw_flat_xgb.yaml").read_text(encoding="utf-8")
    )
    assert cfg["split"]["protocol"] == "leaky"
    assert cfg["window"]["length_s"] == 5.0
    assert cfg["window"]["hop_s"] == 1.0
    assert cfg["tracking"]["protocol_name"] == "A2"
    assert cfg["model"]["name"] == "xgboost"
    params = cfg["model"]["params"]
    for key, value in STUDENT_XGB_PARAMS.items():
        assert params[key] == value
    assert params["device"] == "cuda"


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config"):
        run_experiment(tmp_path / "does_not_exist.yaml")


def test_a1_length_samples_wins_over_default_length_s(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for subject_id in (1600, 1601):
        _write_session(processed, subject_id, "A", ax=1.0)
        _write_session(processed, subject_id, "B", ax=-1.0)

    config = _write_config(
        tmp_path,
        processed,
        "xgboost",
        window={"length_samples": 80, "hop_samples": 40, "min_coverage": 0.0},
        protocol_name="A1",
    )
    payload = json.loads(run_experiment(config).read_text(encoding="utf-8"))
    assert payload["n_timesteps"] == 80
    assert payload["n_features"] == 80 * 6
    assert payload["protocol_name"] == "A1"


def test_groupkfold_logs_per_fold_subjects(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for subject_id in (1600, 1601, 1602):
        _write_session(processed, subject_id, "A", ax=1.0)
        _write_session(processed, subject_id, "B", ax=-1.0)

    config = _write_config(
        tmp_path,
        processed,
        "dummy",
        split={"protocol": "groupkfold", "n_splits": 3},
        protocol_name="B",
    )
    payload = json.loads(run_experiment(config).read_text(encoding="utf-8"))
    assert payload["n_folds"] == 3
    assert payload["pooled_oof"] is True
    assert len(payload["folds"]) == 3
    union_test = sorted({s for fold in payload["folds"] for s in fold["subjects_test"]})
    assert payload["subjects_test"] == union_test
    assert "mean_fold_macro_f1" in payload
    for fold in payload["folds"]:
        assert fold["subjects_train"]
        assert fold["subjects_test"]
        assert set(fold["subjects_train"]).isdisjoint(fold["subjects_test"])


def test_groupkfold_logreg_logs_model_device_features(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for subject_id in (1600, 1601, 1602):
        _write_session(processed, subject_id, "A", ax=1.0)
        _write_session(processed, subject_id, "B", ax=-1.0)

    config = _write_config(
        tmp_path,
        processed,
        "logreg",
        split={"protocol": "groupkfold", "n_splits": 3},
        protocol_name="B",
        model_params={"max_iter": 500},
    )
    payload = json.loads(run_experiment(config).read_text(encoding="utf-8"))
    assert payload["n_folds"] == 3
    assert payload["model"] == "logreg"
    assert payload["device"] == "phone"
    assert payload["features"] == "raw_flat"
    assert isinstance(payload["macro_f1"], float)
    for fold in payload["folds"]:
        assert set(fold["subjects_train"]).isdisjoint(fold["subjects_test"])


def test_watch_device_loads_watch_parquet(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for subject_id in (1600, 1601):
        _write_session(processed, subject_id, "A", ax=1.0, device="watch")
        _write_session(processed, subject_id, "B", ax=-1.0, device="watch")

    config = _write_config(
        tmp_path,
        processed,
        "dummy",
        device="watch",
        protocol_name="B",
    )
    payload = json.loads(run_experiment(config).read_text(encoding="utf-8"))
    assert payload["device"] == "watch"
    assert payload["n_windows"] > 0
    assert payload["n_channels"] == 6


def test_device_both_concat_yields_windows_from_both(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for subject_id in (1600, 1601):
        for device in ("phone", "watch"):
            _write_session(processed, subject_id, "A", ax=1.0, device=device)
            _write_session(processed, subject_id, "B", ax=-1.0, device=device)

    phone_cfg = _write_config(
        tmp_path,
        processed,
        "dummy",
        device="phone",
        filename="phone.yaml",
        protocol_name="B",
    )
    both_cfg = _write_config(
        tmp_path,
        processed,
        "dummy",
        device="both",
        filename="both.yaml",
        protocol_name="B",
    )
    phone = json.loads(run_experiment(phone_cfg).read_text(encoding="utf-8"))
    both = json.loads(run_experiment(both_cfg).read_text(encoding="utf-8"))
    watch_cfg = _write_config(
        tmp_path,
        processed,
        "dummy",
        device="watch",
        filename="watch.yaml",
        protocol_name="B",
    )
    watch = json.loads(run_experiment(watch_cfg).read_text(encoding="utf-8"))
    assert both["device"] == "both"
    assert both["n_channels"] == 6
    assert both["n_windows"] == 2 * phone["n_windows"]
    assert watch["device"] == "watch"
    assert watch["n_windows"] == phone["n_windows"]


def test_grouped_holdout_has_no_subject_overlap_and_n_repeats_folds(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for subject_id in (1600, 1601, 1602, 1603, 1604):
        _write_session(processed, subject_id, "A", ax=1.0)
        _write_session(processed, subject_id, "B", ax=-1.0)

    config = _write_config(
        tmp_path,
        processed,
        "dummy",
        split={"protocol": "grouped_holdout", "n_test": 2, "n_repeats": 3},
        protocol_name="C",
    )
    payload = json.loads(run_experiment(config).read_text(encoding="utf-8"))
    assert payload["protocol"] == "grouped_holdout"
    assert payload["n_folds"] == 3
    assert payload["pooled_oof"] is True
    assert len(payload["folds"]) == 3
    union_train: set[int] = set()
    union_test: set[int] = set()
    fold_f1: list[float] = []
    for fold in payload["folds"]:
        assert len(fold["subjects_test"]) == 2
        assert set(fold["subjects_train"]).isdisjoint(fold["subjects_test"])
        union_train.update(fold["subjects_train"])
        union_test.update(fold["subjects_test"])
        fold_f1.append(float(fold["macro_f1"]))
    assert payload["subjects_train"] == sorted(union_train)
    assert payload["subjects_test"] == sorted(union_test)
    assert payload["mean_fold_macro_f1"] == pytest.approx(float(np.mean(fold_f1)))
    assert payload["macro_f1"] == pytest.approx(payload["mean_fold_macro_f1"])


TRAIN_LADDER = (
    "protocol_a1_phone_raw_flat_xgb.yaml",
    "protocol_a2_phone_raw_flat_xgb.yaml",
    "protocol_b_phone_raw_flat_xgb.yaml",
    "protocol_b_phone_stat_dummy.yaml",
    "protocol_b_phone_stat_logreg.yaml",
    "protocol_b_phone_stat_rf.yaml",
    "protocol_b_phone_stat_xgb.yaml",
    "protocol_b_watch_stat_xgb.yaml",
    "protocol_b_concat_stat_xgb.yaml",
    "protocol_c_phone_stat_xgb.yaml",
)


def test_train_ladder_filenames_are_complete() -> None:
    configs = REPO / "configs"
    found = tuple(sorted(p.name for p in configs.glob("protocol_*.yaml")))
    assert found == tuple(sorted(TRAIN_LADDER))
    for stale in (
        "phone_xgb.yaml",
        "watch_xgb.yaml",
        "fusion_xgb.yaml",
        "protocol_a_leaky.yaml",
        "protocol_b_dummy.yaml",
        "protocol_b_logreg.yaml",
        "protocol_b_rf.yaml",
        "protocol_b_groupkfold.yaml",
        "protocol_b_raw_flat.yaml",
        "protocol_c_loso.yaml",
    ):
        assert not (configs / stale).exists()


@pytest.mark.parametrize(
    ("filename", "protocol", "device", "features", "model", "protocol_name"),
    [
        ("protocol_a1_phone_raw_flat_xgb.yaml", "leaky", "phone", "raw_flat", "xgboost", "A1"),
        ("protocol_a2_phone_raw_flat_xgb.yaml", "leaky", "phone", "raw_flat", "xgboost", "A2"),
        ("protocol_b_phone_stat_xgb.yaml", "groupkfold", "phone", "statistical", "xgboost", "B"),
        ("protocol_b_phone_raw_flat_xgb.yaml", "groupkfold", "phone", "raw_flat", "xgboost", "B"),
        ("protocol_b_phone_stat_dummy.yaml", "groupkfold", "phone", "statistical", "dummy", "B"),
        ("protocol_b_phone_stat_logreg.yaml", "groupkfold", "phone", "statistical", "logreg", "B"),
        ("protocol_b_phone_stat_rf.yaml", "groupkfold", "phone", "statistical", "rf", "B"),
        ("protocol_b_watch_stat_xgb.yaml", "groupkfold", "watch", "statistical", "xgboost", "B"),
        ("protocol_b_concat_stat_xgb.yaml", "groupkfold", "both", "statistical", "xgboost", "B"),
        ("protocol_c_phone_stat_xgb.yaml", "grouped_holdout", "phone", "statistical", "xgboost", "C"),
    ],
)
def test_train_yaml_matches_protocol_device_features(
    filename: str,
    protocol: str,
    device: str,
    features: str,
    model: str,
    protocol_name: str,
) -> None:
    cfg = yaml.safe_load((REPO / "configs" / filename).read_text(encoding="utf-8"))
    assert cfg["split"]["protocol"] == protocol
    assert cfg["data"]["device"] == device
    assert cfg["features"]["kind"] == features
    assert cfg["model"]["name"] == model
    assert cfg["tracking"]["protocol_name"] == protocol_name
    if protocol == "groupkfold":
        assert cfg["split"]["n_splits"] == 5
    if protocol == "grouped_holdout":
        assert cfg["split"]["n_test"] == 5
        assert cfg["split"]["n_repeats"] == 3
    if filename == "protocol_a1_phone_raw_flat_xgb.yaml":
        assert cfg["window"]["length_samples"] == 80
        assert cfg["window"]["hop_samples"] == 40
        params = cfg["model"]["params"]
        for key, value in STUDENT_XGB_PARAMS.items():
            assert params[key] == value
        assert params["device"] == "cuda"
        return
    if filename.endswith("_raw_flat_xgb.yaml"):
        assert cfg["window"]["length_s"] == 5.0
        assert cfg["window"]["hop_s"] == 1.0
        params = cfg["model"]["params"]
        for key, value in STUDENT_XGB_PARAMS.items():
            assert params[key] == value
        assert params["device"] == "cuda"
    elif model == "xgboost":
        params = cfg["model"]["params"]
        assert 100 <= params["n_estimators"] <= 200
        assert params["max_depth"] == 6
        assert params["device"] == "cuda"


def test_hierarchical_run_experiment_logs_model(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    axes = {"A": 1.0, "D": 2.0, "F": -1.0, "H": -2.0}
    for subject_id in (1600, 1601):
        for activity, ax in axes.items():
            _write_session(processed, subject_id, activity, ax=ax)

    config = _write_config(
        tmp_path,
        processed,
        "hierarchical",
        features="statistical",
        protocol_name="B",
        split={"protocol": "leaky", "leaky_test_size": 0.25},
    )
    payload = json.loads(run_experiment(config).read_text(encoding="utf-8"))
    assert payload["model"] == "hierarchical"
    assert payload["n_windows"] > 0
    assert isinstance(payload["macro_f1"], float)


def test_magnitude_features_are_32_from_two_mags(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for subject_id in (1600, 1601):
        _write_session(processed, subject_id, "A", ax=1.0)
        _write_session(processed, subject_id, "B", ax=-1.0)

    config = _write_config(
        tmp_path,
        processed,
        "dummy",
        features="magnitude",
        protocol_name="B",
    )
    payload = json.loads(run_experiment(config).read_text(encoding="utf-8"))
    assert payload["features"] == "magnitude"
    assert payload["n_channels"] == 6
    assert payload["n_features"] == 32


def test_trim_start_reduces_windows(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for subject_id in (1600, 1601):
        _write_session(processed, subject_id, "A", ax=1.0, duration_s=20.0)
        _write_session(processed, subject_id, "B", ax=-1.0, duration_s=20.0)

    window = {"length_s": 5.0, "hop_s": 1.0, "min_coverage": 0.0}
    full = json.loads(
        run_experiment(
            _write_config(
                tmp_path,
                processed,
                "dummy",
                window=window,
                filename="full.yaml",
                protocol_name="B",
            )
        ).read_text(encoding="utf-8")
    )
    trimmed = json.loads(
        run_experiment(
            _write_config(
                tmp_path,
                processed,
                "dummy",
                window=window,
                filename="trim.yaml",
                protocol_name="B",
                repair={"trim_start_s": 15.0},
            )
        ).read_text(encoding="utf-8")
    )
    assert trimmed["n_windows"] < full["n_windows"]
    assert trimmed["n_windows"] > 0


def test_reorient_aligned_swaps_phone_accel_leaves_gyro() -> None:
    n = 8
    channels = np.zeros((n, 6), dtype=np.float32)
    channels[:, 0] = 9.8
    channels[:, 3] = 1.0
    channels[:, 4] = 2.0
    channels[:, 5] = 3.0
    session = AlignedSession(
        subject_id=1600,
        activity="A",
        device="phone",
        timestamps_ns=np.arange(n, dtype=np.int64),
        channels=channels,
        hz=TARGET_HZ,
    )
    out = _reorient_aligned(session)
    np.testing.assert_allclose(out.channels[:, 1], 9.8)
    np.testing.assert_allclose(out.channels[:, 0], 0.0)
    np.testing.assert_allclose(out.channels[:, 3:], channels[:, 3:])
    watch = AlignedSession(
        subject_id=1600,
        activity="A",
        device="watch",
        timestamps_ns=session.timestamps_ns,
        channels=channels.copy(),
        hz=TARGET_HZ,
    )
    np.testing.assert_array_equal(_reorient_aligned(watch).channels, channels)


ABLATION_FILES = (
    "window_2s.yaml",
    "window_10s.yaml",
    "trim_15s.yaml",
    "reorient_on.yaml",
    "magnitude.yaml",
    "hierarchical.yaml",
)


def test_ablation_yaml_files_match_research_knobs() -> None:
    root = REPO / "configs" / "ablations"
    found = tuple(sorted(p.name for p in root.glob("*.yaml")))
    assert found == tuple(sorted(ABLATION_FILES))
    merged = {name: _load_config(root / name, REPO) for name in ABLATION_FILES}
    for name, cfg in merged.items():
        assert cfg["split"]["protocol"] == "groupkfold"
        assert cfg["data"]["device"] == "phone"
        assert cfg["tracking"]["protocol_name"] == "B"
        assert cfg["tracking"]["output_dir"] == "docs/reports/ablations"
        assert cfg["window"]["hop_s"] == 1.0
        assert cfg["model"]["params"]["n_estimators"] == 200
    assert merged["window_2s.yaml"]["window"]["length_s"] == 2.0
    assert merged["window_10s.yaml"]["window"]["length_s"] == 10.0
    assert merged["trim_15s.yaml"]["repair"]["trim_start_s"] == 15.0
    assert merged["trim_15s.yaml"]["repair"]["reorient"] is False
    assert merged["reorient_on.yaml"]["repair"]["reorient"] is True
    assert merged["reorient_on.yaml"]["repair"]["trim_start_s"] == 0.0
    assert merged["magnitude.yaml"]["features"]["kind"] == "magnitude"
    assert merged["hierarchical.yaml"]["model"]["name"] == "hierarchical"
    assert merged["hierarchical.yaml"]["features"]["kind"] == "statistical"
    for name in ("trim_15s.yaml", "reorient_on.yaml", "magnitude.yaml", "hierarchical.yaml"):
        assert merged[name]["window"]["length_s"] == 5.0
