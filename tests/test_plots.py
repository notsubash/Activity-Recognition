import json
from pathlib import Path

from har.constants import LABEL_ORDER
from har.eval.plots import class_f1_by_name, sync_mlflow_run_names, write_readme_figures
from har.train import run_experiment
from test_train import _write_config, _write_session


def _report(**overrides: object) -> dict[str, object]:
    per_class = {str(i): 0.1 * ((i % 5) + 1) for i in range(len(LABEL_ORDER))}
    row: dict[str, object] = {
        "macro_f1": 0.32,
        "accuracy": 0.33,
        "protocol": "groupkfold",
        "protocol_name": "B",
        "model": "xgboost",
        "device": "phone",
        "features": "statistical",
        "per_class_f1": per_class,
        "per_group_f1": {
            "locomotion": 0.88,
            "posture": 0.37,
            "hand": 0.60,
            "eating": 0.49,
        },
    }
    row.update(overrides)
    return row


def _write(path: Path, **overrides: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_report(**overrides)) + "\n", encoding="utf-8")


def test_class_f1_by_name_maps_index_keys() -> None:
    named = class_f1_by_name({"0": 0.7, "1": 0.2, str(len(LABEL_ORDER) - 1): 0.4})
    assert named["walking"] == 0.7
    assert named["jogging"] == 0.2
    assert named["folding clothes"] == 0.4


def test_write_readme_figures_creates_pngs(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _write(
        reports / "protocol_a2_phone_raw_flat_xgb.json",
        protocol="leaky",
        protocol_name="A2",
        features="raw_flat",
        macro_f1=0.89,
    )
    _write(
        reports / "protocol_b_phone_raw_flat_xgb.json",
        features="raw_flat",
        macro_f1=0.29,
    )
    for name, device, model, f1 in (
        ("protocol_b_phone_stat_dummy.json", "phone", "dummy", 0.02),
        ("protocol_b_phone_stat_logreg.json", "phone", "logreg", 0.28),
        ("protocol_b_phone_stat_rf.json", "phone", "rf", 0.31),
        ("protocol_b_phone_stat_xgb.json", "phone", "xgboost", 0.33),
        ("protocol_b_concat_stat_xgb.json", "both", "xgboost", 0.52),
        ("protocol_b_watch_stat_xgb.json", "watch", "xgboost", 0.70),
    ):
        _write(reports / name, device=device, model=model, macro_f1=f1)
    _write(reports / "ablations" / "window_10s.json", macro_f1=0.34)
    _write(reports / "ablations" / "window_2s.json", macro_f1=0.30)
    _write(reports / "ablations" / "trim_15s.json", macro_f1=0.32)
    _write(reports / "ablations" / "reorient_on.json", macro_f1=0.32)
    _write(reports / "ablations" / "magnitude.json", macro_f1=0.31)
    _write(reports / "ablations" / "hierarchical.json", macro_f1=0.33, model="hierarchical")

    out = tmp_path / "figures"
    written = write_readme_figures(reports, out)
    names = {path.name for path in written}
    assert names == {
        "leakage_macro_f1.png",
        "protocol_b_ladder.png",
        "per_class_f1_phone_watch.png",
        "per_group_f1_phone_watch.png",
        "ablations_macro_f1.png",
        "sampling_rate_modes.png",
    }
    for path in written:
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_write_readme_figures_skips_missing_reports(tmp_path: Path) -> None:
    written = write_readme_figures(tmp_path / "reports", tmp_path / "figures")
    assert {path.name for path in written} == {"sampling_rate_modes.png"}


def test_run_experiment_uses_config_stem_as_mlflow_run_name(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for subject_id in (1600, 1601):
        _write_session(processed, subject_id, "A", ax=1.0)
        _write_session(processed, subject_id, "B", ax=-1.0)

    metrics_path = run_experiment(
        _write_config(tmp_path, processed, "dummy", filename="protocol_b_phone_stat_dummy.yaml")
    )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    import mlflow

    run = mlflow.MlflowClient(tracking_uri=payload["mlflow_tracking_uri"]).get_run(
        payload["mlflow_run_id"]
    )
    assert run.info.run_name == "protocol_b_phone_stat_dummy"
    assert "fold_macro_f1" in run.data.metrics
    assert "group_f1_locomotion" in run.data.metrics


def test_sync_mlflow_run_names_renames_existing_run(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for subject_id in (1600, 1601):
        _write_session(processed, subject_id, "A", ax=1.0)
        _write_session(processed, subject_id, "B", ax=-1.0)

    from har.train import run_experiment

    metrics_path = run_experiment(_write_config(tmp_path, processed, "dummy", filename="exp.yaml"))
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    import mlflow

    client = mlflow.MlflowClient(tracking_uri=payload["mlflow_tracking_uri"])
    client.set_tag(payload["mlflow_run_id"], "mlflow.runName", "amusing-gibbon-7")
    reports = metrics_path.parent
    reports.joinpath("protocol_b_watch_stat_xgb.json").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )
    metrics_path.unlink()
    updated = sync_mlflow_run_names(reports)
    assert updated == ["protocol_b_watch_stat_xgb"]
    run = client.get_run(payload["mlflow_run_id"])
    assert run.info.run_name == "protocol_b_watch_stat_xgb"
