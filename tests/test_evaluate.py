import json
from pathlib import Path

import pytest

from har.evaluate import main, run_ladder
from test_train import _write_config, _write_session


def test_run_ladder_writes_summary_with_two_rows(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for subject_id in (1600, 1601):
        _write_session(processed, subject_id, "A", ax=1.0)
        _write_session(processed, subject_id, "B", ax=-1.0)

    cfg_a = _write_config(
        tmp_path,
        processed,
        "dummy",
        filename="ladder_a.yaml",
        protocol_name="B",
    )
    cfg_b = _write_config(
        tmp_path,
        processed,
        "dummy",
        filename="ladder_b.yaml",
        protocol_name="B",
        device="phone",
    )
    summary_path = run_ladder([cfg_a, cfg_b])
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_path.name == "ladder_summary.json"
    assert len(data["table"]) == 2
    assert len(data["runs"]) == 2
    for row in data["table"]:
        assert row["protocol"]
        assert row["device"]
        assert row["features"]
        assert row["model"] == "dummy"
        assert isinstance(row["macro_f1"], float)
        assert isinstance(row["accuracy"], float)


def test_from_reports_rebuilds_summary(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    for i, name in enumerate(("one", "two")):
        (reports / f"{name}.json").write_text(
            json.dumps(
                {
                    "protocol": "groupkfold",
                    "protocol_name": "B",
                    "device": "phone",
                    "features": "statistical",
                    "model": "dummy",
                    "macro_f1": 0.1 + i,
                    "accuracy": 0.2 + i,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    (reports / "ladder_summary.json").write_text(
        json.dumps({"runs": [], "table": []}) + "\n",
        encoding="utf-8",
    )
    out = main(["--from-reports", str(reports)])
    assert out is not None
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["table"]) == 2
    assert {row["macro_f1"] for row in data["table"]} == {0.1, 1.1}


def test_unknown_model_still_errors(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for subject_id in (1600, 1601):
        _write_session(processed, subject_id, "A", ax=1.0)
        _write_session(processed, subject_id, "B", ax=-1.0)

    config = _write_config(tmp_path, processed, "not_a_model")
    with pytest.raises(ValueError, match="unknown model"):
        run_ladder([config])
