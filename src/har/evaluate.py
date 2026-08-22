"""Ladder CLI: run configs, rebuild a summary table, print one MLflow run."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from har.eval.plots import write_readme_figures
from har.train import _repo_root, _tracking_uri, run_experiment

log = logging.getLogger(__name__)

LADDER_NAME = "ladder_summary.json"
TABLE_KEYS = ("protocol", "device", "features", "model", "macro_f1", "accuracy")


def run_ladder(config_paths: list[Path]) -> Path:
    """Run each config via ``run_experiment`` and write ``ladder_summary.json``."""
    runs: list[dict[str, Any]] = []
    table: list[dict[str, Any]] = []
    out_dir: Path | None = None
    for raw in config_paths:
        metrics_path = run_experiment(Path(raw))
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if out_dir is None:
            out_dir = metrics_path.parent
        runs.append(payload)
        table.append(_table_row(payload, source=metrics_path.name))
    if out_dir is None:
        out_dir = _repo_root() / "docs" / "reports"
    return _write_summary(out_dir, runs, table)


def summarize_reports(reports_dir: Path) -> Path:
    """Rebuild the ladder table from existing metrics JSON files."""
    reports_dir = Path(reports_dir)
    runs: list[dict[str, Any]] = []
    table: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob("*.json")):
        if path.name == LADDER_NAME:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "macro_f1" not in payload:
            continue
        runs.append(payload)
        table.append(_table_row(payload, source=path.name))
    summary = _write_summary(reports_dir, runs, table)
    write_readme_figures(reports_dir, _figure_dir(reports_dir))
    return summary


def _figure_dir(reports_dir: Path) -> Path:
    if reports_dir.name == "reports":
        return reports_dir.parent / "figures"
    return reports_dir / "figures"


def main(argv: list[str] | None = None) -> Path | None:
    parser = argparse.ArgumentParser(description="Run or assemble the HAR evaluation ladder.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--configs", nargs="+", type=Path)
    group.add_argument("--from-reports", type=Path)
    group.add_argument("--run-id")
    parser.add_argument("--tracking-uri")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.configs:
        out = run_ladder(args.configs)
        print(out)
        return out
    if args.from_reports:
        out = summarize_reports(args.from_reports)
        print(out)
        return out
    _print_run(str(args.run_id), args.tracking_uri)
    return None


def _table_row(payload: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    row = {key: payload.get(key) for key in TABLE_KEYS}
    row["protocol_name"] = payload.get("protocol_name")
    row["source"] = source
    return row


def _write_summary(out_dir: Path, runs: list[dict[str, Any]], table: list[dict[str, Any]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / LADDER_NAME
    out.write_text(json.dumps({"runs": runs, "table": table}, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s rows=%d", out, len(table))
    return out


def _print_run(run_id: str, tracking_uri: object) -> None:
    import mlflow

    uri = _tracking_uri(tracking_uri, _repo_root())
    run = mlflow.MlflowClient(tracking_uri=uri).get_run(run_id)
    print(
        json.dumps(
            {"params": dict(run.data.params), "metrics": dict(run.data.metrics)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
