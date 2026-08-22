import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
MAKEFILE = REPO / "Makefile"
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
REQUIRED_MAKE_TARGETS = ("install", "test", "audit", "prepare", "train", "eval", "serve")


def test_makefile_exposes_pipeline_targets():
    raw = MAKEFILE.read_bytes()
    assert b"\r" not in raw
    text = raw.decode("utf-8")
    targets = set(re.findall(r"^([a-zA-Z][\w-]*)\s*:", text, flags=re.M))
    assert set(REQUIRED_MAKE_TARGETS) <= targets
    assert "pip install" in text
    assert "pytest" in text
    assert "har.data.audit" in text
    assert "har.data.repair" in text
    assert "har.train" in text
    assert "har.evaluate" in text
    assert "uvicorn" in text


def test_ci_runs_ruff_and_pytest_without_wisdm():
    text = WORKFLOW.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    jobs = payload.get("jobs") or {}
    assert jobs, "ci.yml needs at least one job"
    blob = text.lower()
    assert "ruff" in blob
    assert "pytest" in blob
    assert "har.data.download" not in blob
    assert "archive.ics.uci.edu" not in blob
    assert "data/external" not in blob
