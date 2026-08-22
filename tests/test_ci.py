import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
MAKEFILE = REPO / "Makefile"
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
PYPROJECT = REPO / "pyproject.toml"
SERVING_README = REPO / "serving" / "README.md"
REQUIRED_MAKE_TARGETS = ("install", "test", "audit", "prepare", "train", "eval", "serve")


def _dep_pin(name: str) -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(rf'"{re.escape(name)}==([^"]+)"', text)
    assert match, name
    return match.group(1)


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
    assert 'python-version: "3.13"' in text


def test_pyarrow_pin_satisfies_mlflow_2_22():
    pyarrow = _dep_pin("pyarrow")
    mlflow = _dep_pin("mlflow")
    major = int(pyarrow.split(".", 1)[0])
    assert mlflow.startswith("2.22."), mlflow
    assert 4 <= major < 20, pyarrow


def test_serving_readme_uses_portable_models_mount():
    text = SERVING_README.read_text(encoding="utf-8")
    assert "C:/Users/" not in text
    assert "$PWD/models:/models" in text


def test_gitignore_ignores_artifact_dir_not_package_models():
    lines = [
        line.strip()
        for line in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "/models/" in lines
    assert "models/" not in lines
    assert (REPO / "src" / "har" / "models" / "export.py").is_file()
