import hashlib
import zipfile
from pathlib import Path

import pytest
import yaml

from har.data.download import download_and_extract

REPO_ROOT = Path(__file__).resolve().parents[1]

FIXTURE_TXT = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "tiny_wisdm"
    / "raw"
    / "phone"
    / "accel"
    / "data_1600_accel_phone.txt"
)
SENTINEL = Path("raw/phone/accel/data_1600_accel_phone.txt")


def _zip_tree(zip_path: Path, source_root: Path, prefix: str) -> str:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in source_root.rglob("*"):
            if path.is_file():
                arcname = prefix + path.relative_to(source_root).as_posix()
                zf.write(path, arcname)
    return hashlib.sha256(zip_path.read_bytes()).hexdigest()


def test_extracts_tiny_zip_and_returns_raw_root(tmp_path: Path):
    src = tmp_path / "src.zip"
    dest = tmp_path / "external"
    digest = _zip_tree(src, FIXTURE_TXT.parents[3], "wisdm-dataset/")

    raw_root = download_and_extract(dest, url=str(src), sha256=digest)

    assert raw_root == dest / "wisdm-dataset"
    assert (raw_root / SENTINEL).is_file()
    assert (raw_root / SENTINEL).read_text(encoding="utf-8") == FIXTURE_TXT.read_text(
        encoding="utf-8"
    )


def test_skips_when_already_extracted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_root = tmp_path / "wisdm-dataset"
    marker = raw_root / SENTINEL
    marker.parent.mkdir(parents=True)
    marker.write_text("already here\n", encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise AssertionError("must not fetch when extracted")

    monkeypatch.setattr("har.data.download._open_source", boom)
    result = download_and_extract(tmp_path, url="http://example.invalid/wisdm.zip")
    assert result == raw_root
    assert marker.read_text(encoding="utf-8") == "already here\n"


def test_extracts_existing_zip_without_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    zip_path = tmp_path / "wisdm-dataset.zip"
    _zip_tree(zip_path, FIXTURE_TXT.parents[3], "wisdm-dataset/")

    def boom(*_args, **_kwargs):
        raise AssertionError("must not fetch when zip is already on disk")

    monkeypatch.setattr("har.data.download._open_source", boom)
    raw_root = download_and_extract(tmp_path, url="http://example.invalid/wisdm.zip")
    assert (raw_root / SENTINEL).is_file()


def test_rejects_wrong_sha256(tmp_path: Path):
    src = tmp_path / "src.zip"
    dest = tmp_path / "external"
    _zip_tree(src, FIXTURE_TXT.parents[3], "wisdm-dataset/")
    with pytest.raises(ValueError, match="sha256"):
        download_and_extract(dest, url=str(src), sha256="0" * 64)


def test_nested_zip_layout_resolves_raw_root(tmp_path: Path):
    src = tmp_path / "src.zip"
    dest = tmp_path / "external"
    _zip_tree(src, FIXTURE_TXT.parents[3], "wisdm-dataset/wisdm-dataset/")

    raw_root = download_and_extract(dest, url=str(src), sha256=None)
    assert raw_root == dest / "wisdm-dataset" / "wisdm-dataset"
    assert (raw_root / SENTINEL).is_file()


def test_audit_yaml_documents_tree_and_checksum_field():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "audit.yaml").read_text(encoding="utf-8"))
    data = cfg["data"]
    assert data["raw_root"] == "data/external/wisdm-dataset"
    assert data["zip_url"].startswith("https://archive.ics.uci.edu/")
    assert "zip_sha256" in data
    assert "raw/phone/accel/data_1600_accel_phone.txt" in data["expected_tree"]
