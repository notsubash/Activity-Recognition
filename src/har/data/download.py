from __future__ import annotations

import argparse
import hashlib
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import BinaryIO

from har.constants import RAW_SENTINEL, UCI_ZIP_URL

log = logging.getLogger(__name__)

CHUNK_BYTES = 1024 * 1024
ZIP_NAME = "wisdm-dataset.zip"
URL_TIMEOUT_S = 60
_SENTINEL = Path(RAW_SENTINEL)


def resolve_raw_root(dest: Path) -> Path | None:
    """Return the directory that contains `raw/`, if the sentinel file is present."""
    dest = dest.resolve()
    candidates = (
        dest,
        dest / "wisdm-dataset",
        dest / "wisdm-dataset" / "wisdm-dataset",
    )
    for candidate in candidates:
        if (candidate / _SENTINEL).is_file():
            return candidate
    return None


def download_and_extract(dest: Path, url: str, sha256: str | None = None) -> Path:
    """Fetch the WISDM zip into dest (unless already extracted), verify, extract, return raw_root."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    existing = resolve_raw_root(dest)
    if existing is not None:
        log.info("already extracted: %s", existing)
        return existing

    expected = None if sha256 in (None, "") else sha256.lower()
    zip_path = dest / ZIP_NAME
    digest: str | None = None
    if zip_path.is_file() and expected is not None:
        digest = _sha256_file(zip_path)
        if digest != expected:
            zip_path.unlink()
            digest = None

    if not zip_path.is_file():
        with _open_source(url) as source:
            digest = _stream_to_file(source, zip_path)
        log.info("zip sha256: %s", digest)

    if expected is not None:
        if digest is None:
            digest = _sha256_file(zip_path)
        if digest != expected:
            zip_path.unlink(missing_ok=True)
            raise ValueError(f"sha256 mismatch: expected {sha256}, got {digest}")

    try:
        _safe_extract(zip_path, dest)
        raw_root = resolve_raw_root(dest)
        if raw_root is None:
            raise FileNotFoundError(
                f"extracted zip but missing {_SENTINEL.as_posix()} under {dest}"
            )
    except BaseException:
        zip_path.unlink(missing_ok=True)
        raise
    log.info("extracted to %s", raw_root)
    return raw_root


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Download and extract WISDM UCI 507. Not for CI.")
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("--sha256", default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repo = _repo_root()
    config_path = args.config or repo / "configs" / "audit.yaml"
    data_cfg = _load_data_config(config_path)

    dest = args.dest or Path(data_cfg.get("dest") or repo / "data" / "external")
    if not dest.is_absolute():
        dest = repo / dest
    url = args.url or data_cfg.get("zip_url") or UCI_ZIP_URL
    sha256 = args.sha256 if args.sha256 is not None else data_cfg.get("zip_sha256")

    raw_root = download_and_extract(dest, url=url, sha256=sha256)
    print(raw_root)
    return raw_root


def _open_source(url: str) -> BinaryIO:
    path = Path(url)
    if path.is_file():
        return path.open("rb")
    request = urllib.request.Request(url, headers={"User-Agent": "har-wisdm-downloader/0.1"})
    return urllib.request.urlopen(request, timeout=URL_TIMEOUT_S)  # noqa: S310


def _stream_to_file(source: BinaryIO, dest: Path) -> str:
    hasher = hashlib.sha256()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with tmp.open("wb") as out:
            while True:
                chunk = source.read(CHUNK_BYTES)
                if not chunk:
                    break
                out.write(chunk)
                hasher.update(chunk)
        tmp.replace(dest)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return hasher.hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_extract(zip_path: Path, dest: Path) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"unsafe path in zip: {info.filename}")
        zf.extractall(dest)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_data_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = loaded.get("data") if isinstance(loaded, dict) else None
    return data if isinstance(data, dict) else {}


if __name__ == "__main__":
    main()
