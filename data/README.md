# Data Directory

Raw WISDM files are gitignored. Do not commit them.

## Layout (this dump)

The official UCI zip extracts to a single `wisdm-dataset/` folder. Place it under `data/external/`:

```text
data/external/wisdm-dataset/
  README.txt
  WISDM-dataset-description.pdf
  activity_key.txt
  raw/phone/accel/data_1600_accel_phone.txt
  raw/phone/gyro/
  raw/watch/accel/
  raw/watch/gyro/
  arff_files/
  arffmagic-master/
```

Checksum of the zip is recorded in `configs/audit.yaml` as `data.zip_sha256` (null until the first scripted download).

## Get the files

Option A: already extracted (this machine):

```text
data/external/wisdm-dataset/raw/phone/accel/data_1600_accel_phone.txt
```

Option B: download for reproducibility (not run in CI):

```bash
python -m har.data.download
# or: python scripts/download_wisdm.py
```

If that path already exists, the script skips the ~295 MB fetch.

Option C: download the zip yourself from [UCI dataset 507](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset) and extract into `data/external/`. Do not unzip into `data/external/wisdm-dataset/`; that creates an extra nested folder.

The archived `DataLoader.ipynb` hardcoded `../data/external/wisdm-dataset/wisdm-dataset`. That extra nested folder is not what the UCI zip produces. If you re-run the notebook, pass `../data/external/wisdm-dataset`.

## Tracked vs generated

- `activity_key.txt` (repo root `data/`): label legend, tracked.
- `external/`: original WISDM dump, gitignored.
- `processed/`: repaired parquet sessions and `manifest.jsonl`, gitignored (keep `.gitkeep`). Archived notebooks still write `raw.csv` / `arff.csv` here.
- `audit/`: generated coverage tables, gitignored (keep `.gitkeep`). Summary lives in `docs/data_card.md`.

## Audit

```bash
python -m har.data.audit
# or: python scripts/audit.py
```

Writes `data/audit/sessions.csv`, `coverage.csv`, `missing_cells.csv`, `hz_by_session.csv` (gitignored) and regenerates `docs/data_card.md`. Does not download data. CI must not run this on the full dump.

## Repair

```bash
python -m har.data.repair
# or: python scripts/prepare.py
```

Resamples each session to 20 Hz by interpolating onto a shared time grid (not every k-th row), aligns accel and gyro by coverage intersection (not an exact-timestamp inner join), optionally reorients phone accel, then trims the start. Default config is `configs/default.yaml` (`reorient: false`). Ablation: `configs/repair_reorient.yaml`.

Writes one parquet per aligned session under `data/processed/{device}/` plus `data/processed/manifest.jsonl` (input path, n_in, n_out, hz_in, hz_out, reorient, trim). Re-running replaces parquet, `phone/`/`watch/` dirs, and `manifest.jsonl`; it leaves other files such as archived-notebook `raw.csv`. CI must not run this on the full dump.
