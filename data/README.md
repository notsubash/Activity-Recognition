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
- `processed/`: generated arrays and CSVs, gitignored.
- `audit/`: generated coverage tables, gitignored (keep `.gitkeep`). Summary lives in `docs/data_card.md`.

## Audit

```bash
python -m har.data.audit
# or: python scripts/audit.py
```

Writes `data/audit/sessions.csv`, `coverage.csv`, `missing_cells.csv`, `hz_by_session.csv` (gitignored) and regenerates `docs/data_card.md`. Does not download data. CI must not run this on the full dump.
