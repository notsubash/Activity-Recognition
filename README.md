# Activity Recognition with WISDM + XGBoost

Notebook-first human activity recognition project using WISDM accelerometer data.

The pipeline includes:
- data ingestion from raw + ARFF exports
- preprocessing and exploratory analysis
- XGBoost model training and evaluation
- confusion matrix and class-wise metric review

## Repository Structure

```text
.
├─ assets/
│  └─ images/
│     ├─ PhoneConf2.png
│     └─ PhoneEval2.png
├─ data/
│  ├─ activity_key.txt
│  ├─ external/            # local source dataset (gitignored)
│  └─ processed/           # generated CSVs (gitignored)
├─ docs/
│  └─ reports/
│     └─ evaluation.txt
├─ notebooks/
│  └─ archive/             # student notebooks
│     ├─ DataLoader.ipynb
│     ├─ analysis.ipynb
│     └─ PhoneXGB2.ipynb
├─ src/
│  └─ har/                 # installable package
├─ tests/
├─ .editorconfig
├─ .gitignore
├─ pyproject.toml
├─ requirements.txt
├─ requirements-dev.txt
├─ LICENSE
└─ README.md
```

## Tech Stack

- Python 3.10+
- Jupyter Notebook
- pandas, numpy, scipy
- scikit-learn
- xgboost
- matplotlib, seaborn
- FastAPI, ONNX Runtime, MLflow (package runtime; not used by archived notebooks)

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For development tooling (pytest, ruff, pre-commit):

```bash
python -m pip install -r requirements-dev.txt
```

## Data Layout and Paths

Raw WISDM files are gitignored. Expected tree after download or a manual extract:

```text
data/external/wisdm-dataset/raw/phone/accel/data_1600_accel_phone.txt
```

```bash
python -m har.data.download   # skips if that file already exists; not used in CI
python -m har.data.audit      # writes gitignored CSVs under data/audit/ and docs/data_card.md
python -m har.data.repair     # resample/align to 20 Hz; writes gitignored parquet under data/processed/
python -m har.train --config configs/protocol_a2_phone_raw_flat_xgb.yaml  # A2 leaky XGBoost; overnight on full WISDM, not CI

python -m mlflow ui --backend-store-uri mlruns # Run Mlflow
```

Zip URL and checksum field live in `configs/audit.yaml`. See `data/README.md`.

Archived notebooks still expect a processed CSV:

1. `notebooks/archive/DataLoader.ipynb` writes `data/processed/raw.csv` and `arff.csv`.
2. `notebooks/archive/analysis.ipynb` reads `data/processed/arff.csv`.
3. `notebooks/archive/PhoneXGB2.ipynb` reads `data/processed/raw.csv`.

The archive loader hardcoded `../data/external/wisdm-dataset/wisdm-dataset`. The UCI zip extracts one level shallower (`data/external/wisdm-dataset`).

## Notebook Workflow

1. Run `notebooks/archive/DataLoader.ipynb`.
2. Run `notebooks/archive/analysis.ipynb`.
3. Run `notebooks/archive/PhoneXGB2.ipynb`.

## Formatting and Quality

- `.editorconfig` for whitespace and newlines
- `ruff` via `pyproject.toml` (line length 100). Dev extras are pytest, ruff, and pre-commit.

```bash
python -m ruff check src tests
python -m ruff format --check src tests
```

## Baseline Result

From `docs/reports/evaluation.txt` (Protocol A, leaky window split, phone-only):
- best run accuracy is around `0.8559`
- class-level metrics highlight weaker classes such as stairs/kicking compared to sitting/writing

See `docs/protocol.md` for A1 (80-sample clone) vs A2 (session-safe leaky) and `docs/limitations.md` for why that number is not subject-independent. MLflow writes `mlruns/` (gitignored). Fixture tests train dummy and a tiny XGBoost; they do not download WISDM.

## Honest results (Task 9)

Primary metric is **macro-F1**. Accuracy is secondary. Every cell names protocol and config. Numbers below are full 51-subject WISDM on repaired 20 Hz parquet. On the same flattened phone windows, leaky A2 is 0.8925 macro-F1 and subject-grouped B is 0.2924. A1 (80-sample clone) is 0.8490 / 0.8475, next to the student 0.8559 accuracy.

### Protocol A1 vs the student notebook (phone, leaky, 80-sample flatten)

| Protocol | Config | Model | macro-F1 | Accuracy |
|----------|--------|-------|----------|----------|
| Student notebook | `docs/reports/evaluation.txt` | xgboost | (not reported) | 0.8559 |
| A1 leaky | `configs/protocol_a1_phone_raw_flat_xgb.yaml` (`docs/reports/protocol_a1_phone_raw_flat_xgb.json`) | xgboost (student 982 trees) | 0.8490 | 0.8475 |

A1 is the clone. It is not the same window geometry as A2.

### Protocol A vs B, same representation (phone, 5 s / 1 s, `raw_flat`)

| Protocol | Config | Model | macro-F1 | Accuracy |
|----------|--------|-------|----------|----------|
| A2 leaky | `configs/protocol_a2_phone_raw_flat_xgb.yaml` (`docs/reports/protocol_a2_phone_raw_flat_xgb.json`) | xgboost (student 982 trees) | 0.8925 | 0.8913 |
| B GroupKFold | `configs/protocol_b_phone_raw_flat_xgb.yaml` (`docs/reports/protocol_b_phone_raw_flat_xgb.json`) | xgboost (student 982 trees) | 0.2924 | 0.3047 |

### Protocol B, statistical features, GroupKFold 5

| Device | Config | Model | macro-F1 | Accuracy |
|--------|--------|-------|----------|----------|
| phone | `configs/protocol_b_phone_stat_dummy.yaml` (`docs/reports/protocol_b_phone_stat_dummy.json`) | dummy | 0.0151 | 0.0551 |
| phone | `configs/protocol_b_phone_stat_logreg.yaml` (`docs/reports/protocol_b_phone_stat_logreg.json`) | logreg | 0.2767 | 0.2799 |
| phone | `configs/protocol_b_phone_stat_rf.yaml` (`docs/reports/protocol_b_phone_stat_rf.json`) | rf | 0.3131 | 0.3252 |
| phone | `configs/protocol_b_phone_stat_xgb.yaml` (`docs/reports/protocol_b_phone_stat_xgb.json`) | xgboost (honest 200 trees) | 0.3272 | 0.3382 |
| watch | `configs/protocol_b_watch_stat_xgb.yaml` (`docs/reports/protocol_b_watch_stat_xgb.json`) | xgboost (honest 200 trees) | 0.7031 | 0.7013 |
| both (concat windows, 6 channels; not 12-channel fusion) | `configs/protocol_b_concat_stat_xgb.yaml` (`docs/reports/protocol_b_concat_stat_xgb.json`) | xgboost (honest 200 trees) | 0.5236 | 0.5267 |

### Protocol C (46/5 x 3 repeats from one seed, not 51-fold LOSO)

| Protocol | Config | Model | macro-F1 | Accuracy |
|----------|--------|-------|----------|----------|
| C grouped_holdout | `configs/protocol_c_phone_stat_xgb.yaml` (`docs/reports/protocol_c_phone_stat_xgb.json`) | xgboost (honest 200 trees) | 0.2985 | 0.3140 |

Rebuild the compact table:

```bash
python -m har.evaluate --from-reports docs/reports
```

## License

MIT License. See `LICENSE`.
