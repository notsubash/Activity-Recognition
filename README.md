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

From `docs/reports/evaluation.txt`:
- best run accuracy is around `0.8559`
- class-level metrics highlight weaker classes such as stairs/kicking compared to sitting/writing

## License

MIT License. See `LICENSE`.
