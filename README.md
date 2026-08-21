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

1. Put original WISDM files under `data/external/`.
2. Run `notebooks/archive/DataLoader.ipynb` to generate:
   - `data/processed/raw.csv`
   - `data/processed/arff.csv`
3. `notebooks/archive/analysis.ipynb` reads from `data/processed/arff.csv`.
4. `notebooks/archive/PhoneXGB2.ipynb` reads from `data/processed/raw.csv`.

> Notebooks were updated to use relative paths for local reproducibility.

## Notebook Workflow

1. Run `notebooks/archive/DataLoader.ipynb`.
2. Run `notebooks/archive/analysis.ipynb`.
3. Run `notebooks/archive/PhoneXGB2.ipynb`.

## Formatting and Quality

This repo includes production-style formatting config:
- `.editorconfig` for whitespace/newline consistency
- `pyproject.toml` for `black`, `isort`, and `ruff` configuration

Run formatting checks:

```bash
python -m nbqa black notebooks
python -m nbqa isort notebooks
python -m nbqa ruff notebooks
```

## Baseline Result

From `docs/reports/evaluation.txt`:
- best run accuracy is around `0.8559`
- class-level metrics highlight weaker classes such as stairs/kicking compared to sitting/writing

## License

MIT License. See `LICENSE`.
