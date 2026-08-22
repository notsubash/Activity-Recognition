PYTHON ?= python
CONFIG ?= configs/protocol_b_watch_stat_xgb.yaml
RAW_ROOT ?= data/external/wisdm-dataset
PROCESSED_DIR ?= data/processed
HOST ?= 0.0.0.0
PORT ?= 8000

.PHONY: install test audit prepare train eval figures serve

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

audit:
	$(PYTHON) -m har.data.audit --raw-root $(RAW_ROOT)

prepare:
	$(PYTHON) -m har.data.repair --raw-root $(RAW_ROOT) --processed-dir $(PROCESSED_DIR)

train:
	$(PYTHON) -m har.train --config $(CONFIG)

eval:
	$(PYTHON) -m har.evaluate --from-reports docs/reports

figures:
	$(PYTHON) -m har.eval.plots --from-reports docs/reports --out docs/figures --sync-mlflow

serve:
	$(PYTHON) -m uvicorn har.serve.app:app --host $(HOST) --port $(PORT)
