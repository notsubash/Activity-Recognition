# Model card: WISDM HAR

Intended use: CPU API that maps one 5 s, 20 Hz IMU window (phone or watch, 6 channels) to one of 18 WISDM activities. Not a medical or safety device.

## Optional DL

Trees on repaired session features were enough. Protocol B GroupKFold phone statistical XGBoost macro-F1 is 0.3272 (`configs/protocol_b_phone_stat_xgb.yaml`), which beats dummy 0.0151 (`configs/protocol_b_phone_stat_dummy.yaml`), logreg 0.2767 (`configs/protocol_b_phone_stat_logreg.yaml`), RF 0.3131 (`configs/protocol_b_phone_stat_rf.yaml`), and flattened raw 0.2924 (`configs/protocol_b_phone_raw_flat_xgb.yaml`). Watch statistical XGBoost is 0.7031 (`configs/protocol_b_watch_stat_xgb.yaml`). Hierarchical is 0.3271 (`configs/ablations/hierarchical.yaml`) and does not beat the flat tree. No `src/har/models/tcn.py`, no `configs/phone_tcn.yaml`, no PyTorch extra.

## Served model (Task 12)

| Field | Value |
|-------|--------|
| Recommended bundle | Watch statistical XGBoost from `configs/protocol_b_watch_stat_xgb.yaml` |
| Protocol (metrics) | B, 5-fold GroupKFold on `subject_id` |
| Protocol B macro-F1 | 0.7031 (`docs/reports/protocol_b_watch_stat_xgb.json`) |
| Protocol B accuracy | 0.7013 |
| Features | statistical, 104 dims, 5.0 s window, 20 Hz, `T=100`, `C=6` |
| Artifact | ONNX (`python -m har.models.export --out models/watch_stat_xgb.onnx`) plus sidecar JSON. joblib is a fallback. |
| ONNX | XGBoost trees via `onnxmltools==1.16.0` / `onnx==1.22.0`. Statistical features stay in Python. `onnxruntime==1.20.1` runs the graph. |
| Abstain | `max(proba) < threshold`; default threshold 0.0 (never abstain). Not temperature-scaled. |
| p95 CPU latency | 2.7 ms over 100 `POST /predict` calls (FastAPI TestClient, 200-tree XGBoost, statistical 100x6 window, this CPU). Not Docker/uvicorn and not a 51-subject watch model. Pytest checks a stub path stays under 500 ms. |

Export fits on all windows from that config (one train subject used only for early stopping). That served fit is not a GroupKFold fold. Cite Protocol B numbers from the metrics JSON, not from the export command.

Phone statistical XGBoost is 0.3272 macro-F1 under the same protocol. Concat fusion is 0.5236 and is stacked 6-channel windows, not 12-channel alignment. Do not send a phone window to a watch bundle; the API returns 422 on `device` mismatch.

## Metrics to cite next to any number

- Primary: macro-F1. Also per-class F1, per-group F1 (locomotion, posture, hand, eating), balanced accuracy.
- Leaky student-style accuracy is Protocol A only. Do not put it next to this API.

## Failure modes

Watch group F1 from `docs/reports/protocol_b_watch_stat_xgb.json`: locomotion 0.9292, posture 0.6606, hand 0.8788, eating 0.8450. Phone eating and posture stay hard (eating group F1 0.4945 on `configs/protocol_b_phone_stat_xgb.yaml`). Stairs, kicking, and the eating cluster are the usual confusions. Hierarchical raises eating group F1 on phone (0.5855) but not 18-way macro-F1.

## Data and limits

Training data is repaired WISDM 20 Hz parquet (UCI 507). No demographics, so no fairness slice. Timestamp deltas are nanoseconds; absolute epoch is unused. Details: `docs/data_card.md`, `docs/protocol.md`, `docs/limitations.md`.

## How to serve

See `serving/README.md`. Endpoints: `GET /health`, `GET /labels`, `POST /predict`. Wrong window length or channel count is 422.
