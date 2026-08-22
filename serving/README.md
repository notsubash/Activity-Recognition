# HAR API

CPU service for one 5 s, 20 Hz window (`T=100`, `C=6`). The v1 artifact is ONNX from `python -m har.models.export` (XGBoost trees via `onnxmltools`; statistical features still run in Python). joblib remains a fallback for stubs and non-XGBoost estimators.

Shipped default to train for serving is watch statistical XGBoost (`configs/protocol_b_watch_stat_xgb.yaml`, Protocol B macro-F1 0.7031). Phone is a different model; do not send phone windows to a watch bundle.

## Train to a bundle

From the repo root, after `data/processed/` exists:

```bash
python -m har.models.export --config configs/protocol_b_watch_stat_xgb.yaml --out models/watch_stat_xgb.onnx
```

That fits on all windows (one subject held out only for XGBoost early stopping), then writes gitignored `models/watch_stat_xgb.onnx` plus sidecar `models/watch_stat_xgb.json`. `data.device` must be `phone` or `watch`, not `both`. Export forces `device: cpu` even if the YAML says `cuda`. Use `--out *.joblib` only if you need a pickle.

Abstain is `max(proba) < threshold`. Default threshold is `0.0` (never abstain) until a calibration pass.

```bash
python -m har.models.export --config configs/protocol_b_watch_stat_xgb.yaml --out models/watch_stat_xgb.onnx --abstain-threshold 0.0
```

## Run locally

```bash
export HAR_MODEL_PATH=models/watch_stat_xgb.onnx
# optional override: HAR_ABSTAIN_THRESHOLD=0.0
uvicorn har.serve.app:app --host 0.0.0.0 --port 8000
```

The sidecar JSON must sit next to the `.onnx` file (same stem).

```http
GET  /health
GET  /labels
POST /predict
```

`POST /predict` needs `T=100` rows (`window_s * hz` at the default 5 s / 20 Hz). Each row is 6 channels in `ax, ay, az, gx, gy, gz` order. Wrong `T` or `C` returns 422. Wrong `device` or `hz` also returns 422.

```bash
python -c "import json,urllib.request; body=json.dumps({'device':'watch','hz':20,'channels':['ax','ay','az','gx','gy','gz'],'samples':[[0,0,0,0,0,0]]*100}).encode(); req=urllib.request.Request('http://127.0.0.1:8000/predict', data=body, headers={'Content-Type':'application/json'}); print(json.load(urllib.request.urlopen(req)))"
```

Response:

```json
{
  "activity_code": "A",
  "activity_name": "walking",
  "group": "locomotion",
  "proba": {"A": 0.81},
  "confidence": 0.81,
  "abstained": false
}
```

(`proba` includes all 18 codes A-S except N.)

p95 on this CPU was 2.7 ms over 100 `POST /predict` calls through FastAPI TestClient with a 200-tree XGBoost on a statistical 100x6 window. Not Docker/uvicorn. Pytest asserts the stub path stays under 500 ms.

```bash
python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')))"
```

## Docker

Build from the repo root (not from `serving/`):

```bash
docker build -f serving/Dockerfile -t har-api .
MSYS_NO_PATHCONV=1 docker run --rm -p 8000:8000 \
  -v "$PWD/models:/models" \
  -e HAR_MODEL_PATH=/models/watch_stat_xgb.onnx \
  har-api
```

Git Bash rewrites `/models/...` to a Git path unless `MSYS_NO_PATHCONV=1` is set. If local uvicorn is already on 8000, map `-p 8001:8000` instead.

The image is CPU inference only. It does not install MLflow, XGBoost, or pyarrow. A full `pip install .` is the train/eval env, not this image.
