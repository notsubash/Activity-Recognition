# Evaluation protocols

Primary metric is **macro-F1**. Also report per-class F1, per-group F1 (locomotion, posture, hand, eating), and balanced accuracy. Accuracy is secondary. Every public number must name its protocol.

A1 and A2 train on **repaired 20 Hz** session-safe parquet from `data/processed/`. They clone the student *split and window geometry*, not the unrepaired concat table. Do not judge overnight A2 against 0.8559 as if parse bugs were held fixed.

Windows are built inside one `(subject_id, device, activity)` session. Scalers, encoders, and early stopping belong on training subjects only, except Protocol A, which copies the student notebook and early-stops on the test set.

## Ladder

| Protocol | Split | When to use |
|----------|--------|-------------|
| A leaky | `train_test_split` on windows, `random_state=42`, 20% test | Student-style leaky number. Label every figure **leaky**. |
| A1 | A, plus 80-sample flatten / hop 40 (row count, not seconds) | Closest clone of `notebooks/archive/PhoneXGB2.ipynb` that this package can run. Config: `configs/phone_xgb.yaml`. |
| A2 | A, plus session-safe 5 s / 1 s hop on the same repaired matrix | Leaky vs GroupKFold on one representation (RQ2). Config: `configs/protocol_a_leaky.yaml`. |
| B | 5-fold GroupKFold on `subject_id` | Main comparison table. Task 9. |
| C | Leave-one-subject-out, or 41/5/5 subjects x seeds if 51 folds are too slow | Variance across people. Task 9. |
| D | Train phone, test watch (and reverse) on the same subjects | Hardware transfer. Later. |

Known clone deltas vs the archive notebook: timestamp deltas as nanoseconds (not `unit="us"`), accel/gyro interpolated onto a shared grid (not an exact-timestamp join), CPU not CUDA, session-safe windows (not `get_frames` on a concat table), no scaler fit on all rows, and flatten layout `(N, T, C)` C-order rather than `(N, 6, 80)`.

Target after A2 exists: leaky accuracy in the same ballpark as 0.86 is possible but not required. Protocol B on the **same** flattened features should drop. That drop is the finding (RQ2).

## Student hyperparameters (Protocol A only)

From `docs/reports/evaluation.txt`. They are pinned in the A1/A2 YAML files. `fit_xgboost` does not inject them. Do not reuse them as the honest-protocol default without a subject-grouped search.

```text
colsample_bytree: 0.9396893641976711
gamma: 0
learning_rate: 0.10241823755571676
max_depth: 6
n_estimators: 982
subsample: 0.8545330472743582
early_stopping_rounds: 10
```

## Commands

Fixture smoke (CI / pytest) trains dummy and a 2-tree XGBoost on synthetic windows. Full WISDM Protocol A is an overnight run, not CI. A missing `--config` path errors; it does not silently run `configs/default.yaml`.

```bash
python -m har.train --config configs/protocol_a_leaky.yaml
python -m har.train --config configs/phone_xgb.yaml
```

Requires repaired parquet under `data/processed/` from `python -m har.data.repair`. MLflow writes `mlruns/` (gitignored). Metrics JSON writes to `docs/reports/`. Logged fields include `macro_f1`, `protocol` / `protocol_name`, subject lists (or `pooled_oof` for multi-fold), `session_hz`, and window shape.
