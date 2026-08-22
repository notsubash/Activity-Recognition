# Evaluation protocols

Primary metric is **macro-F1**. Also report per-class F1, per-group F1 (locomotion, posture, hand, eating), and balanced accuracy. Accuracy is secondary. Every public number must name its protocol.

A1 and A2 train on **repaired 20 Hz** session-safe parquet from `data/processed/`. They clone the student *split and window geometry*, not the unrepaired concat table. Do not judge overnight A2 against 0.8559 as if parse bugs were held fixed.

Windows are built inside one `(subject_id, device, activity)` session. Scalers, encoders, and early stopping belong on training subjects only, except Protocol A, which copies the student notebook and early-stops on the test set.

## Ladder

| Protocol | Split | When to use |
|----------|--------|-------------|
| A leaky | `train_test_split` on windows, `random_state=42`, 20% test | Student-style leaky number. Label every figure **leaky**. |
| A1 | A, plus 80-sample flatten / hop 40 (row count, not seconds) | Closest clone of `notebooks/archive/PhoneXGB2.ipynb`. Config: `configs/protocol_a1_phone_raw_flat_xgb.yaml`. |
| A2 | A, plus session-safe 5 s / 1 s hop on the same repaired matrix | Leaky vs GroupKFold on one representation (RQ2). Config: `configs/protocol_a2_phone_raw_flat_xgb.yaml`. |
| B | 5-fold GroupKFold on `subject_id` | Main comparison table. Configs: `configs/protocol_b_*.yaml`. |
| C | Grouped holdout: 5 test subjects, rest train, 3 repeats from one seed (46/5 x 3). Not 51-fold LOSO. | Variance across people. Config: `configs/protocol_c_phone_stat_xgb.yaml`. |
| D | Train phone, test watch (and reverse) on the same subjects | Hardware transfer. Later. |

Known clone deltas vs the archive notebook: timestamp deltas as nanoseconds (not `unit="us"`), accel/gyro interpolated onto a shared grid (not an exact-timestamp join), session-safe windows (not `get_frames` on a concat table), no scaler fit on all rows, and flatten layout `(N, T, C)` C-order rather than `(N, 6, 80)`. YAML files request `device: cuda`; `fit_xgboost` falls back to CPU if CUDA is not visible.

Target after A2 exists: leaky accuracy in the same ballpark as 0.86 is possible but not required. Protocol B on the **same** flattened features should drop. That drop is the finding (RQ2). Config: `configs/protocol_b_phone_raw_flat_xgb.yaml` (student XGBoost params, GroupKFold).

**Concat** in this repo is phone and watch windows stacked as extra rows. Each window is still 6 channels. `data.device: both` loads both devices. It is not a 12-channel time-aligned phone+watch fusion. Config: `configs/protocol_b_concat_stat_xgb.yaml`.

Protocol C keeps `loso()` in code for a true 51-fold run later. The shipped C config uses `grouped_holdout` (5 test subjects, 3 repeats) because 51-fold XGBoost is too slow. Nested XGBoost validation is still one held-out **train** subject (`_train_val`), not a separate 5-subject val split.

## Training config names

Filenames are `protocol_{rung}_{device}_{features}_{model}.yaml`. `stat` means `features.kind: statistical`. `concat` means `data.device: both` (stacked 6-channel windows). Metrics JSON uses the same stem under `docs/reports/`.

| Config | Role | Run on disk |
|--------|------|-------------|
| `protocol_a1_phone_raw_flat_xgb.yaml` | Student clone, 80-sample flatten, leaky | yes |
| `protocol_a2_phone_raw_flat_xgb.yaml` | 5 s flatten, leaky (RQ2 leaky side) | yes |
| `protocol_b_phone_raw_flat_xgb.yaml` | Same flatten as A2, GroupKFold (RQ2 honest side) | yes |
| `protocol_b_phone_stat_dummy.yaml` | Chance floor | yes |
| `protocol_b_phone_stat_logreg.yaml` | Linear baseline | yes |
| `protocol_b_phone_stat_rf.yaml` | Tree baseline | yes |
| `protocol_b_phone_stat_xgb.yaml` | Honest phone XGBoost | yes |
| `protocol_b_watch_stat_xgb.yaml` | Honest watch XGBoost | yes |
| `protocol_b_concat_stat_xgb.yaml` | Honest concat phone+watch windows | yes |
| `protocol_c_phone_stat_xgb.yaml` | Phone grouped holdout 46/5 x 3 | yes |

Not missing, on purpose:

- No watch A1/A2. The student notebook is phone-only.
- No watch dummy/logreg/RF. One 18-class chance floor is enough.
- No watch raw_flat. RQ3 (features vs flatten) is the phone A2 vs B flatten pair plus phone statistical B.
- No Protocol C on watch or concat. C checks that phone holdout tracks phone GroupKFold. Watch GroupKFold already has 5 subject folds.
- No 12-channel aligned fusion. That is a new pipeline, not a YAML rename.

## Student hyperparameters (Protocol A, and Protocol B raw_flat only)

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

Honest B/C XGBoost (phone statistical, watch, concat, Protocol C) uses a smaller family: `n_estimators: 200`, `max_depth: 6`, `device: cuda`. Do not treat those runs as a 982-tree reproduction.

## Commands

Fixture smoke (CI / pytest) trains dummy and a 2-tree XGBoost on synthetic windows. Full WISDM Protocol A is an overnight run, not CI. A missing `--config` path errors; it does not silently run `configs/default.yaml`.

```bash
python -m har.train --config configs/protocol_a1_phone_raw_flat_xgb.yaml
python -m har.train --config configs/protocol_a2_phone_raw_flat_xgb.yaml
python -m har.train --config configs/protocol_b_phone_stat_dummy.yaml
python -m har.train --config configs/protocol_b_phone_stat_logreg.yaml
python -m har.train --config configs/protocol_b_phone_stat_rf.yaml
python -m har.train --config configs/protocol_b_phone_stat_xgb.yaml
python -m har.train --config configs/protocol_b_phone_raw_flat_xgb.yaml
python -m har.train --config configs/protocol_b_watch_stat_xgb.yaml
python -m har.train --config configs/protocol_b_concat_stat_xgb.yaml
python -m har.train --config configs/protocol_c_phone_stat_xgb.yaml

python -m har.evaluate --configs configs/protocol_b_phone_stat_dummy.yaml configs/protocol_b_phone_stat_logreg.yaml
python -m har.evaluate --from-reports docs/reports
python -m har.evaluate --run-id <mlflow_run_id>
# or: python scripts/evaluate.py --from-reports docs/reports
```

Requires repaired parquet under `data/processed/` from `python -m har.data.repair`. MLflow writes `mlruns/` (gitignored). Metrics JSON writes to `docs/reports/`. Logged fields include `macro_f1`, `mean_fold_macro_f1`, `pooled_macro_f1`, `protocol` / `protocol_name`, `model`, `device`, `features`, subject unions across folds (or `pooled_oof` in MLflow for multi-fold), `session_hz`, and window shape. `docs/reports/ladder_summary.json` is the compact table. Protocol C headline `macro_f1` is the unweighted mean of repeats, not pooled windows.
