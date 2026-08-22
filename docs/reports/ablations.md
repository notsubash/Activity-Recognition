# Protocol B ablations

Phone, repaired 20 Hz parquet, GroupKFold 5, statistical features unless noted. Primary metric is **macro-F1**. Control is `configs/protocol_b_phone_stat_xgb.yaml` (`docs/reports/protocol_b_phone_stat_xgb.json`): 5.0 s / 1.0 s hop, XYZ, no reorient, trim 0 s, flat 18-way XGBoost (200 trees).

Trim and reorient run at train time on that same parquet. They do not overwrite `data/processed/`.

| Setting | Config | n windows | T | Features | Model | macro-F1 | Accuracy | Eating group F1 |
|---------|--------|-----------|---|----------|-------|----------|----------|-----------------|
| Control (5 s, XYZ, trim 0, flat 18-way) | `configs/protocol_b_phone_stat_xgb.yaml` | 158798 | 100 | statistical (104) | xgboost | 0.3272 | 0.3382 | 0.4945 |
| Window 2 s | `configs/ablations/window_2s.yaml` | 161519 | 40 | statistical (104) | xgboost | 0.2951 | 0.3048 | 0.4610 |
| Window 10 s | `configs/ablations/window_10s.yaml` | 154263 | 200 | statistical (104) | xgboost | 0.3422 | 0.3528 | 0.5151 |
| Trim first 15 s | `configs/ablations/trim_15s.yaml` | 145193 | 100 | statistical (104) | xgboost | 0.3247 | 0.3357 | 0.4712 |
| Phone-accel reorient | `configs/ablations/reorient_on.yaml` | 158798 | 100 | statistical (104) | xgboost | 0.3230 | 0.3360 | 0.4830 |
| Magnitude only (accel+gyro) | `configs/ablations/magnitude.yaml` | 158798 | 100 | magnitude (32) | xgboost | 0.3142 | 0.3279 | 0.4516 |
| Hierarchical (group then expert) | `configs/ablations/hierarchical.yaml` | 158798 | 100 | statistical (104) | hierarchical | 0.3271 | 0.3302 | 0.5855 |

JSON for the six knobs: `docs/reports/ablations/<stem>.json`. Hop is 1.0 s in every row.

Per-group F1 (locomotion / posture / hand / eating):

| Setting | Locomotion | Posture | Hand | Eating |
|---------|------------|---------|------|--------|
| Control | 0.8873 | 0.3709 | 0.6002 | 0.4945 |
| Window 2 s | 0.8685 | 0.3584 | 0.5716 | 0.4610 |
| Window 10 s | 0.8927 | 0.3791 | 0.6023 | 0.5151 |
| Trim 15 s | 0.8877 | 0.3660 | 0.5978 | 0.4712 |
| Reorient | 0.8895 | 0.3726 | 0.5926 | 0.4830 |
| Magnitude | 0.8698 | 0.2580 | 0.6103 | 0.4516 |
| Hierarchical | 0.8964 | 0.3178 | 0.6380 | 0.5855 |

## Repair knobs on this 18-class phone split

Resample-to-20 Hz and accel/gyro align are already in the control parquet. On top of that:

- rWISDM-style phone-accel reorient does not raise Protocol B macro-F1 (0.3230 vs 0.3272). Default `reorient: false` stays. This ablation reorients already-aligned phone accel (`channels[:, :3]`); `prepare.py` would reorient raw accel before interpolating onto the gyro grid.
- Dropping the first 15 s does not raise macro-F1 (0.3247) and lowers eating group F1. Default `trim_start_s: 0.0` stays.

## Two-stage head

Hierarchical does **not** beat flat 18-way on macro-F1 (0.3271 vs 0.3272). It does raise eating group F1 (0.5855 vs 0.4945) and hand group F1 (0.6380 vs 0.6002), while posture drops (0.3178 vs 0.3709). Eating class F1 moves H +0.047, I +0.027, J +0.062, K -0.031, L +0.029.

Experts train on the true group. Inference routes by the predicted group. Locomotion labels include M (index 12), so experts remap XGBoost's local `0..K-1` back to `LABEL_ORDER` indices.

## Other knobs (modeling policy, not extra XGBoost search)

- 10 s windows are the only row that clearly beats the 5 s control on macro-F1 (0.3422). 2 s windows are worse (0.2951).
- Magnitude-only (two Euclidean channels, then the same statistical extractor) is worse than XYZ (0.3142). Axis stats matter.

No XGBoost grid search. Same 200-tree family as the honest phone B control.

```bash
python -m har.train --config configs/ablations/window_2s.yaml
python -m har.train --config configs/ablations/window_10s.yaml
python -m har.train --config configs/ablations/trim_15s.yaml
python -m har.train --config configs/ablations/reorient_on.yaml
python -m har.train --config configs/ablations/magnitude.yaml
python -m har.train --config configs/ablations/hierarchical.yaml
```
