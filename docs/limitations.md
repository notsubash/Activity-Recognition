# Limitations of the archived student pipeline

This is a description of the 2024 notebook pipeline in `notebooks/archive/`, not of the rebuilt `src/har/` package. Keep these defects in mind when you read `docs/reports/evaluation.txt` (phone accuracy `0.855905403547367`). That number is Protocol A (leaky). It is not a subject-independent HAR result.

## Subject leakage

`Subject-id` is dropped, then `train_test_split` shuffles windows. The same person can appear on both sides of the split. A tree can memorize gait, phone placement, and watch laterality instead of the activity. Protocol B (GroupKFold) and Protocol C (grouped holdout, not 51-fold LOSO) exist so this cannot happen by accident.

## Temporal leakage

Windows are 80 samples with hop 40, then shuffled. Overlapping windows from the same bout land in train and test. Early stopping watches the test set (`eval_set=[(X_test, y_test)]`). Protocol A copies that setup on purpose. Honest runs must hold out **subjects** for validation, not random windows from people already in train.

## Cross-session windows

`get_frames` slides over the concatenated table. A window can mix two subjects or two activities at a file boundary. Session-safe windowing only slides inside one `(subject, device, activity)` run.

## Timestamp unit bug

Sample deltas are about `5.03e7`. Treated as nanoseconds that is ~50.3 ms, about 20 Hz. `pd.to_datetime(..., unit="us")` turns those deltas into ~50 s gaps and decorative 2019 datetimes. Deltas are usable. Absolute epoch is not (rWISDM: it does not match 2017).

## Scaling fitted on all rows

A scaler fit on the full matrix before the split sees test windows. Fit scalers, encoders, and early stopping only on training subjects.

## Weak tree features

XGBoost is given `80 x 6 = 480` raw samples. The official ARFF already had distribution bins, peaks, MFCCs, and axis correlations. Flattened raw windows remain the A1/A2 reproduction. Statistical features are the Protocol B default, compared later (RQ3).

## Exact-timestamp accel/gyro merge

IMU clocks rarely share sample instants. An inner join on the raw timestamp drops and distorts rows. Repair interpolates both streams onto a shared 20 Hz grid per session.

## Unused deep-learning imports

The phone notebook imports Keras CNN pieces. TensorFlow was in `requirements.txt` and is not a v1 runtime dependency. No CNN ships until a logged table shows XGBoost losing on GroupKFold macro-F1.

## Watch data unused

The student result is phone-only with no stated product choice. The dump has watch accel and gyro. Phone vs watch vs fusion is RQ4, under a subject-independent protocol.

## Evaluation writeup

`docs/reports/evaluation.txt` is a generic metric walkthrough, not a model card. It does not name the split, the window identity rules, or the timestamp handling. Rebuilt numbers live next to a protocol name in the README and in MLflow.

## What this rebuild does not claim yet

Task 9 filled the Protocol B/C table in the README from `docs/reports/`. A2 is `docs/reports/protocol_a_leaky.json`. Do not treat A2 vs 0.8559 as a leakage-only delta: the student matrix was unrepaired and concat-windowed. Do not treat statistical GroupKFold XGBoost as the same-representation A vs B gap; that cell is `protocol_b_raw_flat.json`.
