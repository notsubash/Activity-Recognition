# Build log (blog / video)

Source notes for public writing. Not a second spec. After each plan task, append a section. Commands are copy-pasteable.

---

## Task 0: Package skeleton, constants, archive notebooks

**Commit:** `369ae56` (skeleton), `64f4350` (Pyright config)

**Story beat:** The student repo had notebooks and an 85.6% number. We turned it into an installable package and locked the 18 WISDM labels before touching models.

**Shipped:**
- `src/har/` with `constants.py` and `types.py`
- `tests/test_constants.py`
- `pyproject.toml` `[project]` + pytest; TensorFlow dropped from runtime deps
- Student notebooks moved to `notebooks/archive/`
- `.gitignore` for `mlruns/`, `data/audit/`, ONNX

**Decision:** Freeze `CODE_TO_NAME` / `GROUP_OF` as `MappingProxyType` and test they match `data/activity_key.txt`, rather than parsing that file on import. Setuptools `packages.find` from `src/` instead of Poetry's `packages = [{ include = "har", from = "src" }]`. Four-group map in code is locomotion A,B,C,M; posture D,E; hand F,G,O,P,Q,R,S; eating H,I,J,K,L. Standing (E) is posture, not locomotion, even though an earlier plan bullet mixed it.

**Gotcha:** There is no activity `N`. Eighteen classes are A–S skipping N. If a video graphic shows A–R or 19 letters, it is wrong.

**Demo clip:**
```bash
python -c "from har.constants import ACTIVITY_CODES; print(len(ACTIVITY_CODES))"
# 18
pytest tests/test_constants.py -q
# 6 passed
```

---

## Task 1: Parser

**Commit:** `e727c5a` (parser), plus follow-up for pandas typing, parse `path:line` errors, and this log

**Story beat:** The archived loader left a semicolon glued to every `z` value. The new parser strips `;` at ingest and splits one subject-sensor file into activity runs instead of sliding windows across the concatenated table.

**Shipped:**
- `src/har/data/parse.py`: `parse_raw_line`, `parse_raw_file` (DataFrame), `load_subject_sensor_file`, `split_activity_runs`
- `tests/test_parse.py`, `tests/fixtures/sample_raw.txt`
- Official Weiss line as the first fixture row

**Decision:** One raw file is one subject × device × sensor with all activities concatenated, so `parse_raw_file` returns a DataFrame, not a `SessionFrame`. Runs split when activity changes, timestamp goes backward, or the gap is greater than 2 s (`>` not `>=`). Also split on `subject_id` change. Line-by-line parse, no `applymap`, no `pd.to_datetime`.

**Gotcha:**
- Official format is `subject-id, activity-code, timestamp, x, y, z;` with a trailing semicolon on `z`. The student `read_csv` kept that semicolon as part of the string.
- Timestamp deltas are about `5.035e7`, which is nanoseconds at 20 Hz, not microseconds. Do not demo `pd.to_datetime(..., unit="us")`.
- The zip on disk extracted to `data/external/wisdm-dataset/raw/...`, not the nested `wisdm-dataset/wisdm-dataset/raw/...` path in the plan. Parser does not care; Task 2 docs must.
- Pyright / pandas-stubs: `pd.DataFrame(rows, columns=list(RAW_COLUMNS))` infers `list[Literal[...]]`, which is not an `Axes`. Annotating `tuple[str, ...]` is not enough either (`tuple.index` vs `SequenceNotStr`). Build the frame from a dict of columns instead.

**Demo clip:**
```text
# tests/fixtures/sample_raw.txt
1600,A,252207666810782,-0.36476135,8.793503,1.0550842;
1600,A,252207717164786,-0.8797302,9.768784,1.0169983;
1600,B,252207767518790,2.0014954,11.10907,2.619156;
```
Second A line is exactly 50,354,004 ns after the official sample (20 Hz). Then B starts a new session.

```bash
pytest tests/test_parse.py -q
# 7 passed in this file; 13 with test_constants.py
python -c "from pathlib import Path; from har.data.parse import load_subject_sensor_file, parse_raw_line; p=Path('data/external/wisdm-dataset/raw/phone/accel/data_1600_accel_phone.txt'); print(parse_raw_line(p.read_text().splitlines()[0])); frames=load_subject_sensor_file(p,'phone','accel'); print(len(frames), [f.key.activity for f in frames])"
# (1600, 'A', 252207666810782, -0.36476135, 8.793503, 1.0550842)
# 18 ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S']
```

---

## Task 2: Download script and data README

**Commit:** `d73675a`

**Story beat:** Reproducibility is a script plus a checksum field, not a zip in git. If the sentinel file is already on disk, the downloader does nothing.

**Shipped:**
- `src/har/data/download.py`: `download_and_extract(dest, url, sha256) -> raw_root`, streaming fetch, optional sha256, skip when extracted
- `scripts/download_wisdm.py` and `python -m har.data.download`
- `configs/audit.yaml` with zip URL, `zip_sha256: null`, expected tree
- `data/README.md` and README data section
- `tests/test_download.py` plus `tests/fixtures/tiny_wisdm/` (one fake txt, zipped in the test)
- `.gitignore` no longer ignores `docs/build-log.md`

**Decision:** Default extract dest is `data/external`, because the UCI zip already contains a top-level `wisdm-dataset/` folder. That matches this dump: `data/external/wisdm-dataset/raw/...`. The plan and archived `DataLoader.ipynb` used `wisdm-dataset/wisdm-dataset/raw/...`; `resolve_raw_root` accepts both. `zip_sha256` stays null until someone runs a real zip download (this machine has an extract, not the zip).

**Gotcha:** Extracting the UCI zip *into* `data/external/wisdm-dataset/` creates the extra nested folder. CI must not call the downloader; the tests use a tiny fixture zip only. A failed extract (HTML 403, truncated body, zip-slip) must delete `wisdm-dataset.zip` or the next run reuses it forever while `zip_sha256` is still null.

**Demo clip:**
```bash
python -m pytest tests/test_download.py -q
# 6 passed
python -m har.data.download
# already extracted: .../data/external/wisdm-dataset
# .../data/external/wisdm-dataset
```

---

## Task 3: Audit (research deliverable)

**Commit:** pending (you add the commit)

**Story beat:** This dump matches Weiss row counts exactly (15,630,426). The student notebook's extra 18,827 rows are not in the raw files. Sampling is mixed 20/25/50/100 Hz, so a 200-row window is not 10 seconds.

**Shipped:**
- `src/har/data/audit.py`: `audit_session`, `audit_dataset`, coverage grid, missing cells, Weiss totals, `write_data_card`
- `scripts/audit.py` and `python -m har.data.audit`
- `tests/test_audit.py` (synthetic 20 vs 50 Hz, 1609-like missing B, Weiss warning, data card)
- `docs/data_card.md` from the real dump
- `data/audit/.gitkeep` (CSVs gitignored)
- `configs/audit.yaml` `audit_dir` / `data_card`

**Decision:** Coverage is the full 51 x 18 x 4 grid with zeros; `missing_cells.csv` is `n_samples == 0`. Implied Hz is `1e9 / median_dt_ns`, NaN if fewer than 2 samples. CSVs stay gitignored (`sessions.csv` is 725 KB); the data card is the tracked summary. Hz modes are 20, 25, 50, and 100, not a 15-25 bin that would hide 25 Hz.

**Gotcha:**
- Official claim is 20 Hz. This dump: 2,838 sessions at 20 Hz, 543 at 25 Hz, 322 at 50 Hz (18 of those round to 51), 14 at 100 Hz. Windowing by row count mixes 10 s and 4 s of real time.
- rWISDM phone-accel gaps (1609 B, 1616 B/F, 1642 C/F) are real. The 18-class grid also lacks 1607 J, 1618 O, 1643 I on phone accel, plus more gyro/watch cells (35 missing cells total).
- 3,717 session runs vs 3,637 occupied cells: 80 extra runs from gap > 2 s or time reversal inside an activity.
- Zero non-monotonic timestamps and zero NaNs in this dump. Weiss totals matched, so no warning. The student 15,649,253 figure is a loader/concat artifact, not this extract.
- Empty or nested-wrong `--raw-root` used to overwrite `docs/data_card.md` with a 3672-row empty grid and exit 0. CLI now uses `resolve_raw_root` and `audit_dataset` raises if no txt files match.

**Demo clip:**
```bash
python -m pytest tests/test_audit.py -q
# 12 passed
python -m har.data.audit
# wrote .../docs/data_card.md
```
The data card answers: 35 missing cells (phone accel: 1607 J, 1609 B, 1616 B,F, 1618 O, 1642 C,F, 1643 I) and Hz modes 20 / 25 / 50 / 100.

---

## Task 4: Repair (resample, align, optional reorient, trim)

**Commit:** pending (you add the commit)

**Story beat:** WISDM is not 20 Hz, and accel/gyro clocks do not share sample instants. Repair interpolates onto a 20 Hz grid and aligns by overlapping coverage, so later windows are 5.0 s of real time, not 200 leftover rows.

**Shipped:**
- `src/har/data/repair.py`: `resample_session`, `reorient_phone_accel`, `align_device`, `trim_start`, `prepare_dataset`
- `scripts/prepare.py` and `python -m har.data.repair`
- `tests/test_repair.py`
- `configs/default.yaml` (`reorient: false`) and `configs/repair_reorient.yaml`
- `pyarrow` pinned for parquet

**Decision:** Resample with `np.interp` onto `t0, t0+1/hz, ... t1`, not every k-th row. Align independently onto the intersection of accel and gyro coverage; an exact-timestamp inner join of offset clocks is empty. Reorient is phone accel only: gravity-ish axis is largest |mean|; if that mean is negative, add `2 * abs(mean)` so the AC waveform is not mirrored, then swap X/Y when |mean_x| > |mean_y| so +Y is upright. `prepare_dataset` pairs overlapping runs of the same subject/activity/device, skips duplicate time coverage, writes `data/processed/{device}/{subject}_{activity}_{run}.parquet`, and a `manifest.jsonl` with input_path, n_in, n_out, hz_in, hz_out, reorient, trim. Re-running replaces parquet, device dirs, and manifest; it does not delete notebook CSVs. `align_accel_gyro: false` raises until an unrepaired path exists.

**Gotcha:**
- Inclusive grid: 3.0 s at 20 Hz is 61 samples (`3*20+1`), which still matches the plan's `len ≈ 60 (±2)`.
- Offset IMU clocks are the point of this task. Inner-joining on timestamp is how the student merge dropped and distorted rows.
- Do not multiply the gravity axis by -1. That flips the oscillation. The shift keeps phase.
- Watch and gyro are not reoriented. Enable reorient only via `configs/repair_reorient.yaml`.
- CI must not run prepare on the full dump. Parquet stays gitignored.

**Demo clip:**
```bash
python -m pytest tests/test_repair.py -q
# 5 passed
python -m har.data.repair
# or: python scripts/prepare.py --config configs/repair_reorient.yaml
```

---

## Task 5: Session-safe windowing

**Commit:** pending (you add the commit)

**Story beat:** The archived loader slid 80-sample windows over a concatenated table, so one window could mix subjects or activities. `make_windows` only slides inside one `AlignedSession`. A mixed `subject_id` or activity table cannot become a session.

**Shipped:**
- `src/har/data/windows.py`: `make_windows`, `stack_windows`, `aligned_session_from_dataframe`
- `tests/test_windows.py`

**Decision:** Window length and hop are sample counts from `round(seconds * session.hz)`, so 5 s / 1 s at 20 Hz is 100 samples with hop 20. A 10 s session is 200 samples and yields `1 + floor((10-5)/1) = 6` windows. Coverage is the fraction of timesteps where every channel is finite; windows below `min_coverage` are dropped (callers pass 0.95 from `configs/default.yaml`; it is not a `make_windows` default). `y` is the index of `activity` in `LABEL_ORDER`. `stack_windows` returns `X (N,T,C)`, `y (N,)`, `groups (N,)` subject IDs. `aligned_session_from_dataframe` is the trust boundary for anyone still holding a concatenated table: unique `subject_id`, `activity`, `device`, and `hz`, or raise.

**Gotcha:**
- Do not concatenate two activities and call `make_windows` on the concat. Window each session, then `stack_windows`.
- `AlignedSession` already stores one subject and one activity as scalars. The mixed-identity test is on the dataframe constructor, which is how the student concat table would enter this path. That constructor also rejects mixed `device` and mixed `hz`.
- A session shorter than `length_s` returns `[]`, not a padded window.
- This assumes repaired `session.hz`. Row-count windows on unrepaired 50 Hz data are still the wrong number of seconds.

**Demo clip:**
```bash
python -m pytest tests/test_windows.py -q
# 13 passed
```

---

## Task 6: Statistical features

**Commit:** pending (you add the commit)

**Story beat:** The student XGBoost ate 80 flattened samples. The official ARFF already had bins, MAD, correlations, and a resultant. We ship that statistical family (without MFCC) plus a raw flatten so Protocol B can compare representations on the same windows.

**Shipped:**
- `src/har/features/statistical.py`: `extract_statistical`, `flatten_raw`, `feature_names`
- `tests/test_features.py`

**Decision:** Per channel: mean, std, MAD as mean-abs-dev from the mean (WISDM "average absolute difference"), min, max, range, then 10 equal-width histogram fractions over that channel's min-max. Accel trio (first 3 channels): mean resultant magnitude and pairwise Pearson corr (xy, xz, yz). Same block for gyro when `C==6`. Constant channels put all histogram mass in bin 0 and corr is 0. Skip peak-interval and MFCC; `spectral.py` waits until Protocol B still confuses eating. `flatten_raw` is C-order `(T*C,)`. Six-channel vector is 104 floats (16 per channel, plus 4 accel, plus 4 gyro).

**Gotcha:**
- `np.histogram(..., density=True)` is a PDF, not bin fractions. Count then divide by the count sum.
- A zero-range channel cannot be binned; special-case it or the 10 bins become noise.
- Corrcoef on a constant axis is NaN. Fill with 0 so trees do not eat NaNs.

**Demo clip:**
```bash
python -m pytest tests/test_features.py -q
# 9 passed
PYTHONPATH=src python -c "import numpy as np; from har.features.statistical import extract_statistical, flatten_raw, feature_names; x=np.ones((100,6)); print(extract_statistical(x).shape, flatten_raw(x).shape, len(feature_names(6)))"
# (104,) (600,) 104
```

---

## Task 7: Splits and metrics

**Commit:** pending (you add the commit)

**Story beat:** The student 85.6% came from shuffling windows, so the same person can be in train and test. Protocol A is allowed to leak. Protocol B (GroupKFold) and C (LOSO) raise if a subject appears on both sides.

**Shipped:**
- `src/har/eval/splits.py`: `Split`, `leaky_split`, `group_kfold`, `loso`, `assert_no_subject_overlap`
- `src/har/eval/metrics.py`: `compute_metrics`
- `src/har/eval/plots.py`: `confusion_counts`, `save_confusion_matrix` (matplotlib imported inside the save path)
- `tests/test_splits.py`, `tests/test_metrics.py`

**Decision:** `leaky_split` takes `groups` even though the plan's one-line signature omitted it, because `Split` stores `groups_train` / `groups_test` and that is how the leak test is proved. GroupKFold and LOSO call `assert_no_subject_overlap` on every fold. `per_group_f1` maps class indices through `LABEL_ORDER` then `GROUP_OF` and scores the four-group problem (locomotion, posture, hand, eating). Protocol D hardware transfer waits. Matplotlib stays out of `pyproject.toml` until a test or the train CLI actually saves figures. After review: the group-split fixture is interleaved `[1,2,1,2]` with a unique marker per window so a row-wise `KFold` cannot pass; public eval APIs are typed; out-of-range integer labels raise `ValueError`, not `IndexError`.

**Gotcha:**
- A 50/50 leaky split on `[1,1,2,2]` can accidentally be subject-clean. Use a 3/1 split (`test_size=0.75`) if you want a guaranteed leak for the unit test.
- `GroupKFold(shuffle=True)` is required before `random_state` does anything.
- Integer `y` is an index into `LABEL_ORDER`. Generic toy labels `0,1,2` are walking/jogging/stairs, all locomotion, so `per_group_f1` is not interesting unless you pick classes from different groups.

**Demo clip:**
```bash
python -m pytest tests/test_splits.py tests/test_metrics.py -q
# 5 passed
```

---

## Task 8: Training CLI, tracking, Protocol A reproduction

**Commit:** pending (you add the commit)

**Story beat:** The 85.6% phone number came from a leaky split and 982-tree XGBoost on flattened windows. We can now run that setup as Protocol A from a YAML file, log macro-F1 and subject lists to MLflow, and keep the full WISDM job off CI.

**Shipped:**
- `src/har/models/baselines.py` (`fit_dummy`) and `src/har/models/xgboost.py` (`fit_xgboost`, student params)
- `src/har/train.py`: `run_experiment(config_path) -> metrics json`
- `scripts/train.py` and `python -m har.train --config ...`
- `configs/protocol_a1_phone_raw_flat_xgb.yaml` (A1: 80-sample flatten, hop 40, leaky)
- `configs/protocol_a2_phone_raw_flat_xgb.yaml` (A2: session-safe 5 s / 1 s, leaky, same XGBoost family)
- `docs/protocol.md`, `docs/limitations.md`
- `tests/test_models.py`, `tests/test_train.py`

**Decision:** Protocol A early-stops on the test set, like `PhoneXGB2.ipynb`. XGBoost 2.1 dropped `early_stopping_rounds` on `fit()`, so we pass `xgboost.callback.EarlyStopping`. Student notebook used `device='cuda'`; we pin `device: cpu`. A1 windows are row counts (`length_samples` / `hop_samples`) converted per session as `samples / hz` so mixed-rate files still yield 80-sample vectors. `run_experiment` overlays `configs/default.yaml`. Dummy is `model.name: dummy`. Full 982-tree WISDM is overnight, not pytest.

**Gotcha:**
- A1 is the closest clone, not bit-identical accuracy. Timestamp unit, accel/gyro join, CUDA vs CPU, and repaired 20 Hz parquet already change the matrix.
- A2 is leaky session-safe windows on the **same repaired** matrix. Protocol B on the same flatten is Task 9. It is not an unrepaired parse-bug isolate.
- Missing `--config` must raise. A silent fall-through would train `default.yaml` (GroupKFold, 982 trees) on `data/processed/`.
- `fit_xgboost` does not inject student hyperparameters; A1/A2 YAML owns them.
- MLflow param values are strings; subject lists are comma-separated on leaky runs and `pooled_oof` on multi-fold. Tracking URI is a `file://` path so Windows pytest tmp dirs resolve.
- Do not run `python -m har.train` in CI against `data/processed/`. Tests write tiny parquet under `tmp_path`.

**Demo clip:**
```bash
python -m pytest tests/test_models.py tests/test_train.py -q
python -m har.train --config configs/protocol_a2_phone_raw_flat_xgb.yaml
# writes docs/reports/protocol_a2_phone_raw_flat_xgb.json and mlruns/ (overnight on full WISDM)
```

---

## Task 9: Honest baselines (Protocols B and C)

**Commit:** pending (you add the commit)

**Story beat:** Protocol A2 leaked subjects and scored ~0.89 macro-F1 on flattened phone windows. Task 9 adds GroupKFold and a 46/5 x 3 grouped holdout so dummy, logreg, RF, and XGBoost can be compared without the same person in train and test.

**Shipped:**
- `fit_logreg` (StandardScaler on train only) and `fit_rf` in `src/har/models/baselines.py`
- `grouped_holdout` in `src/har/eval/splits.py` (5 test subjects, 3 repeats). `loso()` kept.
- `src/har/evaluate.py`: `run_ladder`, `--from-reports`, `--run-id`; `scripts/evaluate.py`
- Configs: `protocol_b_phone_stat_{dummy,logreg,rf,xgb}.yaml`, `protocol_b_phone_raw_flat_xgb.yaml`, `protocol_b_watch_stat_xgb.yaml`, `protocol_b_concat_stat_xgb.yaml` (concat 6-channel windows, not 12-channel align), `protocol_c_phone_stat_xgb.yaml` (`grouped_holdout`)
- `notebooks/01_audit_eda.ipynb` loads audit CSVs and reports JSON only
- README results table; `docs/protocol.md` B/C/fusion/honest-xgb notes
- Metrics JSON now includes `model`, `device`, `features`
- Full-WISDM json: dummy/logreg/rf/honest-phone-xgb, `protocol_b_watch_stat_xgb.json` (macro-F1 0.7031), `protocol_b_concat_stat_xgb.json` (0.5236), `protocol_c_phone_stat_xgb.json` (0.2985 mean of 3 repeats; pooled windows were 0.2996), `protocol_b_phone_raw_flat_xgb.json` (0.2924 vs A2 leaky 0.8925).

**Decision:** Concat is phone and watch windows stacked (still 6 channels). Protocol C is not 51-fold LOSO; 51-fold XGBoost is too slow, so C is 46/5 x 3 repeats from one seed. Student 982-tree params stay on A and on `protocol_b_phone_raw_flat_xgb.yaml` (RQ2). Other B/C XGBoost YAML uses 200 trees, max_depth 6, cuda. Evaluate loops `run_experiment`; no window cache.

**Gotcha:**
- Do not present a fixture or 10-subject debug number as the 51-subject result.
- Logreg scaler belongs in a Pipeline fit on train windows only. RF has no scaler.
- `python -m har.evaluate --from-reports docs/reports` skips `ladder_summary.json` itself.
- Do not present a statistical GroupKFold XGBoost number as the A2 vs B leakage gap. A2 leaky `raw_flat` is 0.8925 macro-F1; the same 982-tree flatten under GroupKFold (`protocol_b_phone_raw_flat_xgb.json`) is 0.2924. That drop is RQ2. Honest phone B statistical XGBoost is a different representation (0.3272).
- Watch statistical GroupKFold (0.70 macro-F1) beat phone (0.33) and concat fusion (0.52) on this 18-class split. Concat fusion is more data, not 12-channel alignment.

**Demo clip:**
```bash
python -m pytest tests/test_models.py tests/test_splits.py tests/test_train.py tests/test_evaluate.py -q
# 27 passed
python -m har.train --config configs/protocol_b_phone_stat_dummy.yaml
python -m har.evaluate --from-reports docs/reports
```

---

## Task 10: Ablations and hierarchy (RQ1, RQ5)

**Commit:** pending (you add the commit)

**Story beat:** Flat 18-way phone XGBoost at 5 s is the honest control. A group-then-expert head does not raise macro-F1, but it does raise eating group F1. Ten-second windows are the only knob that clearly beats the 5 s control.

**Shipped:**
- `src/har/models/hierarchical.py`: group XGBoost plus four experts fit on the true group; inference routes by predicted group
- `to_magnitude` in `src/har/features/statistical.py`; `features.kind: magnitude`
- Train-time `repair.trim_start_s` and `repair.reorient` so ablations do not rebuild parquet
- `configs/ablations/{window_2s,window_10s,trim_15s,reorient_on,magnitude,hierarchical}.yaml`
- `docs/reports/ablations.md` plus `docs/reports/ablations/*.json`
- Tests: `tests/test_hierarchical.py` (routing shapes, M=12 remap), train trim/magnitude/YAML checks

**Decision:** Keep default 5 s, trim 0, reorient off. Hierarchical stays an ablation, not the shipped 18-way head. Experts remap local `0..K-1` because locomotion includes M (label 12); XGBoost sklearn rejects `[0,1,2,12]`. Magnitude is two Euclidean channels (accel, gyro), then the same statistical extractor (32 features, no XYZ trio corr).

**Gotcha:**
- A one-class-per-group fixture never hits the M=12 XGBoost error. The first full-WISDM hierarchical run failed on fold 1 until the remap existed.
- The first routing test sliced a class-blocked matrix, so test windows were only eating (and leftover hand). It now splits by class so every expert is hit.
- Train-time reorient is rWISDM gravity repair on phone accel of already-aligned sessions. It is not a second `prepare.py` into a new processed tree.
- Ablation JSON lives under `docs/reports/ablations/` so `python -m har.evaluate --from-reports docs/reports` does not mix them into `ladder_summary.json`. The YAML test now asserts that path after merge with `default.yaml`.

**Demo clip:**
```bash
python -m pytest tests/test_hierarchical.py tests/test_train.py tests/test_features.py -q
# routing + magnitude + ablation YAML
python -m har.train --config configs/ablations/hierarchical.yaml
# writes docs/reports/ablations/hierarchical.json (macro-F1 0.3271, eating group F1 0.5855)
```

---

## Config rename: `protocol_{rung}_{device}_{features}_{model}`

**Commit:** pending (you add the commit)

**Story beat:** `phone_xgb.yaml` and `watch_xgb.yaml` looked like a device pair. They were not. Phone was the leaky student clone; watch was honest GroupKFold. Names now encode protocol, device, features, and model.

**Shipped:** ten ladder YAML files plus matching `docs/reports/*.json`. README includes A1 (0.8490 / 0.8475). `docs/protocol.md` lists the ladder and the intentional omissions (no watch A1, no watch dummy, no watch Protocol C).

**Decision:** Do not add unrun configs to look complete. Task 9 already had every required cell. `stat` means statistical features. `concat` means `device: both`, not 12-channel fusion. Protocol C filename no longer says `loso`.

**Gotcha:** Retrain writes `docs/reports/<config-stem>.json`. Old stems (`phone_xgb.json`, `watch_xgb.json`, `protocol_c_loso.json`) are gone.

**Demo clip:**
```bash
python -m pytest tests/test_train.py -q -k train_ladder
ls configs/protocol_*.yaml
```

---

## Task 11: Optional DL (skipped; trees won)

**Commit:** pending (you add the commit)

**Story beat:** The plan only allows a 1D-CNN/TCN if XGBoost loses on Protocol B macro-F1. It did not, so we did not add PyTorch.

**Shipped:**
- `docs/model_card.md` with the skip paragraph and cited Protocol B configs
- No `src/har/models/tcn.py`, no `configs/phone_tcn.yaml`, no `dl` extra in `pyproject.toml`

**Decision:** Treat "XGBoost wins" as the logged classical ladder, not as an absolute 18-class score. Phone statistical XGBoost (0.3272) beat dummy (0.0151), logreg (0.2767), RF (0.3131), and flattened raw (0.2924). Watch statistical XGBoost is 0.7031. Hierarchical is 0.3271. That is enough to skip DL. Task 12 still owns serving, ONNX, and the rest of the model card.

**Gotcha:** A later TCN run is not a success claim unless it uses the same GroupKFold splits and the same windows, with a side-by-side row next to `configs/protocol_b_phone_stat_xgb.yaml`. Do not add torch "just in case."

**Demo clip:**
```bash
python -c "import tomllib; from pathlib import Path; p=tomllib.loads(Path('pyproject.toml').read_text()); print('torch' in str(p).lower(), 'tcn' in str(p).lower()); print(Path('docs/model_card.md').read_text().splitlines()[6][:80])"
# False False
# Trees on repaired session features were enough. ...
test ! -e src/har/models/tcn.py && test ! -e configs/phone_tcn.yaml && echo skipped
# skipped
```

---

## Task 12: Export, API, Docker, calibration

**Commit:** pending (you add the commit)

**Story beat:** The product surface is a CPU FastAPI that scores one 5 s window. XGBoost trees ship as ONNX; features stay in Python.

**Shipped:**
- `src/har/models/export.py`: `ModelBundle`, joblib and ONNX (`onnxmltools` + sidecar JSON), `predict_window`, `export_from_config`, `python -m har.models.export`
- `src/har/serve/schema.py`, `src/har/serve/app.py`: `GET /health`, `GET /labels`, `POST /predict`
- `serving/Dockerfile`, `serving/README.md`, `.dockerignore`
- `tests/test_serve.py` (stub 422/happy path, device/hz/channel names, body cap, joblib and ONNX roundtrip, fixture export with YAML `cuda` forced to CPU, `HAR_MODEL_PATH` startup for both artifacts, 100-request p95 bound)
- `docs/model_card.md` (Task 11 skip paragraph plus serving contract and p95)
- `httpx==0.28.1` on the dev extra for FastAPI `TestClient`
- `onnx==1.22.0`, `onnxmltools==1.16.0`, `skl2onnx==1.20.0` on runtime deps (`onnx` 1.22 has a wheel; 1.17 did not)

**Decision:** Prefer ONNX for the served XGBoost head. `onnx==1.17.0` had no wheel and tried a source build; `onnx==1.22.0` is `cp312-abi3` and works on Python 3.13. `onnxmltools.convert_xgboost` matches sklearn `predict_proba` on a fixture. joblib remains for stubs. Abstain is `max(proba) < threshold` with default 0.0 (never abstain). The bundle to train for serving is watch statistical XGBoost (`configs/protocol_b_watch_stat_xgb.yaml`). Export refuses `data.device: both` and `hierarchical`. Export always fits with `device: cpu` even if the YAML says `cuda`. `activity_code` is argmax over the 18-way padded `proba`.

**Gotcha:**
- `POST /predict` `samples` length is `T=100` at 5 s / 20 Hz, not one row. A one-row JSON body is 422.
- GroupKFold metrics in the model card are not the export fit. Export refits on all windows (one subject held out only for early stopping).
- Wrong `T` or `C` is 422; so is `device` or `hz` mismatch. Do not send phone windows to a watch bundle.
- Module-level `har.serve.app:app` loads `HAR_MODEL_PATH` on startup. An `.onnx` path also needs the sidecar `.json`.
- joblib pickle must be exported with the same CPython minor as `serving/Dockerfile` (3.13). ONNX does not have that pickle constraint.
- Bodies over 1 MiB return 413. `samples` is capped at 512 rows.

**Demo clip:**
```bash
python -m pytest tests/test_serve.py -q
# 19 passed
python -c "from pathlib import Path; print(Path('docs/model_card.md').read_text().splitlines()[6][:80])"
# Trees on repaired session features were enough. Protocol B GroupKFold phone sta
python -m har.models.export --config configs/protocol_b_watch_stat_xgb.yaml --out models/watch_stat_xgb.onnx
# writes gitignored onnx + json; then:
# HAR_MODEL_PATH=models/watch_stat_xgb.onnx uvicorn har.serve.app:app --host 0.0.0.0 --port 8000
```

