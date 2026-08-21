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
# 7 passed
```
