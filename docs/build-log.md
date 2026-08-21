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
