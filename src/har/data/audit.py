from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from har.constants import ACTIVITY_CODES, SUBJECT_ID_MAX, SUBJECT_ID_MIN
from har.data.download import resolve_raw_root
from har.data.parse import load_subject_sensor_file
from har.types import Device, Sensor, SessionAudit, SessionFrame

log = logging.getLogger(__name__)

WEISS_ROWS = {
    ("phone", "accel"): 4_804_403,
    ("phone", "gyro"): 3_608_635,
    ("watch", "accel"): 3_777_046,
    ("watch", "gyro"): 3_440_342,
}
WEISS_TOTAL = 15_630_426
STUDENT_TOTAL = 15_649_253

DEVICES: tuple[Device, ...] = ("phone", "watch")
SENSORS: tuple[Sensor, ...] = ("accel", "gyro")
_RAW_NAME = re.compile(r"^data_\d+_(accel|gyro)_(phone|watch)\.txt$")

SESSION_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "activity",
    "device",
    "sensor",
    "n_samples",
    "duration_s",
    "median_dt_ns",
    "p05_dt_ns",
    "p95_dt_ns",
    "implied_hz",
    "n_non_monotonic",
    "n_nan",
    "mean_x",
    "mean_y",
    "mean_z",
    "source_path",
)


def audit_session(frame: SessionFrame, source_path: str = "") -> SessionAudit:
    n = int(frame.timestamps_ns.shape[0])
    xyz = frame.xyz
    n_nan = int(np.isnan(xyz).sum())
    if n == 0:
        mean_x = mean_y = mean_z = float("nan")
    else:
        mean_x = float(np.nanmean(xyz[:, 0]))
        mean_y = float(np.nanmean(xyz[:, 1]))
        mean_z = float(np.nanmean(xyz[:, 2]))

    if n < 2:
        return SessionAudit(
            key=frame.key,
            n_samples=n,
            duration_s=0.0,
            median_dt_ns=float("nan"),
            p05_dt_ns=float("nan"),
            p95_dt_ns=float("nan"),
            implied_hz=float("nan"),
            n_non_monotonic=0,
            n_nan=n_nan,
            mean_x=mean_x,
            mean_y=mean_y,
            mean_z=mean_z,
            source_path=source_path,
        )

    ts = np.asarray(frame.timestamps_ns, dtype=np.int64)
    dt = np.diff(ts)
    median_dt = float(np.median(dt))
    implied_hz = (1e9 / median_dt) if median_dt > 0 else float("nan")
    return SessionAudit(
        key=frame.key,
        n_samples=n,
        duration_s=float(ts[-1] - ts[0]) / 1e9,
        median_dt_ns=median_dt,
        p05_dt_ns=float(np.percentile(dt, 5)),
        p95_dt_ns=float(np.percentile(dt, 95)),
        implied_hz=float(implied_hz),
        n_non_monotonic=int(np.count_nonzero(dt < 0)),
        n_nan=n_nan,
        mean_x=mean_x,
        mean_y=mean_y,
        mean_z=mean_z,
        source_path=source_path,
    )


def iter_raw_sensor_files(raw_root: Path) -> Iterator[tuple[Path, Device, Sensor]]:
    raw = Path(raw_root) / "raw"
    if not raw.is_dir():
        return
    for path in sorted(raw.glob("*/*/*.txt")):
        match = _RAW_NAME.match(path.name)
        if match is None:
            continue
        device = path.parent.parent.name
        sensor = path.parent.name
        if device not in DEVICES or sensor not in SENSORS:
            continue
        name_sensor, name_device = match.group(1), match.group(2)
        if name_device != device or name_sensor != sensor:
            log.warning(
                "filename %s does not match directory %s/%s",
                path.name,
                device,
                sensor,
            )
        yield path, device, sensor  # type: ignore[misc]


def audit_dataset(raw_root: Path) -> pd.DataFrame:
    raw_root = Path(raw_root)
    files = list(iter_raw_sensor_files(raw_root))
    if not files:
        raise FileNotFoundError(f"no raw sensor txt files under {raw_root}")
    rows: list[dict[str, object]] = []
    for path, device, sensor in files:
        try:
            rel = path.relative_to(raw_root).as_posix()
        except ValueError:
            rel = path.as_posix()
        log.info("auditing %s", rel)
        for frame in load_subject_sensor_file(path, device, sensor):
            rows.append(_audit_to_row(audit_session(frame, source_path=rel)))
    df = pd.DataFrame({col: [row[col] for row in rows] for col in SESSION_COLUMNS})
    _warn_weiss_mismatch(df)
    return df


def stream_totals(sessions: pd.DataFrame) -> dict[tuple[str, str], int]:
    totals = {(d, s): 0 for d in DEVICES for s in SENSORS}
    if sessions.empty:
        return totals
    grouped = sessions.groupby(["device", "sensor"], sort=False)["n_samples"].sum()
    for key, value in grouped.items():
        totals[key] = int(value)  # type: ignore[index]
    return totals


def coverage_grid(sessions: pd.DataFrame, subjects: list[int] | None = None) -> pd.DataFrame:
    subjects = _subjects_for_grid(sessions, subjects)
    index = pd.MultiIndex.from_product(
        [subjects, list(ACTIVITY_CODES), list(DEVICES), list(SENSORS)],
        names=["subject_id", "activity", "device", "sensor"],
    )
    if sessions.empty:
        n_samples = pd.Series(0, index=index, dtype="int64")
        duration_s = pd.Series(0.0, index=index, dtype="float64")
        implied_hz = pd.Series(np.nan, index=index, dtype="float64")
    else:
        grouped = sessions.groupby(["subject_id", "activity", "device", "sensor"], sort=False)
        n_samples = grouped["n_samples"].sum().reindex(index).fillna(0).astype("int64")
        duration_s = grouped["duration_s"].sum().reindex(index).fillna(0.0)
        implied_hz = grouped["implied_hz"].median().reindex(index)
    coverage = pd.DataFrame(
        {
            "n_samples": n_samples,
            "duration_s": duration_s,
            "implied_hz": implied_hz,
        }
    )
    return coverage.reset_index()


def missing_cells(coverage: pd.DataFrame) -> pd.DataFrame:
    missing = coverage.loc[
        coverage["n_samples"] <= 0, ["subject_id", "activity", "device", "sensor"]
    ]
    return missing.reset_index(drop=True)


def write_audit_tables(
    sessions: pd.DataFrame,
    audit_dir: Path,
    *,
    subjects: list[int] | None = None,
) -> None:
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(audit_dir / "sessions.csv", index=False)
    coverage = coverage_grid(sessions, subjects=subjects)
    coverage.to_csv(audit_dir / "coverage.csv", index=False)
    missing_cells(coverage).to_csv(audit_dir / "missing_cells.csv", index=False)
    hz_cols = [
        "subject_id",
        "activity",
        "device",
        "sensor",
        "n_samples",
        "duration_s",
        "implied_hz",
        "median_dt_ns",
        "source_path",
    ]
    sessions.reindex(columns=hz_cols).to_csv(audit_dir / "hz_by_session.csv", index=False)


def hz_mode(hz: float) -> str:
    if not np.isfinite(hz):
        return "unknown"
    if 17.5 <= hz < 22.5:
        return "20"
    if 22.5 <= hz < 27.5:
        return "25"
    if 45 <= hz < 55:
        return "50"
    if 90 <= hz < 110:
        return "100"
    return "other"


def write_data_card(
    sessions: pd.DataFrame,
    path: Path,
    *,
    subjects: list[int] | None = None,
) -> None:
    path = Path(path)
    coverage = coverage_grid(sessions, subjects=subjects)
    missing = missing_cells(coverage)
    totals = stream_totals(sessions)
    dump_total = sum(totals.values())
    n_subjects = len(_subjects_for_grid(sessions, subjects))

    lines = [
        "# WISDM data card",
        "",
        "Generated by `python scripts/audit.py`. Sampling rate is implied Hz: "
        "`1e9 / median_dt_ns` from timestamp diffs inside a session. "
        "Absolute Unix epoch is not used.",
        "",
        "Dataset: [WISDM Smartphone and Smartwatch Activity and Biometrics Dataset "
        "(UCI 507)](https://archive.ics.uci.edu/dataset/507/"
        "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset).",
        "",
        "## Row counts vs Weiss",
        "",
        "| Stream | This dump | Weiss | Delta |",
        "|--------|-----------|-------|-------|",
    ]
    for key, expected in WEISS_ROWS.items():
        device, sensor = key
        actual = totals[key]
        lines.append(f"| {device} {sensor} | {actual:,} | {expected:,} | {actual - expected:,} |")
    lines.append(
        f"| **total** | **{dump_total:,}** | **{WEISS_TOTAL:,}** | **{dump_total - WEISS_TOTAL:,}** |"
    )
    lines.extend(
        [
            "",
            f"Student notebook total was {STUDENT_TOTAL:,} "
            f"(delta vs Weiss {STUDENT_TOTAL - WEISS_TOTAL:,}).",
            "",
            "## Sessions vs coverage cells",
            "",
        ]
    )
    occupied = int((coverage["n_samples"] > 0).sum())
    extra_runs = int(len(sessions) - occupied) if not sessions.empty else 0
    lines.extend(
        [
            f"- Session runs after activity/gap splits: **{len(sessions):,}**",
            f"- Occupied subject x activity x stream cells: **{occupied:,}**",
            f"- Extra runs from in-activity splits (gap > 2 s or time reversal): **{extra_runs:,}**",
            "",
            "## Sample quality",
            "",
            f"- NaN values in xyz: **{_sum_int(sessions, 'n_nan'):,}**",
            f"- Non-monotonic timestamp diffs: **{_sum_int(sessions, 'n_non_monotonic'):,}**",
            "",
            "## Missing cells",
            "",
            f"Expected grid: {n_subjects} subjects x {len(ACTIVITY_CODES)} activities x "
            f"{len(DEVICES)} devices x {len(SENSORS)} sensors. "
            "A cell is missing when that subject/activity/stream has zero samples.",
            "",
            f"Missing cells: **{len(missing)}**.",
            "",
        ]
    )
    if missing.empty:
        lines.append("None.")
    else:
        lines.append("| subject_id | activity | device | sensor |")
        lines.append("|------------|----------|--------|--------|")
        shown = missing
        extra = 0
        if len(missing) > 200:
            shown = missing.head(200)
            extra = len(missing) - 200
        for subject_id, activity, device, sensor in shown.itertuples(index=False):
            lines.append(f"| {subject_id} | {activity} | {device} | {sensor} |")
        if extra:
            lines.append("")
            lines.append(f"... and {extra} more (see `data/audit/missing_cells.csv`).")

        phone_accel = missing[(missing["device"] == "phone") & (missing["sensor"] == "accel")]
        if not phone_accel.empty:
            lines.extend(["", "### Phone accel missing activities by subject", ""])
            by_subj = phone_accel.groupby("subject_id", sort=True)["activity"].agg(
                lambda s: ",".join(s)
            )
            lines.append("| subject_id | missing activities |")
            lines.append("|------------|--------------------|")
            for sid, acts in by_subj.items():
                lines.append(f"| {sid} | {acts} |")
            lines.extend(
                [
                    "",
                    "rWISDM reported phone accel gaps at 1609 B, 1616 B and F, and 1642 C and F. "
                    "The 18-class grid also lists other missing cells in the table above.",
                ]
            )

    lines.extend(
        [
            "",
            "## Sampling-rate modes",
            "",
            "Session implied Hz clustered around the observed peaks: 20, 25, 50, and 100. "
            "Do not window by row count; a 200-row official ARFF window is 10 s at 20 Hz and 4 s at 50 Hz.",
            "",
            "| Mode | n_sessions | n_samples |",
            "|------|------------|-----------|",
        ]
    )
    if sessions.empty:
        lines.append("| (no sessions) | 0 | 0 |")
    else:
        tmp = sessions.assign(hz_mode=sessions["implied_hz"].map(hz_mode))
        for mode in ("20", "25", "50", "100", "other", "unknown"):
            part = tmp[tmp["hz_mode"] == mode]
            n_samples = int(part["n_samples"].sum()) if len(part) else 0
            lines.append(f"| {mode} | {len(part):,} | {n_samples:,} |")

        finite = sessions.loc[np.isfinite(sessions["implied_hz"]), "implied_hz"]
        if len(finite):
            counts = finite.round().astype(int).value_counts().sort_index()
            lines.extend(
                [
                    "",
                    "### Rounded implied Hz (sessions)",
                    "",
                    "| Hz | n_sessions |",
                    "|----|------------|",
                ]
            )
            for hz, n in counts.items():
                lines.append(f"| {hz} | {n:,} |")

    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- No demographics (gender, handedness, height, phone model) in this dump, "
            "so no fairness slices.",
            "- Start-of-trial seconds may not match the labeled activity; trim is a later repair flag.",
            "- Accel and gyro clocks are not assumed to share sample instants; "
            "alignment is a later repair step.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Audit raw WISDM sessions. Writes CSV under data/audit/."
    )
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--audit-dir", type=Path, default=None)
    parser.add_argument("--data-card", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repo = _repo_root()
    data_cfg = _load_data_config(args.config or repo / "configs" / "audit.yaml")

    raw_root = args.raw_root or Path(
        data_cfg.get("raw_root") or repo / "data" / "external" / "wisdm-dataset"
    )
    if not raw_root.is_absolute():
        raw_root = repo / raw_root
    audit_dir = args.audit_dir or Path(data_cfg.get("audit_dir") or repo / "data" / "audit")
    if not audit_dir.is_absolute():
        audit_dir = repo / audit_dir
    data_card = args.data_card or Path(data_cfg.get("data_card") or repo / "docs" / "data_card.md")
    if not data_card.is_absolute():
        data_card = repo / data_card

    resolved = resolve_raw_root(raw_root)
    if resolved is not None:
        raw_root = resolved
    log.info("auditing %s", raw_root)
    sessions = audit_dataset(raw_root)
    write_audit_tables(sessions, audit_dir)
    write_data_card(sessions, data_card)
    log.info("wrote %s (%s sessions)", audit_dir, len(sessions))
    log.info("wrote %s", data_card)
    print(data_card)
    return data_card


def _audit_to_row(audit: SessionAudit) -> dict[str, object]:
    key = audit.key
    return {
        "subject_id": key.subject_id,
        "activity": key.activity,
        "device": key.device,
        "sensor": key.sensor,
        "n_samples": audit.n_samples,
        "duration_s": audit.duration_s,
        "median_dt_ns": audit.median_dt_ns,
        "p05_dt_ns": audit.p05_dt_ns,
        "p95_dt_ns": audit.p95_dt_ns,
        "implied_hz": audit.implied_hz,
        "n_non_monotonic": audit.n_non_monotonic,
        "n_nan": audit.n_nan,
        "mean_x": audit.mean_x,
        "mean_y": audit.mean_y,
        "mean_z": audit.mean_z,
        "source_path": audit.source_path,
    }


def _warn_weiss_mismatch(sessions: pd.DataFrame) -> None:
    totals = stream_totals(sessions)
    dump_total = sum(totals.values())
    for key, expected in WEISS_ROWS.items():
        actual = totals[key]
        if actual != expected:
            device, sensor = key
            log.warning(
                "%s %s rows %s vs Weiss %s (delta %s)",
                device,
                sensor,
                actual,
                expected,
                actual - expected,
            )
    if dump_total != WEISS_TOTAL:
        log.warning(
            "dump total %s vs Weiss %s (delta %s); student notebook had %s",
            dump_total,
            WEISS_TOTAL,
            dump_total - WEISS_TOTAL,
            STUDENT_TOTAL,
        )


def _sum_int(sessions: pd.DataFrame, column: str) -> int:
    if sessions.empty or column not in sessions.columns:
        return 0
    return int(sessions[column].sum())


def _subjects_for_grid(sessions: pd.DataFrame, subjects: list[int] | None) -> list[int]:
    if subjects is not None:
        return list(subjects)
    expected = set(range(SUBJECT_ID_MIN, SUBJECT_ID_MAX + 1))
    if not sessions.empty:
        expected |= set(sessions["subject_id"].astype(int))
    return sorted(expected)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_data_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = loaded.get("data") if isinstance(loaded, dict) else None
    return data if isinstance(data, dict) else {}


if __name__ == "__main__":
    main()
