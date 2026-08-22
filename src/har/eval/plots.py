"""Confusion matrices and README / MLflow figures from metrics JSON."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from har.constants import CODE_TO_NAME, LABEL_ORDER

matplotlib.use("Agg", force=True)

log = logging.getLogger(__name__)

PHONE = "#0072B2"
WATCH = "#D55E00"
CONCAT = "#009E73"
MUTED = "#7A7A7A"
INK = "#222222"

HZ_SESSION_COUNTS = ((20, 2838), (25, 543), (50, 322), (100, 14))

LADDER_ROWS = (
    ("protocol_b_phone_stat_dummy.json", "Dummy (phone)"),
    ("protocol_b_phone_stat_logreg.json", "LogReg (phone)"),
    ("protocol_b_phone_stat_rf.json", "Random forest (phone)"),
    ("protocol_b_phone_stat_xgb.json", "XGBoost (phone)"),
    ("protocol_b_concat_stat_xgb.json", "XGBoost (concat)"),
    ("protocol_b_watch_stat_xgb.json", "XGBoost (watch)"),
)

ABLATION_ROWS = (
    ("protocol_b_phone_stat_xgb.json", "Control 5 s XYZ"),
    ("ablations/window_2s.json", "Window 2 s"),
    ("ablations/window_10s.json", "Window 10 s"),
    ("ablations/trim_15s.json", "Trim 15 s"),
    ("ablations/reorient_on.json", "Reorient on"),
    ("ablations/magnitude.json", "Magnitude only"),
    ("ablations/hierarchical.json", "Hierarchical"),
)


def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int | str],
) -> np.ndarray:
    return np.asarray(sklearn_confusion_matrix(y_true, y_pred, labels=list(labels)))


def class_f1_by_name(per_class_f1: Mapping[str, float]) -> dict[str, float]:
    """Map metrics JSON keys (``"0"``..``"17"`` or activity codes) to class names."""
    out: dict[str, float] = {}
    for key, value in per_class_f1.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            code = str(key)
            out[str(CODE_TO_NAME.get(code, code))] = float(value)
            continue
        if 0 <= idx < len(LABEL_ORDER):
            out[CODE_TO_NAME[LABEL_ORDER[idx]]] = float(value)
        else:
            out[str(key)] = float(value)
    return out


def save_confusion_matrix(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Sequence[int | str],
    class_names: list[str] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    counts = confusion_counts(y_true, y_pred, labels)
    names = class_names if class_names is not None else [str(x) for x in labels]
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    image = ax.imshow(counts, interpolation="nearest", cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(names)), labels=names, rotation=55, ha="right")
    ax.set_yticks(range(len(names)), labels=names)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return _savefig(fig, path)


def save_run_summary_figure(path: Path, payload: Mapping[str, Any]) -> Path:
    """Per-class F1 bars for one run (MLflow artifact)."""
    import matplotlib.pyplot as plt

    rows = _ordered_class_f1(payload)
    names = [name for name, _ in rows]
    values = [value for _, value in rows]
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    y = np.arange(len(names))
    ax.barh(y, values, color=PHONE, height=0.72)
    ax.set_yticks(y, labels=names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title(_run_title(payload))
    ax.axvline(
        float(payload["macro_f1"]), color=WATCH, linewidth=1.2, linestyle="--", label="macro-F1"
    )
    ax.legend(frameon=False, loc="lower right")
    return _savefig(fig, path)


def write_readme_figures(reports_dir: Path, out_dir: Path) -> list[Path]:
    """Build the README figure set from metrics JSON. Skip charts whose files are missing."""
    reports_dir = Path(reports_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    a2 = _load(reports_dir / "protocol_a2_phone_raw_flat_xgb.json")
    b_flat = _load(reports_dir / "protocol_b_phone_raw_flat_xgb.json")
    if a2 and b_flat:
        written.append(
            _labeled_bars(
                out_dir / "leakage_macro_f1.png",
                title="Same phone windows: leaky split vs GroupKFold",
                xlabel="macro-F1",
                labels=(
                    "A2 leaky\n(flattened 5 s)",
                    "B GroupKFold\n(flattened 5 s)",
                ),
                values=(float(a2["macro_f1"]), float(b_flat["macro_f1"])),
                colors=(MUTED, PHONE),
            )
        )

    ladder = [(label, _load(reports_dir / name)) for name, label in LADDER_ROWS]
    if all(row[1] for row in ladder):
        written.append(
            _labeled_bars(
                out_dir / "protocol_b_ladder.png",
                title="Protocol B GroupKFold, statistical features unless noted",
                xlabel="macro-F1",
                labels=tuple(label for label, _ in ladder),
                values=tuple(float(row["macro_f1"]) for _, row in ladder if row),
                colors=(MUTED, MUTED, MUTED, PHONE, CONCAT, WATCH),
                invert=False,
            )
        )

    phone = _load(reports_dir / "protocol_b_phone_stat_xgb.json")
    watch = _load(reports_dir / "protocol_b_watch_stat_xgb.json")
    if phone and watch:
        written.append(_grouped_class_f1(out_dir / "per_class_f1_phone_watch.png", phone, watch))
        written.append(_grouped_group_f1(out_dir / "per_group_f1_phone_watch.png", phone, watch))

    ablation_loaded = [(label, _load(reports_dir / name)) for name, label in ABLATION_ROWS]
    if all(row[1] for row in ablation_loaded):
        written.append(
            _labeled_bars(
                out_dir / "ablations_macro_f1.png",
                title="Phone statistical XGBoost ablations (Protocol B)",
                xlabel="macro-F1",
                labels=tuple(label for label, _ in ablation_loaded),
                values=tuple(float(row["macro_f1"]) for _, row in ablation_loaded if row),
                colors=(PHONE,) + (MUTED,) * (len(ablation_loaded) - 1),
            )
        )

    written.append(_hz_modes(out_dir / "sampling_rate_modes.png"))
    return written


def sync_mlflow_run_names(reports_dir: Path) -> list[str]:
    """Set MLflow run names to the metrics JSON stem. Skip missing tracking stores."""
    import mlflow

    updated: list[str] = []
    for path in _metric_json_paths(Path(reports_dir)):
        payload = _load(path)
        if not payload:
            continue
        run_id = payload.get("mlflow_run_id")
        uri = payload.get("mlflow_tracking_uri")
        if not run_id or not uri:
            continue
        try:
            client = mlflow.MlflowClient(tracking_uri=str(uri))
            name = path.stem
            if path.parent.name == "ablations":
                name = f"ablation_{name}"
            client.set_tag(str(run_id), "mlflow.runName", name)
            _backfill_run_metrics(client, str(run_id), payload)
        except Exception as exc:  # noqa: BLE001 - tracking store may be absent
            log.warning("skip mlflow rename %s: %s", path.name, exc)
            continue
        updated.append(name)
    return updated


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description="Write README figures from metrics JSON.")
    parser.add_argument("--from-reports", type=Path, default=Path("docs/reports"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--sync-mlflow",
        action="store_true",
        help="Rename existing MLflow runs to the report filename stem.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    written = write_readme_figures(args.from_reports, args.out)
    for path in written:
        log.info("wrote %s", path)
        print(path)
    if args.sync_mlflow:
        for name in sync_mlflow_run_names(args.from_reports):
            log.info("named mlflow run %s", name)
    return written


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _metric_json_paths(reports_dir: Path) -> list[Path]:
    paths = [p for p in sorted(reports_dir.glob("*.json")) if p.name != "ladder_summary.json"]
    paths.extend(sorted((reports_dir / "ablations").glob("*.json")))
    return paths


def _backfill_run_metrics(client: Any, run_id: str, payload: Mapping[str, Any]) -> None:
    existing = client.get_run(run_id).data.metrics
    if "fold_macro_f1" not in existing:
        for i, fold in enumerate(payload.get("folds") or [], start=1):
            client.log_metric(run_id, "fold_macro_f1", float(fold["macro_f1"]), step=i)
            if "accuracy" in fold:
                client.log_metric(run_id, "fold_accuracy", float(fold["accuracy"]), step=i)
    groups = payload.get("per_group_f1") or {}
    for name, value in groups.items():
        key = f"group_f1_{name}"
        if key not in existing:
            client.log_metric(run_id, key, float(value))


def _ordered_class_f1(payload: Mapping[str, Any]) -> list[tuple[str, float]]:
    raw = payload.get("per_class_f1") or {}
    rows: list[tuple[str, float]] = []
    for i, code in enumerate(LABEL_ORDER):
        if str(i) in raw:
            value = raw[str(i)]
        elif i in raw:
            value = raw[i]
        elif code in raw:
            value = raw[code]
        else:
            value = 0.0
        rows.append((CODE_TO_NAME[code], float(value)))
    return rows


def _run_title(payload: Mapping[str, Any]) -> str:
    protocol = payload.get("protocol_name") or payload.get("protocol") or "run"
    device = payload.get("device") or "?"
    model = payload.get("model") or "?"
    return f"Per-class F1 · {protocol} · {device} {model}"


def _style(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.set_axisbelow(True)


def _savefig(fig: Any, path: Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200, facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out


def _labeled_bars(
    path: Path,
    *,
    title: str,
    xlabel: str,
    labels: Sequence[str],
    values: Sequence[float],
    colors: Sequence[str],
    invert: bool = True,
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(labels) + 1.6))
    y = np.arange(len(labels))
    ax.barh(y, values, color=list(colors), height=0.66)
    ax.set_yticks(y, labels=list(labels))
    if invert:
        ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for yi, value in zip(y, values, strict=True):
        ax.text(value + 0.02, float(yi), f"{value:.3f}", va="center", color=INK, fontsize=9)
    _style(ax)
    return _savefig(fig, path)


def _grouped_class_f1(path: Path, phone: Mapping[str, Any], watch: Mapping[str, Any]) -> Path:
    import matplotlib.pyplot as plt

    phone_rows = _ordered_class_f1(phone)
    watch_map = dict(_ordered_class_f1(watch))
    names = [name for name, _ in phone_rows]
    phone_vals = [value for _, value in phone_rows]
    watch_vals = [watch_map[name] for name in names]
    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    y = np.arange(len(names))
    ax.barh(y + 0.18, phone_vals, 0.36, label="Phone", color=PHONE)
    ax.barh(y - 0.18, watch_vals, 0.36, label="Watch", color=WATCH)
    ax.set_yticks(y, labels=names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title("Per-class F1, Protocol B statistical XGBoost")
    ax.legend(frameon=False, loc="lower right")
    _style(ax)
    return _savefig(fig, path)


def _grouped_group_f1(path: Path, phone: Mapping[str, Any], watch: Mapping[str, Any]) -> Path:
    import matplotlib.pyplot as plt

    order = ("locomotion", "hand", "eating", "posture")
    phone_g = {str(k): float(v) for k, v in (phone.get("per_group_f1") or {}).items()}
    watch_g = {str(k): float(v) for k, v in (watch.get("per_group_f1") or {}).items()}
    labels = [name.capitalize() for name in order]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(order))
    ax.bar(x - 0.18, [phone_g[k] for k in order], 0.36, label="Phone", color=PHONE)
    ax.bar(x + 0.18, [watch_g[k] for k in order], 0.36, label="Watch", color=WATCH)
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("Activity-group F1, Protocol B statistical XGBoost")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.set_axisbelow(True)
    return _savefig(fig, path)


def _hz_modes(path: Path) -> Path:
    import matplotlib.pyplot as plt

    labels = [f"{hz} Hz" for hz, _ in HZ_SESSION_COUNTS]
    values = [count for _, count in HZ_SESSION_COUNTS]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(labels))
    ax.bar(x, values, color=PHONE)
    ax.set_xticks(x, labels=labels)
    ax.set_ylabel("Sessions")
    ax.set_title("Implied sampling-rate modes in raw WISDM")
    for xi, value in zip(x, values, strict=True):
        ax.text(float(xi), value + 40, f"{value:,}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(values) * 1.15)
    return _savefig(fig, path)


if __name__ == "__main__":
    main()
