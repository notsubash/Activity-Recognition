"""Macro-F1, per-class F1, and per-group F1. Accuracy is secondary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict, cast

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from har.constants import GROUP_LABELS, GROUP_OF, LABEL_ORDER
Label = int | str


class MetricsDict(TypedDict):
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    per_class_f1: dict[Label, float]
    per_group_f1: dict[str, float]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[Label]) -> MetricsDict:
    """Score predictions against ``labels``.

    ``y_true``, ``y_pred``, and ``labels`` must be ``LABEL_ORDER`` indices
    (``WindowRecord.y``) or activity codes (A-S, no N). Do not remap a class
    subset to ``0..K-1`` or group F1 will attach those ints to walking/jogging.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    label_list = list(labels)
    per_class = _f1(y_true_arr, y_pred_arr, label_list, average=None)
    y_true_g = np.array([_group_of(v) for v in y_true_arr])
    y_pred_g = np.array([_group_of(v) for v in y_pred_arr])
    per_group = _f1(y_true_g, y_pred_g, list(GROUP_LABELS), average=None)
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(_f1(y_true_arr, y_pred_arr, label_list, average="macro")),
        "per_class_f1": {
            _label_key(label): float(score)
            for label, score in zip(label_list, per_class, strict=True)
        },
        "per_group_f1": {
            name: float(score) for name, score in zip(GROUP_LABELS, per_group, strict=True)
        },
    }


def _f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[Label],
    *,
    average: str | None,
) -> np.ndarray:
    # sklearn-stubs type average as str and zero_division as str; None and 0 are valid.
    result = cast(Any, f1_score)(
        y_true, y_pred, labels=list(labels), average=average, zero_division=0
    )
    return np.asarray(result, dtype=np.float64)


def _label_key(label: object) -> Label:
    if isinstance(label, str):
        return label
    return int(cast(Any, label))


def _group_of(label: object) -> str:
    if isinstance(label, str):
        code = label
    else:
        idx = int(cast(Any, label))
        try:
            code = LABEL_ORDER[idx]
        except IndexError as exc:
            raise ValueError(f"label {idx} is not a LABEL_ORDER index") from exc
    try:
        return GROUP_OF[code]
    except KeyError as exc:
        raise ValueError(f"unknown activity {code!r}") from exc
