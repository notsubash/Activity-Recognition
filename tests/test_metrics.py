from typing import Any, cast

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from har.constants import GROUP_OF, LABEL_ORDER
from har.eval.metrics import GROUP_LABELS, compute_metrics
from har.eval.plots import confusion_counts


def _sklearn_f1(y_true, y_pred, labels, *, average: str | None) -> np.ndarray:
    result = cast(Any, f1_score)(y_true, y_pred, labels=labels, average=average, zero_division=0)
    return np.asarray(result, dtype=np.float64)


def test_compute_metrics_matches_sklearn_on_toy_3_class():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    labels = [0, 1, 2]
    got = compute_metrics(y_true, y_pred, labels)
    assert got["accuracy"] == accuracy_score(y_true, y_pred)
    assert got["balanced_accuracy"] == balanced_accuracy_score(y_true, y_pred)
    assert got["macro_f1"] == float(_sklearn_f1(y_true, y_pred, labels, average="macro"))
    per_class = _sklearn_f1(y_true, y_pred, labels, average=None)
    assert got["per_class_f1"] == {
        label: float(score) for label, score in zip(labels, per_class, strict=True)
    }
    assert "per_group_f1" in got


def test_per_group_f1_collapses_wisdm_classes():
    # A locomotion, D posture, H eating
    a, d, h = LABEL_ORDER.index("A"), LABEL_ORDER.index("D"), LABEL_ORDER.index("H")
    y_true = np.array([a, a, d, d, h, h])
    y_pred = np.array([a, d, d, d, h, a])
    labels = [a, d, h]
    got = compute_metrics(y_true, y_pred, labels)

    y_true_g = np.array([GROUP_OF[LABEL_ORDER[i]] for i in y_true])
    y_pred_g = np.array([GROUP_OF[LABEL_ORDER[i]] for i in y_pred])
    expected = _sklearn_f1(y_true_g, y_pred_g, list(GROUP_LABELS), average=None)
    assert got["per_group_f1"] == {
        name: float(score) for name, score in zip(GROUP_LABELS, expected, strict=True)
    }


def test_group_labels_match_group_of():
    assert set(GROUP_LABELS) == set(GROUP_OF.values())


def test_compute_metrics_rejects_out_of_range_label():
    with pytest.raises(ValueError, match="LABEL_ORDER"):
        compute_metrics(np.array([99]), np.array([0]), [0, 99])


def test_confusion_counts_matches_sklearn():
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 2])
    labels = [0, 1, 2]
    np.testing.assert_array_equal(
        confusion_counts(y_true, y_pred, labels),
        confusion_matrix(y_true, y_pred, labels=labels),
    )
