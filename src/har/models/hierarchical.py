"""Two-stage HAR: group classifier, then a per-group expert.

Experts are fit on the true group during training. Inference routes by the
predicted group.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from har.constants import GROUP_LABELS, GROUP_OF, LABEL_ORDER
from har.models.xgboost import fit_xgboost


@dataclass
class HierarchicalModel:
    group_model: Any
    experts: dict[str, Any]

    def predict_groups(self, X: np.ndarray) -> np.ndarray:
        idx = np.asarray(self.group_model.predict(np.asarray(X)), dtype=np.int64)
        return np.asarray([GROUP_LABELS[int(i)] for i in idx])

    def predict(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X)
        group_pred = self.predict_groups(arr)
        out = np.empty(arr.shape[0], dtype=np.int64)
        for name, expert in self.experts.items():
            mask = group_pred == name
            if not np.any(mask):
                continue
            out[mask] = np.asarray(expert.predict(arr[mask]), dtype=np.int64)
        missing = ~np.isin(group_pred, list(self.experts))
        if np.any(missing):
            raise RuntimeError("predicted group has no expert")
        return out


class _ConstantPredictor:
    def __init__(self, label: int) -> None:
        self.classes_ = np.asarray([int(label)], dtype=np.int64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(np.asarray(X)), int(self.classes_[0]), dtype=np.int64)


@dataclass
class _MappedPredictor:
    """XGBoost wrapper: local 0..K-1 labels in, original ``classes_`` out."""

    model: Any
    classes_: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        local = np.asarray(self.model.predict(np.asarray(X)), dtype=np.int64)
        return np.asarray(self.classes_, dtype=np.int64)[local]


def fit_hierarchical(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    params: Mapping[str, Any] | None,
) -> HierarchicalModel:
    x_tr = np.asarray(X_train)
    y_tr = np.asarray(y_train)
    if x_tr.shape[0] == 0 or y_tr.shape[0] == 0:
        raise ValueError("train set is empty")
    y_group = _group_index(y_tr)
    x_val, y_val_g, y_val_arr = _val_arrays(X_val, y_val)
    x_val_g, y_val_g = _filter_val(x_val, y_val_g, y_group)
    if int(np.unique(y_group).size) < 2:
        group_model = _ConstantPredictor(int(y_group[0]))
    else:
        group_model = _fit_mapped(x_tr, y_group, x_val_g, y_val_g, params)
    experts: dict[str, Any] = {}
    for name in GROUP_LABELS:
        mask = y_group == GROUP_LABELS.index(name)
        if not np.any(mask):
            experts[name] = _ConstantPredictor(_default_label(name))
            continue
        x_g = x_tr[mask]
        y_g = y_tr[mask]
        x_g_val, y_g_val = _filter_val(*_group_val(x_val, y_val_arr, name), y_g)
        if np.unique(y_g).size < 2:
            experts[name] = _ConstantPredictor(int(y_g[0]))
            continue
        experts[name] = _fit_mapped(x_g, y_g, x_g_val, y_g_val, params)
    return HierarchicalModel(group_model=group_model, experts=experts)


def _fit_mapped(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    params: Mapping[str, Any] | None,
) -> _MappedPredictor:
    classes, y_local = np.unique(y, return_inverse=True)
    x_v, y_v = _local_val(X_val, y_val, classes)
    model = fit_xgboost(X, y_local.astype(np.int64), x_v, y_v, params)
    return _MappedPredictor(model=model, classes_=np.asarray(classes, dtype=np.int64))


def _local_val(
    X_val: np.ndarray | None, y_val: np.ndarray | None, classes: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if X_val is None or y_val is None:
        return None, None
    lookup = {int(c): i for i, c in enumerate(classes)}
    local = np.array([lookup.get(int(v), -1) for v in y_val], dtype=np.int64)
    mask = local >= 0
    if not np.any(mask):
        return None, None
    return X_val[mask], local[mask]


def _group_index(y: np.ndarray) -> np.ndarray:
    return np.asarray([GROUP_LABELS.index(GROUP_OF[LABEL_ORDER[int(i)]]) for i in y], dtype=np.int64)


def _default_label(group: str) -> int:
    for i, code in enumerate(LABEL_ORDER):
        if GROUP_OF[code] == group:
            return i
    raise ValueError(f"unknown group {group!r}")


def _val_arrays(
    X_val: np.ndarray | None, y_val: np.ndarray | None
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if X_val is None or y_val is None or len(np.asarray(X_val)) == 0:
        return None, None, None
    y_arr = np.asarray(y_val)
    return np.asarray(X_val), _group_index(y_arr), y_arr


def _filter_val(
    X_val: np.ndarray | None, y_val: np.ndarray | None, y_train: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if X_val is None or y_val is None:
        return None, None
    mask = np.isin(y_val, np.unique(y_train))
    if not np.any(mask):
        return None, None
    return X_val[mask], y_val[mask]


def _group_val(
    X_val: np.ndarray | None, y_val: np.ndarray | None, group: str
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if X_val is None or y_val is None:
        return None, None
    mask = _group_index(y_val) == GROUP_LABELS.index(group)
    if not np.any(mask):
        return None, None
    return X_val[mask], y_val[mask]
