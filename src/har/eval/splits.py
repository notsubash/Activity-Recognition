"""Protocol A/B/C splits. Random window splits may leak subjects; grouped splits must not."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, train_test_split


@dataclass(frozen=True)
class Split:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    groups_train: np.ndarray
    groups_test: np.ndarray


def assert_no_subject_overlap(train_groups: np.ndarray, test_groups: np.ndarray) -> None:
    overlap = set(np.unique(train_groups).tolist()) & set(np.unique(test_groups).tolist())
    if overlap:
        raise ValueError(f"subject overlap between train and test: {sorted(overlap)}")


def leaky_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int | None,
    groups: np.ndarray,
) -> Split:
    """Protocol A: `train_test_split` on windows. Subjects may appear in both sides."""
    X, y, groups = _align(X, y, groups)
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=test_size, random_state=seed
    )
    return Split(
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train),
        np.asarray(y_test),
        np.asarray(groups_train),
        np.asarray(groups_test),
    )


def group_kfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int | None,
) -> Iterator[Split]:
    """Protocol B: GroupKFold on subject_id. Each fold is checked for subject overlap."""
    X, y, groups = _align(X, y, groups)
    cv = GroupKFold(n_splits=n_splits, shuffle=seed is not None, random_state=seed)
    for train_idx, test_idx in cv.split(X, y, groups):
        split = _take(X, y, groups, train_idx, test_idx)
        assert_no_subject_overlap(split.groups_train, split.groups_test)
        yield split


def loso(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Iterator[Split]:
    """Protocol C: leave one subject out."""
    X, y, groups = _align(X, y, groups)
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        split = _take(X, y, groups, train_idx, test_idx)
        assert_no_subject_overlap(split.groups_train, split.groups_test)
        yield split


def _align(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)
    n = y_arr.shape[0]
    if X_arr.shape[0] != n or groups_arr.shape[0] != n:
        raise ValueError("X, y, and groups must share the first dimension")
    return X_arr, y_arr, groups_arr


def _take(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Split:
    return Split(
        X_train=X[train_idx],
        X_test=X[test_idx],
        y_train=y[train_idx],
        y_test=y[test_idx],
        groups_train=groups[train_idx],
        groups_test=groups[test_idx],
    )
