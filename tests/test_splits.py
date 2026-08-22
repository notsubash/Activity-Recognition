import numpy as np
import pytest

from har.data.windows import stack_windows
from har.eval.splits import (
    assert_no_subject_overlap,
    group_kfold,
    leaky_split,
    loso,
)
from har.types import WindowRecord


def _windows(subject_ids: list[int]):
    records = []
    for i, sid in enumerate(subject_ids):
        x = np.zeros((8, 6), dtype=np.float32)
        x[0, 0] = float(i)
        records.append(
            WindowRecord(
                subject_id=sid,
                activity="A",
                device="phone",
                start_ns=i,
                end_ns=i + 1,
                x=x,
                y=i % 2,
            )
        )
    return stack_windows(records)


def _assert_split_rows_match_groups(split, X, y, groups) -> None:
    train_i = split.X_train[:, 0, 0].astype(int)
    test_i = split.X_test[:, 0, 0].astype(int)
    np.testing.assert_array_equal(split.X_train, X[train_i])
    np.testing.assert_array_equal(split.X_test, X[test_i])
    np.testing.assert_array_equal(split.y_train, y[train_i])
    np.testing.assert_array_equal(split.y_test, y[test_i])
    np.testing.assert_array_equal(split.groups_train, groups[train_i])
    np.testing.assert_array_equal(split.groups_test, groups[test_i])


def test_leaky_split_may_overlap_subjects_group_split_must_not():
    X_leak, y_leak, g_leak = _windows([1, 1, 2, 2])
    np.testing.assert_array_equal(g_leak, [1, 1, 2, 2])

    # 3/1 split always puts the held-out subject's other window in test.
    leaky = leaky_split(X_leak, y_leak, test_size=0.75, seed=0, groups=g_leak)
    leaky_overlap = set(leaky.groups_train) & set(leaky.groups_test)
    assert leaky_overlap, "Protocol A must be allowed to leak subjects"
    _assert_split_rows_match_groups(leaky, X_leak, y_leak, g_leak)

    # Interleaved subjects: a row-wise KFold would mix subjects in both sides.
    X, y, groups = _windows([1, 2, 1, 2])
    np.testing.assert_array_equal(groups, [1, 2, 1, 2])
    for split in group_kfold(X, y, groups, n_splits=2, seed=0):
        assert_no_subject_overlap(split.groups_train, split.groups_test)
        assert set(split.groups_train).isdisjoint(split.groups_test)
        _assert_split_rows_match_groups(split, X, y, groups)


def test_assert_no_subject_overlap_raises_on_shared_subject():
    with pytest.raises(ValueError, match="overlap"):
        assert_no_subject_overlap(np.array([1, 1, 2]), np.array([2, 3]))

    assert_no_subject_overlap(np.array([1, 1]), np.array([2, 3]))


def test_loso_holds_out_one_subject_per_fold():
    X, y, groups = _windows([1, 2, 1, 2])
    folds = list(loso(X, y, groups))
    assert len(folds) == 2
    held_out = {int(np.unique(split.groups_test)[0]) for split in folds}
    assert held_out == {1, 2}
    for split in folds:
        assert_no_subject_overlap(split.groups_train, split.groups_test)
        assert np.unique(split.groups_test).shape == (1,)
        _assert_split_rows_match_groups(split, X, y, groups)
