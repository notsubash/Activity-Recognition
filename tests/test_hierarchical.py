import numpy as np
import pytest

from har.constants import GROUP_LABELS, GROUP_OF, LABEL_ORDER
from har.models.hierarchical import fit_hierarchical


def _four_group_xy(n_per: int = 24, n_features: int = 8):
    """One class per Weiss group, separable on feature 0. Not WISDM."""
    codes = [LABEL_ORDER.index(c) for c in ("A", "D", "F", "H")]
    y = np.repeat(np.asarray(codes, dtype=np.int64), n_per)
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.05, size=(y.size, n_features)).astype(np.float32)
    for i, code in enumerate(codes):
        x[y == code, 0] += float(i) * 8.0
    return x, y


def _tiny_params() -> dict:
    return {
        "n_estimators": 8,
        "max_depth": 2,
        "n_jobs": 1,
        "device": "cpu",
    }


def _split_by_class(x: np.ndarray, y: np.ndarray, n_train: int, n_val: int):
    x_tr, y_tr, x_va, y_va, x_te, y_te = [], [], [], [], [], []
    for code in np.unique(y):
        xi, yi = x[y == code], y[y == code]
        x_tr.append(xi[:n_train])
        y_tr.append(yi[:n_train])
        x_va.append(xi[n_train : n_train + n_val])
        y_va.append(yi[n_train : n_train + n_val])
        x_te.append(xi[n_train + n_val :])
        y_te.append(yi[n_train + n_val :])
    return (
        np.concatenate(x_tr),
        np.concatenate(y_tr),
        np.concatenate(x_va),
        np.concatenate(y_va),
        np.concatenate(x_te),
        np.concatenate(y_te),
    )


def test_hierarchical_predict_shape_matches_n_samples() -> None:
    x_tr, y_tr, x_va, y_va, x_te, y_te = _split_by_class(*_four_group_xy(), 16, 4)
    model = fit_hierarchical(x_tr, y_tr, x_va, y_va, _tiny_params())
    pred = np.asarray(model.predict(x_te))
    assert pred.shape == (y_te.shape[0],)


def test_hierarchical_group_and_expert_routing_shapes() -> None:
    x_tr, y_tr, x_va, y_va, x_te, y_te = _split_by_class(*_four_group_xy(), 16, 4)
    model = fit_hierarchical(x_tr, y_tr, x_va, y_va, _tiny_params())

    group_pred = np.asarray(model.predict_groups(x_te))
    assert group_pred.shape == (x_te.shape[0],)
    assert set(group_pred) == set(GROUP_LABELS)

    routed = np.asarray(model.predict(x_te))
    assert routed.shape == (y_te.shape[0],)
    n_routed = 0
    for name, expert in model.experts.items():
        mask = group_pred == name
        n_mask = int(mask.sum())
        assert n_mask > 0
        expert_out = np.asarray(expert.predict(x_te[mask]))
        assert expert_out.shape == (n_mask,)
        np.testing.assert_array_equal(routed[mask], expert_out)
        n_routed += n_mask
    assert n_routed == x_te.shape[0]


def test_hierarchical_experts_trained_on_true_group_labels() -> None:
    x, y = _four_group_xy()
    model = fit_hierarchical(x, y, None, None, _tiny_params())
    assert set(model.experts) == set(GROUP_LABELS)
    for name, expert in model.experts.items():
        classes = np.asarray(expert.classes_)
        assert classes.size >= 1
        for label in classes:
            assert GROUP_OF[LABEL_ORDER[int(label)]] == name


def test_experts_keep_true_group_when_group_head_would_confuse() -> None:
    """One A window looks like H. True-group experts must still own A, not eating."""
    x, y = _four_group_xy()
    x = x.copy()
    a_i = int(np.flatnonzero(y == LABEL_ORDER.index("A"))[0])
    h_i = int(np.flatnonzero(y == LABEL_ORDER.index("H"))[0])
    x[a_i] = x[h_i]
    model = fit_hierarchical(x, y, None, None, _tiny_params())
    eating = {int(c) for c in np.asarray(model.experts["eating"].classes_)}
    loco = {int(c) for c in np.asarray(model.experts["locomotion"].classes_)}
    assert LABEL_ORDER.index("A") in loco
    assert LABEL_ORDER.index("A") not in eating
    assert LABEL_ORDER.index("H") in eating


def test_locomotion_expert_returns_global_m_label() -> None:
    """A=0 and M=12 are one group but not 0..K-1; predict must emit 12, not 1."""
    n_per, n_features = 20, 6
    codes = [LABEL_ORDER.index(c) for c in ("A", "M", "D", "F", "H")]
    y = np.repeat(np.asarray(codes, dtype=np.int64), n_per)
    rng = np.random.default_rng(1)
    x = rng.normal(scale=0.05, size=(y.size, n_features)).astype(np.float32)
    for i, code in enumerate(codes):
        x[y == code, 0] += float(i) * 6.0
    x_tr, y_tr, x_va, y_va, x_te, y_te = _split_by_class(x, y, 12, 4)
    model = fit_hierarchical(x_tr, y_tr, x_va, y_va, _tiny_params())
    pred = np.asarray(model.predict(x_te))
    assert pred.shape == y_te.shape
    assert LABEL_ORDER.index("M") in np.unique(pred)
    m_code = LABEL_ORDER.index("M")
    loco = np.asarray(model.experts["locomotion"].predict(x_te[y_te == m_code][:1]))
    assert loco.shape == (1,)
    assert int(loco[0]) in (LABEL_ORDER.index("A"), m_code)


def test_single_group_train_still_predicts() -> None:
    n, n_features = 20, 4
    y = np.full(n, LABEL_ORDER.index("A"), dtype=np.int64)
    x = np.random.default_rng(0).normal(size=(n, n_features)).astype(np.float32)
    model = fit_hierarchical(x[:12], y[:12], None, None, _tiny_params())
    pred = np.asarray(model.predict(x[12:]))
    assert pred.shape == (8,)
    assert set(pred.tolist()) == {LABEL_ORDER.index("A")}


def test_hierarchical_rejects_empty_train() -> None:
    x = np.zeros((0, 4), dtype=np.float32)
    y = np.zeros((0,), dtype=np.int64)
    with pytest.raises(ValueError, match="train"):
        fit_hierarchical(x, y, None, None, _tiny_params())
