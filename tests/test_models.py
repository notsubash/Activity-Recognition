import numpy as np

from har.models.baselines import fit_dummy
from har.models.xgboost import STUDENT_XGB_PARAMS, fit_xgboost


def _fixture_windows(n: int = 24, n_features: int = 8):
    """Tiny separable 2-class matrix. Not WISDM."""
    y = np.array([0, 1] * (n // 2), dtype=np.int64)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    X[y == 1] += 4.0
    return X, y


def test_dummy_and_xgboost_run_on_fixture_windows():
    X, y = _fixture_windows()
    X_train, y_train = X[:16], y[:16]
    X_val, y_val = X[16:], y[16:]

    dummy = fit_dummy(X_train, y_train)
    model = fit_xgboost(
        X_train,
        y_train,
        X_val,
        y_val,
        {"n_estimators": 2, "max_depth": 2, "n_jobs": 1},
    )

    y_dummy = np.asarray(dummy.predict(X_val), dtype=np.int64)
    y_xgb = np.asarray(model.predict(X_val), dtype=np.int64)
    assert y_dummy.shape == y_val.shape
    assert y_xgb.shape == y_val.shape
    assert set(np.unique(y_dummy)).issubset({0, 1})
    assert set(np.unique(y_xgb)).issubset({0, 1})
    assert model.get_params()["n_estimators"] == 2
    assert model.get_params()["subsample"] != STUDENT_XGB_PARAMS["subsample"]
