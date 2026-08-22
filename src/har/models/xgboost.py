"""XGBoost fit wrapper. Protocol A student hyperparameters live in YAML."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

# From docs/reports/evaluation.txt. Protocol A reproduction only.
STUDENT_XGB_PARAMS: dict[str, Any] = {
    "colsample_bytree": 0.9396893641976711,
    "gamma": 0,
    "learning_rate": 0.10241823755571676,
    "max_depth": 6,
    "n_estimators": 982,
    "subsample": 0.8545330472743582,
}


def fit_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    params: Mapping[str, Any] | None,
) -> XGBClassifier:
    """Fit an sklearn-style XGBoost classifier. Returns a predict()-able model.

    ``params`` is the source of truth. Protocol A student values live in
    ``STUDENT_XGB_PARAMS`` and the A1/A2 YAML files, not in this default merge.
    """
    cfg = dict(params or {})
    early_stopping_rounds = cfg.pop("early_stopping_rounds", None)
    random_state = cfg.pop("random_state", 42)
    n_classes = int(np.unique(y_train).size)
    eval_metric = cfg.pop("eval_metric", "mlogloss" if n_classes > 2 else "logloss")
    has_val = X_val is not None and y_val is not None and len(np.asarray(X_val)) > 0
    callbacks = list(cfg.pop("callbacks", None) or [])
    if has_val and early_stopping_rounds:
        callbacks.append(EarlyStopping(rounds=int(early_stopping_rounds)))
    model = XGBClassifier(
        eval_metric=eval_metric,
        tree_method=cfg.pop("tree_method", "hist"),
        device=cfg.pop("device", "cpu"),
        random_state=random_state,
        verbosity=cfg.pop("verbosity", 0),
        callbacks=callbacks or None,
        **cfg,
    )
    fit_kwargs: dict[str, Any] = {"verbose": False}
    if has_val:
        fit_kwargs["eval_set"] = [(np.asarray(X_val), np.asarray(y_val))]
    model.fit(np.asarray(X_train), np.asarray(y_train), **fit_kwargs)
    return model
