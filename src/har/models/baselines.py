"""Dummy and other sklearn baselines."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier


def fit_dummy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    strategy: str = "most_frequent",
    seed: int | None = None,
) -> DummyClassifier:
    """Majority or stratified dummy. Fit scalers never belong here."""
    clf = DummyClassifier(strategy=strategy, random_state=seed)
    clf.fit(np.asarray(X_train), np.asarray(y_train))
    return clf
