"""Dummy and other sklearn baselines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def fit_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Mapping[str, Any] | None = None,
) -> Pipeline:
    """StandardScaler (train only) then LogisticRegression."""
    clf = LogisticRegression(**dict(params or {}))
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(np.asarray(X_train), np.asarray(y_train))
    return pipe


def fit_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Mapping[str, Any] | None = None,
) -> RandomForestClassifier:
    """RandomForestClassifier. No scaler."""
    clf = RandomForestClassifier(**dict(params or {}))
    clf.fit(np.asarray(X_train), np.asarray(y_train))
    return clf
