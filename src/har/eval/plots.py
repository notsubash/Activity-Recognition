"""Confusion-matrix helpers for experiment reports."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int | str],
) -> np.ndarray:
    return np.asarray(sklearn_confusion_matrix(y_true, y_pred, labels=list(labels)))


def save_confusion_matrix(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Sequence[int | str],
    class_names: list[str] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    counts = confusion_counts(y_true, y_pred, labels)
    names = class_names if class_names is not None else [str(x) for x in labels]
    fig, ax = plt.subplots()
    image = ax.imshow(counts, interpolation="nearest")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(range(len(names)), labels=names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), labels=names)
    fig.colorbar(image, ax=ax)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
