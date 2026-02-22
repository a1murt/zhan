"""
Evaluation metrics for myopia progression risk classification.

Computed every epoch for both train and validation splits:
  • Macro one-vs-rest ROC-AUC
  • Macro precision
  • Macro recall
  • 3×3 confusion matrix (Low / Moderate / High Risk)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# Ordered to match integer class labels 0, 1, 2
CLASS_NAMES = ["Low Risk", "Moderate Risk", "High Risk"]


def compute_metrics(
    all_labels: np.ndarray,   # (N,)        integer class indices
    all_probs: np.ndarray,    # (N, 3)      softmax probabilities
    all_preds: np.ndarray,    # (N,)        argmax predicted classes
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics from epoch-level accumulated arrays.

    Returns
    -------
    dict with keys:
        macro_auc        : float
        macro_precision  : float
        macro_recall     : float
        confusion_matrix : np.ndarray shape (3, 3)
    """
    # ------------------------------------------------------------------
    # Macro ROC-AUC (one-vs-rest)
    # Guard against epochs where not all classes are represented in labels
    # (can happen with very small validation sets).
    # ------------------------------------------------------------------
    try:
        macro_auc = float(
            roc_auc_score(
                all_labels,
                all_probs,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError as exc:
        logger.warning(
            "ROC-AUC could not be computed (likely missing class in split): %s. "
            "Falling back to 0.0.",
            exc,
        )
        macro_auc = 0.0

    # ------------------------------------------------------------------
    # Macro precision & recall (zero_division=0 silences ill-defined warnings
    # for classes absent from predictions in early epochs)
    # ------------------------------------------------------------------
    macro_precision = float(
        precision_score(all_labels, all_preds, average="macro", zero_division=0)
    )
    macro_recall = float(
        recall_score(all_labels, all_preds, average="macro", zero_division=0)
    )

    # ------------------------------------------------------------------
    # Confusion matrix — always 3×3 regardless of which classes appeared
    # ------------------------------------------------------------------
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    return {
        "macro_auc": macro_auc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "confusion_matrix": cm,
    }


def log_epoch_metrics(
    epoch: int,
    metrics: Dict[str, Any],
    split: str = "val",
) -> None:
    """Pretty-print metrics to the logger at INFO level."""
    header = f"[Epoch {epoch:>3d}] {split.upper():<5}"
    logger.info(
        "%s | AUC: %.4f | Precision: %.4f | Recall: %.4f",
        header,
        metrics["macro_auc"],
        metrics["macro_precision"],
        metrics["macro_recall"],
    )
    # Format confusion matrix with class name annotations
    cm: np.ndarray = metrics["confusion_matrix"]
    max_name_len = max(len(n) for n in CLASS_NAMES)
    header_row = " " * (max_name_len + 2) + "  ".join(
        f"{n[:6]:>6}" for n in CLASS_NAMES
    )
    rows = [header_row]
    for i, row_name in enumerate(CLASS_NAMES):
        cells = "  ".join(f"{cm[i, j]:>6d}" for j in range(len(CLASS_NAMES)))
        rows.append(f"  {row_name:<{max_name_len}}  {cells}")
    logger.info("Confusion Matrix:\n%s", "\n".join(rows))
