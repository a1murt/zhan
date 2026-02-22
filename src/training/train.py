"""
Training pipeline for MultimodalMyopiaClassifier.

Key design choices
------------------
• Mixed-precision training via torch.amp.autocast + GradScaler
  (VRAM-efficient; falls back gracefully to fp32 on CPU).
• Patient-level GroupShuffleSplit prevents left/right eye images from the
  same patient appearing in both train and val (data-leakage prevention).
• Focal Loss (γ=2, α=0.25) down-weights easy negatives and focuses the
  model on rare High-Risk cases — critical for a medical screening system.
• Imputers and label-encoders are fit *only* on the training split and
  persisted in the checkpoint for identical preprocessing at inference time.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from src.data.dataset import (
    LABEL_COL,
    PATIENT_ID_COL,
    MyopiaDataset,
)
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.multimodal import MultimodalMyopiaClassifier
from src.training.metrics import compute_metrics, log_epoch_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    FL(p_t) = -α · (1 − p_t)^γ · log(p_t)

    Args:
        gamma: Focusing parameter.  Higher γ down-weights easy examples more
               aggressively.  γ=2 is standard per Lin et al. (2017).
        alpha: Scalar weighting factor applied uniformly across classes.
               For per-class weighting pass a Tensor of length num_classes.
        num_classes: Number of output classes (used for one-hot conversion).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(
        self,
        logits: torch.Tensor,   # (B, C)
        targets: torch.Tensor,  # (B,) integer class indices
    ) -> torch.Tensor:
        # Cross-entropy gives −log(p_t) element-wise
        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # (B,)
        # p_t = exp(−CE)
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


# ---------------------------------------------------------------------------
# Patient-safe train / val split
# ---------------------------------------------------------------------------

def split_by_patient(
    df: pd.DataFrame,
    val_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into train/val subsets ensuring that all records for
    a given Patient_ID land in exactly one subset (no leakage between eyes).

    Returns
    -------
    train_df, val_df — both reset-indexed.
    """
    gss = GroupShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state
    )
    groups = df[PATIENT_ID_COL].values
    train_idx, val_idx = next(gss.split(df, df[LABEL_COL], groups=groups))
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Single epoch runner (train or eval)
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    is_train: bool,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one full pass over *loader*.

    Returns
    -------
    (avg_loss, all_labels, all_probs, all_preds)
    """
    model.train(is_train)

    use_amp = device.type == "cuda"
    grad_ctx = nullcontext() if is_train else torch.no_grad()

    total_loss = 0.0
    all_labels_list: list[np.ndarray] = []
    all_probs_list:  list[np.ndarray] = []
    all_preds_list:  list[np.ndarray] = []

    with grad_ctx:
        for images, tabular, labels in loader:
            images  = images.to(device, non_blocking=True)
            tabular = tabular.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images, tabular)          # (B, 3)
                loss   = criterion(logits, labels)

            if is_train:
                assert optimizer is not None and scaler is not None
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                # Unscale before clipping so gradient norms are meaningful
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * images.size(0)

            # Detach and move to CPU before numpy conversion
            probs = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
            preds = logits.detach().argmax(dim=1).cpu().numpy()

            all_labels_list.append(labels.cpu().numpy())
            all_probs_list.append(probs)
            all_preds_list.append(preds)

    avg_loss   = total_loss / len(loader.dataset)  # type: ignore[arg-type]
    all_labels = np.concatenate(all_labels_list)
    all_probs  = np.concatenate(all_probs_list)
    all_preds  = np.concatenate(all_preds_list)

    return avg_loss, all_labels, all_probs, all_preds


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------

def train(
    df: pd.DataFrame,
    checkpoint_dir: str = "checkpoints",
    num_epochs: int = 50,
    batch_size: int = 16,
    num_workers: int = 4,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    val_size: float = 0.20,
    random_state: int = 42,
) -> MultimodalMyopiaClassifier:
    """
    Full training loop.

    Saves the best checkpoint (by val macro-AUC) to
    ``{checkpoint_dir}/best_model.pt``.  The checkpoint includes imputers
    and label-encoders so the API can replicate train-time preprocessing
    without any additional state files.

    Returns
    -------
    The model loaded with the best validation weights.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_df, val_df = split_by_patient(df, val_size=val_size, random_state=random_state)
    logger.info(
        "Split → train: %d samples (%d patients)  |  val: %d samples (%d patients)",
        len(train_df), train_df[PATIENT_ID_COL].nunique(),
        len(val_df),   val_df[PATIENT_ID_COL].nunique(),
    )

    train_dataset = MyopiaDataset(
        train_df,
        transform=get_train_transforms(),
        fit_imputers=True,
    )
    val_dataset = MyopiaDataset(
        val_df,
        transform=get_val_transforms(),
        fit_imputers=False,
        numerical_imputer=train_dataset.numerical_imputer,
        categorical_imputer=train_dataset.categorical_imputer,
        label_encoders=train_dataset.label_encoders,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,          # keeps BatchNorm stable at the last partial batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # Model, loss, optimiser, scheduler, scaler
    # ------------------------------------------------------------------
    model = MultimodalMyopiaClassifier(
        tabular_input_dim=train_dataset.tabular_input_dim,
    ).to(device)

    criterion = FocalLoss(gamma=2.0, alpha=0.25, num_classes=3)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # T_0=10: first restart after 10 epochs; T_mult=2: each cycle doubles
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_auc = 0.0
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    for epoch in range(1, num_epochs + 1):
        train_loss, t_labels, t_probs, t_preds = _run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, is_train=True
        )
        val_loss, v_labels, v_probs, v_preds = _run_epoch(
            model, val_loader, criterion, device, optimizer=None, scaler=None, is_train=False
        )

        # Scheduler step uses the fractional epoch for smooth annealing
        scheduler.step(epoch - 1 + len(train_loader) / len(train_loader))

        train_metrics = compute_metrics(t_labels, t_probs, t_preds)
        val_metrics   = compute_metrics(v_labels, v_probs, v_preds)

        logger.info(
            "[Epoch %d/%d]  Train Loss: %.4f  |  Val Loss: %.4f  |  LR: %.2e",
            epoch, num_epochs,
            train_loss, val_loss,
            optimizer.param_groups[0]["lr"],
        )
        log_epoch_metrics(epoch, train_metrics, split="train")
        log_epoch_metrics(epoch, val_metrics,   split="val")

        # ------------------------------------------------------------------
        # Checkpoint — persist everything needed for inference
        # ------------------------------------------------------------------
        if val_metrics["macro_auc"] > best_auc:
            best_auc = val_metrics["macro_auc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                    # Preprocessing objects (serialised via pickle inside torch.save)
                    "tabular_input_dim":  train_dataset.tabular_input_dim,
                    "numerical_imputer":  train_dataset.numerical_imputer,
                    "categorical_imputer": train_dataset.categorical_imputer,
                    "label_encoders":     train_dataset.label_encoders,
                },
                best_checkpoint_path,
            )
            logger.info(
                "  ✓ New best val AUC: %.4f — checkpoint saved to %s",
                best_auc, best_checkpoint_path,
            )

    # ------------------------------------------------------------------
    # Reload best weights before returning
    # ------------------------------------------------------------------
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Training complete. Best val AUC: %.4f", best_auc)
    return model
