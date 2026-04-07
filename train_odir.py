"""
ODIR-5K Binary Myopia Training Launcher
========================================
Reads  : data/processed/odir_clean.csv
Images : IMAGE_DIR (set below — folder containing 0_left.jpg, 0_right.jpg, …)
Outputs: checkpoints/odir_best.pt

Run:
    python train_odir.py
    python train_odir.py --image-dir /path/to/images --batch-size 8 --epochs 5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from src.data.odir_dataset import OdirDataset
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.multimodal import MultimodalMyopiaClassifier

from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------------------------
# Binary-specific metrics (replaces the 3-class compute_metrics from metrics.py)
# ---------------------------------------------------------------------------

def compute_binary_metrics(
    all_labels: np.ndarray,   # (N,)   integer 0/1
    all_probs:  np.ndarray,   # (N, 2) softmax probabilities
    all_preds:  np.ndarray,   # (N,)   argmax predictions
) -> dict:
    # AUC: use probability of the positive class (column 1)
    try:
        auc = float(roc_auc_score(all_labels, all_probs[:, 1]))
    except ValueError as exc:
        logger.warning("AUC failed: %s. Returning 0.0", exc)
        auc = 0.0

    precision = float(precision_score(all_labels, all_preds, average="macro", zero_division=0))
    recall    = float(recall_score(all_labels, all_preds,    average="macro", zero_division=0))
    cm        = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    return {"macro_auc": auc, "macro_precision": precision,
            "macro_recall": recall, "confusion_matrix": cm}


def log_binary_metrics(epoch: int, metrics: dict, split: str = "val") -> None:
    logger.info(
        "[Epoch %3d] %s | AUC: %.4f | Precision: %.4f | Recall: %.4f",
        epoch, split.upper(), metrics["macro_auc"],
        metrics["macro_precision"], metrics["macro_recall"],
    )
    cm = metrics["confusion_matrix"]
    logger.info(
        "  Confusion Matrix:\n"
        "                Pred=0  Pred=1\n"
        "  True=0 (neg)  %6d  %6d\n"
        "  True=1 (pos)  %6d  %6d",
        cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1],
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_odir.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — edit IMAGE_DIR if not passing via CLI
# ---------------------------------------------------------------------------
CSV_PATH   = Path("data/processed/odir_external_real.csv")
CKPT_DIR   = Path("checkpoints")
IMAGE_DIR  = Path("data/raw/images")   # <-- default; override via --image-dir

# ---------------------------------------------------------------------------
# Focal Loss (binary-compatible: num_classes=2)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, reduction="none")
        pt   = torch.exp(-ce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * ce
        return loss.mean()


# ---------------------------------------------------------------------------
# Patient-safe train/val split
# ---------------------------------------------------------------------------

# FIX: use stratified split to avoid validation having only one class
# WHY: previous random split caused AUC = nan
from sklearn.model_selection import train_test_split

def split_patients(
    df: pd.DataFrame, val_size: float = 0.20, seed: int = 42
):
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df["Myopia"],  
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler:    torch.cuda.amp.GradScaler | None,
    is_train:  bool,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.train(is_train)
    use_amp  = device.type == "cuda"
    grad_ctx = nullcontext() if is_train else torch.no_grad()

    total_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    n_batches = len(loader)
    t0 = time.time()

    with grad_ctx:
        for batch_idx, (images, tabular, labels) in enumerate(loader, 1):
            images  = images.to(device, non_blocking=True)
            tabular = tabular.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images, tabular)
                loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

                if scaler is not None:  # GPU case
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                else:  # CPU case
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
            preds = logits.detach().argmax(dim=1).cpu().numpy()
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)
            all_preds.append(preds)

            # Progress heartbeat every 20 batches
            if batch_idx % 20 == 0 or batch_idx == n_batches:
                elapsed = time.time() - t0
                eta     = elapsed / batch_idx * (n_batches - batch_idx)
                phase   = "TRAIN" if is_train else "VAL  "
                logger.info(
                    "  [%s] batch %d/%d | loss=%.4f | elapsed=%.0fs | ETA=%.0fs",
                    phase, batch_idx, n_batches, loss.item(), elapsed, eta,
                )

    avg_loss = total_loss / len(loader.dataset)  # type: ignore[arg-type]
    return (
        avg_loss,
        np.concatenate(all_labels),
        np.concatenate(all_probs),
        np.concatenate(all_preds),
    )


# ---------------------------------------------------------------------------
# Training entry-point (called with a specific batch_size to allow OOM retry)
# ---------------------------------------------------------------------------

def train_with_batch_size(
    df:         pd.DataFrame,
    batch_size: int,
    num_epochs: int,
    image_dir:  Path,
) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | batch_size: %d | epochs: %d", device, batch_size, num_epochs)

    # ── Split ──────────────────────────────────────────────────────────────
    train_df, val_df = split_patients(df)
    # CHANGED: removed patient-based logging
    # WHY: dataset does not have patient IDs anymore
    logger.info(
        "Split -> train: %d rows | val: %d rows",
        len(train_df),
        len(val_df),
)

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = OdirDataset(train_df, transform=get_train_transforms(), fit_imputers=True)
    val_ds   = OdirDataset(
        val_df,
        transform=get_val_transforms(),
        fit_imputers=False,
        numerical_imputer=train_ds.numerical_imputer,
        scaler=train_ds.scaler, # FIX: removed categorical_imputer (not used anymore)
        label_encoders=train_ds.label_encoders,
    )

    # num_workers=0 on Windows avoids multiprocessing spawn issues
    nw = 0 if sys.platform == "win32" else 4
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = MultimodalMyopiaClassifier(
        tabular_input_dim=train_ds.tabular_input_dim,
        num_classes=2,
    ).to(device)

    # FREEZE vision branch — train only tabular branch + fusion head
    for param in model.vision_branch.parameters():
        param.requires_grad = False
    logger.info("Vision branch frozen (requires_grad=False)")

    logger.info(
        "Model: tabular_input_dim=%d | num_classes=2", train_ds.tabular_input_dim
    )

    

    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(df["Myopia"]),
        y=df["Myopia"]
    )

    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    # FIX: safer AMP initialization (works across PyTorch versions)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_auc  = 0.0
    ckpt_path = CKPT_DIR / "odir_best.pt"

    # ── Loop ──────────────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        logger.info("=" * 60)
        logger.info("EPOCH %d / %d", epoch, num_epochs)

        t_loss, t_labels, t_probs, t_preds = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, is_train=True
        )
        v_loss, v_labels, v_probs, v_preds = run_epoch(
            model, val_loader,   criterion, device, None, None, is_train=False
        )

        scheduler.step(epoch - 1 + 1)

        t_metrics = compute_binary_metrics(t_labels, t_probs, t_preds)
        v_metrics = compute_binary_metrics(v_labels, v_probs, v_preds)

        logger.info(
            "[Epoch %d/%d] Train Loss: %.4f | Val Loss: %.4f | LR: %.2e",
            epoch, num_epochs, t_loss, v_loss, optimizer.param_groups[0]["lr"],
        )
        log_binary_metrics(epoch, t_metrics, split="train")
        log_binary_metrics(epoch, v_metrics, split="val")

        if v_metrics["macro_auc"] > best_auc:
            best_auc = v_metrics["macro_auc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_auc": best_auc,
                    "tabular_input_dim": train_ds.tabular_input_dim,
                    "numerical_imputer": train_ds.numerical_imputer,
                    "scaler": train_ds.scaler,
                    "label_encoders": train_ds.label_encoders,
                    "num_classes": 2,
                },
                ckpt_path,
            )
            logger.info(
                "  [CKPT] New best val AUC: %.4f -> saved to %s",
                best_auc, ckpt_path,
            )

    logger.info("Training complete. Best val AUC: %.4f", best_auc)


# ---------------------------------------------------------------------------
# Main — OOM-aware retry loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ODIR-5K binary myopia model.")
    parser.add_argument(
        "--image-dir", type=Path, default=IMAGE_DIR,
        help="Directory containing 0_left.jpg, 0_right.jpg … (default: data/raw/images)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs",     type=int, default=5)
    args = parser.parse_args()

    image_dir: Path = args.image_dir
    batch_size: int = args.batch_size

    # ── Validate CSV ───────────────────────────────────────────────────────
    if not CSV_PATH.exists():
        logger.error("CSV not found: %s  — run data/prep_odir.py first.", CSV_PATH)
        sys.exit(1)

    # ── Validate image directory ───────────────────────────────────────────
    # if not image_dir.exists():
    #     logger.error(
    #         "Image directory not found: %s\n"
    #         "Download the ODIR-5K fundus images and set --image-dir to the folder "
    #         "containing 0_left.jpg, 0_right.jpg, etc.",
    #         image_dir,
    #     )
    #     sys.exit(1)

    # sample_imgs = list(image_dir.glob("*.jpg"))[:3]
    # if not sample_imgs:
    #     logger.error(
    #         "No .jpg files found in %s — wrong directory?", image_dir
    #     )
    #     sys.exit(1)
    # logger.info("Image directory OK: %s  (%d+ images found)", image_dir, len(sample_imgs))

    # Skipping image directory check (tabular-only mode)
    logger.info("Skipping image directory validation (no images used)")

    # ── Load CSV and build full Image_Path ────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    # df["Image_Path"] = df["Image_Name"].apply(lambda fn: str(image_dir / fn))
    logger.info("Loaded %d rows from %s", len(df), CSV_PATH)
    logger.info(
        "Myopia distribution: pos=%d (%.1f%%) | neg=%d (%.1f%%)",
        df["Myopia"].sum(), df["Myopia"].mean() * 100,
        (df["Myopia"] == 0).sum(), (1 - df["Myopia"].mean()) * 100,
    )

    # ── OOM-aware training loop ────────────────────────────────────────────
    min_batch = 1
    while batch_size >= min_batch:
        try:
            logger.info("Starting training with batch_size=%d ...", batch_size)
            train_with_batch_size(df, batch_size, args.epochs, image_dir)
            break  # success
        except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as exc:
            err_str = str(exc).lower()
            is_oom  = (
                isinstance(exc, torch.cuda.OutOfMemoryError)
                or "out of memory" in err_str
                or "alloc" in err_str
            )
            if is_oom and batch_size > min_batch:
                new_bs = max(batch_size // 2, min_batch)
                logger.warning(
                    "OOM detected (batch_size=%d). Retrying with batch_size=%d ...",
                    batch_size, new_bs,
                )
                batch_size = new_bs
                torch.cuda.empty_cache()
            else:
                logger.error("Training failed: %s", exc)
                raise


if __name__ == "__main__":
    main()
