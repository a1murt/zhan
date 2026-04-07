"""
Multimodal Late-Fusion Architecture for Myopia Progression Classification.

Three-class output: 0 = Low Risk, 1 = Moderate Risk, 2 = High Risk.

Vision branch  : SwinTransformerV2-Tiny (pretrained, head removed) → 768-d vector
Tabular branch : MLP  [input_dim → 128 → 64]
Fusion head    : concat(768 + 64 = 832) → 256 → 3 logits
"""

import torch
import torch.nn as nn
import timm


class TabularBranch(nn.Module):
    """
    Two-layer MLP for structured clinical data.

    Architecture: input_dim → 128 → 64
    Each block: Linear → BatchNorm1d → GELU → Dropout(0.3)
    """

    def __init__(self, input_dim: int, dropout: float = 0.3) -> None:
        super().__init__()

        # ✅ This remains unchanged — already well designed
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MultimodalMyopiaClassifier(nn.Module):

    # Feature sizes (kept constant for clarity and debugging)
    _VISION_FEATURE_DIM: int = 768
    _TABULAR_FEATURE_DIM: int = 64

    def __init__(
        self,
        tabular_input_dim: int,
        num_classes: int = 3,
        dropout_fusion: float = 0.4,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Vision branch
        # ------------------------------------------------------------------
        self.vision_branch: nn.Module = timm.create_model(
            "swinv2_tiny_window16_256",
            pretrained=True,
            num_classes=0,
        )

        # ✅ ADDED: Safety check (already present but KEEP IT)
        # WHY: prevents silent dimension mismatch bugs later
        actual_vision_dim: int = self.vision_branch.num_features
        assert actual_vision_dim == self._VISION_FEATURE_DIM, (
            f"Unexpected SwinV2-Tiny feature dim: expected "
            f"{self._VISION_FEATURE_DIM}, got {actual_vision_dim}"
        )

        # ------------------------------------------------------------------
        # Tabular branch
        # ------------------------------------------------------------------
        self.tabular_branch = TabularBranch(
            input_dim=tabular_input_dim,
            dropout=0.3,
        )

        # ------------------------------------------------------------------
        # Fusion head
        # ------------------------------------------------------------------
        concat_dim: int = self._VISION_FEATURE_DIM + self._TABULAR_FEATURE_DIM  # 768 + 64 = 832

        self.fusion_head = nn.Sequential(
            nn.Linear(concat_dim, 256),

            # ✅ CHANGED: Added BatchNorm (optional but recommended)
            # WHY: stabilizes training when combining modalities
            nn.BatchNorm1d(256),

            nn.GELU(),

            nn.Dropout(p=dropout_fusion),

            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:

        # ✅ ADDED: explicit type casting
        # WHY: avoids dtype mismatch (VERY common PyTorch bug)
        tabular = tabular.float()

        # Vision features
        vision_features = self.vision_branch(image)  # (B, 768)

        # Tabular features
        tabular_features = self.tabular_branch(tabular)  # (B, 64)

        # ✅ ADDED: shape safety check
        # WHY: catches silent bugs early instead of crashing deep in training
        assert vision_features.shape[0] == tabular_features.shape[0], \
            "Batch size mismatch between image and tabular data"

        # Fusion
        fused = torch.cat([vision_features, tabular_features], dim=1)  # (B, 832)

        logits = self.fusion_head(fused)  # (B, num_classes)

        return logits