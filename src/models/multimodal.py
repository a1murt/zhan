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
    """
    Late-fusion multimodal classifier combining fundus images and clinical
    tabular data to predict myopia progression risk.

    Args:
        tabular_input_dim: Number of tabular features fed to the MLP branch.
            For the default schema (Age, Baseline_SE, Axial_Length, Gender)
            this is 4.
        num_classes:       Number of progression-risk classes (default 3).
        dropout_fusion:    Dropout rate used inside the fusion head (default 0.4).
    """

    # SwinV2-Tiny feature dimension (fixed by architecture)
    _VISION_FEATURE_DIM: int = 768
    # MLP branch output dimension (fixed by TabularBranch)
    _TABULAR_FEATURE_DIM: int = 64

    def __init__(
        self,
        tabular_input_dim: int,
        num_classes: int = 3,
        dropout_fusion: float = 0.4,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Vision branch: SwinTransformerV2-Tiny, 256×256 input
        # num_classes=0 removes the classification head; forward() returns
        # a (B, 768) global-average-pooled feature vector.
        # ------------------------------------------------------------------
        self.vision_branch: nn.Module = timm.create_model(
            "swinv2_tiny_window16_256",
            pretrained=True,
            num_classes=0,          # strip classifier → raw feature vector
        )

        # Verify the feature dim at construction time so mis-matches surface
        # immediately rather than at runtime during the first forward pass.
        actual_vision_dim: int = self.vision_branch.num_features
        assert actual_vision_dim == self._VISION_FEATURE_DIM, (
            f"Unexpected SwinV2-Tiny feature dim: expected "
            f"{self._VISION_FEATURE_DIM}, got {actual_vision_dim}. "
            "Check your timm version."
        )

        # ------------------------------------------------------------------
        # Tabular branch: MLP [input_dim → 128 → 64]
        # ------------------------------------------------------------------
        self.tabular_branch = TabularBranch(
            input_dim=tabular_input_dim,
            dropout=0.3,
        )

        # ------------------------------------------------------------------
        # Fusion head: concat → 256 → num_classes
        # ------------------------------------------------------------------
        concat_dim: int = self._VISION_FEATURE_DIM + self._TABULAR_FEATURE_DIM  # 832
        self.fusion_head = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_fusion),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        image: torch.Tensor,    # (B, 3, 256, 256)
        tabular: torch.Tensor,  # (B, tabular_input_dim)
    ) -> torch.Tensor:          # (B, num_classes)  — raw logits
        vision_features = self.vision_branch(image)      # (B, 768)
        tabular_features = self.tabular_branch(tabular)  # (B, 64)

        fused = torch.cat([vision_features, tabular_features], dim=1)  # (B, 832)
        logits = self.fusion_head(fused)                               # (B, 3)
        return logits
