"""
Albumentations pipelines for fundus image preprocessing.

Image size is fixed at 256×256 to satisfy the SwinV2-Tiny window constraint:
  patch_size=4 → feature map 64×64; window_size=16 → 64/16 = 4 (integer ✓).
  Using 224×224 would produce a 56×56 feature map (56/16 = 3.5 → runtime error).

ImageNet normalisation statistics are used for transfer-learning compatibility.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet channel statistics (RGB order)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# Input resolution required by swinv2_tiny_window16_256
_INPUT_SIZE = 256


def get_train_transforms() -> A.Compose:
    """
    Augmentation pipeline for the training split.

    Stages
    ------
    1. Resize to 256×256
    2. RandomRotate90      — fundus images have no canonical orientation
    3. ShiftScaleRotate    — moderate geometric jitter
    4. CLAHE               — local contrast enhancement for retinal vessel clarity
    5. Normalize           — ImageNet stats
    6. ToTensorV2          — HWC uint8 → CHW float32 tensor
    """
    return A.Compose(
        [
            A.Resize(_INPUT_SIZE, _INPUT_SIZE),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=15,
                border_mode=0,   # constant (black) border → avoids wrap-around artefacts
                p=0.60,
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.50),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms() -> A.Compose:
    """
    Deterministic pipeline for validation and test splits (and inference).

    Stages
    ------
    1. Resize to 256×256
    2. Normalize — ImageNet stats
    3. ToTensorV2
    """
    return A.Compose(
        [
            A.Resize(_INPUT_SIZE, _INPUT_SIZE),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )
