"""
FastAPI inference service for multimodal myopia progression prediction.

Endpoint
--------
POST /predict/
    Body : multipart/form-data
        fundus_image  : UploadFile  — JPEG or PNG fundus photograph
        age           : float       — patient age (years)
        gender        : str         — "M" or "F"
        baseline_se   : float       — spherical equivalent (diopters)
        axial_length  : float       — axial length (mm)

    Response : JSON
        risk_class        : str   — "Low Risk" | "Moderate Risk" | "High Risk"
        confidence_score  : float — softmax probability of predicted class [0, 1]
        inference_time_ms : float — wall-clock time from request receipt to response

Additional endpoints
--------------------
GET /health  — liveness check; returns device info.
GET /        — service metadata.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.dataset import CATEGORICAL_COLS, NUMERICAL_COLS, TABULAR_FEATURE_COLS
from src.data.transforms import get_val_transforms
from src.models.multimodal import MultimodalMyopiaClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = "checkpoints/best_model.pt"
RISK_CLASS_MAP: Dict[int, str] = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk",
}

# ---------------------------------------------------------------------------
# Global model state (populated in lifespan)
# ---------------------------------------------------------------------------

_MODEL_STATE: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup, release at shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the trained model and preprocessing objects on application startup."""
    import os

    if not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(
            f"Model checkpoint not found: '{CHECKPOINT_PATH}'. "
            "Run training before starting the API."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model checkpoint from %s onto %s …", CHECKPOINT_PATH, device)

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location=device, weights_only=False
    )

    net = MultimodalMyopiaClassifier(
        tabular_input_dim=checkpoint["tabular_input_dim"]
    )
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device)
    net.eval()

    _MODEL_STATE["model"]               = net
    _MODEL_STATE["device"]              = device
    _MODEL_STATE["numerical_imputer"]   = checkpoint["numerical_imputer"]
    _MODEL_STATE["categorical_imputer"] = checkpoint["categorical_imputer"]
    _MODEL_STATE["label_encoders"]      = checkpoint["label_encoders"]
    _MODEL_STATE["transform"]           = get_val_transforms()

    logger.info(
        "Model loaded successfully. Best val AUC from training: %.4f",
        checkpoint.get("best_auc", float("nan")),
    )

    yield  # ← server is running here

    _MODEL_STATE.clear()
    logger.info("Model resources released.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Myopia Progression Prediction API",
    description=(
        "Multimodal AI system for early diagnosis and clinical decision support "
        "in paediatric and adolescent myopia (ages 5–20). "
        "Classifies fundus images combined with clinical data into three "
        "progression-risk categories: Low, Moderate, and High."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    risk_class: str = Field(
        ...,
        description="Predicted progression-risk category.",
        examples=["High Risk"],
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Softmax probability of the predicted class.",
        examples=[0.8341],
    )
    inference_time_ms: float = Field(
        ...,
        description="End-to-end inference wall-clock time in milliseconds.",
        examples=[47.3],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preprocess_tabular(
    age: float,
    gender: str,
    baseline_se: float,
    axial_length: float,
) -> torch.Tensor:
    """
    Apply train-time imputation and encoding to a single-row DataFrame,
    then return a (1, 4) float32 tensor ready for the tabular branch.
    """
    raw = pd.DataFrame(
        [{"Age": age, "Baseline_SE": baseline_se, "Axial_Length": axial_length, "Gender": gender}]
    )

    # Impute numerical columns (using train-fitted median statistics)
    raw[NUMERICAL_COLS] = _MODEL_STATE["numerical_imputer"].transform(raw[NUMERICAL_COLS])

    # Impute categorical columns (using train-fitted most_frequent statistics)
    raw[CATEGORICAL_COLS] = _MODEL_STATE["categorical_imputer"].transform(
        raw[CATEGORICAL_COLS].astype(str)
    )

    # Ordinal encode categorical columns
    for col in CATEGORICAL_COLS:
        le = _MODEL_STATE["label_encoders"][col]
        try:
            raw[col] = le.transform(raw[col].astype(str))
        except ValueError:
            # Unknown category at inference time → fall back to most frequent class index
            raw[col] = 0

    values = raw[TABULAR_FEATURE_COLS].values.astype(np.float32)
    device: torch.device = _MODEL_STATE["device"]
    return torch.tensor(values, dtype=torch.float32).to(device)  # (1, 4)


def _decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into an (H, W, 3) uint8 RGB numpy array."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(
            status_code=422,
            detail="Image could not be decoded. Ensure the file is a valid JPEG or PNG.",
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root() -> Dict[str, str]:
    return {
        "service": "Myopia Progression Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", summary="Liveness check")
def health_check() -> Dict[str, str]:
    device: torch.device = _MODEL_STATE.get("device", torch.device("cpu"))
    return {"status": "ok", "device": str(device)}


@app.post(
    "/predict/",
    response_model=PredictionResponse,
    summary="Predict myopia progression risk",
    response_description="Risk class, confidence score, and inference latency.",
)
async def predict(
    fundus_image: UploadFile = File(
        ..., description="Fundus photograph of the patient's eye (JPEG or PNG)."
    ),
    age: float = Form(
        ..., description="Patient age in years (expected range: 5–20).", ge=1.0, le=100.0
    ),
    gender: str = Form(
        ..., description='Patient biological sex: "M" (male) or "F" (female).'
    ),
    baseline_se: float = Form(
        ...,
        description="Baseline spherical equivalent refraction in diopters (e.g. −3.50).",
    ),
    axial_length: float = Form(
        ...,
        description="Axial length measured by biometry in millimetres (e.g. 24.5).",
    ),
) -> PredictionResponse:
    t_start = time.perf_counter()

    # ---- Validate content type -----------------------------------------
    if fundus_image.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type '{fundus_image.content_type}'. "
                   "Upload a JPEG or PNG fundus photograph.",
        )

    # ---- Read and decode image -----------------------------------------
    raw_bytes = await fundus_image.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    rgb_image = _decode_image(raw_bytes)

    # ---- Apply val transforms ------------------------------------------
    transform = _MODEL_STATE["transform"]
    image_tensor: torch.Tensor = transform(image=rgb_image)["image"]  # (3, 256, 256)
    image_tensor = image_tensor.unsqueeze(0).to(
        _MODEL_STATE["device"], non_blocking=True
    )  # (1, 3, 256, 256)

    # ---- Preprocess tabular data ---------------------------------------
    tabular_tensor = _preprocess_tabular(age, gender, baseline_se, axial_length)

    # ---- Inference -----------------------------------------------------
    model: MultimodalMyopiaClassifier = _MODEL_STATE["model"]
    device: torch.device = _MODEL_STATE["device"]
    use_amp = device.type == "cuda"

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(image_tensor, tabular_tensor)           # (1, 3)

    probs = F.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy()  # (3,)
    predicted_class = int(np.argmax(probs))
    confidence      = float(probs[predicted_class])

    inference_time_ms = (time.perf_counter() - t_start) * 1_000.0

    logger.info(
        "Prediction: %s (conf=%.3f, %.1f ms) | age=%.1f gender=%s SE=%.2f AL=%.2f",
        RISK_CLASS_MAP[predicted_class], confidence, inference_time_ms,
        age, gender, baseline_se, axial_length,
    )

    return PredictionResponse(
        risk_class=RISK_CLASS_MAP[predicted_class],
        confidence_score=round(confidence, 4),
        inference_time_ms=round(inference_time_ms, 2),
    )
