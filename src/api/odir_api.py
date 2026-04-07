"""
FastAPI inference service for ODIR-5K binary myopia classification.

Endpoint
--------
POST /predict/
    Body : multipart/form-data
        fundus_image       : UploadFile  -- JPEG or PNG fundus photograph
        age                : float       -- patient age (years)
        refraction_without : float       -- refraction without cycloplegia (diopters)
        refraction_with    : float       -- refraction with cycloplegia (diopters)
        axl_current        : float       -- current axial length (mm)
        axl_6m_ago         : float       -- axial length 6 months ago (mm)
        family_history     : int         -- number of myopic parents (0, 1, or 2)
        screen_hours       : float       -- daily screen time (hours)
        outdoor_hours      : float       -- daily outdoor time (hours)

    Response : JSON
        diagnosis         : str   -- "No Myopia" | "Myopia"
        confidence_score  : float -- probability of predicted class [0, 1]
        p_myopia          : float -- raw myopia probability
        inference_time_ms : float -- wall-clock time in ms

GET /health  -- liveness check
GET /        -- service metadata
GET /docs    -- Swagger UI
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.transforms import get_val_transforms
from src.models.multimodal import MultimodalMyopiaClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "checkpoints/odir_best.pt"
LABEL_MAP: Dict[int, str] = {0: "No Myopia", 1: "Myopia"}

_STATE: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    import os
    if not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(
            f"Checkpoint not found: '{CHECKPOINT_PATH}'. Train the model first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading checkpoint from %s on %s ...", CHECKPOINT_PATH, device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    model = MultimodalMyopiaClassifier(
        tabular_input_dim=ckpt["tabular_input_dim"],
        num_classes=ckpt["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    _STATE["model"]     = model
    _STATE["device"]    = device
    _STATE["scaler"]    = ckpt["scaler"]
    _STATE["transform"] = get_val_transforms()

    logger.info(
        "Model ready. Best val AUC from training: %.4f (epoch %d)",
        ckpt.get("best_auc", float("nan")),
        ckpt.get("epoch", -1),
    )
    yield
    _STATE.clear()
    logger.info("Model released.")


# ---------------------------------------------------------------------------
app = FastAPI(
    title="ODIR-5K Myopia Detection API",
    description=(
        "Binary myopia classifier trained on ODIR-5K fundus images. "
        "Inputs: fundus photograph + age + gender. "
        "Output: No Myopia / Myopia with confidence."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://zhan-ai.kz",
        "https://www.zhan-ai.kz",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    diagnosis: str = Field(..., examples=["Myopia"])
    confidence_score: float = Field(..., ge=0.0, le=1.0, examples=[0.8341])
    p_myopia: float = Field(..., ge=0.0, le=1.0, examples=[0.8341])
    inference_time_ms: float = Field(..., examples=[12.4])


# ---------------------------------------------------------------------------
def _preprocess_tabular(
    refraction_without: float,
    refraction_with: float,
    axl_current: float,
    axl_6m_ago: float,
    age: float,
    genetics: int,
    screen_hours: float,
    outdoor_hours: float,
) -> torch.Tensor:
    """Return a (1, 8) float32 tensor scaled with the training StandardScaler.

    Feature order must match ODIR_TABULAR_COLS in odir_dataset.py:
    [refraction_without, refraction_with, axl_current, axl_delta,
     age, genetics, screen_hours, outdoor_hours]
    """
    axl_delta = axl_current - axl_6m_ago

    raw = np.array([[
        refraction_without,
        refraction_with,
        axl_current,
        axl_delta,
        age,
        float(genetics),
        screen_hours,
        outdoor_hours,
    ]], dtype=np.float32)

    scaled = _STATE["scaler"].transform(raw).astype(np.float32)
    return torch.tensor(scaled, dtype=torch.float32).to(_STATE["device"])


def _decode_image(raw_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode image. Send JPEG or PNG.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root() -> Dict[str, str]:
    return {"service": "ODIR Myopia Detection API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health", summary="Liveness check")
def health() -> Dict[str, str]:
    device = _STATE.get("device", torch.device("cpu"))
    return {"status": "ok", "device": str(device)}


@app.post(
    "/predict/",
    response_model=PredictionResponse,
    summary="Detect myopia from fundus image",
)
async def predict(
    fundus_image:       UploadFile = File(..., description="Fundus photograph (JPEG or PNG)."),
    age:                float = Form(..., ge=1.0,    le=110.0, description="Patient age in years."),
    refraction_without: float = Form(..., ge=-30.0,  le=30.0,  description="Refraction without cycloplegia (diopters)."),
    refraction_with:    float = Form(..., ge=-30.0,  le=30.0,  description="Refraction with cycloplegia (diopters)."),
    axl_current:        float = Form(..., ge=10.0,   le=40.0,  description="Current axial length (mm)."),
    axl_6m_ago:         float = Form(..., ge=10.0,   le=40.0,  description="Axial length 6 months ago (mm)."),
    family_history:     int   = Form(..., ge=0,      le=2,     description="Number of myopic parents (0, 1, or 2)."),
    screen_hours:       float = Form(..., ge=0.0,   le=24.0,  description="Daily screen time (hours)."),
    outdoor_hours:      float = Form(..., ge=0.0,   le=24.0,  description="Daily outdoor time (hours)."),
) -> PredictionResponse:
    t0 = time.perf_counter()

    if fundus_image.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content-type '{fundus_image.content_type}'. Use JPEG or PNG.",
        )

    raw_bytes = await fundus_image.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")

    rgb = _decode_image(raw_bytes)
    image_t = _STATE["transform"](image=rgb)["image"].unsqueeze(0).to(_STATE["device"])
    tab_t   = _preprocess_tabular(
        refraction_without=refraction_without,
        refraction_with=refraction_with,
        axl_current=axl_current,
        axl_6m_ago=axl_6m_ago,
        age=age,
        genetics=family_history,
        screen_hours=screen_hours,
        outdoor_hours=outdoor_hours,
    )

    model: MultimodalMyopiaClassifier = _STATE["model"]
    use_amp = _STATE["device"].type == "cuda"

    with torch.no_grad():
        with torch.amp.autocast(device_type=_STATE["device"].type, enabled=use_amp):
            logits = model(image_t, tab_t)

    probs     = F.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy()
    pred_cls  = int(np.argmax(probs))
    confidence = float(probs[pred_cls])
    p_myopia  = float(probs[1])
    elapsed   = (time.perf_counter() - t0) * 1000.0

    logger.info(
        "Prediction: %s  conf=%.3f  p_myopia=%.3f  %.1f ms | age=%.0f axl_delta=%.2f genetics=%d",
        LABEL_MAP[pred_cls], confidence, p_myopia, elapsed,
        age, axl_current - axl_6m_ago, family_history,
    )

    return PredictionResponse(
        diagnosis=LABEL_MAP[pred_cls],
        confidence_score=round(confidence, 4),
        p_myopia=round(p_myopia, 4),
        inference_time_ms=round(elapsed, 2),
    )
