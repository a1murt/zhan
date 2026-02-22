"""
FastAPI inference service for ODIR-5K binary myopia classification.

Endpoint
--------
POST /predict/
    Body : multipart/form-data
        fundus_image : UploadFile  -- JPEG or PNG fundus photograph
        age          : float       -- patient age (years)
        gender       : str         -- "M" or "F"

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

    _STATE["model"]               = model
    _STATE["device"]              = device
    _STATE["numerical_imputer"]   = ckpt["numerical_imputer"]
    _STATE["categorical_imputer"] = ckpt["categorical_imputer"]
    _STATE["label_encoders"]      = ckpt["label_encoders"]
    _STATE["transform"]           = get_val_transforms()

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
def _preprocess_tabular(age: float, gender_str: str) -> torch.Tensor:
    """Return a (1, 2) float32 tensor [age_imputed, gender_encoded]."""
    gender_int = 1 if gender_str.strip().upper() == "M" else 0

    age_arr = _STATE["numerical_imputer"].transform(np.array([[age]]))
    age_val = float(age_arr[0, 0])

    gen_imp = _STATE["categorical_imputer"].transform([[str(gender_int)]])[0, 0]
    gen_enc = int(_STATE["label_encoders"]["Gender"].transform([str(gen_imp)])[0])

    tab = torch.tensor([[age_val, float(gen_enc)]], dtype=torch.float32)
    return tab.to(_STATE["device"])


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
    fundus_image: UploadFile = File(..., description="Fundus photograph (JPEG or PNG)."),
    age: float = Form(..., ge=1.0, le=110.0, description="Patient age in years."),
    gender: str = Form(..., description='"M" (male) or "F" (female).'),
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
    tab_t   = _preprocess_tabular(age, gender)

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
        "Prediction: %s  conf=%.3f  p_myopia=%.3f  %.1f ms | age=%.0f gender=%s",
        LABEL_MAP[pred_cls], confidence, p_myopia, elapsed, age, gender,
    )

    return PredictionResponse(
        diagnosis=LABEL_MAP[pred_cls],
        confidence_score=round(confidence, 4),
        p_myopia=round(p_myopia, 4),
        inference_time_ms=round(elapsed, 2),
    )
