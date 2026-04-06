"""
Mock API for local frontend/backend development.
Mimics odir_api.py responses without requiring a model checkpoint.

Run:
    uvicorn src.api.mock_api:app --host 0.0.0.0 --port 8050 --reload
"""

import random
import time

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="ZhanAI Mock API",
    description="Mock inference API for frontend development. Returns fake predictions.",
    version="mock",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    diagnosis: str = Field(..., examples=["Myopia"])
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    p_myopia: float = Field(..., ge=0.0, le=1.0)
    inference_time_ms: float


@app.get("/", include_in_schema=False)
def root():
    return {"service": "ZhanAI Mock API", "version": "mock", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "device": "mock"}


@app.post("/predict/", response_model=PredictionResponse)
async def predict(
    fundus_image: UploadFile = File(...),
    age: float = Form(...),
    gender: str = Form(...),
    refraction_without: float = Form(...),
    refraction_with: float = Form(...),
    axl_current: float = Form(...),
    axl_6m_ago: float = Form(...),
    family_history: int = Form(...),
    screen_hours: float = Form(...),
    outdoor_hours: float = Form(...),
):
    # Simulate a small processing delay
    time.sleep(0.3)

    p_myopia = round(random.uniform(0.1, 0.95), 4)
    diagnosis = "Myopia" if p_myopia >= 0.5 else "No Myopia"
    confidence = p_myopia if diagnosis == "Myopia" else round(1 - p_myopia, 4)

    return PredictionResponse(
        diagnosis=diagnosis,
        confidence_score=confidence,
        p_myopia=p_myopia,
        inference_time_ms=round(random.uniform(10, 80), 2),
    )
