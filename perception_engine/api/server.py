"""
FastAPI server for Perception Lab inference.

Run with: uvicorn perception_engine.api.server:app --reload --port 8000

Environment variables:
  PERCEPTION_WEIGHTS: path to best_model_v6.pth (or v5)
  PERCEPTION_PRIOR: path to joint_histograms.pkl (optional)
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from perception_engine.api.pipeline import run_pipeline

app = FastAPI(
    title="Perception Engine API",
    description="Inference pipeline: Segmentation → Cost Map → Safe Path",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEIGHTS_PATH = os.environ.get(
    "PERCEPTION_WEIGHTS",
    os.path.join(os.path.dirname(__file__), "..", "..", "weights", "best_model_v6.pth"),
)
PRIOR_PATH = os.environ.get(
    "PERCEPTION_PRIOR",
    os.path.join(os.path.dirname(__file__), "..", "..", "weights", "joint_histograms.pkl"),
)


@app.get("/health")
def health():
    return {"status": "ok", "weights": WEIGHTS_PATH}


@app.post("/api/infer")
async def infer(image: UploadFile = File(...)):
    """Run perception pipeline on uploaded image. Returns base64-encoded PNGs."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (PNG, JPG, etc.)")

    # Limit size (e.g. 10MB)
    contents = await image.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    result = run_pipeline(
        image_bytes=contents,
        weights_path=WEIGHTS_PATH,
        prior_path=PRIOR_PATH if os.path.exists(PRIOR_PATH) else None,
        use_prior=os.path.exists(PRIOR_PATH),
        original_filename=image.filename,
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return result
