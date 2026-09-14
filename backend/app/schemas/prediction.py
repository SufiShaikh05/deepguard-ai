"""
The exact shape of every API response. The frontend matches these
field names exactly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassProbabilities(BaseModel):
    real: float = Field(..., ge=0.0, le=1.0, description="Probability the image is authentic")
    fake: float = Field(..., ge=0.0, le=1.0, description="Probability the image is manipulated")


class PredictionResponse(BaseModel):
    label: str = Field(..., description="Either 'real' or 'fake'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: ClassProbabilities
    heatmap_image: str = Field(..., description="Base64 PNG data URI of the Grad-CAM overlay")


class FastPredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: ClassProbabilities
