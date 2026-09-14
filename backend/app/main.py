"""
Application entry point: creates the app, configures CORS, and loads
both models once via the lifespan context manager.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# ml_core lives as a sibling folder to backend/. Adding it to sys.path
# lets us import its code directly - one copy of the truth, never a
# pasted-in second copy that could quietly drift out of sync.
_ML_CORE_PATH = Path(__file__).resolve().parents[2] / "ml_core"
if str(_ML_CORE_PATH) not in sys.path:
    sys.path.insert(0, str(_ML_CORE_PATH))

import torch
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.services.gradcam_service import GradCAMService
from app.services.inference_service import InferenceService
from app.utils.exceptions import (
    InvalidImageError,
    ModelNotReadyError,
    generic_exception_handler,
    invalid_image_handler,
    model_not_ready_handler,
)
from models.dual_stream_net import DualStreamNet


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device(settings.device)
    app.state.gradcam_service = None
    app.state.inference_service = None
    app.state.image_size = 224

    try:
        with open(settings.ml_config_path, "r", encoding="utf-8") as file:
            ml_config = yaml.safe_load(file)

        pytorch_model = DualStreamNet.from_config(ml_config)
        state_dict = torch.load(settings.checkpoint_path, map_location=device, weights_only=True)
        pytorch_model.load_state_dict(state_dict)
        pytorch_model.to(device)

        app.state.gradcam_service = GradCAMService(pytorch_model, device)
        app.state.inference_service = InferenceService(settings.onnx_model_path)
        app.state.image_size = ml_config["data"]["image_size"]
        print("Models loaded - DeepGuard AI is ready.")
    except FileNotFoundError as error:
        print(f"Model files not found ({error}) - app is up, but /predict will return 503 until they exist.")

    yield
    print("Shutting down.")


app = FastAPI(title="DeepGuard AI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(InvalidImageError, invalid_image_handler)
app.add_exception_handler(ModelNotReadyError, model_not_ready_handler)
app.add_exception_handler(Exception, generic_exception_handler)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
