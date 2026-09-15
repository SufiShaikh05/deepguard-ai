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

import logging
import time
import uuid

import torch
import yaml
from fastapi import FastAPI, Request
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

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("deepguard.main")


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
        logger.info("Models loaded successfully - DeepGuard AI is ready.")
    except FileNotFoundError as error:
        logger.warning("Model files not found (%s) - app is up, but /predict will return 503 until they exist.", error)

    yield
    logger.info("DeepGuard AI shutting down.")


app = FastAPI(title="DeepGuard AI", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id

    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s - %d - %.2fms [request_id=%s]",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
        return response
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.exception(
            "Unhandled exception on %s %s - %.2fms [request_id=%s]: %s",
            request.method,
            request.url.path,
            duration_ms,
            request_id,
            exc,
        )
        raise exc


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
