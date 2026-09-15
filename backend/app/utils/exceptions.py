"""
Custom exceptions and the FastAPI handlers that turn them into JSON.
"""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("deepguard.exceptions")


class InvalidImageError(Exception):
    """Raised when an uploaded file isn't a usable image."""


class ModelNotReadyError(Exception):
    """Raised if a route is hit before the models have finished loading."""


async def invalid_image_handler(request: Request, exc: InvalidImageError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning("Invalid image on %s [request_id=%s]: %s", request.url.path, request_id, exc)
    return JSONResponse(status_code=400, content={"error": "invalid_image", "message": str(exc)})


async def model_not_ready_handler(request: Request, exc: ModelNotReadyError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning("Model not ready on %s [request_id=%s]: %s", request.url.path, request_id, exc)
    return JSONResponse(status_code=503, content={"error": "model_not_ready", "message": str(exc)})


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("Internal error on %s [request_id=%s]: %s", request.url.path, request_id, exc)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "Something went wrong while processing your request."},
    )
