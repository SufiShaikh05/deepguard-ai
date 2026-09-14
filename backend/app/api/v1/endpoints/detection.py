from __future__ import annotations

from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.schemas.prediction import FastPredictionResponse, PredictionResponse
from app.services.gradcam_service import GradCAMService
from app.services.image_service import preprocess_for_model, read_and_validate_image
from app.services.inference_service import InferenceService
from app.utils.exceptions import ModelNotReadyError

router = APIRouter()


def get_gradcam_service(request: Request) -> GradCAMService:
    if request.app.state.gradcam_service is None:
        raise ModelNotReadyError("The explainability model isn't loaded yet.")
    return request.app.state.gradcam_service


def get_inference_service(request: Request) -> InferenceService:
    if request.app.state.inference_service is None:
        raise ModelNotReadyError("The classification model isn't loaded yet.")
    return request.app.state.inference_service


def get_image_size(request: Request) -> int:
    return request.app.state.image_size


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    gradcam_service: GradCAMService = Depends(get_gradcam_service),
    image_size: int = Depends(get_image_size),
) -> PredictionResponse:
    """Full explainable prediction: classification + Grad-CAM heatmap."""
    original_image = await read_and_validate_image(file)
    rgb_tensor, freq_tensor = preprocess_for_model(original_image, image_size)

    result = gradcam_service.predict_and_explain(rgb_tensor, freq_tensor, original_image)
    return PredictionResponse(**result)


@router.post("/predict/fast", response_model=FastPredictionResponse)
async def predict_fast(
    file: UploadFile = File(...),
    inference_service: InferenceService = Depends(get_inference_service),
    image_size: int = Depends(get_image_size),
) -> FastPredictionResponse:
    """Classification only, via ONNX Runtime - faster, no heatmap."""
    original_image = await read_and_validate_image(file)
    rgb_tensor, freq_tensor = preprocess_for_model(original_image, image_size)

    result = inference_service.predict(rgb_tensor.numpy(), freq_tensor.numpy())
    return FastPredictionResponse(**result)
