"""
The fast, explanation-free path using ONNX Runtime.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from utils.constants import CLASS_NAMES


class InferenceService:
    def __init__(self, onnx_model_path: str) -> None:
        self.session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        # Swapping in a GPU box later? Install onnxruntime-gpu and use
        # providers=["CUDAExecutionProvider", "CPUExecutionProvider"] instead.

    def predict(self, rgb_array: np.ndarray, freq_array: np.ndarray) -> dict:
        rgb_batch = np.expand_dims(rgb_array, axis=0).astype(np.float32)
        freq_batch = np.expand_dims(freq_array, axis=0).astype(np.float32)

        logits = self.session.run(None, {"rgb": rgb_batch, "freq": freq_batch})[0]
        probabilities = _softmax(logits[0])
        predicted_class = int(np.argmax(probabilities))

        return {
            "label": CLASS_NAMES[predicted_class],
            "confidence": float(probabilities[predicted_class]),
            "probabilities": {
                CLASS_NAMES[0]: float(probabilities[0]),
                CLASS_NAMES[1]: float(probabilities[1]),
            },
        }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)  # numerical stability
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()
