"""
End-to-end demo: load an image, run it through the model, generate
its Grad-CAM heatmap, blend it with OpenCV, and save a PNG.

Run with: python gradcam_demo.py --image path/to/any_face.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from explainability.gradcam_engine import GradCAMExplainer
from explainability.heatmap_blender import blend_heatmap_with_image
from models.dual_stream_net import DualStreamNet
from utils.constants import CLASS_NAMES
from utils.dct_utils import image_to_dct, normalize_dct
from utils.transforms import normalize_rgb, resize_image, to_chw_tensor_array


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> DualStreamNet:
    model = DualStreamNet.from_config(config)
    checkpoint_file = Path(checkpoint_path)

    if checkpoint_file.is_file():
        state_dict = torch.load(checkpoint_file, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded trained weights from {checkpoint_file}")
    else:
        print(
            f"No checkpoint found at '{checkpoint_file}' - using an untrained model "
            "(matching config.yaml's architecture). The heatmap will run correctly, "
            "but won't be meaningful until you train."
        )

    return model.to(device)


def preprocess_image(image_path: str, image_size: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    image = resize_image(image, image_size)
    image_array = np.array(image)

    rgb_tensor = torch.from_numpy(to_chw_tensor_array(normalize_rgb(image_array))).float()
    dct_array = normalize_dct(image_to_dct(image_array))
    freq_tensor = torch.from_numpy(to_chw_tensor_array(dct_array)).float()

    return rgb_tensor, freq_tensor, image_array


def main(image_path: str, checkpoint_path: str, config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint_path, config, device)
    explainer = GradCAMExplainer(model, device)

    rgb_tensor, freq_tensor, original_image = preprocess_image(image_path, config["data"]["image_size"])
    result = explainer.explain(rgb_tensor, freq_tensor)

    blended = blend_heatmap_with_image(original_image, result.heatmap)

    output_path = Path("gradcam_demo_output.png")
    cv2.imwrite(str(output_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    print(f"Prediction: {CLASS_NAMES[result.predicted_class]}  (confidence: {result.confidence:.2%})")
    print(f"Probabilities: {result.probabilities}")
    print(f"Heatmap saved to {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Grad-CAM on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to any face photo.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    main(args.image, args.checkpoint, args.config)
