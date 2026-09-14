"""
Converts a trained checkpoint into an ONNX file the backend can load,
and verifies the export actually matches the original model's output.

Run with: python -m export.export_onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

from models.dual_stream_net import DualStreamNet


def load_trained_model(checkpoint_path: str, config: dict, device: torch.device) -> DualStreamNet:
    model = DualStreamNet.from_config(config)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def export_to_onnx(model: DualStreamNet, output_path: Path, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    dummy_rgb = torch.randn(1, 3, image_size, image_size)
    dummy_freq = torch.randn(1, 3, image_size, image_size)

    onnx_program = torch.onnx.export(
        model,
        (dummy_rgb, dummy_freq),
        input_names=["rgb", "freq"],
        output_names=["logits"],
        dynamo=True,
    )
    onnx_program.optimize()
    onnx_program.save(str(output_path))

    return dummy_rgb, dummy_freq


def verify_export(
    model: DualStreamNet, onnx_path: Path, dummy_rgb: torch.Tensor, dummy_freq: torch.Tensor, tolerance: float = 1e-4
) -> None:
    with torch.no_grad():
        torch_output = model(dummy_rgb, dummy_freq).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"rgb": dummy_rgb.numpy(), "freq": dummy_freq.numpy()})[0]

    max_difference = float(np.abs(torch_output - onnx_output).max())

    if max_difference > tolerance:
        raise ValueError(f"ONNX export mismatch: {max_difference:.6f} exceeds tolerance {tolerance}.")

    print(f"Export verified - max difference from PyTorch: {max_difference:.8f}")


def main(checkpoint_path: str, config_path: str, output_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device("cpu")  # export is a one-time offline step
    model = load_trained_model(checkpoint_path, config, device)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dummy_rgb, dummy_freq = export_to_onnx(model, output_file, config["data"]["image_size"])
    verify_export(model, output_file, dummy_rgb, dummy_freq)

    print(f"Saved ONNX model to {output_file.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the trained model to ONNX.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output", type=str, default="../backend/app/ml_assets/model.onnx")
    args = parser.parse_args()

    main(args.checkpoint, args.config, args.output)
