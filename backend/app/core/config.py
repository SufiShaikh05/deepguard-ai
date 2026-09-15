"""
Central, typed app configuration, loaded from environment variables.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    cors_allowed_origins: list[str] = ["http://localhost:5173"]

    # model.onnx is a build artifact exported specifically for this
    # backend (see ml_core/export/export_onnx.py) and lives locally.
    onnx_model_path: str = "app/ml_assets/model.onnx"

    # The checkpoint and its config are read directly from ml_core -
    # one copy of the truth, not a duplicate.
    checkpoint_path: str = "../ml_core/checkpoints/best_model.pth"
    ml_config_path: str = "../ml_core/configs/config.yaml"

    device: str = "cpu"
    log_level: str = "INFO"


settings = Settings()
