from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHOTO_EDIT_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False
    model_dir: Path = ROOT / "models"
    output_dir: Path = ROOT / "outputs"
    lora_dir: Path = ROOT / "models" / "loras"
    hf_home: Path = ROOT / "models" / "huggingface"
    max_concurrency: int = 1
    auth_user: str | None = None
    auth_password: str | None = None

    def prepare(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        self.hf_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.hf_home.resolve()))
        os.environ.setdefault("HF_HUB_CACHE", str((self.hf_home / "hub").resolve()))
        os.environ.setdefault("TRANSFORMERS_CACHE", str((self.hf_home / "transformers").resolve()))
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


settings = Settings()
settings.prepare()
