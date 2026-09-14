from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True, slots=True)
class LoraSpec:
    name: str
    path: Path
    weight: float
    adapter_name: str


@dataclass(slots=True)
class GenerationRequest:
    model_key: str
    prompt: str
    negative_prompt: str
    images: list[Image.Image]
    mask: Image.Image | None
    width: int
    height: int
    steps: int
    guidance: float
    true_cfg: float
    strength: float
    seed: int
    count: int = 1
    preserve_identity: bool = True
    size_multiplier: int = 1
    compose: bool = False
    output_resolution: str = "1K"
    loras: list[LoraSpec] = field(default_factory=list)


@dataclass(slots=True)
class GenerationResult:
    images: list[Image.Image]
    seed: int
    elapsed_seconds: float
    model_key: str
    saved_paths: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    raw: Any = None
