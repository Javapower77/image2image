from __future__ import annotations

import json
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path

from PIL import PngImagePlugin

from photo_edit_studio.config import settings
from photo_edit_studio.image_utils import (
    MAX_OUTPUT_SIDE,
    combine_output_size,
    composite_with_mask,
    normalize_image,
    scaled_output_size,
)
from photo_edit_studio.models import MODEL_SPECS, model_manager
from photo_edit_studio.models.memory import release_cuda
from photo_edit_studio.restoration import restore_faces
from photo_edit_studio.types import GenerationRequest, GenerationResult


def generate(
    request: GenerationRequest, restoration: str, restore_weight: float
) -> GenerationResult:
    if not request.images:
        if request.compose:
            raise ValueError("Upload 1–3 images to combine.")
        raise ValueError("Upload a source image.")
    spec = MODEL_SPECS[request.model_key]
    if request.compose:
        request.width, request.height = combine_output_size(request.output_resolution)
        request.mask = None
        input_max_side = MAX_OUTPUT_SIDE
    else:
        request.width, request.height = scaled_output_size(
            request.images[0].size, request.size_multiplier
        )
        input_max_side = max(request.width, request.height)
    request.images = [
        normalize_image(image, max_side=input_max_side)
        for image in request.images[: spec.max_images]
    ]
    if request.seed < 0:
        request.seed = secrets.randbelow(2**31 - 1)

    started = time.perf_counter()
    adapter = model_manager.get(request.model_key)
    images = adapter.generate(request)
    notes: list[str] = []
    if request.compose:
        notes.append(
            f"Output {request.width}×{request.height} ({request.output_resolution} canvas)."
        )
        notes.append(f"Combined {len(request.images)} reference image(s) into one new picture.")
    else:
        notes.append(
            f"Output {request.width}×{request.height} from image-1 aspect ×{request.size_multiplier}."
        )
    if request.mask is not None:
        images = [composite_with_mask(request.images[0], image, request.mask) for image in images]
        notes.append("Mask composited after generation; white areas contain the edit.")
    if restoration != "Off":
        release_cuda()
        images = [restore_faces(image, restoration, restore_weight) for image in images]
        notes.append(f"Applied {restoration} post-processing.")
        release_cuda()
    if request.loras:
        notes.append(
            "LoRAs: " + ", ".join(f"{spec.name}@{spec.weight:g}" for spec in request.loras)
        )

    elapsed = time.perf_counter() - started
    result = GenerationResult(images, request.seed, elapsed, request.model_key, notes=notes)
    result.saved_paths = _save(result, request)
    return result


def _save(result: GenerationResult, request: GenerationRequest) -> list[Path]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = settings.output_dir / f"{stamp}_{request.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "model": MODEL_SPECS[request.model_key].repo_id,
        "seed": result.seed,
        "width": request.width,
        "height": request.height,
        "size_multiplier": request.size_multiplier,
        "compose": request.compose,
        "output_resolution": request.output_resolution if request.compose else None,
        "steps": request.steps,
        "guidance": request.guidance,
        "true_cfg": request.true_cfg,
        "strength": request.strength,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "input_count": len(request.images),
        "loras": [{"name": spec.name, "weight": spec.weight} for spec in request.loras],
        "prompt_stored": False,
    }
    paths: list[Path] = []
    for index, image in enumerate(result.images, start=1):
        path = run_dir / f"result_{index}.png"
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("generation", json.dumps(metadata))
        image.save(path, pnginfo=pnginfo)
        paths.append(path)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths


def metadata_text(result: GenerationResult) -> str:
    payload = {
        "model": MODEL_SPECS[result.model_key].label,
        "seed": result.seed,
        "elapsed_seconds": round(result.elapsed_seconds, 2),
        "outputs": [str(path) for path in result.saved_paths],
        "notes": result.notes,
    }
    return json.dumps(payload, indent=2)
