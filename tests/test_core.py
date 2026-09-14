from __future__ import annotations

import pytest
from PIL import Image

from photo_edit_studio.engine import generate
from photo_edit_studio.image_utils import composite_with_mask, normalize_image
from photo_edit_studio.models import MODEL_SPECS, ModelManager
from photo_edit_studio.types import GenerationRequest
from photo_edit_studio.ui import (
    COMBINE_MODE,
    EDIT_MODE,
    _collect_images,
    _combine_size_preview,
    _run,
    _size_preview,
)


def test_registry_has_requested_models() -> None:
    assert set(MODEL_SPECS) == {
        "qwen-2511",
        "qwen-2511-aio",
        "firered-1.1",
        "flux-klein-4b",
        "flux-klein-9b",
    }


def test_manager_selects_qwen_aio_adapter() -> None:
    from photo_edit_studio.models.diffusers_adapters import QwenAioAdapter

    adapter = ModelManager().get("qwen-2511-aio")
    assert isinstance(adapter, QwenAioAdapter)
    assert MODEL_SPECS["qwen-2511-aio"].default_steps == 4


def test_manager_starts_unloaded() -> None:
    assert ModelManager().status == "No model loaded"


def test_normalize_image_converts_and_resizes() -> None:
    image = Image.new("RGBA", (3000, 1000), "red")
    result = normalize_image(image, max_side=1000)
    assert result.mode == "RGB"
    assert result.size == (1000, 333)


def test_mask_composite() -> None:
    source = Image.new("RGB", (8, 8), "red")
    generated = Image.new("RGB", (8, 8), "blue")
    white = Image.new("L", (8, 8), 255)
    assert composite_with_mask(source, generated, white, feather=0).getpixel((0, 0)) == (0, 0, 255)


def test_empty_prompt_rejected() -> None:
    with pytest.raises(ValueError, match="instruction"):
        _run(
            mode=EDIT_MODE,
            source=Image.new("RGB", (8, 8), "red"),
            references=None,
            combine_1=None,
            combine_2=None,
            combine_3=None,
            mask=None,
            model_key="qwen-2511",
            prompt=" ",
            negative_prompt="",
            size_multiplier=1,
            output_resolution="1K",
            steps=4,
            guidance=1.0,
            true_cfg=4.0,
            strength=0.8,
            seed=0,
            count=1,
            preserve_identity=True,
            restoration="Off",
            restore_weight=0.35,
        )


def test_combine_mode_collects_up_to_three_images() -> None:
    one = Image.new("RGB", (64, 32), "red")
    two = Image.new("RGB", (32, 64), "blue")
    three = Image.new("RGB", (48, 48), "green")
    images = _collect_images(COMBINE_MODE, None, [one], one, two, three)
    assert images == [one, two, three]


def test_edit_mode_uses_source_and_references() -> None:
    source = Image.new("RGB", (64, 32), "red")
    reference = Image.new("RGB", (32, 32), "blue")
    images = _collect_images(EDIT_MODE, source, [reference], None, None, None)
    assert images == [source, reference]


def test_size_preview_uses_image_one_aspect() -> None:
    source = Image.new("RGB", (1024, 768), "red")
    text = _size_preview(EDIT_MODE, 2, "1K", source, None, None, None)
    assert "2048 × 1536" in text


def test_size_preview_combine_uses_resolution_canvas() -> None:
    text = _combine_size_preview("2K", None, None, None)
    assert "2048 × 2048" in text
    assert "2K" in text


def test_compose_without_images_is_rejected() -> None:
    request = GenerationRequest(
        model_key="qwen-2511",
        prompt="combine them",
        negative_prompt="",
        images=[],
        mask=None,
        width=1024,
        height=1024,
        steps=4,
        guidance=1.0,
        true_cfg=4.0,
        strength=0.8,
        seed=0,
        compose=True,
    )
    with pytest.raises(ValueError, match="1"):
        generate(request, "Off", 0.35)
