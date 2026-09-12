from __future__ import annotations

import pytest
from PIL import Image

from photo_edit_studio.image_utils import composite_with_mask, normalize_image
from photo_edit_studio.models import MODEL_SPECS, ModelManager
from photo_edit_studio.ui import _run


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
            source=Image.new("RGB", (8, 8), "red"),
            references=None,
            mask=None,
            model_key="qwen-2511",
            prompt=" ",
            negative_prompt="",
            width=1024,
            height=1024,
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
