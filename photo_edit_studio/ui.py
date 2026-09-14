from __future__ import annotations

from typing import Any

import gradio as gr
from PIL import Image

from photo_edit_studio.engine import generate, metadata_text
from photo_edit_studio.image_utils import combine_output_size, scaled_output_size
from photo_edit_studio.loras import (
    MAX_LORAS,
    NONE_CHOICE,
    assign_new_slots,
    import_lora_files,
    library_status,
    lora_choices,
    selected_loras,
)
from photo_edit_studio.models import MODEL_SPECS, model_manager
from photo_edit_studio.types import GenerationRequest

EDIT_MODE = "Edit source"
COMBINE_MODE = "Combine images"
EDIT_PLACEHOLDER = (
    "Replace the background with a softly lit Paris café; preserve the "
    "person's exact identity, pose, and skin texture."
)
COMBINE_PLACEHOLDER = (
    "Create a new picture: the person from image 1 wearing the outfit from "
    "image 2, standing in the scene from image 3."
)

CSS = """
.gradio-container {max-width: 1480px !important; background: #0b1020;}
.hero {padding:24px;border-radius:20px;background:linear-gradient(135deg,#172554,#312e81);}
.hero h1 {margin: 0; font-size: 2.2rem;}
.panel {border: 1px solid #263454 !important; border-radius: 16px !important;}
.primary {background: linear-gradient(90deg,#4f46e5,#7c3aed) !important;}
.mode-hidden {display: none !important;}
"""

HIDDEN = ["mode-hidden"]


def _visibility(hidden: bool) -> Any:
    return gr.update(elem_classes=HIDDEN if hidden else [])


def _model_changed(key: str) -> tuple[Any, ...]:
    spec = MODEL_SPECS[key]
    choices = lora_choices(key)
    return (
        gr.update(value=spec.default_steps),
        gr.update(value=spec.default_guidance),
        gr.update(value=spec.default_true_cfg),
        f"**{spec.label}** · up to {spec.max_images} inputs · {spec.license_note}",
        library_status(key),
        *[gr.update(choices=choices, value=NONE_CHOICE) for _ in range(MAX_LORAS)],
    )


def _upload_loras(model_key: str, files: Any, *current_names: str) -> tuple[Any, ...]:
    saved = import_lora_files(model_key, files)
    names = assign_new_slots(list(current_names), saved)
    choices = lora_choices(model_key)
    return (
        library_status(model_key, saved),
        *[gr.update(choices=choices, value=name) for name in names],
    )


def _as_pil_images(value: Any) -> list[Image.Image]:
    if value is None:
        return []
    if isinstance(value, Image.Image):
        return [value]
    images: list[Image.Image] = []
    for item in value:
        if isinstance(item, Image.Image):
            images.append(item)
        elif isinstance(item, (tuple, list)) and item and isinstance(item[0], Image.Image):
            images.append(item[0])
    return images


def _collect_images(
    mode: str,
    source: Image.Image | None,
    references: Any,
    combine_1: Image.Image | None,
    combine_2: Image.Image | None,
    combine_3: Image.Image | None,
) -> list[Image.Image]:
    if mode == COMBINE_MODE:
        return [image for image in (combine_1, combine_2, combine_3) if image is not None]
    images = [source] if source is not None else []
    images.extend(_as_pil_images(references))
    return images


def _size_preview(
    mode: str,
    multiplier: int,
    resolution: str,
    source: Image.Image | None,
    combine_1: Image.Image | None,
    combine_2: Image.Image | None,
    combine_3: Image.Image | None,
) -> str:
    if mode == COMBINE_MODE:
        width, height = combine_output_size(str(resolution))
        return f"Output **{width} × {height}** · {resolution} canvas"
    images = _collect_images(mode, source, None, combine_1, combine_2, combine_3)
    if not images:
        return "Upload an image to lock output size to its aspect ratio."
    width, height = scaled_output_size(images[0].size, int(multiplier))
    src_w, src_h = images[0].size
    return (
        f"Output **{width} × {height}** · source {src_w} × {src_h} · "
        f"aspect preserved at ×{int(multiplier)}"
    )


def _edit_size_preview(multiplier: int, source: Image.Image | None) -> str:
    return _size_preview(EDIT_MODE, int(multiplier), "1K", source, None, None, None)


def _combine_size_preview(
    resolution: str,
    combine_1: Image.Image | None,
    combine_2: Image.Image | None,
    combine_3: Image.Image | None,
) -> str:
    return _size_preview(COMBINE_MODE, 1, str(resolution), None, combine_1, combine_2, combine_3)


def _mode_changed(mode: str) -> tuple[Any, ...]:
    is_edit = mode == EDIT_MODE
    preview = (
        "Upload an image to lock output size to its aspect ratio."
        if is_edit
        else _combine_size_preview("1K", None, None, None)
    )
    return (
        _visibility(hidden=not is_edit),
        _visibility(hidden=is_edit),
        gr.update(
            placeholder=EDIT_PLACEHOLDER if is_edit else COMBINE_PLACEHOLDER,
            label="Editing instruction" if is_edit else "Combination instruction",
        ),
        gr.update(value="Generate edit" if is_edit else "Generate combination"),
        _visibility(hidden=not is_edit),
        _visibility(hidden=is_edit),
        preview,
    )


def _run(
    mode: str,
    source: Image.Image | None,
    references: Any,
    combine_1: Image.Image | None,
    combine_2: Image.Image | None,
    combine_3: Image.Image | None,
    mask: Image.Image | None,
    model_key: str,
    prompt: str,
    negative_prompt: str,
    size_multiplier: int,
    output_resolution: str,
    steps: int,
    guidance: float,
    true_cfg: float,
    strength: float,
    seed: int,
    count: int,
    preserve_identity: bool,
    restoration: str,
    restore_weight: float,
    lora_name_1: str = NONE_CHOICE,
    lora_name_2: str = NONE_CHOICE,
    lora_name_3: str = NONE_CHOICE,
    lora_name_4: str = NONE_CHOICE,
    lora_name_5: str = NONE_CHOICE,
    lora_weight_1: float = 1.0,
    lora_weight_2: float = 1.0,
    lora_weight_3: float = 1.0,
    lora_weight_4: float = 1.0,
    lora_weight_5: float = 1.0,
) -> tuple[list[Image.Image], str, str]:
    if not prompt.strip():
        raise ValueError("Enter an instruction.")
    compose = mode == COMBINE_MODE
    request = GenerationRequest(
        model_key=model_key,
        prompt=prompt,
        negative_prompt=negative_prompt,
        images=_collect_images(mode, source, references, combine_1, combine_2, combine_3),
        mask=None if compose else mask,
        width=1024,
        height=1024,
        steps=int(steps),
        guidance=guidance,
        true_cfg=true_cfg,
        strength=strength,
        seed=int(seed),
        count=int(count),
        preserve_identity=preserve_identity,
        size_multiplier=int(size_multiplier),
        compose=compose,
        output_resolution=str(output_resolution),
        loras=selected_loras(
            model_key,
            [lora_name_1, lora_name_2, lora_name_3, lora_name_4, lora_name_5],
            [lora_weight_1, lora_weight_2, lora_weight_3, lora_weight_4, lora_weight_5],
        ),
    )
    result = generate(request, restoration, restore_weight)
    return result.images, metadata_text(result), model_manager.status


def build_app() -> gr.Blocks:
    model_choices = [(spec.label, key) for key, spec in MODEL_SPECS.items()]
    with gr.Blocks(title="Local Photo Edit Studio", css=CSS) as app:
        gr.HTML(
            "<section class='hero'><h1>Local Photo Edit Studio</h1>"
            "<p>Private H100 photo editing · lazy-loaded models · local outputs</p>"
            "</section>"
        )
        with gr.Row():
            with gr.Column(scale=5, elem_classes="panel"):
                mode = gr.Radio(
                    [EDIT_MODE, COMBINE_MODE],
                    value=EDIT_MODE,
                    label="Workflow",
                )
                with gr.Group() as edit_group:
                    with gr.Row():
                        source = gr.Image(type="pil", label="Source photo", height=390)
                        mask = gr.Image(
                            type="pil",
                            image_mode="L",
                            label="Optional edit mask (white = edit)",
                            height=390,
                        )
                    references = gr.Gallery(
                        label="Optional references · clean face crop, clothing, object, or scene",
                        type="pil",
                        columns=4,
                        rows=1,
                        height=180,
                    )
                with gr.Group(elem_classes=HIDDEN) as combine_group:
                    gr.Markdown(
                        "Upload 1–3 images and describe how to combine them into one new picture. "
                        "Refer to them as image 1, image 2, and image 3. Choose 1K, 2K, or 4K for the output canvas."
                    )
                    with gr.Row():
                        combine_1 = gr.Image(type="pil", label="Image 1", height=260)
                        combine_2 = gr.Image(type="pil", label="Image 2", height=260)
                        combine_3 = gr.Image(type="pil", label="Image 3", height=260)
                prompt = gr.Textbox(
                    label="Editing instruction",
                    placeholder=EDIT_PLACEHOLDER,
                    lines=4,
                )
                negative_prompt = gr.Textbox(
                    label="Negative prompt (model-dependent)",
                    placeholder="identity drift, altered facial structure, waxy skin, extra fingers",
                    lines=2,
                )
                with gr.Accordion("LoRAs (up to 5)", open=False):
                    lora_upload = gr.File(
                        label="Upload LoRA files for the selected model family",
                        file_count="multiple",
                        file_types=[".safetensors", ".pt", ".bin"],
                        type="filepath",
                    )
                    lora_status = gr.Markdown()
                    lora_names: list[gr.Dropdown] = []
                    lora_weights: list[gr.Slider] = []
                    for index in range(1, MAX_LORAS + 1):
                        with gr.Row():
                            lora_names.append(
                                gr.Dropdown(
                                    choices=[NONE_CHOICE],
                                    value=NONE_CHOICE,
                                    label=f"LoRA {index}",
                                    scale=3,
                                )
                            )
                            lora_weights.append(
                                gr.Slider(
                                    0,
                                    2,
                                    value=1.0,
                                    step=0.05,
                                    label=f"Weight {index}",
                                    scale=2,
                                )
                            )
                    gr.Markdown(
                        "LoRAs stay on disk under `models/loras/<family>/`. "
                        "Use adapters trained for the selected family; weights of 0 skip a slot."
                    )
                with gr.Row():
                    generate_button = gr.Button(
                        "Generate edit", variant="primary", elem_classes="primary"
                    )
                    clear_button = gr.ClearButton(value="Clear")
            with gr.Column(scale=3, elem_classes="panel"):
                model_key = gr.Dropdown(model_choices, value="qwen-2511", label="Model")
                model_info = gr.Markdown()
                size_multiplier = gr.Radio(
                    choices=[("×1", 1), ("×2", 2), ("×3", 3)],
                    value=1,
                    label="Size multiplier (keeps the uploaded image aspect ratio)",
                )
                output_resolution = gr.Radio(
                    choices=["1K", "2K", "4K"],
                    value="1K",
                    label="Output resolution",
                    elem_classes=HIDDEN,
                )
                size_info = gr.Markdown("Upload an image to lock output size to its aspect ratio.")
                steps = gr.Slider(1, 60, value=40, step=1, label="Inference steps")
                guidance = gr.Slider(0, 10, value=1.0, step=0.1, label="Guidance / CFG")
                true_cfg = gr.Slider(0, 10, value=4.0, step=0.1, label="True CFG (Qwen/FireRed)")
                strength = gr.Slider(
                    0.05,
                    1.0,
                    value=0.8,
                    step=0.05,
                    label="Edit strength (when supported)",
                )
                with gr.Row():
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
                    count = gr.Slider(1, 4, value=1, step=1, label="Outputs")
                preserve_identity = gr.Checkbox(
                    value=True, label="Identity-preservation prompt"
                )
                with gr.Accordion("Face restoration", open=False):
                    restoration = gr.Radio(["Off", "GFPGAN"], value="Off", label="Post-process")
                    restore_weight = gr.Slider(
                        0, 1, value=0.35, step=0.05, label="Restoration fidelity weight"
                    )
                    gr.Markdown(
                        "Use sparingly: restoration can improve micro-details but may also "
                        "shift identity."
                    )
                unload_button = gr.Button("Unload model / release VRAM")
                status = gr.Markdown("No model loaded")
        output = gr.Gallery(label="Results", columns=2, object_fit="contain", height="auto")
        metadata = gr.Code(label="Run metadata (prompts are not stored)", language="json")

        inputs = [
            mode,
            source,
            references,
            combine_1,
            combine_2,
            combine_3,
            mask,
            model_key,
            prompt,
            negative_prompt,
            size_multiplier,
            output_resolution,
            steps,
            guidance,
            true_cfg,
            strength,
            seed,
            count,
            preserve_identity,
            restoration,
            restore_weight,
            *lora_names,
            *lora_weights,
        ]
        generate_button.click(_run, inputs=inputs, outputs=[output, metadata, status])
        model_outputs = [steps, guidance, true_cfg, model_info, lora_status, *lora_names]
        model_key.change(_model_changed, inputs=model_key, outputs=model_outputs)
        mode.change(
            _mode_changed,
            inputs=mode,
            outputs=[
                edit_group,
                combine_group,
                prompt,
                generate_button,
                size_multiplier,
                output_resolution,
                size_info,
            ],
        )
        size_multiplier.change(
            _edit_size_preview, inputs=[size_multiplier, source], outputs=size_info
        )
        source.change(_edit_size_preview, inputs=[size_multiplier, source], outputs=size_info)
        combine_size_inputs = [output_resolution, combine_1, combine_2, combine_3]
        for control in combine_size_inputs:
            control.change(
                _combine_size_preview, inputs=combine_size_inputs, outputs=size_info
            )
        lora_upload.upload(
            _upload_loras,
            inputs=[model_key, lora_upload, *lora_names],
            outputs=[lora_status, *lora_names],
        )
        unload_button.click(model_manager.unload, outputs=status)
        clear_button.add(
            [
                source,
                references,
                combine_1,
                combine_2,
                combine_3,
                mask,
                prompt,
                negative_prompt,
                output,
                metadata,
            ]
        )
        app.load(_model_changed, inputs=model_key, outputs=model_outputs)
    return app
