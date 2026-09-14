# Changelog

All notable changes are documented here.

## [0.4.3] - 2026-09-14

### Fixed

- Changing a LoRA weight no longer crashes with `Setting requires_grad=True on inference tensor`. Adapter weights are applied in inference mode, and generation uses `torch.no_grad()` instead of `torch.inference_mode()`.

## [0.4.2] - 2026-09-14

### Fixed

- Combine-images size preview no longer crashes with `needed: 7, got: 6` when some image slots are empty. Hidden Gradio widgets are not used as event inputs.

## [0.4.1] - 2026-09-14

### Changed

- Combine-images output size is now a 1K / 2K / 4K canvas (1024, 2048, or 4096 square). Edit source still uses ×1 / ×2 / ×3 from the source aspect ratio.

## [0.4.0] - 2026-09-14

### Added

- Output size now follows the first uploaded image's aspect ratio. Choose ×1, ×2, or ×3 instead of independent width/height sliders (clamped to 512–2048 and aligned to 64 px).
- Combine-images workflow: upload up to three references and a prompt to generate one new picture. Image 1 still sets the aspect ratio.

## [0.3.2] - 2026-09-12

### Fixed

- Rapid AIO load no longer builds the transformer on the meta device, which caused `Cannot copy out of meta tensor; no data!` when moving RoPE caches or enabling CPU offload.

## [0.3.1] - 2026-09-12

### Fixed

- CUDA out-of-memory on Qwen 2511 / Rapid AIO: model CPU offload, VAE tiling, sequential batch images, allocator `expandable_segments`, and Rapid AIO no longer loads two transformers.

## [0.3.0] - 2026-09-12

### Added

- Qwen Image Edit 2511 Rapid AIO as a selectable model. It loads the distilled ComfyUI transformer from `models/repos/Qwen--Qwen-Image-Edit-2511-AIO` onto the official 2511 pipeline.

## [0.2.0] - 2026-09-12

### Added

- Local LoRA library with upload, up to five selected adapters per run, and per-LoRA weights.
- Qwen/FireRed and FLUX.2 Klein family LoRA folders under `models/loras/`.

## [0.1.1] - 2026-09-12

### Changed

- Removed extra application-level content filters and the permission checkbox.
- Left model-native and license-required safeguards enabled.

## [0.1.0] - 2026-09-12

### Added

- Local Gradio photo editing studio with a modern two-panel interface.
- Lazy, one-at-a-time adapters for Qwen Image Edit 2511, FireRed Image Edit 1.1, and FLUX.2 [klein] 4B/9B.
- Multi-reference inputs, identity-preservation prompting, seed, dimensions, steps, CFG, true CFG, strength, batching, and optional mask compositing.
- Optional lazy-loaded GFPGAN post-processing.
- Project-local snapshot downloader and strictly local runtime loading.
- Python 3.11/H100 setup and run scripts.
- Privacy-conscious metadata without prompt or input persistence.
- Setup, model, architecture, usage, license/safety, and troubleshooting documentation.
