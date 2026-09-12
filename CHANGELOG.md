# Changelog

All notable changes are documented here.

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
