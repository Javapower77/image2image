# Local Photo Edit Studio

A private, local Gradio studio for high-quality instruction-based photo editing on an NVIDIA H100. It supports Qwen Image Edit 2511, Qwen Rapid AIO, FireRed Image Edit 1.1, and FLUX.2 [klein] 4B/9B through lazy-loaded Diffusers adapters.

## Features

- Background, clothing, object, scene, style, and localized edits.
- Source image plus multiple face/clothing/scene references.
- Combine up to three images with a prompt into one new picture at 1K, 2K, or 4K.
- Edit output size follows the source aspect ratio via ×1 / ×2 / ×3.
- Identity-preservation prompt mode, deterministic seeds, CFG, true CFG, steps, strength, masks, and batches.
- Optional local GFPGAN post-processing.
- Upload and apply up to five local LoRAs on the selected base model, each with its own weight.
- One model in GPU memory at a time; explicit unload control.
- Local model cache and output storage. Prompts and source images are not written to logs or metadata.
- No cloud inference, API inference, telemetry, or share link by default.

## Requirements

- Linux Azure VM with an NVIDIA H100 and a recent NVIDIA driver.
- Python 3.11, Git, and at least 200 GB free disk if downloading every model.
- Hugging Face account/token for gated repositories. FLUX.2 [klein] 9B requires accepting its model terms and is non-commercial.

## Quick start

1. Run `bash scripts/setup.sh`.
2. Activate: `source .venv/bin/activate`.
3. Authenticate only if needed: `huggingface-cli login` (the token remains in the local Hugging Face cache).
4. Download selected models, for example: `python scripts/download_models.py qwen-2511 flux-klein-4b --restorers`.
5. Optionally install GFPGAN support: `pip install -e '.[restore]'`.
6. Start with `bash scripts/run.sh`.
7. From the VM itself, open `http://127.0.0.1:7860`. From a workstation, use SSH port forwarding: `ssh -L 7860:127.0.0.1:7860 USER@VM`.

To download every registered snapshot, run `python scripts/download_models.py`. Rapid AIO also needs `qwen-2511`. Once downloaded, use `HF_HUB_OFFLINE=1 bash scripts/run.sh` for strict offline operation.

## Recommended editing workflow

1. Choose **Edit source** or **Combine images**.
2. For edits, put the unmodified photo in **Source photo**. For combinations, upload up to three images and refer to them as image 1, 2, and 3.
3. Optionally add a tight, clean face crop first in **References**, followed by clothing/object references.
4. For edits, pick ×1, ×2, or ×3 to keep the source aspect ratio (max 2048 px). For combinations, pick 1K, 2K, or 4K.
5. Describe only the intended change and explicitly name what must remain unchanged.
6. Start with the model defaults. Qwen/FireRed generally use 40 steps and true CFG 4; Rapid AIO and distilled FLUX Klein use 4 steps and guidance 1.
7. Reuse the same seed while tuning the prompt.
8. Optionally upload LoRAs for the selected model family, choose up to five, and set each weight.
9. Apply GFPGAN only when facial micro-detail needs correction; low weights reduce identity drift.

Example instruction:

> Replace only the jacket with the garment shown in image 2. Preserve the person’s exact face, body proportions, pose, hands, hair, lighting, camera angle, background, and skin texture.

## Important limitations

- Generic image-editing models cannot mathematically guarantee identity. A clean face crop and constrained prompt usually improve consistency.
- The mask is composited after generation in this release; it is not native latent inpainting.
- `strength`, dimensions, negative prompt, and batch count are passed only when supported by the selected pipeline API.
- FireRed supports 1–3 native inputs. This project intentionally does not use FireRed’s cloud-based recaption agent.
- Model APIs are recent and require the current Diffusers source build.
- GFPGAN can beautify or alter identity. Compare restored and original outputs.
- LoRAs must match the selected model family. A FLUX.1 adapter will not load on FLUX.2 Klein, and Qwen adapters will not load on Klein.

## Privacy and licenses

This app is local and has no extra application-level content filters. It does not log prompts or images. Local use still does not waive laws, consent requirements, or model licenses. Model-native safeguards and license-required filters (especially FLUX.2 [klein] 9B) are left intact and are not disabled.

FLUX.2 [klein] 9B is gated and non-commercial. Qwen 2511, Rapid AIO, FireRed 1.1, and FLUX.2 [klein] 4B identify Apache-2.0 model licenses; verify current upstream terms before deployment.

See `docs/SETUP.md`, `docs/MODELS.md`, `docs/ARCHITECTURE.md`, `docs/USAGE.md`, and `docs/TROUBLESHOOTING.md`.
