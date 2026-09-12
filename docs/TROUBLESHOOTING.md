# Troubleshooting

## Pipeline class unavailable

Run `pip install --upgrade --force-reinstall 'diffusers @ git+https://github.com/huggingface/diffusers.git'`. QwenImageEditPlusPipeline and Flux2KleinPipeline are recent.

## Local model not found

Run `python scripts/download_models.py MODEL_KEY`. Runtime loading uses `local_files_only=True` and never downloads implicitly.

## Rapid AIO file is HTML or tiny

`Qwen-Rapid-AIO.safetensors` must be a real ~28 GB checkpoint. A Hugging Face webpage saved under that name will fail. Delete it and run `python scripts/download_models.py qwen-2511 qwen-2511-aio`. The official Qwen 2511 snapshot must already be present.

## Gated repository error

Accept the model terms on Hugging Face, authenticate locally, and retry. Klein 9B is gated and non-commercial.

## LoRA failed to load

Confirm the file is a LoRA for the selected family (Qwen/FireRed vs FLUX.2 Klein), not a full checkpoint and not a FLUX.1 adapter on Klein. Re-upload the `.safetensors` file, keep weights near 1.0, and unload/reload the base model if a previous adapter failed mid-load.

## CUDA out of memory

Qwen 2511 and Rapid AIO keep a large transformer in VRAM during denoising. The studio now offloads the text encoder and VAE when they are idle, tiles VAE encode/decode, and generates batch images one at a time.

If you still see `torch.OutOfMemoryError`:

1. Click **Unload model** and retry.
2. Keep **count** at 1 and start at 1024×1024 or smaller.
3. Close other GPU processes (`nvidia-smi`).
4. Restart the app so the CUDA allocator can use `expandable_segments`.
5. Rapid AIO must not load the official 2511 transformer at the same time; restart after updating if an older process is still running.

## Identity changes

Use a cleaner face crop as the first reference, lower requested edit scope, explicitly preserve facial geometry and skin texture, and reuse the same seed. Turn restoration off while diagnosing. No generic model guarantees biometric identity.

## GFPGAN errors

Install the restoration extra and download the checkpoint with `--restorers`. Some modern Torch versions expose compatibility problems in BasicSR; keep restoration optional rather than blocking core generation.

## Remote browser cannot connect

The app binds to `127.0.0.1`. Use SSH forwarding. If intentionally exposing it, configure authentication and the Azure NSG first.
