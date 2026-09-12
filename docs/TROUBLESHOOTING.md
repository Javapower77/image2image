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

Click **Unload model**, keep queue concurrency at one, reduce output dimensions/count, and close other GPU workloads. H100 80 GB should run each listed BF16 model independently, but loading two pipelines at once is intentionally prevented.

## Identity changes

Use a cleaner face crop as the first reference, lower requested edit scope, explicitly preserve facial geometry and skin texture, and reuse the same seed. Turn restoration off while diagnosing. No generic model guarantees biometric identity.

## GFPGAN errors

Install the restoration extra and download the checkpoint with `--restorers`. Some modern Torch versions expose compatibility problems in BasicSR; keep restoration optional rather than blocking core generation.

## Remote browser cannot connect

The app binds to `127.0.0.1`. Use SSH forwarding. If intentionally exposing it, configure authentication and the Azure NSG first.
