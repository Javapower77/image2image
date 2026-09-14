# Models and tuning

| UI model | Local repository | Typical steps | CFG | Inputs | Notes |
|---|---|---:|---:|---:|---|
| Qwen Image Edit 2511 | `Qwen/Qwen-Image-Edit-2511` | 40 | guidance 1, true CFG 4 | up to 3 in this app | Strong character consistency and practical edits. |
| Qwen Image Edit 2511 Rapid AIO | `Qwen/Qwen-Image-Edit-2511-AIO` | 4 | guidance 1, true CFG 1 | up to 3 in this app | Distilled ComfyUI Rapid AIO transformer on the official 2511 text encoder/VAE. |
| FireRed Image Edit 1.1 | `FireRedTeam/FireRed-Image-Edit-1.1` | 40 | guidance 1, true CFG 4 | 1–3 native | Strong portrait consistency and multi-element fusion. |
| FLUX.2 [klein] 4B | `black-forest-labs/FLUX.2-klein-4B` | 4 | 1 | multi-reference | Fast distilled Apache-2.0 model. |
| FLUX.2 [klein] 9B | `black-forest-labs/FLUX.2-klein-9B` | 4 | 1 | multi-reference | Gated, non-commercial, roughly 29 GB VRAM. |

All models load in BF16 and from local Safetensors snapshots. Pipelines use model CPU offload and VAE tiling so the text encoder is not kept on the GPU during denoising. The current Diffusers source build is required because these pipeline classes are newer than many stable releases.

Rapid AIO is a single ComfyUI checkpoint (`Qwen-Rapid-AIO.safetensors`). The studio keeps the official Qwen 2511 tokenizer, text encoder, VAE, and scheduler, and swaps in the AIO transformer. Download it with `python scripts/download_models.py qwen-2511 qwen-2511-aio` (the official 2511 snapshot is required). A 100 KB HTML file in that folder is a failed Hugging Face page save, not weights. Defaults are 4 steps because the merge is Lightning-distilled; raising steps toward 8 is sometimes useful, but 40-step Qwen settings are the wrong starting point.

## Reference ordering

In **Edit source**, image 1 is the photo being changed. Image 2 should be the highest-priority reference—usually a clean face crop for identity-critical work or the target garment for virtual try-on. Later images can supply additional objects or environment cues. In **Combine images**, all three slots are references for a new picture; still refer to them as “image 1”, “image 2”, and “image 3”. Edit output follows image 1's aspect ratio; combine output uses the selected 1K/2K/4K canvas.

## Parameter behavior

The adapter inspects the active pipeline signature and sends only supported parameters. Therefore controls remain stable across upstream pipeline variations. Qwen-family models use `true_cfg_scale`; Rapid AIO and Klein distilled defaults are intentionally low. Increasing those distilled models far above four to eight steps generally adds latency without guaranteed quality improvement.

## LoRAs

The studio applies up to five local LoRAs through Diffusers (`load_lora_weights` + `set_adapters`) on the currently loaded pipeline. Libraries are family-scoped:

| Family | Models | Directory |
|---|---|---|
| `qwen` | Qwen Image Edit 2511, Rapid AIO, FireRed Image Edit 1.1 | `models/loras/qwen/` |
| `flux` | FLUX.2 [klein] 4B and 9B | `models/loras/flux/` |

Use adapters trained for that architecture. Qwen-family LoRAs target `QwenImageTransformer2DModel`; Klein LoRAs target `Flux2Transformer2DModel` (not FLUX.1). PEFT is required.

## Local-only FireRed behavior

The upstream FireRed agent can preprocess more than three images and optionally recaption through hosted LLM APIs. This project limits FireRed to three direct inputs and does not integrate that agent, preserving the no-cloud requirement.

## License checks

Model terms can change. Review the license files stored in each downloaded snapshot. In particular, do not use Klein 9B commercially without an appropriate license, and retain its required filtering/manual-review behavior.
