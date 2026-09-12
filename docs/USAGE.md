# Editing guide

## Identity-preserving edits

Use a high-resolution source without heavy compression. Add a sharp, front-facing face crop as image 2 when the source face is small. Keep identity preservation enabled and phrase the request as a constrained delta: state the desired change, then list face, pose, body proportions, camera, lighting, and background elements that must stay unchanged.

Avoid asking for many unrelated changes in one run. Make one structural edit first, reuse its output as a new source, and then make a smaller refinement.

## Clothing changes

Use the person as image 1 and a clean product or worn-garment photo as image 2. Mention fabric, cut, sleeve length, closures, logos, and which image supplies the garment. Ask to preserve body proportions, pose, hands, face, and scene. Clothing replacement is generative, not measurement-accurate virtual fitting.

## Background and scene edits

Describe perspective, time of day, light direction, depth of field, and desired contact shadows. Explicitly preserve the subject, foreground edges, and camera framing. A white mask can constrain final compositing, but it does not currently condition denoising.

## LoRAs

Upload `.safetensors`, `.pt`, or `.bin` adapters in the **LoRAs** panel. Files are copied into `models/loras/<family>/` and stay on disk for later sessions. Qwen Image Edit, Rapid AIO, and FireRed share one library; FLUX.2 [klein] 4B and 9B share another.

Choose up to five existing files and set each weight (typically 0.6–1.2; 0 skips the slot). LoRAs must match the selected family: a FLUX.2 Klein adapter will not load on Qwen, and a Qwen Image Edit adapter will not load on Klein. Changing the base model reloads that family’s library.

LoRA names and weights are stored in run metadata. Prompt text is still not stored.

## Restoration

Generate without restoration first. If eyes, mouth, or pores are weak, enable GFPGAN at about 0.2–0.4. Higher values can produce smooth skin or identity drift. A future FaceDetailer-style implementation should use local detection, a padded face crop, low-denoise regeneration, and feathered paste-back; it is not falsely represented as available in this release.

## Reproducibility

Set a fixed non-negative seed. Metadata is saved beside outputs. Prompt text is deliberately excluded for privacy, so retain it separately if exact reruns matter.
