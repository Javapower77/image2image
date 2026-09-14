from __future__ import annotations

from PIL import Image, ImageFilter, ImageOps

MIN_OUTPUT_SIDE = 512
MAX_OUTPUT_SIDE = 2048
OUTPUT_ALIGN = 64
SIZE_MULTIPLIERS = (1, 2, 3)
COMBINE_RESOLUTIONS = {
    "1K": (1024, 1024),
    "2K": (2048, 2048),
    "4K": (4096, 4096),
}


def combine_output_size(resolution: str) -> tuple[int, int]:
    try:
        return COMBINE_RESOLUTIONS[resolution]
    except KeyError as exc:
        raise ValueError(
            f"Combine resolution must be one of {tuple(COMBINE_RESOLUTIONS)}."
        ) from exc


def scaled_output_size(
    size: tuple[int, int],
    multiplier: int,
    *,
    min_side: int = MIN_OUTPUT_SIDE,
    max_side: int = MAX_OUTPUT_SIDE,
    align: int = OUTPUT_ALIGN,
) -> tuple[int, int]:
    """Scale an image size by ×1/×2/×3 while keeping aspect ratio within model limits."""
    if multiplier not in SIZE_MULTIPLIERS:
        raise ValueError(f"Size multiplier must be one of {SIZE_MULTIPLIERS}.")
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Image has invalid dimensions.")
    width *= multiplier
    height *= multiplier
    longest = max(width, height)
    if longest > max_side:
        scale = max_side / longest
        width *= scale
        height *= scale
    longest = max(width, height)
    if longest < min_side:
        scale = min_side / longest
        width *= scale
        height *= scale
    longest = max(width, height)
    if longest > max_side:
        scale = max_side / longest
        width *= scale
        height *= scale
    width = int(max(align, round(width / align) * align))
    height = int(max(align, round(height / align) * align))
    width = min(max(align, (width // align) * align), max_side)
    height = min(max(align, (height // align) * align), max_side)
    return width, height


def normalize_image(image: Image.Image, max_side: int = 2048) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def fit_reference(image: Image.Image, width: int, height: int) -> Image.Image:
    return ImageOps.contain(normalize_image(image), (width, height), Image.Resampling.LANCZOS)


def composite_with_mask(
    source: Image.Image, generated: Image.Image, mask: Image.Image, feather: int = 16
) -> Image.Image:
    target_size = generated.size
    source = ImageOps.fit(normalize_image(source), target_size, Image.Resampling.LANCZOS)
    mask = ImageOps.fit(mask.convert("L"), target_size, Image.Resampling.LANCZOS)
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return Image.composite(generated.convert("RGB"), source, mask)
