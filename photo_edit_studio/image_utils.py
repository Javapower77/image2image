from __future__ import annotations

from PIL import Image, ImageFilter, ImageOps


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
