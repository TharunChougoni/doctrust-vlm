from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw, ImageFilter

Box = tuple[float, float, float, float]


def box_to_pixels(box: Box, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Convert a normalized box to a clipped integer pixel box."""
    width, height = size
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, round(x1 * width))),
        max(0, min(height - 1, round(y1 * height))),
        max(1, min(width, round(x2 * width))),
        max(1, min(height, round(y2 * height))),
    )


def jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    if not 1 <= quality <= 95:
        raise ValueError("JPEG quality must be between 1 and 95")
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").copy()


def gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    if radius < 0:
        raise ValueError("Blur radius cannot be negative")
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))


def evidence_occlusion(image: Image.Image, box: Box, fill: int = 128) -> Image.Image:
    if not 0 <= fill <= 255:
        raise ValueError("fill must be between 0 and 255")
    result = image.convert("RGB").copy()
    ImageDraw.Draw(result).rectangle(box_to_pixels(box, result.size), fill=(fill,) * 3)
    return result


def distractor_box(box: Box) -> Box:
    """Move a box to the opposite page quadrant while preserving its size."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    target_x1 = 0.05 if x1 >= 0.5 else 0.95 - width
    target_y1 = 0.05 if y1 >= 0.5 else 0.95 - height
    return (target_x1, target_y1, target_x1 + width, target_y1 + height)


def distractor_occlusion(image: Image.Image, box: Box, fill: int = 128) -> Image.Image:
    return evidence_occlusion(image, distractor_box(box), fill=fill)


def apply_variant(image: Image.Image, box: Box, spec: dict) -> Image.Image:
    """Dispatch one declarative corruption specification."""
    transform = spec.get("transform")
    if transform is None:
        return image.convert("RGB").copy()
    if transform == "jpeg":
        return jpeg_compression(image, quality=int(spec["quality"]))
    if transform == "gaussian_blur":
        return gaussian_blur(image, radius=float(spec["radius"]))
    if transform == "evidence_occlusion":
        return evidence_occlusion(image, box, fill=int(spec.get("fill", 128)))
    if transform == "distractor_occlusion":
        return distractor_occlusion(image, box, fill=int(spec.get("fill", 128)))
    raise ValueError(f"Unknown transform: {transform}")
