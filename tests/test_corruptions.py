import pytest
from PIL import Image

from doctrust.corruptions import (
    apply_variant,
    box_to_pixels,
    distractor_box,
    evidence_occlusion,
    gaussian_blur,
    jpeg_compression,
)


def test_box_to_pixels() -> None:
    assert box_to_pixels((0.1, 0.2, 0.5, 0.6), (100, 200)) == (10, 40, 50, 120)


def test_transforms_preserve_size() -> None:
    image = Image.new("RGB", (100, 80), "white")
    box = (0.2, 0.2, 0.4, 0.4)
    assert gaussian_blur(image, 1.5).size == image.size
    assert jpeg_compression(image, 35).size == image.size
    assert evidence_occlusion(image, box).size == image.size


def test_distractor_box_preserves_dimensions() -> None:
    original = (0.6, 0.7, 0.8, 0.8)
    moved = distractor_box(original)
    assert round(moved[2] - moved[0], 6) == round(original[2] - original[0], 6)
    assert round(moved[3] - moved[1], 6) == round(original[3] - original[1], 6)


def test_non_clean_variant_cannot_silently_become_clean() -> None:
    image = Image.new("RGB", (100, 80), "white")
    with pytest.raises(ValueError, match="missing a supported transform"):
        apply_variant(
            image,
            (0.2, 0.2, 0.4, 0.4),
            {"name": "evidence_occlusion", "mode": "evidence_occlusion"},
        )
