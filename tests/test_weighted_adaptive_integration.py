
import numpy as np
import pytest

from moire.adaptive import process_overlapping_patches
from moire.datatypes import Peak, PatchResult


def test_weighted_adaptive_integration():
    size = 128
    yy, xx = np.mgrid[:size, :size]

    pattern = np.cos(2 * np.pi * 8 * xx / size)

    image = np.stack([
        100 + 30 * pattern,
        120 + 40 * pattern,
        140 + 50 * pattern,
    ], axis=2)

    detections = [
        PatchResult(
            patch_id=0,
            y=0,
            x=0,
            size=128,
            peaks=[Peak(0.0, 8 / size, 10.0)],
            intensity=0.8,
        ),
        PatchResult(
            patch_id=1,
            y=0,
            x=128,
            size=128,
            peaks=[],
            intensity=0.2,
        ),
    ]

    # Both modes select the first detection region's peaks.
    best = process_overlapping_patches(
        image,
        detections,
        patch_size=128,
        stride=64,
        intensity_mode="best",
    )

    weighted = process_overlapping_patches(
        image,
        detections,
        patch_size=128,
        stride=64,
        intensity_mode="weighted",
    )

    # The image overlaps only the first detection region.
    # Therefore, weighted and best intensity are identical.
    assert np.allclose(best, weighted, atol=1e-10)

    assert best.shape == image.shape
    assert np.all(np.isfinite(best))
    assert np.all(np.isfinite(weighted))

    # Confirm that the original detection objects were not modified.
    assert detections[0].intensity == pytest.approx(0.8)
    assert detections[1].intensity == pytest.approx(0.2)


def test_invalid_intensity_mode():
    image = np.zeros((128, 128, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        process_overlapping_patches(
            image,
            [],
            patch_size=128,
            stride=64,
            intensity_mode="invalid",
        )
