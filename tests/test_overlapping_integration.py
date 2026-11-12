import numpy as np

from moire.adaptive import process_overlapping_patches
from moire.datatypes import Peak, PatchResult


def test_overlapping_integration():
    height, width = 128, 192

    yy, xx = np.mgrid[:height, :width]
    pattern = np.cos(2 * np.pi * 8 * xx / 128)

    image = np.stack([
        100 + 30 * pattern,
        120 + 40 * pattern,
        140 + 50 * pattern
    ], axis=2)

    # A simulated detection region.
    detections = [
        PatchResult(
            patch_id=0,
            y=0,
            x=0,
            size=128,
            peaks=[Peak(0.0, 8 / 128, 10.0)],
            intensity=0.8
        )
    ]

    filtered = process_overlapping_patches(
        image,
        detections,
        patch_size=128,
        stride=64,
        sigma=2.0,
        weight_mode="triangular"
    )

    assert filtered.shape == image.shape
    assert np.all(np.isfinite(filtered))

    # The periodic component should decrease.
    for channel in range(3):
        assert (
            np.std(filtered[:, :, channel])
            < np.std(image[:, :, channel])
        )

    # No detection: original image must be preserved.
    unchanged = process_overlapping_patches(
        image,
        [],
        patch_size=128,
        stride=64
    )

    assert np.allclose(unchanged, image, atol=1e-10)

    print("Overlapping Integration Test passed!")
