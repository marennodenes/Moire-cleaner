import numpy as np

from moire.adaptive import filter_patch_with_detection
from moire.datatypes import Peak, PatchResult


def test_adaptive_notch():
    size = 128

    x = np.arange(size)
    xx, yy = np.meshgrid(x, x)

    pattern = np.cos(2 * np.pi * 8 * xx / size)

    image_bgr = np.stack([
        100 + 30 * pattern,
        120 + 40 * pattern,
        140 + 50 * pattern,
    ], axis=2)

    # Simulated Member B detection result.
    detection = PatchResult(
        patch_id=0,
        y=0,
        x=0,
        size=size,
        peaks=[Peak(fy=0.0, fx=8 / size, ratio=10.0)],
        intensity=0.8,
    )

    # Filtering enabled.
    filtered = filter_patch_with_detection(
        image_bgr,
        detection,
        sigma=2.0
    )

    assert filtered.shape == image_bgr.shape
    assert np.all(np.isfinite(filtered))

    for channel in range(3):
        assert (
            np.std(filtered[:, :, channel])
            < np.std(image_bgr[:, :, channel])
        )

    # Filtering disabled by low intensity.
    detection.intensity = 0.1

    unchanged = filter_patch_with_detection(
        image_bgr,
        detection
    )

    assert np.allclose(unchanged, image_bgr)

    # No detection result.
    no_detection = filter_patch_with_detection(
        image_bgr,
        None
    )

    assert np.allclose(no_detection, image_bgr)

    print("Adaptive Notch Filtering Test passed!")
