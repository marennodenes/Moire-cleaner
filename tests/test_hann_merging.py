import numpy as np

from moire.patch_processing import (
    create_hann_weights,
    divide_into_patches,
    merge_patches,
)


def test_hann_merging():
    rng = np.random.default_rng(42)

    image = rng.integers(
        0, 256, size=(128, 160, 3), dtype=np.uint8
    )

    patches = divide_into_patches(
        image, patch_size=32, stride=16
    )

    weights = create_hann_weights(32)

    assert weights.shape == (32, 32)
    assert np.all(weights > 0)
    assert np.allclose(weights, weights[::-1, ::-1])

    restored = merge_patches(
        patches, image.shape, weight_mode="hann"
    )

    assert restored.shape == image.shape
    assert np.all(np.isfinite(restored))
    assert np.allclose(restored, image, atol=1e-10)

    # Also verify grayscale reconstruction.
    grayscale = image[:, :, 0]
    gray_patches = divide_into_patches(
        grayscale, patch_size=32, stride=16
    )

    gray_restored = merge_patches(
        gray_patches, grayscale.shape, weight_mode="hann"
    )

    assert np.allclose(gray_restored, grayscale, atol=1e-10)

    print("Hann Merging Test passed!")
