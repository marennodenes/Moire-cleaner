import numpy as np

from moire.patch_processing import (
    divide_into_patches,
    merge_patches,
)


def test_bgr_patch_reconstruction():
    rng = np.random.default_rng(42)

    # Create a synthetic BGR image.
    image = rng.integers(
        0, 256, size=(128, 128, 3), dtype=np.uint8
    )

    original = image.copy()

    # Create overlapping patches.
    patches = divide_into_patches(
        image, patch_size=32, stride=16
    )

    assert len(patches) == 49

    # Reconstruct using both merging methods.
    for mode in ("uniform", "triangular"):
        restored = merge_patches(
            patches,
            image.shape,
            weight_mode=mode
        )

        assert restored.shape == image.shape
        assert np.all(np.isfinite(restored))
        assert np.allclose(restored, image, atol=1e-10)

        error = np.max(
            np.abs(restored - image.astype(np.float64))
        )

        print(f"{mode}: max reconstruction error = {error}")

    # Verify that patch extraction preserved the original.
    assert np.array_equal(image, original)

    print("BGR Patch Reconstruction Test passed!")


if __name__ == "__main__":
    test_bgr_patch_reconstruction()
