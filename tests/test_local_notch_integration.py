import numpy as np

from moire.fft import apply_fft, apply_ifft
from moire.notch import apply_notch_filter
from moire.datatypes import Peak


def test_local_notch_integration():
    size = 128

    # Create a synthetic periodic pattern.
    x = np.arange(size)
    xx, yy = np.meshgrid(x, x)

    image = 128 + 40 * np.cos(
        2 * np.pi * 8 * xx / size
    )

    # Step 1: Apply FFT.
    fft_image = apply_fft(image)

    # Step 2: Define the known frequency peak.
    peaks = [
        Peak(fy=0.0, fx=8 / size, ratio=10.0)
    ]

    # Step 3: Apply Gaussian Notch Filter.
    filtered_fft = apply_notch_filter(
        fft_image,
        peaks,
        sigma=2.0
    )

    # Step 4: Apply IFFT.
    filtered_image = apply_ifft(filtered_fft)

    # Step 5: Verify the result.
    assert filtered_image.shape == image.shape
    assert np.all(np.isfinite(filtered_image))

    original_std = np.std(image)
    filtered_std = np.std(filtered_image)

    assert filtered_std < original_std

    print("Local Notch Integration Test passed!")
    print("Original std:", original_std)
    print("Filtered std:", filtered_std)


if __name__ == "__main__":
    test_local_notch_integration()
