import numpy as np

from moire.fft import apply_fft, apply_ifft
from moire.notch import apply_notch_filter
from moire.datatypes import Peak


def test_bgr_notch_integration():
    size = 128

    x = np.arange(size)
    xx, yy = np.meshgrid(x, x)

    # Create three channels with the same periodic
    # frequency but different amplitudes.
    b = 100 + 30 * np.cos(2 * np.pi * 8 * xx / size)
    g = 120 + 40 * np.cos(2 * np.pi * 8 * xx / size)
    r = 140 + 50 * np.cos(2 * np.pi * 8 * xx / size)

    image_bgr = np.stack([b, g, r], axis=2)

    peaks = [
        Peak(fy=0.0, fx=8 / size, ratio=10.0)
    ]

    filtered_channels = []

    for channel_idx in range(3):
        channel = image_bgr[:, :, channel_idx]

        fft_image = apply_fft(channel)

        filtered_fft = apply_notch_filter(
            fft_image,
            peaks,
            sigma=2.0
        )

        filtered_channel = apply_ifft(filtered_fft)
        filtered_channels.append(filtered_channel)

    filtered_bgr = np.stack(filtered_channels, axis=2)

    assert filtered_bgr.shape == image_bgr.shape
    assert np.all(np.isfinite(filtered_bgr))

    for channel_idx in range(3):
        original_std = np.std(image_bgr[:, :, channel_idx])
        filtered_std = np.std(filtered_bgr[:, :, channel_idx])

        assert filtered_std < original_std

    print("BGR Notch Integration Test passed!")


if __name__ == "__main__":
    test_bgr_notch_integration()
