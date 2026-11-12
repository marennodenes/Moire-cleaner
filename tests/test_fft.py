import numpy as np

from moire.fft import apply_fft, apply_ifft, get_magnitude_spectrum


def test_fft_roundtrip_preserves_shape_and_values():
    image = np.arange(64, dtype=np.float32).reshape(8, 8)

    fft = apply_fft(image)
    restored = apply_ifft(fft)

    assert fft.shape == image.shape
    assert restored.shape == image.shape
    assert np.isfinite(restored).all()
    assert np.allclose(restored, image, atol=1e-5)


def test_magnitude_spectrum_is_finite_and_same_shape():
    image = np.ones((8, 8), dtype=np.float32)

    magnitude = get_magnitude_spectrum(apply_fft(image))

    assert magnitude.shape == image.shape
    assert np.isfinite(magnitude).all()
