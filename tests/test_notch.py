import numpy as np
import pytest

from moire.notch import (
    apply_notch_filter,
    gaussian_notch,
    peak_to_coordinates,
)


class DummyPeak:
    def __init__(self, fy, fx):
        self.fy = fy
        self.fx = fx


def test_peak_to_coordinates_returns_mirror_pair():
    peak = DummyPeak(0.125, 0.25)

    p1, p2 = peak_to_coordinates((64, 64), peak)

    assert p1 == (40, 48)
    assert p2 == (24, 16)


def test_gaussian_notch_is_zero_at_center_and_increases_outward():
    mask = gaussian_notch((64, 64), (40, 48), sigma=5)

    assert mask.shape == (64, 64)
    assert mask[40, 48] == pytest.approx(0.0)
    assert mask[32, 32] > 0.0
    assert mask[0, 0] > 0.0


def test_apply_notch_filter_uses_peak_locations():
    peak = DummyPeak(0.125, 0.25)
    fft_image = np.ones((64, 64), dtype=np.float32)

    filtered = apply_notch_filter(fft_image, [peak], sigma=5)

    assert filtered.shape == fft_image.shape
    assert filtered[40, 48] == pytest.approx(0.0)
    assert np.isfinite(filtered).all()
    assert filtered[0, 0] > 0.0