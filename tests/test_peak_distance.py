
import pytest

from moire.adaptive import calculate_peak_distance
from moire.datatypes import Peak


def test_identical_peaks():
    a = Peak(0.125, 0.015625, 5.0)
    b = Peak(0.125, 0.015625, 8.0)

    assert calculate_peak_distance(a, b) == pytest.approx(0.0)


def test_nearby_peaks():
    a = Peak(0.12109375, 0.015625, 5.0)
    b = Peak(0.12109375, 0.013671875, 8.0)

    assert calculate_peak_distance(a, b) == pytest.approx(
        1 / 512
    )


def test_symmetric_peaks():
    a = Peak(0.125, 0.02, 5.0)
    b = Peak(-0.125, -0.02, 8.0)

    assert calculate_peak_distance(a, b) == pytest.approx(0.0)


def test_different_peaks():
    a = Peak(0.125, 0.0, 5.0)
    b = Peak(0.0, 0.125, 8.0)

    assert calculate_peak_distance(a, b) > 0.1
