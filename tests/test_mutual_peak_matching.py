
import pytest

from moire.datatypes import Peak
from moire.adaptive import match_mutual_peaks


def test_identical_peak_matching():
    a = [Peak(0.125, 0.015625, 5.0)]
    b = [Peak(0.125, 0.015625, 8.0)]

    matches = match_mutual_peaks(a, b)

    assert len(matches) == 1
    assert matches[0][0:2] == (0, 0)
    assert matches[0][2] == pytest.approx(0.0)


def test_one_bin_matching():
    a = [Peak(0.125, 0.015625, 5.0)]
    b = [Peak(0.125, 0.013671875, 8.0)]

    matches = match_mutual_peaks(a, b)

    assert len(matches) == 1
    assert matches[0][2] == pytest.approx(1.0)


def test_distant_peaks_not_matched():
    a = [Peak(0.125, 0.0, 5.0)]
    b = [Peak(0.0, 0.125, 8.0)]

    matches = match_mutual_peaks(a, b)

    assert matches == []


def test_symmetric_peak_matching():
    a = [Peak(0.125, 0.02, 5.0)]
    b = [Peak(-0.125, -0.02, 8.0)]

    matches = match_mutual_peaks(a, b)

    assert len(matches) == 1
    assert matches[0][2] == pytest.approx(0.0)


def test_invalid_threshold():
    with pytest.raises(ValueError):
        match_mutual_peaks([], [], max_distance_bins=-1)
