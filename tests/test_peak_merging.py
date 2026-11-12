
import pytest

from moire.datatypes import Peak
from moire.adaptive import merge_matched_peaks


def test_identical_peaks_merged():
    a = [Peak(0.125, 0.02, 5.0)]
    b = [Peak(0.125, 0.02, 8.0)]

    result = merge_matched_peaks(a, b)

    assert len(result) == 1
    assert result[0] == b[0]


def test_one_bin_peaks_merged():
    a = [Peak(0.125, 0.015625, 5.0)]
    b = [Peak(0.125, 0.013671875, 8.0)]

    result = merge_matched_peaks(a, b)

    assert len(result) == 1
    assert result[0] == b[0]


def test_distant_peaks_preserved():
    a = [Peak(0.125, 0.0, 5.0)]
    b = [Peak(0.0, 0.125, 8.0)]

    result = merge_matched_peaks(a, b)

    assert len(result) == 2
    assert a[0] in result
    assert b[0] in result


def test_symmetric_peaks_merged():
    a = [Peak(0.125, 0.02, 5.0)]
    b = [Peak(-0.125, -0.02, 8.0)]

    result = merge_matched_peaks(a, b)

    assert len(result) == 1
    assert result[0] == b[0]


def test_unmatched_peaks_preserved():
    a = [
        Peak(0.125, 0.015625, 5.0),
        Peak(0.250, 0.0, 7.0)
    ]

    b = [
        Peak(0.125, 0.013671875, 8.0),
        Peak(0.375, 0.0, 6.0)
    ]

    result = merge_matched_peaks(a, b)

    assert len(result) == 3
    assert b[0] in result
    assert a[1] in result
    assert b[1] in result
    assert a[0] not in result


def test_empty_peak_lists():
    peak = Peak(0.125, 0.02, 5.0)

    assert merge_matched_peaks([], []) == []
    assert merge_matched_peaks([peak], []) == [peak]
    assert merge_matched_peaks([], [peak]) == [peak]


def test_input_lists_unchanged():
    a = [Peak(0.125, 0.02, 5.0)]
    b = [Peak(0.125, 0.02, 8.0)]

    original_a = a.copy()
    original_b = b.copy()

    merge_matched_peaks(a, b)

    assert a == original_a
    assert b == original_b


def test_invalid_distance_threshold():
    with pytest.raises(ValueError):
        merge_matched_peaks([], [], max_distance_bins=-1)
