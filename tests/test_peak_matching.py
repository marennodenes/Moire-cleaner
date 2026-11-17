import pytest

from moire.adaptive import (
    calculate_peak_distance,
    match_mutual_peaks,
    merge_matched_peaks,
)
from moire.datatypes import Peak


def test_identical_peaks_have_zero_distance():
    a = Peak(0.125, 0.015625, 5.0)
    b = Peak(0.125, 0.015625, 8.0)

    assert calculate_peak_distance(a, b) == pytest.approx(0.0)


def test_nearby_peaks_have_expected_distance():
    a = Peak(0.12109375, 0.015625, 5.0)
    b = Peak(0.12109375, 0.013671875, 8.0)

    assert calculate_peak_distance(a, b) == pytest.approx(
        1 / 512
    )


def test_symmetric_peaks_have_zero_distance():
    a = Peak(0.125, 0.02, 5.0)
    b = Peak(-0.125, -0.02, 8.0)

    assert calculate_peak_distance(a, b) == pytest.approx(0.0)


def test_different_peaks_have_greater_distance():
    a = Peak(0.125, 0.0, 5.0)
    b = Peak(0.0, 0.125, 8.0)

    assert calculate_peak_distance(a, b) > 0.1


def test_identical_peak_matching():
    a = [Peak(0.125, 0.015625, 5.0)]
    b = [Peak(0.125, 0.015625, 8.0)]

    matches = match_mutual_peaks(a, b)

    assert len(matches) == 1
    assert matches[0][0:2] == (0, 0)
    assert matches[0][2] == pytest.approx(0.0)


def test_one_bin_peak_matching():
    a = [Peak(0.125, 0.015625, 5.0)]
    b = [Peak(0.125, 0.013671875, 8.0)]

    matches = match_mutual_peaks(a, b)

    assert len(matches) == 1
    assert matches[0][2] == pytest.approx(1.0)


def test_distant_peaks_are_not_matched():
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


def test_peak_matching_rejects_negative_threshold():
    with pytest.raises(ValueError):
        match_mutual_peaks([], [], max_distance_bins=-1)


def test_identical_peaks_are_merged():
    a = [Peak(0.125, 0.02, 5.0)]
    b = [Peak(0.125, 0.02, 8.0)]

    result = merge_matched_peaks(a, b)

    assert len(result) == 1
    assert result[0] == b[0]


def test_one_bin_peaks_are_merged():
    a = [Peak(0.125, 0.015625, 5.0)]
    b = [Peak(0.125, 0.013671875, 8.0)]

    result = merge_matched_peaks(a, b)

    assert len(result) == 1
    assert result[0] == b[0]


def test_distant_peaks_are_preserved():
    a = [Peak(0.125, 0.0, 5.0)]
    b = [Peak(0.0, 0.125, 8.0)]

    result = merge_matched_peaks(a, b)

    assert len(result) == 2
    assert a[0] in result
    assert b[0] in result


def test_symmetric_peaks_are_merged():
    a = [Peak(0.125, 0.02, 5.0)]
    b = [Peak(-0.125, -0.02, 8.0)]

    result = merge_matched_peaks(a, b)

    assert len(result) == 1
    assert result[0] == b[0]


def test_unmatched_peaks_are_preserved():
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


def test_empty_peak_lists_are_handled():
    peak = Peak(0.125, 0.02, 5.0)

    assert merge_matched_peaks([], []) == []
    assert merge_matched_peaks([peak], []) == [peak]
    assert merge_matched_peaks([], [peak]) == [peak]


def test_peak_merging_does_not_modify_input_lists():
    a = [Peak(0.125, 0.02, 5.0)]
    b = [Peak(0.125, 0.02, 8.0)]

    original_a = a.copy()
    original_b = b.copy()

    merge_matched_peaks(a, b)

    assert a == original_a
    assert b == original_b


def test_peak_merging_rejects_negative_threshold():
    with pytest.raises(ValueError):
        merge_matched_peaks([], [], max_distance_bins=-1)
