
import pytest

from moire.datatypes import Peak, PatchResult
from moire.adaptive import merge_local_detection_peaks


def make_detection(region_id, peaks):
    return PatchResult(
        patch_id=region_id,
        y=0,
        x=region_id * 512,
        size=512,
        peaks=peaks,
        intensity=0.5
    )


def test_three_identical_peaks_merged():
    regions = [
        make_detection(0, [Peak(0.125, 0.02, 5.0)]),
        make_detection(1, [Peak(0.125, 0.02, 8.0)]),
        make_detection(2, [Peak(0.125, 0.02, 10.0)])
    ]

    result = merge_local_detection_peaks(regions)

    assert len(result) == 1
    assert result[0] == regions[2].peaks[0]


def test_close_three_peaks_merged():
    regions = [
        make_detection(0, [Peak(0.125, 0.0, 5.0)]),
        make_detection(1, [Peak(0.125 + 0.25/512, 0.0, 8.0)]),
        make_detection(2, [Peak(0.125 + 0.5/512, 0.0, 10.0)])
    ]

    result = merge_local_detection_peaks(regions)

    assert len(result) == 1
    assert result[0] == regions[2].peaks[0]


def test_chain_does_not_merge_all():
    regions = [
        make_detection(0, [Peak(0.125, 0.0, 5.0)]),
        make_detection(1, [Peak(0.125 + 1/512, 0.0, 8.0)]),
        make_detection(2, [Peak(0.125 + 2/512, 0.0, 10.0)])
    ]

    result = merge_local_detection_peaks(regions)

    assert len(result) == 2
    assert regions[1].peaks[0] in result
    assert regions[2].peaks[0] in result


def test_different_frequencies_preserved():
    regions = [
        make_detection(0, [Peak(0.125, 0.0, 5.0)]),
        make_detection(1, [Peak(0.0, 0.125, 8.0)])
    ]

    result = merge_local_detection_peaks(regions)

    assert len(result) == 2
    assert regions[0].peaks[0] in result
    assert regions[1].peaks[0] in result


def test_symmetric_peaks_merged():
    regions = [
        make_detection(0, [Peak(0.125, 0.02, 5.0)]),
        make_detection(1, [Peak(-0.125, -0.02, 8.0)])
    ]

    result = merge_local_detection_peaks(regions)

    assert len(result) == 1
    assert result[0] == regions[1].peaks[0]


def test_same_region_peaks_not_merged():
    regions = [
        make_detection(0, [
            Peak(0.125, 0.0, 5.0),
            Peak(0.125 + 0.5/512, 0.0, 7.0)
        ])
    ]

    result = merge_local_detection_peaks(regions)

    assert len(result) == 2


def test_empty_detection_results():
    assert merge_local_detection_peaks([]) == []

    regions = [make_detection(0, [])]
    assert merge_local_detection_peaks(regions) == []


def test_input_unchanged():
    regions = [
        make_detection(0, [Peak(0.125, 0.0, 5.0)]),
        make_detection(1, [Peak(0.125, 0.0, 8.0)])
    ]

    original = [r.peaks.copy() for r in regions]

    merge_local_detection_peaks(regions)

    for region, peaks in zip(regions, original):
        assert region.peaks == peaks


def test_invalid_threshold():
    with pytest.raises(ValueError):
        merge_local_detection_peaks(
            [],
            max_distance_bins=-1
        )


def test_invalid_patch_size():
    with pytest.raises(ValueError):
        merge_local_detection_peaks(
            [],
            patch_size=0
        )
