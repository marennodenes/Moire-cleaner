import pytest

from moire.adaptive import get_detection_overlaps
from moire.datatypes import PatchResult


def make_detection(patch_id, y, x, size, intensity=0.5):
    return PatchResult(
        patch_id=patch_id,
        y=y,
        x=x,
        size=size,
        peaks=[],
        intensity=intensity,
    )


def test_multiple_detection_overlaps():
    detections = [
        make_detection(0, 0, 0, 512),
        make_detection(1, 0, 512, 512),
    ]

    overlaps = get_detection_overlaps(
        patch_y=0,
        patch_x=256,
        patch_size=512,
        detection_results=detections,
    )

    assert len(overlaps) == 2

    first, second = overlaps

    assert first[0].patch_id == 0
    assert first[1] == 512 * 256
    assert first[2] == pytest.approx(0.5)

    assert second[0].patch_id == 1
    assert second[1] == 512 * 256
    assert second[2] == pytest.approx(0.5)


def test_partial_coverage():
    detections = [
        make_detection(0, 0, 0, 512),
    ]

    overlaps = get_detection_overlaps(
        patch_y=256,
        patch_x=256,
        patch_size=512,
        detection_results=detections,
    )

    assert len(overlaps) == 1
    assert overlaps[0][1] == 256 * 256
    assert overlaps[0][2] == pytest.approx(0.25)


def test_no_overlap():
    detections = [
        make_detection(0, 0, 0, 512),
    ]

    overlaps = get_detection_overlaps(
        patch_y=1024,
        patch_x=1024,
        patch_size=512,
        detection_results=detections,
    )

    assert overlaps == []


def test_invalid_patch_size():
    with pytest.raises(ValueError):
        get_detection_overlaps(0, 0, 0, [])
