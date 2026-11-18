import pytest

from moire.adaptive import calculate_weighted_intensity
from moire.datatypes import PatchResult


def make_detection(patch_id, y, x, size, intensity):
    return PatchResult(
        patch_id=patch_id,
        y=y,
        x=x,
        size=size,
        peaks=[],
        intensity=intensity
    )


def test_weighted_intensity():
    detections = [
        make_detection(0, 0, 0, 512, 0.8),
        make_detection(1, 0, 512, 512, 0.2),
    ]

    # 75% overlap with A and 25% with B.
    result = calculate_weighted_intensity(
        patch_y=0,
        patch_x=128,
        patch_size=512,
        detection_results=detections
    )

    assert result == pytest.approx(0.65)


def test_equal_overlap():
    detections = [
        make_detection(0, 0, 0, 512, 0.8),
        make_detection(1, 0, 512, 512, 0.2),
    ]

    result = calculate_weighted_intensity(
        patch_y=0,
        patch_x=256,
        patch_size=512,
        detection_results=detections
    )

    assert result == pytest.approx(0.5)


def test_no_detection():
    result = calculate_weighted_intensity(
        patch_y=1024,
        patch_x=1024,
        patch_size=512,
        detection_results=[]
    )

    assert result == 0.0
