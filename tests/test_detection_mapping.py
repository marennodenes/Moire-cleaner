from moire.adaptive import find_best_detection
from moire.datatypes import PatchResult, Peak


def test_detection_mapping():
    detection_results = [
        PatchResult(
            patch_id=0,
            y=0,
            x=0,
            size=512,
            peaks=[Peak(0.0, 0.1, 10.0)],
            intensity=0.8,
        ),
        PatchResult(
            patch_id=1,
            y=0,
            x=512,
            size=512,
            peaks=[Peak(0.0, 0.2, 8.0)],
            intensity=0.5,
        ),
    ]

    # Overlaps detection 0 more than detection 1.
    result = find_best_detection(
        patch_y=0,
        patch_x=256,
        patch_size=512,
        detection_results=detection_results,
    )

    assert result is not None
    assert result.patch_id == 0
    assert result.intensity == 0.8

    # Does not overlap any detection region.
    result = find_best_detection(
        patch_y=1024,
        patch_x=1024,
        patch_size=512,
        detection_results=detection_results,
    )

    assert result is None

    print("Detection Mapping Test passed!")


if __name__ == "__main__":
    test_detection_mapping()
