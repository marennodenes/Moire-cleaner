
import numpy as np
import pytest

import moire.adaptive as adaptive
from moire.datatypes import Peak, PatchResult


def make_image():
    image = np.zeros((4, 8, 3), dtype=np.uint8)

    for x in range(8):
        image[:, x, :] = x

    return image


def make_detections():
    return [
        PatchResult(
            patch_id=0,
            y=0,
            x=0,
            size=4,
            peaks=[Peak(0.125, 0.0, 5.0)],
            intensity=0.6
        ),
        PatchResult(
            patch_id=1,
            y=0,
            x=4,
            size=4,
            peaks=[Peak(0.125, 0.5, 8.0)],
            intensity=0.6
        )
    ]


def run_with_capture(monkeypatch, peak_mode=None):
    captured = {}

    def fake_filter(patch, detection_result,
                    sigma=2.0, threshold=0.2):
        x = int(patch[0, 0, 0])

        captured[x] = (
            [] if detection_result is None
            else list(detection_result.peaks)
        )

        return patch.astype(np.float64)

    monkeypatch.setattr(
        adaptive,
        "filter_patch_with_detection",
        fake_filter
    )

    kwargs = dict(
        image_bgr=make_image(),
        detection_results=make_detections(),
        patch_size=4,
        stride=2
    )

    if peak_mode is not None:
        kwargs["peak_mode"] = peak_mode

    result = adaptive.process_overlapping_patches(
        **kwargs
    )

    return captured, result


def test_default_mode_preserves_best(monkeypatch):
    captured, result = run_with_capture(
        monkeypatch
    )

    assert len(captured[2]) == 1
    assert captured[2][0].ratio == 5.0
    assert result.shape == (4, 8, 3)


def test_explicit_best_mode(monkeypatch):
    captured, _ = run_with_capture(
        monkeypatch,
        peak_mode="best"
    )

    assert len(captured[2]) == 1
    assert captured[2][0].ratio == 5.0


def test_merged_mode_keeps_distinct_peaks(monkeypatch):
    captured, result = run_with_capture(
        monkeypatch,
        peak_mode="merged"
    )

    # The patch beginning at x=2 overlaps both regions.
    assert len(captured[2]) == 2

    assert {p.ratio for p in captured[2]} == {
        5.0, 8.0
    }

    assert result.shape == (4, 8, 3)


def test_merged_mode_uses_only_overlaps(monkeypatch):
    captured, _ = run_with_capture(
        monkeypatch,
        peak_mode="merged"
    )

    # Patches beginning at x=0 and x=4
    # each overlap only one detection region.
    assert len(captured[0]) == 1
    assert len(captured[4]) == 1


def test_invalid_peak_mode():
    with pytest.raises(ValueError):
        adaptive.process_overlapping_patches(
            make_image(),
            make_detections(),
            patch_size=4,
            stride=2,
            peak_mode="invalid"
        )


def test_merged_mode_preserves_detection_results(
    monkeypatch
):
    detections = make_detections()
    original_peaks = [
        list(d.peaks) for d in detections
    ]

    def fake_filter(patch, detection_result,
                    sigma=2.0, threshold=0.2):
        return patch.astype(np.float64)

    monkeypatch.setattr(
        adaptive,
        "filter_patch_with_detection",
        fake_filter
    )

    adaptive.process_overlapping_patches(
        make_image(),
        detections,
        patch_size=4,
        stride=2,
        peak_mode="merged"
    )

    for detection, original in zip(
        detections, original_peaks
    ):
        assert detection.peaks == original

def test_merged_mode_removes_duplicate_peaks(monkeypatch):
    detections = make_detections()

    # Make the two detected frequencies identical.
    detections[1].peaks = [
        Peak(0.125, 0.0, 8.0)
    ]

    captured = {}

    def fake_filter(patch, detection_result,
                    sigma=2.0, threshold=0.2):
        x = int(patch[0, 0, 0])
        captured[x] = list(detection_result.peaks)
        return patch.astype(np.float64)

    monkeypatch.setattr(
        adaptive,
        "filter_patch_with_detection",
        fake_filter
    )

    adaptive.process_overlapping_patches(
        make_image(),
        detections,
        patch_size=4,
        stride=2,
        peak_mode="merged"
    )

    assert len(captured[2]) == 1
    assert captured[2][0].ratio == 8.0
