from types import SimpleNamespace

import numpy as np

from moire.pipeline import process_image


# Create test image
image = np.full((100, 100, 3), 128, dtype=np.uint8)

# Test 1: No filter
result = process_image(image)

assert np.array_equal(image, result)
print("Test 1 passed: Identity filter")


# Test 2: Dummy filter
def dummy_filter(img):
    return np.clip(
        img.astype(np.int16) + 20, 0, 255
    ).astype(np.uint8)


result = process_image(image, dummy_filter)

assert result.shape == image.shape
assert np.all(result == 148)

print("Test 2 passed: Dummy filter")
print("Pipeline test completed!")

# Full chain: detection (B) -> adaptive patches (C) -> notch (A)

from pathlib import Path

import cv2
import pytest

from moire.pipeline import moire_filter

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def _load(name):
    data = np.fromfile(str(SAMPLES / name), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / mse)


def test_moire_filter_improves_synthetic_moire():
    # Top-left corner, where the detector finds the stripes.
    noisy = _load("global_moire.png")[:1024, :1024]
    clean = _load("clean.png")[:1024, :1024]

    result = process_image(noisy, moire_filter)

    assert result.shape == noisy.shape
    assert result.dtype == np.uint8
    assert _psnr(result, clean) > _psnr(noisy, clean) + 5


def test_moire_filter_leaves_clean_image_alone():
    clean = _load("clean.png")[:1024, :1024]
    result = moire_filter(clean)
    assert np.abs(result.astype(int) - clean).mean() < 0.5


@pytest.mark.parametrize("size", [(100, 100), (300, 200), (130, 900)])
def test_moire_filter_small_images(size):
    image = np.full((*size, 3), 128, dtype=np.uint8)
    result = moire_filter(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_combined_filter_applies_screen_clean_to_notch_result(monkeypatch):
    import moire.pipeline as pipeline

    image = np.full((512, 512, 3), 128, dtype=np.uint8)
    detections = [SimpleNamespace(peaks=[object()])]
    notch_result = np.full_like(image, 100)
    mask = np.ones(image.shape[:2], dtype=np.float32)
    cleaned = np.full_like(image, 90)
    calls = {}

    monkeypatch.setattr(pipeline, "analyze_image", lambda img, patch_size: detections)
    monkeypatch.setattr(
        pipeline,
        "process_overlapping_patches",
        lambda img, results, **kwargs: notch_result.astype(np.float32),
    )

    def fake_mask(img, patch_size, detections):
        calls["mask"] = (img, patch_size, detections)
        return mask

    def fake_clean(img, mask):
        calls["clean"] = (img, mask)
        return cleaned

    monkeypatch.setattr(pipeline, "moire_mask", fake_mask)
    monkeypatch.setattr(pipeline, "clean_screen_moire", fake_clean)

    result = pipeline.moire_filter_and_screen_clean(image, patch_size=256)

    assert np.array_equal(result, cleaned)
    assert calls["mask"][0] is image
    assert calls["mask"][1] == 256
    assert calls["mask"][2] is detections
    assert np.array_equal(calls["clean"][0], notch_result)
    assert calls["clean"][1] is mask


def test_combined_filter_uses_detected_moire_mask(monkeypatch):
    import moire.pipeline as pipeline

    image = _load("global_moire.png")[:1024, :1024]
    calls = {}

    monkeypatch.setattr(
        pipeline,
        "process_overlapping_patches",
        lambda img, detections, **kwargs: img.astype(np.float32),
    )

    def fake_clean(img, mask):
        calls["image"] = img
        calls["mask"] = mask
        return img

    monkeypatch.setattr(pipeline, "clean_screen_moire", fake_clean)

    result = pipeline.moire_filter_and_screen_clean(image)

    assert result.shape == image.shape
    assert np.array_equal(calls["image"], image)
    assert calls["mask"].shape == image.shape[:2]
    assert np.any(calls["mask"] > 0)
