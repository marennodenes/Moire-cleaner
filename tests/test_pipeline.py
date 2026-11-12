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
