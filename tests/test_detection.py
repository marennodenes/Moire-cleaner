from pathlib import Path

import cv2
import numpy as np
import pytest

from moire.datatypes import Peak
from moire.detection import analyze_patch, detect_peaks, estimate_intensity

SAMPLE = Path(__file__).parent.parent / "data" / "samples" / "cameraman.png"


def add_wave(img, fy, fx, amp=0.15):
    H, W = img.shape
    y, x = np.mgrid[0:H, 0:W]
    return img + amp * np.cos(2 * np.pi * (fy * y / H + fx * x / W))


def same_pair(peak, fy, fx, tol=0):
    """True if peak is (fy, fx) or its mirror (-fy, -fx), within tol."""
    return any(abs(peak.fy - a) <= tol and abs(peak.fx - b) <= tol
               for a, b in ((fy, fx), (-fy, -fx)))


@pytest.fixture(scope="module")
def clean():
    img = cv2.imread(str(SAMPLE), cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32)[:256, :256] / 255


@pytest.fixture(scope="module")
def images(clean):
    one = add_wave(clean, 20, 35)
    return {
        "clean": clean,
        "one": one,
        "two": add_wave(one, -15, 50, amp=0.10),
        "frac": add_wave(clean, 20.5, 35.5),
        "weak": add_wave(clean, 20, 35, amp=0.02),
    }


def test_clean_has_no_peaks(images):
    assert detect_peaks(images["clean"]) == []


def test_one_wave(images):
    peaks = detect_peaks(images["one"])
    assert len(peaks) == 1
    assert same_pair(peaks[0], 20, 35)


def test_two_waves(images):
    peaks = detect_peaks(images["two"])
    assert len(peaks) == 2
    assert same_pair(peaks[0], 20, 35)        # stronger one first
    assert same_pair(peaks[1], -15, 50)


def test_non_integer_frequency(images):
    peaks = detect_peaks(images["frac"])
    assert len(peaks) == 1
    assert same_pair(peaks[0], 20.5, 35.5, tol=1)


def test_weak_wave(images):
    peaks = detect_peaks(images["weak"])
    assert len(peaks) == 1
    assert same_pair(peaks[0], 20, 35)


def test_one_peak_per_mirror_pair(images):
    for peaks in map(detect_peaks, images.values()):
        for p in peaks:
            assert p.fy > 0 or (p.fy == 0 and p.fx > 0)


def test_sorted_strongest_first(images):
    ratios = [p.ratio for p in detect_peaks(images["two"])]
    assert ratios == sorted(ratios, reverse=True)


def test_intensity_in_range(images):
    for img in images.values():
        value = estimate_intensity(detect_peaks(img))
        assert 0.0 <= value <= 1.0


def test_intensity_order(images):
    def intensity(name):
        img = images[name]
        return estimate_intensity(detect_peaks(img))

    assert intensity("clean") == 0.0
    assert 0.0 < intensity("weak") < intensity("one")


def test_no_peaks_gives_zero():
    assert estimate_intensity([]) == 0.0


def test_huge_ratio_is_clipped():
    assert estimate_intensity([Peak(1, 1, 1e6)]) == 1.0


def test_accepts_uint8_and_non_square(clean):
    img = (np.clip(add_wave(clean, 20, 35), 0, 1) * 255).astype(np.uint8)[:, :200]
    peaks = detect_peaks(img)
    # 200 px wide: fx = 35 cycles per 256 px -> 35 * 200 / 256 ~ 27.3
    assert len(peaks) == 1
    assert same_pair(peaks[0], 20, 27.3, tol=1)


def test_bad_ksize_raises(clean):
    with pytest.raises(ValueError):
        detect_peaks(clean, ksize=7)


def test_analyze_patch_with_moire(images):
    result = analyze_patch(images["one"], patch_id=3)
    assert result.patch_id == 3
    assert len(result.peaks) == 1
    assert result.intensity > 0


def test_analyze_patch_without_moire(images):
    result = analyze_patch(images["clean"], patch_id=0)
    assert result.peaks == []
    assert result.intensity == 0.0
