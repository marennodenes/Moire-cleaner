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


# ---------- robustness ----------

@pytest.mark.parametrize("fy, fx", [(0, 30), (30, 0), (-40, 40), (100, 110), (127, 0)])
def test_wave_directions_and_high_frequencies(clean, fy, fx):
    peaks = detect_peaks(add_wave(clean, fy, fx))
    assert len(peaks) == 1
    assert same_pair(peaks[0], fy, fx)


def test_gaussian_noise_is_not_moire(clean):
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.1, clean.shape).astype(np.float32)
    assert detect_peaks(clean + noise) == []
    peaks = detect_peaks(add_wave(clean, 20, 35) + noise)
    assert len(peaks) == 1
    assert same_pair(peaks[0], 20, 35)


@pytest.mark.parametrize("patch", [np.zeros((64, 64)), np.full((64, 64), 0.5)])
def test_flat_patch_gives_nothing(patch):
    result = analyze_patch(patch, patch_id=0)
    assert result.peaks == []
    assert result.intensity == 0.0


def test_intensity_grows_with_amplitude(clean):
    values = [analyze_patch(add_wave(clean, 20, 35, amp), 0).intensity
              for amp in (0.02, 0.05, 0.15, 0.3)]
    assert values == sorted(values)
    assert values[0] > 0


def test_square_wave_finds_harmonics(clean):
    """Real moiré is not a pure cosine: overtones must be found too."""
    H, W = clean.shape
    y, x = np.mgrid[0:H, 0:W]
    img = clean + 0.1 * np.sign(np.cos(2 * np.pi * (20 * y / H + 35 * x / W)))
    peaks = detect_peaks(img)
    assert any(same_pair(p, 20, 35) for p in peaks)    # fundamental
    assert any(same_pair(p, 60, 105) for p in peaks)   # 3rd harmonic


def test_128_patches(clean):
    """Every 128x128 patch: no false peaks without moiré, correct peak with it."""
    wavy = add_wave(clean, 20, 35)
    for i in (0, 64, 128):
        for j in (0, 64, 128):
            assert detect_peaks(clean[i:i + 128, j:j + 128]) == []
            peaks = detect_peaks(wavy[i:i + 128, j:j + 128])
            assert len(peaks) == 1
            assert same_pair(peaks[0], 10, 17.5, tol=1)
