from pathlib import Path

import cv2
import numpy as np
import pytest

from moire.datatypes import Peak
from moire.detection import analyze_image, detect_peaks, estimate_intensity

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
    data = np.fromfile(str(SAMPLE), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
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


@pytest.mark.parametrize("fy, fx", [(64, 20), (20, 64), (64, 0)])
def test_nyquist_peak_is_inside_spectrum(clean, fy, fx):
    """Peaks at the highest frequency must still be valid spectrum indices."""
    patch = add_wave(clean, 2 * fy, 2 * fx)[:128, :128]   # 128 patch: halve freq
    peaks = detect_peaks(patch)
    assert peaks
    for p in peaks:
        assert -64 <= p.fy < 64 and -64 <= p.fx < 64


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
    # (clipping to 0..255 adds a weak overtone, so only check the strongest)
    assert same_pair(peaks[0], 20, 27.3, tol=1)


def test_bad_ksize_raises(clean):
    with pytest.raises(ValueError):
        detect_peaks(clean, ksize=7)


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
    assert detect_peaks(patch) == []


def test_intensity_grows_with_amplitude(clean):
    values = [estimate_intensity(detect_peaks(add_wave(clean, 20, 35, amp)))
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


# ---------- analyze_image: whole colour image ----------

@pytest.fixture(scope="module")
def big_clean():
    """Cameraman as a 1024 x 2048 colour image without moiré."""
    data = np.fromfile(str(SAMPLE), dtype=np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    big = cv2.resize(gray, (2048, 1024), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)


def colour_wave(img, fy, fx, amp=25.0, max_x=None):
    """Add green/magenta stripes (like screen moiré) at (fy, fx) cycles/pixel.

    Green goes up where red and blue go down, so the stripes are mostly
    colour and only a little brightness.
    """
    H, W = img.shape[:2]
    y, x = np.mgrid[0:H, 0:W]
    wave = amp * np.cos(2 * np.pi * (fy * y + fx * x))
    if max_x is not None:
        wave[:, max_x:] = 0
    out = img.astype(np.float32)
    out[..., 1] += wave          # G
    out[..., 0] -= wave          # B
    out[..., 2] -= wave          # R
    return np.clip(out, 0, 255).astype(np.uint8)


def near(peak, fy, fx, size):
    """Peak within 1.5 FFT bins (of a size x size region) of (fy, fx) or its mirror."""
    tol = 1.5 / size
    return any(abs(peak.fy - s * fy) < tol and abs(peak.fx - s * fx) < tol for s in (1, -1))


def test_layout_of_results(big_clean):
    results = analyze_image(big_clean, patch_size=512)
    assert len(results) == 2 * 4
    assert [r.patch_id for r in results] == list(range(8))
    assert [(r.y, r.x) for r in results[:5]] == [(0, 0), (0, 512), (0, 1024), (0, 1536), (512, 0)]
    assert all(r.size == 512 for r in results)


def test_no_moire_gives_empty_results(big_clean):
    for r in analyze_image(big_clean):
        assert r.peaks == []
        assert r.intensity == 0.0


def test_colour_moire_is_found_everywhere(big_clean):
    img = colour_wave(big_clean, 0.03, 0.05)
    for r in analyze_image(img):
        assert r.peaks, f"patch {r.patch_id} missed"
        assert near(r.peaks[0], 0.03, 0.05, r.size)
        assert 0 < r.intensity <= 1


def test_peaks_are_in_cycles_per_pixel(big_clean):
    """The filter gets the FFT bin of its own region with round(f * N)."""
    img = colour_wave(big_clean, 0.03, 0.05)
    for r in analyze_image(img):
        for p in r.peaks:
            assert -0.5 <= p.fy <= 0.5 and -0.5 <= p.fx <= 0.5
            assert p.fy > 0 or (p.fy == 0 and p.fx > 0)       # one per mirror pair
        p = r.peaks[0]
        for N in (256, 512, 1024):
            assert (round(p.fy * N), round(p.fx * N)) in {
                (round(0.03 * N) + dy, round(0.05 * N) + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}


def test_moire_only_where_it_is(big_clean):
    """Stripes only in the left half: left found, far right clean.

    Columns next to the border may be marked too, because the 2x and 4x
    windows reach over the border (known limit).
    """
    img = colour_wave(big_clean, 0.06, 0.08, max_x=1024)
    grid = {(r.y // 256, r.x // 256): bool(r.peaks) for r in analyze_image(img, patch_size=256)}
    assert all(grid[(row, col)] for row in range(4) for col in range(3))
    assert not any(grid[(row, col)] for row in range(4) for col in (6, 7))


def test_brightness_only_stripes_are_ignored(big_clean):
    """Known limit: stripes without colour (like wood grain) are not reported."""
    gray = cv2.cvtColor(big_clean, cv2.COLOR_BGR2GRAY).astype(np.float32)
    y, x = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
    gray += 25 * np.cos(2 * np.pi * (0.03 * y + 0.05 * x))
    img = cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    assert all(r.peaks == [] for r in analyze_image(img))


def test_coloured_straight_edges_are_not_moire(big_clean):
    """A sharp blue band (like on a printed sign) gives a line, not a peak."""
    img = big_clean.copy()
    for x0 in range(100, 2048, 400):
        img[:, x0:x0 + 60] = (200, 60, 20)       # blue in BGR
    assert all(r.peaks == [] for r in analyze_image(img))


@pytest.mark.parametrize("bad", [np.zeros((1024, 1024)), np.zeros((1024, 1024, 4))])
def test_needs_colour_image(bad):
    with pytest.raises(ValueError):
        analyze_image(bad)


@pytest.mark.parametrize("size", [100, 130, 64])
def test_bad_patch_size(big_clean, size):
    with pytest.raises(ValueError):
        analyze_image(big_clean, patch_size=size)


# ---------- grey (colourless) moiré ----------

def _load_sample(name):
    data = np.fromfile(str(SAMPLE.parent / name), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def test_strong_grey_stripes_are_found():
    """global_moire.png: grey stripes at (0.08, 0.08) everywhere."""
    results = analyze_image(_load_sample("global_moire.png"))
    found = [r for r in results
             if any(same_pair(p, 0.08, 0.08, tol=0.005) for p in r.peaks)]
    assert len(found) >= 18


def test_clean_photo_has_no_grey_peaks():
    results = analyze_image(_load_sample("clean.png"))
    assert all(not r.peaks for r in results)
