import cv2
import numpy as np
import pytest

from moire.screen_clean import clean_screen_moire

N = 512


def _lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def _from_lab(lab):
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


@pytest.fixture
def text_page():
    """Light page with dark and blue text-like bars."""
    img = np.full((N, N, 3), 235, np.uint8)
    for y in range(60, N - 40, 40):
        cv2.rectangle(img, (40, y), (300, y + 12), (30, 30, 30), -1)
    cv2.rectangle(img, (320, 60), (470, 100), (200, 60, 30), -1)   # blue (BGR)
    return img


def _colour_bands(img, period=48, amp=5):
    """Add diagonal colour moiré: a/b oscillate, L stays the same."""
    lab = _lab(img)
    yy, xx = np.mgrid[:N, :N]
    phase = 2 * np.pi * (xx + yy) / period
    lab[..., 1] += amp * np.sin(phase)
    lab[..., 2] += amp * np.cos(phase)
    return _from_lab(lab)


# ---------- input / output ----------

def test_shape_and_dtype(text_page):
    out = clean_screen_moire(text_page)
    assert out.shape == text_page.shape and out.dtype == np.uint8


@pytest.mark.parametrize("bad", [np.zeros((N, N), np.uint8), np.zeros((100, 100, 3), np.uint8)])
def test_bad_input(bad):
    with pytest.raises(ValueError):
        clean_screen_moire(bad)


def test_flat_image_is_unchanged():
    img = np.full((N, N, 3), (120, 140, 160), np.uint8)
    out = clean_screen_moire(img)
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


# ---------- colour bands ----------

def test_colour_bands_are_removed(text_page):
    noisy = _colour_bands(text_page)
    out = clean_screen_moire(noisy)
    bg = (slice(150, 450), slice(320, 500))   # background area without text
    assert _lab(out)[bg][..., 1:].std() < 0.3 * _lab(noisy)[bg][..., 1:].std()


def test_text_stays_sharp_and_coloured(text_page):
    out = clean_screen_moire(_colour_bands(text_page))
    # dark bar stays dark, background stays light
    assert _lab(out)[66, 150, 0] < 60 and _lab(out)[90, 150, 0] > 200
    # blue box keeps its colour (b channel well below neutral 128)
    assert _lab(out)[80, 395, 2] < _lab(text_page)[80, 395, 2] + 10


def test_strong_colour_edges_do_not_bleed():
    """Red tile next to a green tile (content on the screen) stays unchanged."""
    img = np.zeros((N, N, 3), np.uint8)
    img[:, :N // 2] = (40, 40, 220)    # red (BGR)
    img[:, N // 2:] = (80, 180, 40)    # green
    out = clean_screen_moire(img)
    assert np.abs(_lab(out)[..., 1:] - _lab(img)[..., 1:]).max() <= 6.5


# ---------- brightness rings ----------

def test_brightness_rings_are_flattened():
    """Wide multiplicative rings on the background are reduced, text stays dark.

    Uses a larger image: ring sizes and masks are tuned for full phone photos.
    """
    M = 1536
    page = np.full((M, M, 3), 235, np.uint8)
    for y in range(100, M - 100, 60):
        cv2.rectangle(page, (60, y), (500, y + 16), (30, 30, 30), -1)
    yy, xx = np.mgrid[:M, :M]
    lab = _lab(page)
    lab[..., 0] *= 1 + 0.06 * np.sin(2 * np.pi * np.hypot(yy - M / 2, xx - 1200) / 600)
    noisy = _from_lab(lab)
    out = clean_screen_moire(noisy)
    bg = (slice(200, 1300), slice(800, 1450))
    assert _lab(out)[bg][..., 0].std() < 0.6 * _lab(noisy)[bg][..., 0].std()
    assert _lab(out)[108, 300, 0] < 60


# ---------- mask ----------

def test_zero_mask_leaves_image_unchanged(text_page):
    noisy = _colour_bands(text_page)
    out = clean_screen_moire(noisy, mask=np.zeros(noisy.shape[:2], np.float32))
    assert np.array_equal(out, noisy)


def test_mask_limits_changes_to_masked_area(text_page):
    noisy = _colour_bands(text_page)
    m = np.zeros(noisy.shape[:2], np.float32)
    m[:, N // 2:] = 1.0
    out = clean_screen_moire(noisy, mask=m)
    assert np.array_equal(out[:, :N // 2], noisy[:, :N // 2])
    assert not np.array_equal(out[:, N // 2:], noisy[:, N // 2:])


def test_detector_mask_skips_photo_without_moire():
    """Smooth gradient 'photo': detector finds nothing, so nothing changes."""
    yy, xx = np.mgrid[:N, :N]
    img = np.stack([xx / N * 200 + 20, yy / N * 180 + 30, np.full((N, N), 90.0)], axis=2).astype(np.uint8)
    out = clean_screen_moire(img, mask=True)
    assert np.array_equal(out, img)
