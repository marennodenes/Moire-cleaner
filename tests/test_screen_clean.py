import cv2
import numpy as np
import pytest

from moire.screen_clean import clean_screen_moire

N = 512


def _lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


@pytest.fixture
def text_page():
    """Light page with dark and blue text-like bars."""
    img = np.full((N, N, 3), 235, np.uint8)
    for y in range(60, N - 40, 40):
        cv2.rectangle(img, (40, y), (300, y + 12), (30, 30, 30), -1)
    cv2.rectangle(img, (320, 60), (470, 100), (200, 60, 30), -1)   # blue (BGR)
    return img


def _colour_bands(img, period=48, amp=12):
    """Add diagonal colour moiré: a/b oscillate, L stays the same."""
    lab = _lab(img)
    yy, xx = np.mgrid[:N, :N]
    phase = 2 * np.pi * (xx + yy) / period
    lab[..., 1] += amp * np.sin(phase)
    lab[..., 2] += amp * np.cos(phase)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def test_shape_and_dtype(text_page):
    out = clean_screen_moire(text_page)
    assert out.shape == text_page.shape and out.dtype == np.uint8


def test_flat_image_is_unchanged():
    img = np.full((N, N, 3), (120, 140, 160), np.uint8)
    out = clean_screen_moire(img)
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


def test_colour_bands_are_removed(text_page):
    noisy = _colour_bands(text_page)
    out = clean_screen_moire(noisy)
    bg = (slice(150, 450), slice(320, 500))   # background area without text
    before = _lab(noisy)[bg][..., 1:].std()
    after = _lab(out)[bg][..., 1:].std()
    assert after < 0.3 * before


def test_text_stays_sharp_and_coloured(text_page):
    out = clean_screen_moire(_colour_bands(text_page))
    # dark bar stays dark, background stays light
    assert _lab(out)[66, 150, 0] < 60 and _lab(out)[90, 150, 0] > 200
    # blue box keeps its colour (b channel well below neutral 128)
    assert _lab(out)[80, 395, 2] < _lab(text_page)[80, 395, 2] + 10


@pytest.mark.parametrize("bad", [np.zeros((N, N), np.uint8), np.zeros((100, 100, 3), np.uint8)])
def test_bad_input(bad):
    with pytest.raises(ValueError):
        clean_screen_moire(bad)
