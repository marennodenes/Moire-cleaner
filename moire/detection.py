"""Detect moiré peaks in a patch spectrum and estimate how strong they are.

Method (see docs/01-explore-spectrum.md):
Hann window -> |fftshift(fft2)| -> divide by local median -> ignore centre
-> threshold -> keep local maxima -> one peak per mirror pair.
"""
import cv2
import numpy as np

from moire.datatypes import Peak, PatchResult

THRESHOLD = 6.0     # clean cameraman max 3.4, weakest wave 7.4
SATURATION = 50.0   # ratio that counts as intensity 1.0


def ratio_map(patch, ksize=5, dc_radius=6):
    """Spectrum magnitude divided by its local median (centred, DC disc = 0).

    ksize must be 3 or 5 (cv2.medianBlur only supports these for float32).
    """
    if ksize not in (3, 5):
        raise ValueError(f"ksize must be 3 or 5, got {ksize}")
    img = np.asarray(patch, dtype=np.float32)
    h, w = img.shape
    img = (img - img.mean()) * np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(img))).astype(np.float32)
    ratio = mag / (cv2.medianBlur(mag, ksize) + 1e-9)
    yy, xx = np.mgrid[-(h // 2):h - h // 2, -(w // 2):w - w // 2]
    ratio[np.hypot(yy, xx) < dc_radius] = 0
    return ratio


def detect_peaks(patch, threshold=THRESHOLD, ksize=5, dc_radius=6):
    """Return moiré peaks in a grayscale patch, strongest first.

    Only one peak per mirror pair is returned, chosen so that fy > 0,
    or fy == 0 and fx > 0.
    """
    ratio = ratio_map(patch, ksize, dc_radius)
    h, w = ratio.shape
    # Hann widens peaks, so neighbours are also high: keep local maxima only.
    is_max = ratio >= cv2.dilate(ratio, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(is_max & (ratio > threshold))

    peaks = {}
    for y, x in zip(ys, xs):
        fy, fx = int(y - h // 2), int(x - w // 2)
        if fy < 0 or (fy == 0 and fx < 0):
            fy, fx = -fy, -fx
        r = float(ratio[y, x])
        if (fy, fx) not in peaks or r > peaks[(fy, fx)].ratio:
            peaks[(fy, fx)] = Peak(fy, fx, r)
    return sorted(peaks.values(), key=lambda p: p.ratio, reverse=True)


def estimate_intensity(peaks, threshold=THRESHOLD, saturation=SATURATION):
    """Moiré strength in [0, 1] from the strongest peak ratio (log scale).

    ratio == threshold -> 0, ratio >= saturation -> 1. No peaks -> 0.0.
    """
    if not peaks:
        return 0.0
    r = max(p.ratio for p in peaks)
    value = np.log(r / threshold) / np.log(saturation / threshold)
    return float(np.clip(value, 0.0, 1.0))


def analyze_patch(patch, patch_id, threshold=THRESHOLD):
    """Run detection on one patch. Always returns a result, also with no moiré."""
    peaks = detect_peaks(patch, threshold)
    return PatchResult(patch_id, peaks, estimate_intensity(peaks, threshold))
