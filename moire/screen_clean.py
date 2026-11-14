"""Moiré removal for photos of screens (experimental, not used by the pipeline yet).

Main entry point: clean_screen_moire(img_bgr) -> img_bgr.

Moiré on real screen photos is mostly broad, curved bands close to DC, not
sharp spectral peaks, so a notch filter only removes a small part of it.
This module works in Lab and treats colour and brightness separately:

1. Colour (a, b): guided filter with brightness (L) as guide. Removes the
   coloured bands but keeps colour edges (text) sharp.
2. Brightness (L): local spectral peak suppression on a downscaled copy of
   log(L) (screen moiré is multiplicative). Only the coarse correction is
   added back, so fine detail (text edges) is untouched. The correction is
   dropped near strong edges and refilled from the surroundings to avoid halos.

Known limits: brightness bands are only partly removed (~15-25 %), and the
colour step slightly changes images without moiré, so it should only be
applied where moiré is detected (no mask yet).
"""
import cv2
import numpy as np

CHROMA_RADIUS = 40      # guided filter radius (pixels) for a/b
CHROMA_EPS = 1e-3       # guided filter eps (L in [0, 1]); higher = blurrier colour
SCALE = 4               # brightness bands are found at 1/SCALE resolution
WINDOW = 64             # window size (pixels, at 1/SCALE) for peak suppression
CLIP = 1.5              # spectral magnitude is clipped to CLIP x local median
DC_RADIUS = 2           # bins around DC left untouched
EDGE_TAU = 0.15         # log-L gradient where the correction is dropped
LOG_OFFSET = 10.0       # log(L + offset) keeps dark areas stable


def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))


def guided_filter(guide, src, radius, eps):
    """Edge-preserving smoothing of `src`, following the edges of `guide`."""
    mean_i, mean_p = _box(guide, radius), _box(src, radius)
    a = (_box(guide * src, radius) - mean_i * mean_p) / (_box(guide * guide, radius) - mean_i**2 + eps)
    b = mean_p - a * mean_i
    return _box(a, radius) * guide + _box(b, radius)


def suppress_peaks(x, window=WINDOW, clip=CLIP, dc_radius=DC_RADIUS):
    """Clip spectral magnitude to `clip` x local median in overlapping Hann windows.

    Weighted overlap-add with hop window // 4. Returns an array like `x`.
    """
    hop = window // 4
    H, W = x.shape
    xp = cv2.copyMakeBorder(x, window, window, window, window, cv2.BORDER_REFLECT)
    w1 = np.hanning(window).astype(np.float32)
    win = np.outer(w1, w1)
    out = np.zeros_like(xp)
    norm = np.zeros_like(xp)
    yy, xx = np.mgrid[-(window // 2):window - window // 2, -(window // 2):window - window // 2]
    keep = np.hypot(yy, xx) < dc_radius
    for y in range(0, xp.shape[0] - window + 1, hop):
        for x0 in range(0, xp.shape[1] - window + 1, hop):
            block = xp[y:y + window, x0:x0 + window]
            mean = block.mean()
            F = np.fft.fftshift(np.fft.fft2((block - mean) * win))
            mag = np.abs(F) + 1e-6
            # local reference level: 5x5 median of log magnitude, smoothed over 9x9
            ref = np.exp(cv2.blur(cv2.medianBlur(np.log(mag).astype(np.float32), 5), (9, 9)))
            gain = cv2.GaussianBlur(np.minimum(1.0, clip * ref / mag).astype(np.float32), (0, 0), 1.0)
            gain[keep] = 1.0
            rec = np.real(np.fft.ifft2(np.fft.ifftshift(F * gain))) + mean * win
            out[y:y + window, x0:x0 + window] += rec * win
            norm[y:y + window, x0:x0 + window] += win * win
    return (out / np.maximum(norm, 1e-6))[window:window + H, window:window + W]


def edge_protect(corr, guide, tau=EDGE_TAU, fill=3.0):
    """Drop the correction near strong edges of `guide` and refill it from around."""
    gx = cv2.Sobel(guide, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(guide, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 1.0)
    w = np.exp(-(edge / tau) ** 2).astype(np.float32)
    filled = cv2.GaussianBlur(corr * w, (0, 0), fill) / (cv2.GaussianBlur(w, (0, 0), fill) + 1e-4)
    return filled * w


def clean_brightness(L, scale=SCALE):
    """Remove coarse moiré bands from L (float32, 0..255)."""
    H, W = L.shape
    X = np.log(L + LOG_OFFSET)
    small = cv2.resize(X, (W // scale, H // scale), interpolation=cv2.INTER_AREA)
    corr = edge_protect(suppress_peaks(small) - small, small)
    corr = cv2.resize(corr, (W, H), interpolation=cv2.INTER_CUBIC)
    return (L + LOG_OFFSET) * np.exp(corr) - LOG_OFFSET


def clean_colour(lab, radius=CHROMA_RADIUS, eps=CHROMA_EPS):
    """Smooth a/b along the edges of L. Changes `lab` in place and returns it."""
    guide = lab[..., 0] / 255
    for ch in (1, 2):
        lab[..., ch] = guided_filter(guide, lab[..., ch] / 255, radius, eps) * 255
    return lab


def clean_screen_moire(img_bgr, brightness=True, colour=True):
    """Remove screen moiré from a BGR uint8 image. Returns a new BGR uint8 image."""
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image (H x W x 3).")
    if min(img_bgr.shape[:2]) < SCALE * WINDOW:
        raise ValueError(f"Image must be at least {SCALE * WINDOW} pixels on each side.")
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0].copy()
    if colour:
        clean_colour(lab)          # guide = original L
    if brightness:
        lab[..., 0] = clean_brightness(L)
    lab = np.clip(np.round(lab), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
