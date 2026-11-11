"""Detect moiré in a photo of a screen, patch by patch.

Main entry point: analyze_image(img_bgr) -> list[PatchResult].

Per patch: find isolated peaks in the spectrum (brightness and colour
channels), keep only coloured peaks that are not edge lines, then use
neighbouring patches to remove lone hits and fill gaps.
"""
from collections import namedtuple

import cv2
import numpy as np

from moire.datatypes import Peak, PatchResult

SCALE = 0.25                # detection runs on the image scaled to this
WINDOWS = (1, 2, 4)         # window sizes (x patch) for the brightness channel
PEAK_THRESHOLD = 5.0        # peak / local median
NEIGHBOUR_THRESHOLD = 3.5   # same, when checking a neighbour's frequency
COLOUR_MIN = 0.4            # colour / brightness energy at the peak
EDGE_STEP = 3               # bins checked inwards/outwards for edge lines
NEIGHBOUR_SUPPORT = 2       # neighbours that must share a frequency
SATURATION = 50.0           # ratio that gives intensity 1.0

SpectrumPeak = namedtuple("SpectrumPeak", "fy fx ratio")   # in FFT bins
_Found = namedtuple("_Found", "k s fy fx ratio")           # k: window, s: its side


# ---------- spectrum of one window ----------

def magnitude(patch):
    """|fftshift(fft2)| of a Hann-windowed patch with the mean removed."""
    img = np.asarray(patch, dtype=np.float32)
    h, w = img.shape
    img = (img - img.mean()) * np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    return np.abs(np.fft.fftshift(np.fft.fft2(img))).astype(np.float32)


def ratio_map(patch, ksize=5, dc_radius=6):
    """Spectrum magnitude divided by its local median (centred, DC disc = 0).

    ksize must be 3 or 5 (cv2.medianBlur only supports these for float32).
    """
    return _ratio_of(magnitude(patch), ksize, dc_radius)


def _ratio_of(mag, ksize=5, dc_radius=6):
    """ratio_map for an already computed magnitude spectrum."""
    if ksize not in (3, 5):
        raise ValueError(f"ksize must be 3 or 5, got {ksize}")
    h, w = mag.shape
    ratio = mag / (cv2.medianBlur(mag, ksize) + 1e-9)
    yy, xx = np.mgrid[-(h // 2):h - h // 2, -(w // 2):w - w // 2]
    ratio[np.hypot(yy, xx) < dc_radius] = 0
    return ratio


def detect_peaks(patch, threshold=PEAK_THRESHOLD, ksize=5, dc_radius=6):
    """Isolated peaks in a single-channel patch, in FFT bins, strongest first.

    One peak per mirror pair (fy > 0, or fy == 0 and fx > 0; at the Nyquist
    row fy can be -h//2 so it stays inside the spectrum).
    """
    return _peaks_in(ratio_map(patch, ksize, dc_radius), threshold)


def _peaks_in(ratio, threshold):
    """detect_peaks for an already computed ratio map."""
    h, w = ratio.shape
    # Hann widens peaks, so neighbours are also high: keep local maxima only.
    is_max = ratio >= cv2.dilate(ratio, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(is_max & (ratio > threshold))

    peaks = {}
    for y, x in zip(ys, xs):
        fy, fx = int(y - h // 2), int(x - w // 2)
        if fy < 0 or (fy == 0 and fx < 0):
            fy, fx = -fy, -fx
        # the mirror of the Nyquist row/column wraps around to the other side
        fy = (fy + h // 2) % h - h // 2
        fx = (fx + w // 2) % w - w // 2
        r = float(ratio[y, x])
        if (fy, fx) not in peaks or r > peaks[(fy, fx)].ratio:
            peaks[(fy, fx)] = SpectrumPeak(fy, fx, r)
    return sorted(peaks.values(), key=lambda p: p.ratio, reverse=True)


def estimate_intensity(peaks, threshold=NEIGHBOUR_THRESHOLD, saturation=SATURATION):
    """Strongest peak ratio on a log scale: threshold -> 0, saturation -> 1."""
    if not peaks:
        return 0.0
    r = max(p.ratio for p in peaks)
    value = np.log(r / threshold) / np.log(saturation / threshold)
    return float(np.clip(value, 0.0, 1.0))


# ---------- whole image ----------

def _window(lab, r, c, k, P):
    """k*P window centred on patch (r, c), kept inside the image, shrunk to P.
    Also returns its side s before shrinking."""
    H, W = lab.shape[:2]
    s = min(k * P, H, W)
    y0 = int(np.clip(r * P + P // 2 - s // 2, 0, H - s))
    x0 = int(np.clip(c * P + P // 2 - s // 2, 0, W - s))
    win = lab[y0:y0 + s, x0:x0 + s]
    if s != P:
        win = cv2.resize(win, (P, P), interpolation=cv2.INTER_AREA)
    return win, s


def _colour(mags, y, x):
    """Colour / brightness energy at spectrum position (y, x)."""
    return np.hypot(mags[1][y, x], mags[2][y, x]) / (mags[0][y, x] + 1e-9)


def _on_line(mag, p, y, x):
    """True if the peak continues inwards/outwards along its direction."""
    P = mag.shape[0]
    rho = np.hypot(p.fy, p.fx)
    for d in (-EDGE_STEP, EDGE_STEP):
        yy = int(round(p.fy * (1 + d / rho))) + P // 2
        xx = int(round(p.fx * (1 + d / rho))) + P // 2
        if 0 <= yy < P and 0 <= xx < P and mag[yy, xx] > 0.5 * mag[y, x]:
            return True
    return False


def _find(lab, r, c, P):
    """Pass 1: accepted peaks of patch (r, c)."""
    found = []
    # bigger windows show coarse moiré; colour channels show moiré behind text
    for k in WINDOWS:
        win, s = _window(lab, r, c, k, P)
        mags = [magnitude(win[..., ch]) for ch in range(3)]
        for ch in ((0, 1, 2) if k == 1 else (0,)):
            ratio = _ratio_of(mags[ch])
            for p in _peaks_in(ratio, PEAK_THRESHOLD):
                y, x = (p.fy + P // 2) % P, (p.fx + P // 2) % P
                if _on_line(mags[ch], p, y, x) or _colour(mags, y, x) <= COLOUR_MIN:
                    continue
                found.append(_Found(k, s, p.fy, p.fx, p.ratio))
    return found


def _has_frequency(lab, r, c, P, f):
    """Pass 2: does patch (r, c) have a coloured peak within +-2 bins of f?"""
    win, s = _window(lab, r, c, f.k, P)
    mags = [magnitude(win[..., ch]) for ch in range(3)]
    y0, x0 = f.fy + P // 2, f.fx + P // 2
    ys = slice(max(0, y0 - 2), min(P, y0 + 3))
    xs = slice(max(0, x0 - 2), min(P, x0 + 3))
    for ch in range(3):
        box = _ratio_of(mags[ch])[ys, xs]
        if box.size == 0 or box.max() < NEIGHBOUR_THRESHOLD:
            continue
        dy, dx = np.unravel_index(np.argmax(box), box.shape)
        y, x = ys.start + dy, xs.start + dx
        if _colour(mags, y, x) > COLOUR_MIN:
            return _Found(f.k, s, y - P // 2, x - P // 2, float(box.max()))
    return None


def _neighbours(r, c):
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def _close(f, g):
    """Same window size and within +-2 bins (also as mirror)."""
    return f.k == g.k and any(abs(f.fy - s * g.fy) <= 2 and abs(f.fx - s * g.fx) <= 2
                              for s in (1, -1))


def _to_peaks(found, patch_size):
    """Convert bins to cycles per original pixel; merge duplicates."""
    peaks = []
    for f in sorted(found, key=lambda f: f.ratio, reverse=True):
        fy, fx = f.fy / f.s * SCALE, f.fx / f.s * SCALE
        if fy < 0 or (fy == 0 and fx < 0):
            fy, fx = -fy, -fx
        tol = 1.5 / patch_size
        if not any(abs(fy - p.fy) < tol and abs(fx - p.fx) < tol for p in peaks):
            peaks.append(Peak(float(fy), float(fx), float(f.ratio)))
    return peaks


def analyze_image(img_bgr, patch_size=512):
    """One PatchResult per patch, row by row (see docs/interface-detection.md).

    img_bgr: colour image from cv2.imread. patch_size: in original pixels,
    multiple of 4 and >= 128. Patches that do not fit at the edges are left out.
    """
    img = np.asarray(img_bgr)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img_bgr must be a colour image (H x W x 3)")
    if patch_size % 4 or patch_size < 128:
        raise ValueError(f"patch_size must be a multiple of 4 and >= 128, got {patch_size}")
    P = int(patch_size * SCALE)
    small = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    if small.dtype == np.uint8:
        small = small.astype(np.float32) / 255
    lab = cv2.cvtColor(small.astype(np.float32), cv2.COLOR_BGR2LAB)
    rows, cols = img.shape[0] // patch_size, img.shape[1] // patch_size

    # pass 1, then drop hits without any neighbouring hit
    found = {(r, c): _find(lab, r, c, P) for r in range(rows) for c in range(cols)}
    hit = {key: bool(v) for key, v in found.items()}
    found = {(r, c): v if any(hit.get(n) for n in _neighbours(r, c)) else []
             for (r, c), v in found.items()}

    # pass 2: frequencies shared by enough neighbours, lower threshold
    extra = {}
    for (r, c), fs in found.items():
        if fs:
            continue
        near = [found.get(n, []) for n in _neighbours(r, c)]
        candidates = {f for group in near for f in group}
        for f in candidates:
            if sum(any(_close(f, g) for g in group) for group in near) < NEIGHBOUR_SUPPORT:
                continue
            g = _has_frequency(lab, r, c, P, f)
            if g:
                extra.setdefault((r, c), []).append(g)
    found.update(extra)

    results = []
    for r in range(rows):
        for c in range(cols):
            peaks = _to_peaks(found[(r, c)], patch_size)
            results.append(PatchResult(
                patch_id=r * cols + c, y=r * patch_size, x=c * patch_size,
                size=patch_size, peaks=peaks, intensity=estimate_intensity(peaks)))
    return results
