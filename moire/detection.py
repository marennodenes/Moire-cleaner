"""Find moiré in a photo of a screen.

Main function: analyze_image(img_bgr) -> list[PatchResult].

The image is made smaller and split into squares (patches).
1. Each patch: look for repeating patterns (sharp peaks in the spectrum).
2. Remove hits in patches with no hits around them (usually not moiré).
3. Patches with no hits get the pattern their neighbours have, if it is
   also weakly there.
"""
from collections import namedtuple

import cv2
import numpy as np

from moire.datatypes import Peak, PatchResult

SCALE = 0.25                # image is made 4x smaller first
WINDOWS = (1, 2, 4)         # also look at 2x and 4x larger areas (coarse moiré)
PEAK_THRESHOLD = 5.0        # how much a peak must stand out
NEIGHBOUR_THRESHOLD = 3.5   # same, lower, in step 3
COLOUR_MIN = 0.4            # how coloured a peak must be
GREY_MIN_RATIO = 30.0       # grey peaks must stand out this much
EDGE_STEP = 3               # used to tell edges from moiré
NEIGHBOUR_SUPPORT = 2       # neighbours that must agree in step 3
SEARCH_RADIUS = 2           # how close two frequencies count as the same
SATURATION = 50.0           # peak strength that gives intensity 1.0

SpectrumPeak = namedtuple("SpectrumPeak", "fy fx ratio")   # position in the spectrum
# a peak found in a patch: which area size, its side in pixels, position, strength
_Hit = namedtuple("_Hit", "window side fy fx ratio")


# ---------- spectrum ----------

def magnitude(patch):
    """Spectrum (FFT magnitude) of a patch, centred."""
    img = np.asarray(patch, dtype=np.float32)
    h, w = img.shape
    img = (img - img.mean()) * np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    return np.abs(np.fft.fftshift(np.fft.fft2(img))).astype(np.float32)


def ratio_map(patch, ksize=5, dc_radius=6):
    """How much each point of the spectrum stands out from its surroundings.

    ksize must be 3 or 5.
    """
    return _ratio_of(magnitude(patch), ksize, dc_radius)


def detect_peaks(patch, threshold=PEAK_THRESHOLD, ksize=5, dc_radius=6):
    """Peaks that stand out in a one-channel patch, strongest first.

    Each peak has a mirror copy in the spectrum; only one of the two is returned.
    """
    return _peaks_in(ratio_map(patch, ksize, dc_radius), threshold)


def estimate_intensity(peaks, threshold=NEIGHBOUR_THRESHOLD, saturation=SATURATION):
    """Moiré strength from 0 (none) to 1 (strong), from the strongest peak."""
    if not peaks:
        return 0.0
    r = max(p.ratio for p in peaks)
    value = np.log(r / threshold) / np.log(saturation / threshold)
    return float(np.clip(value, 0.0, 1.0))


def _ratio_of(mag, ksize=5, dc_radius=6):
    """Same as ratio_map, from a spectrum."""
    if ksize not in (3, 5):
        raise ValueError(f"ksize must be 3 or 5, got {ksize}")
    h, w = mag.shape
    ratio = mag / (cv2.medianBlur(mag, ksize) + 1e-9)
    yy, xx = np.mgrid[-(h // 2):h - h // 2, -(w // 2):w - w // 2]
    ratio[np.hypot(yy, xx) < dc_radius] = 0
    return ratio


def _peaks_in(ratio, threshold):
    """Same as detect_peaks, from a ratio map."""
    h, w = ratio.shape
    # keep only the highest point of each peak
    is_max = ratio >= cv2.dilate(ratio, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(is_max & (ratio > threshold))

    peaks = {}
    for y, x in zip(ys, xs):
        fy, fx = int(y - h // 2), int(x - w // 2)
        if fy < 0 or (fy == 0 and fx < 0):
            fy, fx = -fy, -fx
        # keep the position inside the spectrum
        fy = (fy + h // 2) % h - h // 2
        fx = (fx + w // 2) % w - w // 2
        r = float(ratio[y, x])
        if (fy, fx) not in peaks or r > peaks[(fy, fx)].ratio:
            peaks[(fy, fx)] = SpectrumPeak(fy, fx, r)
    return sorted(peaks.values(), key=lambda p: p.ratio, reverse=True)


# ---------- is a peak moiré? ----------

def _colour(mags, y, x):
    """How coloured the peak at (y, x) is."""
    return np.hypot(mags[1][y, x], mags[2][y, x]) / (mags[0][y, x] + 1e-9)


def _accepted(mags, y, x, ratio):
    """Moiré is coloured; grey peaks only count if very strong (wood, fabric are weaker)."""
    return _colour(mags, y, x) > COLOUR_MIN or ratio >= GREY_MIN_RATIO


def _on_line(mag, peak):
    """True if the peak is part of a line in the spectrum: an edge, not moiré."""
    size = mag.shape[0]
    y, x = (peak.fy + size // 2) % size, (peak.fx + size // 2) % size
    rho = np.hypot(peak.fy, peak.fx)
    for d in (-EDGE_STEP, EDGE_STEP):
        yy = int(round(peak.fy * (1 + d / rho))) + size // 2
        xx = int(round(peak.fx * (1 + d / rho))) + size // 2
        if 0 <= yy < size and 0 <= xx < size and mag[yy, xx] > 0.5 * mag[y, x]:
            return True
    return False


# ---------- one patch ----------

def _spectra(lab, row, col, window, patch):
    """Spectra (L, a, b) of the area around patch (row, col).

    The area is `window` times the patch size, scaled down to patch size.
    Returns (area side in pixels, spectra).
    """
    H, W = lab.shape[:2]
    side = min(window * patch, H, W)
    y0 = int(np.clip(row * patch + patch // 2 - side // 2, 0, H - side))
    x0 = int(np.clip(col * patch + patch // 2 - side // 2, 0, W - side))
    win = lab[y0:y0 + side, x0:x0 + side]
    if side != patch:
        win = cv2.resize(win, (patch, patch), interpolation=cv2.INTER_AREA)
    return side, [magnitude(win[..., ch]) for ch in range(3)]


def _find_hits(lab, row, col, patch):
    """Step 1: moiré peaks in patch (row, col)."""
    hits = []
    for window in WINDOWS:
        side, mags = _spectra(lab, row, col, window, patch)
        # colour channels find moiré behind text; larger areas use brightness only
        channels = (0, 1, 2) if window == 1 else (0,)
        for ch in channels:
            for p in _peaks_in(_ratio_of(mags[ch]), PEAK_THRESHOLD):
                y, x = (p.fy + patch // 2) % patch, (p.fx + patch // 2) % patch
                if not _on_line(mags[ch], p) and _accepted(mags, y, x, p.ratio):
                    hits.append(_Hit(window, side, p.fy, p.fx, p.ratio))
    return hits


def _weak_hit_near(spectra, hit, patch):
    """Step 3: a weaker moiré peak close to `hit`, or None."""
    side, mags = spectra
    y0, x0 = hit.fy + patch // 2, hit.fx + patch // 2
    ys = slice(max(0, y0 - SEARCH_RADIUS), min(patch, y0 + SEARCH_RADIUS + 1))
    xs = slice(max(0, x0 - SEARCH_RADIUS), min(patch, x0 + SEARCH_RADIUS + 1))
    for ch in range(3):
        box = _ratio_of(mags[ch])[ys, xs]
        if box.size == 0 or box.max() < NEIGHBOUR_THRESHOLD:
            continue
        dy, dx = np.unravel_index(np.argmax(box), box.shape)
        y, x = ys.start + dy, xs.start + dx
        if _accepted(mags, y, x, float(box.max())):
            return _Hit(hit.window, side, y - patch // 2, x - patch // 2, float(box.max()))
    return None


# ---------- neighbours ----------

def _neighbours(row, col):
    return [(row + dr, col + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def _same_frequency(a, b):
    """True if two hits are the same pattern."""
    return a.window == b.window and any(
        abs(a.fy - s * b.fy) <= SEARCH_RADIUS and abs(a.fx - s * b.fx) <= SEARCH_RADIUS
        for s in (1, -1))


def _drop_lonely(hits):
    """Step 2: remove hits in patches with no hits around them."""
    return {key: found if any(hits.get(n) for n in _neighbours(*key)) else []
            for key, found in hits.items()}


def _fill_gaps(lab, hits, patch):
    """Step 3: patches with no hits get the pattern their neighbours share."""
    extra = {}
    for (row, col), found in hits.items():
        if found:
            continue
        near = [hits.get(n, []) for n in _neighbours(row, col)]
        spectra = {}   # computed once per area size
        for candidate in {h for group in near for h in group}:
            support = sum(any(_same_frequency(candidate, h) for h in group) for group in near)
            if support < NEIGHBOUR_SUPPORT:
                continue
            if candidate.window not in spectra:
                spectra[candidate.window] = _spectra(lab, row, col, candidate.window, patch)
            weak = _weak_hit_near(spectra[candidate.window], candidate, patch)
            if weak:
                extra.setdefault((row, col), []).append(weak)
    return {**hits, **extra}


# ---------- whole image ----------

def _to_peaks(hits, patch_size):
    """Hits -> Peak list in cycles per pixel, strongest first, no duplicates."""
    peaks = []
    tol = 1.5 / patch_size
    for h in sorted(hits, key=lambda h: h.ratio, reverse=True):
        fy, fx = h.fy / h.side * SCALE, h.fx / h.side * SCALE
        if fy < 0 or (fy == 0 and fx < 0):
            fy, fx = -fy, -fx
        if not any(abs(fy - p.fy) < tol and abs(fx - p.fx) < tol for p in peaks):
            peaks.append(Peak(float(fy), float(fx), float(h.ratio)))
    return peaks


def analyze_image(img_bgr, patch_size=512):
    """One PatchResult per patch, row by row (see docs/interface-detection.md).

    img_bgr: colour image from cv2.imread.
    patch_size: in pixels, a multiple of 4 and at least 128.
    Patches that don't fit at the right/bottom edge are left out.
    """
    img = np.asarray(img_bgr)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img_bgr must be a colour image (H x W x 3)")
    if patch_size % 4 or patch_size < 128:
        raise ValueError(f"patch_size must be a multiple of 4 and >= 128, got {patch_size}")

    small = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    if small.dtype == np.uint8:
        small = small.astype(np.float32) / 255
    lab = cv2.cvtColor(small.astype(np.float32), cv2.COLOR_BGR2LAB)
    patch = int(patch_size * SCALE)                      # patch side in the small image
    rows, cols = img.shape[0] // patch_size, img.shape[1] // patch_size

    hits = {(r, c): _find_hits(lab, r, c, patch) for r in range(rows) for c in range(cols)}
    hits = _drop_lonely(hits)
    hits = _fill_gaps(lab, hits, patch)

    results = []
    for r in range(rows):
        for c in range(cols):
            peaks = _to_peaks(hits[(r, c)], patch_size)
            results.append(PatchResult(
                patch_id=r * cols + c, y=r * patch_size, x=c * patch_size,
                size=patch_size, peaks=peaks, intensity=estimate_intensity(peaks)))
    return results
