"""Remove moiré from photos of screens.

Main entry point: clean_screen_moire(img_bgr, mask=True) -> img_bgr.

Works in Lab, in two steps:
1. Colour bands (a, b): guided filter with L as guide, so colour edges stay sharp.
2. Brightness rings (L): estimate the background without text, divide out the rings.
With mask=True, only areas where moire.detection finds moiré are changed.
"""
import cv2
import numpy as np

CHROMA_RADIUS = 120      # colour smoothing radius (px)
CHROMA_MAX_CHANGE = 6.0  # colour change limit near tiles
RING_SCALE = 4           # downscale for ring estimate
RING_TEXT_THR = 15.0     # text vs ring threshold
MASK_PATCH = 512         # detector patch size (px)


def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))


def guided_filter(guide, src, radius, eps):
    """Edge-preserving smoothing of `src` along the edges of `guide` (He et al.)."""
    mean_i, mean_p = _box(guide, radius), _box(src, radius)
    a = (_box(guide * src, radius) - mean_i * mean_p) / (_box(guide * guide, radius) - mean_i**2 + eps)
    b = mean_p - a * mean_i
    return _box(a, radius) * guide + _box(b, radius)


def _ramp(x, start, width):
    """0 below `start`, 1 above `start + width`."""
    return np.clip((x - start) / width, 0, 1).astype(np.float32)


def _blur_ignoring(x, weight, sigma):
    """Gaussian blur of x that only uses pixels with weight > 0."""
    return cv2.GaussianBlur(x * weight, (0, 0), sigma) / (cv2.GaussianBlur(weight, (0, 0), sigma) + 1e-6)


def clean_colour(lab):
    """Remove colour bands from a float32 Lab image (in place).

    Neutral areas get a large radius (removes the bands), saturated areas a
    small one (keeps coloured text). Next to large coloured areas the change
    is limited, so neighbouring tiles don't bleed into each other.
    """
    before = lab[..., 1:].copy()
    guide = lab[..., 0] / 255

    def smooth(radius):
        return np.stack([guided_filter(guide, lab[..., c] / 255, radius, 1e-3) * 255 for c in (1, 2)], axis=2)

    big, small = smooth(CHROMA_RADIUS), smooth(CHROMA_RADIUS // 3)
    saturated = _ramp(np.hypot(small[..., 0] - 128, small[..., 1] - 128), 12, 15)
    use_small = cv2.GaussianBlur(saturated, (0, 0), 3)[..., None]
    change = use_small * small + (1 - use_small) * big - before

    near_tiles = _ramp(cv2.GaussianBlur(saturated, (0, 0), 40), 0.15, 0.15)[..., None]
    limited = np.clip(change, -CHROMA_MAX_CHANGE, CHROMA_MAX_CHANGE)
    lab[..., 1:] = before + near_tiles * limited + (1 - near_tiles) * change
    return lab


def flatten_rings(L):
    """Remove wide brightness rings from flat background (float32 L, 0..255)."""
    H, W = L.shape
    small = cv2.resize(L, (W // RING_SCALE, H // RING_SCALE), interpolation=cv2.INTER_AREA)

    # text = pixels far from the local median (blurred first, so the screen grid isn't text)
    blurred = cv2.GaussianBlur(small, (0, 0), 1.5)
    median = cv2.medianBlur(np.clip(blurred, 0, 255).astype(np.uint8), 15).astype(np.float32)
    text = cv2.dilate((np.abs(blurred - median) > RING_TEXT_THR).astype(np.uint8), np.ones((5, 5), np.uint8))
    background = 1.0 - text.astype(np.float32)

    # rings = background with rings / background without rings (edge-aware, no halos)
    fine = _blur_ignoring(small, background, 3)
    coarse = guided_filter(fine / 255, fine / 255, 100, 0.02) * 255
    gain = np.clip(coarse / np.maximum(fine, 1.0), 0.8, 1.25)
    gain[fine <= 5] = 1.0                                       # leave black areas alone

    flat = _ramp(cv2.GaussianBlur(background, (0, 0), 15), 0.5, 0.5)   # mostly background around
    gain = 1 + (gain - 1) * flat
    return L * cv2.resize(gain, (W, H), interpolation=cv2.INTER_CUBIC)


def _clean_grid(grid):
    """Patch grid (1 = moiré) -> one solid area per screen.

    Drops small clusters (false hits), closes gaps and fills the convex hull
    of each area (a screen is convex).
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    keep = np.isin(labels, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 4]).astype(np.uint8)

    padded = cv2.copyMakeBorder(keep, 3, 3, 3, 3, cv2.BORDER_REPLICATE)   # no shrinking at edges
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))[3:-3, 3:-3]

    n, labels = cv2.connectedComponents(closed, connectivity=8)
    for i in range(1, n):
        ys, xs = np.nonzero(labels == i)
        cv2.fillConvexPoly(closed, cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32)), 1)
    return closed


def moire_mask(img_bgr, patch_size=MASK_PATCH, detections=None):
    """Soft mask (H x W, 0..1) of where moire.detection finds moiré."""
    from moire.detection import analyze_image

    H, W = img_bgr.shape[:2]
    size = min(patch_size, H, W) // 4 * 4
    if detections is None:
        detections = analyze_image(img_bgr, patch_size=size)
    rows = max(r.y for r in detections) // size + 1
    cols = max(r.x for r in detections) // size + 1
    grid = np.zeros((rows, cols), np.uint8)
    for r in detections:
        grid[r.y // size, r.x // size] = bool(r.peaks)

    full = cv2.resize(_clean_grid(grid).astype(np.float32), (cols * size, rows * size),
                      interpolation=cv2.INTER_NEAREST)
    full = cv2.copyMakeBorder(full, 0, max(0, H - rows * size), 0, max(0, W - cols * size),
                              cv2.BORDER_REPLICATE)[:H, :W]
    return np.clip(cv2.GaussianBlur(full, (0, 0), size * 0.35), 0, 1)


def clean_screen_moire(img_bgr, mask=None, colour=True, rings=True):
    """Remove screen moiré from a BGR uint8 image. Returns a new image.

    mask: None = whole image, True = only where moiré is detected (recommended),
    or an H x W array with values 0..1.
    """
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image (H x W x 3).")
    if min(img_bgr.shape[:2]) < 256:
        raise ValueError("Image must be at least 256 pixels on each side.")
    if mask is True:
        mask = moire_mask(img_bgr)
    if mask is not None and not np.any(mask):
        return img_bgr.copy()

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if colour:
        clean_colour(lab)
    if rings:
        lab[..., 0] = flatten_rings(lab[..., 0])
    out = cv2.cvtColor(np.clip(np.round(lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    if mask is not None:   # blend in BGR so pixels with mask 0 stay exactly the original
        m = np.asarray(mask, np.float32)[..., None]
        out = np.round(m * out + (1 - m) * img_bgr).astype(np.uint8)
    return out
