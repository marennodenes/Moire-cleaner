"""Does looking at a bigger window help with coarse moiré?

For every patch we also run the detector on a window 2x and 4x as big,
centred on the patch and shrunk back to the patch size. Coarse stripes then
get enough periods to give a peak. A patch counts as "found" if any of the
windows finds a peak.

Usage (from the repo root):
    python experiments/multiscale.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # find moire/
from moire.detection import detect_peaks

SCALE, P = 0.25, 128
PHOTOS = ["assignment", "github", "opencv", "eclass"]


def window(img, r, c, k):
    """k*P window centred on patch (r, c), pushed inside the image, shrunk to P."""
    H, W = img.shape
    size = min(k * P, H, W)
    cy, cx = r * P + P // 2, c * P + P // 2
    y0 = int(np.clip(cy - size // 2, 0, H - size))
    x0 = int(np.clip(cx - size // 2, 0, W - size))
    win = img[y0:y0 + size, x0:x0 + size]
    return cv2.resize(win, (P, P), interpolation=cv2.INTER_AREA) if size != P else win


def hit_grid(img, ks):
    rows, cols = img.shape[0] // P, img.shape[1] // P
    grid = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            grid[r, c] = any(detect_peaks(window(img, r, c, k)) for k in ks)
    return grid


def load(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255


photos = {name: load(f"data/real/{name}_screen.jpg") for name in PHOTOS}
# Photos without screens (same phone): every peak found here is a false hit.
cleans = {p.stem.removeprefix("clean_"): load(p) for p in sorted(Path("data/real").glob("clean_*.jpg"))}

print(f"share of patches with a peak (scale {SCALE}, patch {P})")
print(f"{'windows':10s} " + " ".join(f"{n:>10s}" for n in PHOTOS) + " |"
      + " ".join(f"{'false ' + n:>13s}" for n in cleans))
for ks in [(1,), (2,), (4,), (1, 2), (1, 2, 4)]:
    rates = [hit_grid(img, ks).mean() for img in photos.values()]
    false = [hit_grid(img, ks).mean() for img in cleans.values()]
    print(f"{'+'.join(f'{k}x' for k in ks):10s} " +
          " ".join(f"{100 * x:9.0f}%" for x in rates) + " |" +
          " ".join(f"{100 * x:12.0f}%" for x in false))

print("\ngithub, windows 1x+2x+4x (#=found):")
for row in hit_grid(photos["github"], (1, 2, 4)):
    print("   " + "".join("#" if x else "." for x in row))
for name, img in cleans.items():
    print(f"\nfalse hits in {name}, 1x / 1x+2x+4x:")
    for a, b in zip(hit_grid(img, (1,)), hit_grid(img, (1, 2, 4))):
        print("   " + "".join("#" if x else "." for x in a) + "   " + "".join("#" if x else "." for x in b))
