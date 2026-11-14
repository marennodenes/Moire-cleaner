"""Compare the current pipeline with moire.screen_clean on the real photos.

Usage (from the repo root):
    python experiments/compare_screen_clean.py                 # all photos in data/real
    python experiments/compare_screen_clean.py github_screen   # one photo

Saves results/screen_clean/<photo>.jpg: original | current pipeline | screen_clean,
as a downscaled overview and a full-resolution crop from the centre.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # find moire/
from moire.pipeline import moire_filter
from moire.screen_clean import clean_screen_moire

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "screen_clean"
TILE = 600


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (255, 255, 255), -1)
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    return img


def overview_and_crop(img):
    H, W = img.shape[:2]
    s = min(H, W)
    over = cv2.resize(img[(H - s) // 2:(H + s) // 2, (W - s) // 2:(W + s) // 2], (TILE, TILE),
                      interpolation=cv2.INTER_AREA)
    c = min(1200, s)
    crop = cv2.resize(img[H // 2 - c // 2:H // 2 + c // 2, W // 2 - c // 2:W // 2 + c // 2], (TILE, TILE),
                      interpolation=cv2.INTER_AREA)
    return over, crop


def main():
    names = sys.argv[1:] or sorted(p.stem for p in (ROOT / "data" / "real").glob("*.jpg"))
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        img = cv2.imread(str(ROOT / "data" / "real" / f"{name}.jpg"))
        if img is None:
            print(f"Could not read data/real/{name}.jpg")
            continue
        t = time.time()
        current = moire_filter(img)
        t_cur = time.time() - t
        t = time.time()
        clean = clean_screen_moire(img)
        t_new = time.time() - t
        rows = [[], []]
        for im, title in ((img, "original"), (current, "current pipeline"), (clean, "screen_clean")):
            over, crop = overview_and_crop(im)
            rows[0].append(label(over, title))
            rows[1].append(label(crop, title + " (crop)"))
        cv2.imwrite(str(OUT / f"{name}.jpg"), np.vstack([np.hstack(r) for r in rows]))
        print(f"{name:20s} current {t_cur:5.1f}s  screen_clean {t_new:5.1f}s  -> results/screen_clean/{name}.jpg")


if __name__ == "__main__":
    main()
