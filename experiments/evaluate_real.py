"""Measure the detector (moire.detection.analyze_image) on the real photos.

Usage (from the repo root):
    python experiments/evaluate_real.py          # photos the settings were tuned on
    python experiments/evaluate_real.py --test   # held-out photos (data/real/test)

Each photo has a box around the part with visible moiré (in fractions of
width/height), or None if it has no visible moiré. Patches fully inside the
box count as moiré, patches fully outside as clean (every hit there is
false). Patches on the box border are skipped.
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # find moire/
from moire.detection import analyze_image

# photo (under data/real): box (x0, y0, x1, y1) with visible moiré, or None
TUNING = {
    "assignment_screen": (0, 0, 1, 1),
    "github_screen": (0, 0, 1, 0.87),
    "opencv_screen": (0, 0.16, 1, 1),
    "eclass_screen": (0, 0.11, 1, 1),
    "clean_wood": None,
    "clean_plants": None,
}
TEST = {
    "test/screen_signage": (0.17, 0.13, 0.82, 0.95),
    "test/screen_laptop_pdf": None,      # screen, but very little visible moiré
    "test/screen_dark_chat": None,       # screen, but very little visible moiré
    "test/paper_sign": None,
    "test/fabric_white": None,
}


def count(name, box, patch_size):
    """[hits, patches] for patches with moiré and for clean patches."""
    bgr = cv2.imread(f"data/real/{name}.jpg")
    if bgr is None:
        raise SystemExit(f"Could not read data/real/{name}.jpg")
    H, W = bgr.shape[:2]
    moire, clean = [0, 0], [0, 0]
    for res in analyze_image(bgr, patch_size):
        y0, x0, y1, x1 = res.y, res.x, res.y + res.size, res.x + res.size
        if box is None:
            n = clean
        else:
            bx0, by0, bx1, by1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
            if bx0 <= x0 and x1 <= bx1 and by0 <= y0 and y1 <= by1:
                n = moire
            elif x1 <= bx0 or x0 >= bx1 or y1 <= by0 or y0 >= by1:
                n = clean
            else:
                continue
        n[0] += bool(res.peaks)
        n[1] += 1
    return moire, clean


def pct(n):
    return f"{100 * n[0] / n[1]:3.0f}% of {n[1]:3d}" if n[1] else "-"


parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="use the held-out photos")
parser.add_argument("--patch", type=int, default=512, help="patch size in original pixels")
args = parser.parse_args()

print(f"{'photo':26s} {'moiré found':>13s} {'false hits':>13s}")
total_moire, total_clean = [0, 0], [0, 0]
for name, box in (TEST if args.test else TUNING).items():
    moire, clean = count(name, box, args.patch)
    for t, n in ((total_moire, moire), (total_clean, clean)):
        t[0] += n[0]
        t[1] += n[1]
    print(f"{name:26s} {pct(moire):>13s} {pct(clean):>13s}")
print(f"{'TOTAL':26s} {pct(total_moire):>13s} {pct(total_clean):>13s}")
