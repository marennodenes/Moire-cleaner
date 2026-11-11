"""Compare the detector with hand-painted masks from label_patches.py.

Usage (from the repo root):
    python experiments/evaluate_labels.py                          # all masks
    python experiments/evaluate_labels.py --patch 64 --scale 0.25
    python experiments/evaluate_labels.py --threshold 5

The mask is resized to the image size used for detection, and each patch
gets a label from how much of it is painted:
    > 50 % painted -> moiré,   < 10 % -> clean,   in between -> skipped (unsure)

  found       = share of moiré patches the detector finds      (recall)
  correct     = share of detector hits that really are moiré   (precision)
  false hits  = share of clean patches the detector marks
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # find moire/
from moire.detection import THRESHOLD, analyze_patch

parser = argparse.ArgumentParser()
parser.add_argument("masks", nargs="*", help="mask files (default: all)")
parser.add_argument("--patch", type=int, default=128)
parser.add_argument("--scale", type=float, default=0.25)
parser.add_argument("--threshold", type=float, default=THRESHOLD)
args = parser.parse_args()

masks = [Path(m) for m in args.masks] or sorted(Path("data/real/labels").glob("*_mask.png"))
if not masks:
    raise SystemExit("No masks. Paint some with experiments/label_patches.py")


def evaluate(mask_file):
    image = Path("data/real") / (mask_file.name.removesuffix("_mask.png") + ".jpg")
    img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Could not read image for {mask_file}: {image}")
    if args.scale != 1.0:
        img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)

    P = args.patch
    tp = fp = fn = tn = 0
    for r in range(img.shape[0] // P):
        for c in range(img.shape[1] // P):
            painted = mask[r * P:(r + 1) * P, c * P:(c + 1) * P].mean()
            if 0.1 <= painted <= 0.5:       # unsure
                continue
            truth = painted > 0.5
            patch = img[r * P:(r + 1) * P, c * P:(c + 1) * P]
            hit = analyze_patch(patch, 0, threshold=args.threshold).intensity > 0
            tp += hit and truth
            fp += hit and not truth
            fn += truth and not hit
            tn += not hit and not truth
    return tp, fp, fn, tn


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


def row(name, tp, fp, fn, tn):
    print(f"{name:28s} {tp + fn:6d} {fp + tn:6d} {pct(tp, tp + fn):>7s} "
          f"{pct(tp, tp + fp):>8s} {pct(fp, fp + tn):>11s}")


print(f"patch {args.patch}, scale {args.scale}, threshold {args.threshold}")
print(f"{'image':28s} {'moiré':>6s} {'clean':>6s} {'found':>7s} {'correct':>8s} {'false hits':>11s}")
total = np.zeros(4, dtype=int)
for m in masks:
    counts = evaluate(m)
    total += counts
    row(m.name.removesuffix("_mask.png"), *counts)
if len(masks) > 1:
    row("TOTAL", *total)
