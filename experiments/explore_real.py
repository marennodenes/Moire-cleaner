"""Run the detector on a real photo and see where it finds moiré.

Usage (from the repo root):
    python experiments/explore_real.py data/real/photo.jpg
    python experiments/explore_real.py data/real/photo.jpg --patch 64 --threshold 8

Shows:
  1. the image with moiré intensity per patch on top
  2. a histogram of the highest ratio in each patch (to tune the threshold)
  3. spectra of the strongest, a middle and the weakest patch, peaks circled

There is no ground truth for real photos: check by eye that patches with
visible moiré light up and clean ones stay dark.
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # find moire/
from moire.detection import THRESHOLD, analyze_patch, ratio_map

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--patch", type=int, default=128, help="patch size in pixels")
parser.add_argument("--threshold", type=float, default=THRESHOLD)
parser.add_argument("--scale", type=float, default=1.0,
                    help="resize before detection, e.g. 0.25 for big phone photos")
args = parser.parse_args()

# ---------- load image (colour -> grayscale, 0..1) ----------
img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise SystemExit(f"Could not read image: {args.image}")
if args.scale != 1.0:
    # INTER_AREA averages pixels, so fine detail is removed instead of aliased
    img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
img = img.astype(np.float32) / 255
P = args.patch

# ---------- split into a grid of patches (no overlap, edges dropped) ----------
# Simple local version; the real one belongs in moire/patches.py.
rows, cols = img.shape[0] // P, img.shape[1] // P
if rows == 0 or cols == 0:
    raise SystemExit(f"Image {img.shape} is smaller than one patch ({P})")

intensity = np.zeros((rows, cols))
max_ratio = np.zeros((rows, cols))
results = {}
for r in range(rows):
    for c in range(cols):
        patch = img[r * P:(r + 1) * P, c * P:(c + 1) * P]
        res = analyze_patch(patch, patch_id=r * cols + c, threshold=args.threshold)
        results[(r, c)] = res
        intensity[r, c] = res.intensity
        max_ratio[r, c] = ratio_map(patch).max()

# ---------- print summary ----------
n = rows * cols
found = int((intensity > 0).sum())
print(f"{args.image}: {img.shape[1]}x{img.shape[0]} px, {n} patches of {P}x{P}")
print(f"threshold {args.threshold}: moiré found in {found}/{n} patches")
print("highest ratio per patch, percentiles:")
for q in (10, 50, 90, 99):
    print(f"  {q:3d}%: {np.percentile(max_ratio, q):6.1f}")

order = sorted(results, key=lambda k: max_ratio[k])
picks = [("strongest", order[-1]), ("middle", order[n // 2]), ("weakest", order[0])]
for label, (r, c) in picks:
    res = results[(r, c)]
    top = [(p.fy, p.fx, round(p.ratio, 1)) for p in res.peaks[:5]]
    print(f"{label:9s} patch ({r},{c}): intensity {res.intensity:.2f}, peaks {top}")

# ---------- plot ----------
fig = plt.figure(figsize=(14, 8))
grid = fig.add_gridspec(2, 3)

ax = fig.add_subplot(grid[0, :2])
ax.imshow(img, cmap="gray")
overlay = ax.imshow(intensity, cmap="magma", alpha=0.5, vmin=0, vmax=1,
                    extent=[0, cols * P, rows * P, 0], interpolation="nearest")
fig.colorbar(overlay, ax=ax, label="moiré intensity")
for label, (r, c) in picks:
    ax.add_patch(plt.Rectangle((c * P, r * P), P, P, fill=False, ec="cyan", lw=1.5))
    ax.text(c * P + 3, r * P + 12, label, color="cyan", fontsize=8)
ax.set_title(f"intensity per {P}x{P} patch")
ax.axis("off")

ax = fig.add_subplot(grid[0, 2])
ax.hist(max_ratio.ravel(), bins=40, color="gray")
ax.axvline(args.threshold, color="red", label=f"threshold {args.threshold}")
ax.set_xlabel("highest ratio in patch")
ax.set_ylabel("patches")
ax.legend()

h = P // 2
ext = [-h - 0.5, P - h - 0.5, P - h - 0.5, -h - 0.5]   # axes in frequency units
for j, (label, (r, c)) in enumerate(picks):
    patch = img[r * P:(r + 1) * P, c * P:(c + 1) * P]
    win = (patch - patch.mean()) * np.outer(np.hanning(P), np.hanning(P))
    S = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(win))))
    ax = fig.add_subplot(grid[1, j])
    ax.imshow(S, cmap="gray", interpolation="nearest", extent=ext)
    for p in results[(r, c)].peaks:
        for fy, fx in ((p.fy, p.fx), (-p.fy, -p.fx)):
            ax.add_patch(plt.Circle((fx, fy), P / 40, fill=False, ec="red"))
    ax.set_title(f"{label} ({r},{c}), intensity {results[(r, c)].intensity:.2f}",
                 fontsize=9)
    ax.set_xlabel("fx", fontsize=8)
    ax.set_ylabel("fy", fontsize=8)
    ax.tick_params(labelsize=7)

plt.tight_layout()
out = Path("docs/figures") / f"explore_real_{Path(args.image).stem}_{P}_s{args.scale:g}.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=120)
print(f"saved {out}")
plt.show()
