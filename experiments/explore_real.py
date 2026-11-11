"""Run the detector on a real photo and see where it finds moiré.

Usage (from the repo root):
    python experiments/explore_real.py data/real/github_screen.jpg
    python experiments/explore_real.py data/real/github_screen.jpg --patch 256

Shows:
  1. the image with moiré intensity per patch on top
  2. a histogram of the intensities
  3. spectra of the strongest, a middle and a patch without moiré,
     with the reported peaks circled

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
from moire.detection import SCALE, analyze_image

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--patch", type=int, default=512, help="patch size in original pixels")
args = parser.parse_args()

bgr = cv2.imread(args.image)
if bgr is None:
    raise SystemExit(f"Could not read image: {args.image}")
results = analyze_image(bgr, patch_size=args.patch)
if not results:
    raise SystemExit(f"Image {bgr.shape[:2]} is smaller than one patch ({args.patch})")

# ---------- summary ----------
S = args.patch
rows = max(r.y for r in results) // S + 1
cols = max(r.x for r in results) // S + 1
intensity = np.array([r.intensity for r in results]).reshape(rows, cols)
found = sum(bool(r.peaks) for r in results)
print(f"{args.image}: {bgr.shape[1]}x{bgr.shape[0]} px, {len(results)} patches of {S}x{S}")
print(f"moiré found in {found}/{len(results)} patches")

order = sorted(results, key=lambda r: (len(r.peaks) > 0, r.intensity))
with_moire = [r for r in order if r.peaks]
picks = [("strongest", order[-1]),
         ("middle", with_moire[len(with_moire) // 2] if with_moire else order[-1]),
         ("none", order[0])]
for label, res in picks:
    top = [(round(p.fy, 3), round(p.fx, 3), round(p.ratio, 1)) for p in res.peaks[:5]]
    print(f"{label:9s} patch {res.patch_id}: intensity {res.intensity:.2f}, "
          f"peaks (cycles/px) {top}")

# ---------- plot ----------
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize=(14, 8))
grid = fig.add_gridspec(2, 3)

ax = fig.add_subplot(grid[0, :2])
ax.imshow(rgb)
overlay = ax.imshow(intensity, cmap="magma", alpha=0.5, vmin=0, vmax=1,
                    extent=[0, cols * S, rows * S, 0], interpolation="nearest")
fig.colorbar(overlay, ax=ax, label="moiré intensity")
for label, res in picks:
    ax.add_patch(plt.Rectangle((res.x, res.y), S, S, fill=False, ec="cyan", lw=1.5))
    ax.text(res.x + S * 0.03, res.y + S * 0.12, label, color="cyan", fontsize=8)
ax.set_title(f"intensity per {S}x{S} patch")
ax.axis("off")

ax = fig.add_subplot(grid[0, 2])
ax.hist([r.intensity for r in results], bins=20, range=(0, 1), color="gray")
ax.set_xlabel("intensity")
ax.set_ylabel("patches")

# Spectrum of each picked patch at detection scale; a peak at f cycles/pixel
# is at bin f * S in this spectrum (see Peak in moire/datatypes.py).
P = int(S * SCALE)
h = P // 2
ext = [-h - 0.5, P - h - 0.5, P - h - 0.5, -h - 0.5]   # axes in bins
for j, (label, res) in enumerate(picks):
    patch = bgr[res.y:res.y + S, res.x:res.x + S]
    patch = cv2.resize(patch, (P, P), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = (gray - gray.mean()) * np.outer(np.hanning(P), np.hanning(P))
    spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(win))))
    ax = fig.add_subplot(grid[1, j])
    ax.imshow(spec, cmap="gray", interpolation="nearest", extent=ext)
    for p in res.peaks:
        for s in (1, -1):
            ax.add_patch(plt.Circle((s * p.fx * S, s * p.fy * S), P / 40, fill=False, ec="red"))
    ax.set_title(f"{label} (patch {res.patch_id}), intensity {res.intensity:.2f}", fontsize=9)
    ax.set_xlabel("fx (bins)", fontsize=8)
    ax.set_ylabel("fy (bins)", fontsize=8)
    ax.tick_params(labelsize=7)

plt.tight_layout()
out = Path("docs/figures") / f"explore_real_{Path(args.image).stem}_{S}.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=120)
print(f"saved {out}")
plt.show()
