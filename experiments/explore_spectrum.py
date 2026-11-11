import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def spectrum(img, window=False):
    """Log-magnitude of the centred spectrum, optionally with a 2D Hann window."""
    if window:
        h, w = img.shape
        img = img * np.outer(np.hanning(h), np.hanning(w))
    F = np.fft.fftshift(np.fft.fft2(img))
    return np.log1p(np.abs(F))


def add_wave(img, fy, fx, amp=0.15):
    """Add a periodic pattern that should give peaks at (fy, fx) and (-fy, -fx)."""
    H, W = img.shape
    y, x = np.mgrid[0:H, 0:W]
    return img + amp * np.cos(2 * np.pi * (fy * y / H + fx * x / W))


def strongest_peak(img, dc_radius=5):
    """Very first detector: position of the brightest point outside the centre."""
    mag = spectrum(img, window=True)
    cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
    mag[cy - dc_radius:cy + dc_radius + 1, cx - dc_radius:cx + dc_radius + 1] = 0
    iy, ix = np.unravel_index(np.argmax(mag), mag.shape)
    return iy - cy, ix - cx


# ---------- load image ----------
path = sys.argv[1] if len(sys.argv) > 1 else "data/samples/cameraman.png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
if img is None:
    sys.exit(f"Could not read image: {path}")
clean = img.astype(np.float32)[:256, :256] / 255     # centre at (128, 128)

# ---------- test images ----------
one = add_wave(clean, 20, 35)
two = add_wave(one, -15, 50, amp=0.10)
frac = add_wave(clean, 20.5, 35.5)           # test 3: non-integer frequency
weak = add_wave(clean, 20, 35, amp=0.02)     # test 4: weak noise

rows = [
    ("clean", clean),
    ("1 wave (20, 35)", one),
    ("2 waves", two),
    ("non-integer (20.5, 35.5)", frac),
    ("weak amp=0.02", weak),
]

# ---------- print strongest peak for each image ----------
for name, im in rows:
    print(f"{name:28s} strongest peak: {strongest_peak(im)}")

# ---------- plot ----------
# One column per test image, three rows: image / spectrum / spectrum + Hann.
# Wide layout so everything fits on a laptop screen.
ZOOM = 64                                   # show frequencies -ZOOM..ZOOM
c = 128                                     # centre of the 256x256 spectrum
ext = [-ZOOM - 0.5, ZOOM + 0.5, ZOOM + 0.5, -ZOOM - 0.5]   # axes in frequency units

fig, ax = plt.subplots(3, len(rows), figsize=(2.6 * len(rows), 7.5))
for j, (name, im) in enumerate(rows):
    ax[0, j].imshow(im, cmap="gray")
    ax[0, j].set_title(name, fontsize=9)
    ax[0, j].axis("off")
    for i, win in ((1, False), (2, True)):
        S = spectrum(im, win)[c - ZOOM:c + ZOOM + 1, c - ZOOM:c + ZOOM + 1]
        ax[i, j].imshow(S, cmap="gray", interpolation="nearest", extent=ext)
        ax[i, j].tick_params(labelsize=7)
ax[1, 0].set_ylabel("spectrum\nfy")
ax[2, 0].set_ylabel("spectrum + Hann\nfy")
for a in ax[2]:
    a.set_xlabel("fx", fontsize=8)
plt.tight_layout()
plt.show()