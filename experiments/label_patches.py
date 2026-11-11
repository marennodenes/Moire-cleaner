"""Paint by hand where an image has visible moiré (ground truth).

Usage (from the repo root):
    python experiments/label_patches.py data/real/github_screen.jpg

Mouse:  left drag = paint moiré,  right drag = erase
Keys:   + / - = bigger / smaller brush,  a = mark all,  c = clear all,
        g = show patch grid,  s = save,  q = save and quit

Tip: moiré usually covers most of the screen. Press "a" to mark everything,
then erase the parts without moiré (dark areas, the screen edge, ...).

The mask is saved as data/real/labels/<image>_mask.png (white = moiré) and
loaded again next time, so you can stop and continue. Because it is a mask
and not a grid, it works for any patch size and scale in evaluate_labels.py.
"""
import argparse
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

SCALE = 0.25    # the mask is painted and stored at this size


def load_image(path, scale=SCALE):
    """Colour image (RGB, 0..1), resized with INTER_AREA."""
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def mask_path(image):
    return Path("data/real/labels") / f"{Path(image).stem}_mask.png"


def load_mask(path, shape):
    if path.exists():
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape == shape:
            return mask
        print(f"{path} has another size, starting over")
    return np.zeros(shape, dtype=np.uint8)


def save_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)
    print(f"saved {path}: {100 * (mask > 0).mean():.0f}% marked as moiré")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--patch", type=int, default=128,
                        help="grid shown with 'g' (in the painted image's pixels)")
    args = parser.parse_args()

    img = load_image(args.image)
    out = mask_path(args.image)
    mask = load_mask(out, img.shape[:2])
    state = {"radius": 25, "paint": None, "grid": [], "last": None, "drawn": 0.0}

    plt.rcParams["keymap.save"] = []        # free up "s" (normally save figure)
    plt.rcParams["keymap.quit"] = []        # "q" is handled below
    plt.rcParams["keymap.grid"] = []        # "g" is handled below
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.imshow(img)
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[..., 0] = 255
    shown = ax.imshow(overlay, interpolation="nearest")
    ax.axis("off")

    def update_overlay(y0=0, y1=None, x0=0, x1=None):
        """Copy the mask into the red overlay (only the given box, for speed)."""
        overlay[y0:y1, x0:x1, 3] = (mask[y0:y1, x0:x1] > 0) * 100

    def redraw():
        shown.set_data(overlay)
        ax.set_title(f"left drag: paint   right drag: erase   +/-: brush ({state['radius']})"
                     f"   a: all   c: clear   g: grid   s: save   q: save + quit", fontsize=8)
        fig.canvas.draw_idle()
        state["drawn"] = time.perf_counter()

    def paint(event):
        if state["paint"] is None or event.inaxes is not ax or event.xdata is None:
            return
        x, y, r = int(event.xdata), int(event.ydata), state["radius"]
        px, py = state["last"] or (x, y)
        value = 255 if state["paint"] else 0
        # a thick line from the last point, so fast strokes have no gaps
        cv2.line(mask, (px, py), (x, y), value, 2 * r)
        cv2.circle(mask, (x, y), r, value, -1)
        state["last"] = (x, y)
        update_overlay(max(0, min(y, py) - r - 1), max(y, py) + r + 2,
                       max(0, min(x, px) - r - 1), max(x, px) + r + 2)
        if time.perf_counter() - state["drawn"] > 0.05:     # max ~20 redraws/s
            redraw()

    def on_press(event):
        if event.button in (1, 3):
            state["paint"] = event.button == 1
            state["last"] = None
            paint(event)

    def on_release(event):
        state["paint"] = None
        redraw()

    def on_key(event):
        if event.key in ("+", "="):
            state["radius"] += 5
        elif event.key == "-":
            state["radius"] = max(3, state["radius"] - 5)
        elif event.key in ("a", "c"):
            mask[:] = 255 if event.key == "a" else 0
            update_overlay()
        elif event.key == "g":
            if state["grid"]:
                for line in state["grid"]:
                    line.remove()
                state["grid"] = []
            else:
                P = args.patch
                state["grid"] = ([ax.axhline(y, color="white", lw=0.5)
                                  for y in range(0, mask.shape[0], P)] +
                                 [ax.axvline(x, color="white", lw=0.5)
                                  for x in range(0, mask.shape[1], P)])
        elif event.key == "s":
            save_mask(out, mask)
        elif event.key == "q":
            save_mask(out, mask)
            plt.close(fig)
            return
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", paint)
    fig.canvas.mpl_connect("key_press_event", on_key)
    update_overlay()
    redraw()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
