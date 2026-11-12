from pathlib import Path

import cv2
import numpy as np

from moire.patch_processing import get_patch_positions


def load_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Cannot load: {path}")

    return image


def get_seam_positions(length, patch_size, stride):
    starts = get_patch_positions(length, patch_size, stride)

    # Include the start and end of every patch.
    positions = set()

    for start in starts:
        positions.add(start)
        positions.add(start + patch_size)

    # Exclude image boundaries.
    return sorted(p for p in positions if 0 < p < length)


def evaluate_seams(original, processed, patch_size, stride):
    if original.shape != processed.shape:
        raise ValueError("Image shapes must match.")

    height, width = original.shape[:2]

    # Changes introduced by filtering and merging.
    change = (
        processed.astype(np.float32)
        - original.astype(np.float32)
    )

    seam_scores = []
    nearby_scores = []

    # Vertical seams.
    for x in get_seam_positions(width, patch_size, stride):
        seam = np.abs(change[:, x] - change[:, x - 1]).mean()
        seam_scores.append(seam)

        for offset in (-8, 8):
            p = x + offset
            if 0 < p < width:
                nearby = np.abs(
                    change[:, p] - change[:, p - 1]
                ).mean()
                nearby_scores.append(nearby)

    # Horizontal seams.
    for y in get_seam_positions(height, patch_size, stride):
        seam = np.abs(change[y] - change[y - 1]).mean()
        seam_scores.append(seam)

        for offset in (-8, 8):
            p = y + offset
            if 0 < p < height:
                nearby = np.abs(
                    change[p] - change[p - 1]
                ).mean()
                nearby_scores.append(nearby)

    return (
        float(np.mean(seam_scores)),
        float(np.mean(nearby_scores))
    )


def main():
    original = load_image(Path("data/samples/test_image.png"))
    result_dir = Path("experiments/results/weight_comparison")

    for mode in ("uniform", "triangular", "hann"):
        processed = load_image(result_dir / f"{mode}.png")

        seam, nearby = evaluate_seams(
            original,
            processed,
            patch_size=512,
            stride=256
        )

        print(f"\n{mode}")
        print(f"  Seam change: {seam:.6f}")
        print(f"  Nearby change: {nearby:.6f}")
        print(f"  Seam excess: {seam - nearby:.6f}")


if __name__ == "__main__":
    main()
