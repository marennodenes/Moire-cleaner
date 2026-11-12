
from pathlib import Path

import cv2
import numpy as np

from moire.detection import analyze_image
from moire.adaptive import (
    find_best_detection,
    get_detection_overlaps,
    calculate_weighted_intensity,
)


def load_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Cannot load image: {path}")

    return image


def inspect_patch(y, x, detections, patch_size=512):
    print(f"\n{'=' * 60}")
    print(f"Filtering Patch: (y={y}, x={x})")

    best = find_best_detection(
        y, x, patch_size, detections
    )

    weighted = calculate_weighted_intensity(
        y, x, patch_size, detections
    )

    overlaps = get_detection_overlaps(
        y, x, patch_size, detections
    )

    if best is None:
        print("No detection matched.")
        return

    print(f"Best Detection ID: {best.patch_id}")
    print(f"Best Intensity: {best.intensity:.4f}")
    print(f"Weighted Intensity: {weighted:.4f}")
    print(f"Best Peak Count: {len(best.peaks)}")

    print("\nOverlapping Detection Regions:")

    for result, area, ratio in sorted(
        overlaps,
        key=lambda item: item[1],
        reverse=True
    ):
        selected = (
            " [SELECTED]"
            if result.patch_id == best.patch_id
            else ""
        )

        print(
            f"\n  ID={result.patch_id}{selected}"
            f"  Position=({result.y}, {result.x})"
        )
        print(
            f"  Overlap={ratio:.3f}"
            f"  Intensity={result.intensity:.4f}"
            f"  Peaks={len(result.peaks)}"
        )

        for peak in result.peaks:
            print(
                f"    fy={peak.fy:+.6f}"
                f"  fx={peak.fx:+.6f}"
                f"  ratio={peak.ratio:.3f}"
            )


def main():
    image = load_image(
        Path("data/samples/test_image.png")
    )

    print("Running detection...")
    detections = analyze_image(
        image,
        patch_size=512
    )

    target_patches = [
        (2304, 2048),
        (1792, 1536),
    ]

    for y, x in target_patches:
        inspect_patch(y, x, detections)


if __name__ == "__main__":
    main()
