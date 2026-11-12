
from pathlib import Path

import cv2
import numpy as np

from moire.detection import analyze_image
from moire.adaptive import (
    get_detection_overlaps,
    calculate_peak_distance,
)
from moire.patch_processing import divide_into_patches


def load_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Cannot load image: {path}")

    return image


def main():
    image = load_image(
        Path("data/samples/test_image.png")
    )

    patch_size = 512
    stride = 256

    detections = analyze_image(
        image,
        patch_size=patch_size
    )

    patches = divide_into_patches(
        image,
        patch_size,
        stride
    )

    nearest_distances = []

    for patch, y, x in patches:
        overlaps = get_detection_overlaps(
            y, x, patch_size, detections
        )

        # Compare peaks only between different
        # overlapping detection regions.
        for i in range(len(overlaps)):
            region_a = overlaps[i][0]

            if not region_a.peaks:
                continue

            for j in range(i + 1, len(overlaps)):
                region_b = overlaps[j][0]

                if not region_b.peaks:
                    continue

                for peak_a in region_a.peaks:
                    nearest_distance = min(
                        calculate_peak_distance(
                            peak_a,
                            peak_b
                        )
                        for peak_b in region_b.peaks
                    )

                    # Convert cycles/pixel to FFT bins.
                    nearest_distances.append(
                        nearest_distance * patch_size
                    )

    if not nearest_distances:
        print("No comparable peak pairs found.")
        return

    distances = np.array(nearest_distances)

    print("Total peak comparisons:", len(distances))
    print("Minimum distance:", distances.min())
    print("Median distance:", np.median(distances))
    print("Mean distance:", distances.mean())
    print("Maximum distance:", distances.max())

    print("\nDistance distribution (FFT bins)")

    ranges = [
        (0, 1),
        (1, 2),
        (2, 4),
        (4, 8),
        (8, 16),
    ]

    for lower, upper in ranges:
        count = np.sum(
            (distances >= lower)
            & (distances < upper)
        )

        print(
            f"{lower:2d} - {upper:2d} bins: {count}"
        )

    print("16+ bins:", np.sum(distances >= 16))


if __name__ == "__main__":
    main()
