
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from moire.detection import analyze_image
from moire.adaptive import (
    get_detection_overlaps,
    calculate_peak_distance,
)
from moire.patch_processing import get_patch_positions


def load_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Cannot load: {path}")

    return image


def main():
    image = load_image(Path("data/samples/test_image.png"))

    patch_size = 512
    stride = 256

    print("Running detection...")
    detections = analyze_image(image, patch_size=patch_size)

    height, width = image.shape[:2]

    y_positions = get_patch_positions(
        height, patch_size, stride
    )
    x_positions = get_patch_positions(
        width, patch_size, stride
    )

    # Deduplicate overlapping detection-region pairs.
    unique_pairs = {}

    for y in y_positions:
        for x in x_positions:
            overlaps = get_detection_overlaps(
                y, x, patch_size, detections
            )

            for i in range(len(overlaps)):
                for j in range(i + 1, len(overlaps)):
                    a = overlaps[i][0]
                    b = overlaps[j][0]

                    key = tuple(sorted(
                        (a.patch_id, b.patch_id)
                    ))

                    unique_pairs[key] = (a, b)

    print("Unique region pairs:", len(unique_pairs))

    distances = []

    for a, b in unique_pairs.values():
        if not a.peaks or not b.peaks:
            continue

        # A -> B
        for peak_a in a.peaks:
            nearest = min(
                calculate_peak_distance(peak_a, peak_b)
                for peak_b in b.peaks
            )
            distances.append(nearest * patch_size)

        # B -> A
        for peak_b in b.peaks:
            nearest = min(
                calculate_peak_distance(peak_b, peak_a)
                for peak_a in a.peaks
            )
            distances.append(nearest * patch_size)

    if not distances:
        print("No comparable peaks found.")
        return

    distances = np.asarray(distances)

    print("Bidirectional nearest comparisons:", len(distances))

    print("\nStatistics (FFT bins)")
    print("Minimum:", distances.min())
    print("Median:", np.median(distances))
    print("Mean:", distances.mean())
    print("Maximum:", distances.max())

    print("\nDistance distribution")

    bins = [0, 1, 2, 4, 8, 16, np.inf]
    counts, _ = np.histogram(distances, bins=bins)

    labels = [
        "0-1",
        "1-2",
        "2-4",
        "4-8",
        "8-16",
        "16+"
    ]

    for label, count in zip(labels, counts):
        print(f"{label} bins: {count}")

    print("\nCandidate threshold coverage")

    for threshold in (1, 2, 3, 4):
        count = int(np.sum(distances <= threshold))
        ratio = 100 * count / len(distances)

        print(
            f"<= {threshold} bins: "
            f"{count} ({ratio:.1f}%)"
        )


if __name__ == "__main__":
    main()
