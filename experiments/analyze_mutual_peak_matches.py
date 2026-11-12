
from pathlib import Path

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


def find_mutual_matches(peaks_a, peaks_b, patch_size):
    """
    Find mutually nearest peak pairs.
    Return (index_a, index_b, distance_in_bins).
    """
    if not peaks_a or not peaks_b:
        return []

    distances = np.array([
        [
            calculate_peak_distance(a, b) * patch_size
            for b in peaks_b
        ]
        for a in peaks_a
    ])

    # For each peak in A, nearest peak in B.
    nearest_b = np.argmin(distances, axis=1)

    # For each peak in B, nearest peak in A.
    nearest_a = np.argmin(distances, axis=0)

    matches = []

    for i, j in enumerate(nearest_b):
        j = int(j)

        if nearest_a[j] == i:
            matches.append(
                (i, j, float(distances[i, j]))
            )

    return matches


def main():
    image = load_image(
        Path("data/samples/test_image.png")
    )

    patch_size = 512
    stride = 256

    print("Running detection...")
    detections = analyze_image(
        image,
        patch_size=patch_size
    )

    height, width = image.shape[:2]

    y_positions = get_patch_positions(
        height, patch_size, stride
    )
    x_positions = get_patch_positions(
        width, patch_size, stride
    )

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

    all_matches = []
    comparable_region_pairs = 0

    for a, b in unique_pairs.values():
        if not a.peaks or not b.peaks:
            continue

        comparable_region_pairs += 1

        matches = find_mutual_matches(
            a.peaks, b.peaks, patch_size
        )

        for i, j, distance in matches:
            all_matches.append((
                a.patch_id,
                b.patch_id,
                i,
                j,
                distance
            ))

    print("\nUnique region pairs:", len(unique_pairs))
    print("Region pairs with peaks in both:",
          comparable_region_pairs)
    print("Mutual nearest matches:", len(all_matches))

    if not all_matches:
        print("No mutual matches found.")
        return

    distances = np.array([
        match[4] for match in all_matches
    ])

    print("\nMutual match distance statistics")
    print("Minimum:", distances.min())
    print("Median:", np.median(distances))
    print("Maximum:", distances.max())

    print("\nCandidate matching thresholds")

    for threshold in (1, 2, 3, 4):
        count = int(np.sum(distances <= threshold + 1e-9))

        print(
            f"<= {threshold} bins: "
            f"{count} / {len(all_matches)}"
        )

    print("\nMatches between 1 and 2 bins")

    for id_a, id_b, i, j, distance in all_matches:
        if 1 < distance <= 2 + 1e-9:
            print(
                f"Detection {id_a} Peak {i} "
                f"<-> Detection {id_b} Peak {j}: "
                f"{distance:.4f} bins"
            )


if __name__ == "__main__":
    main()
