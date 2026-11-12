
from pathlib import Path

import numpy as np

from moire.detection import analyze_image
from moire.patch_processing import divide_into_patches
from moire.adaptive import (
    find_best_detection,
    calculate_weighted_intensity,
)


def load_image(path):
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Cannot load: {path}")

    return image


def main():
    image = load_image(
        Path("data/samples/test_image.png")
    )

    patch_size = 512
    stride = 256
    threshold = 0.2

    detections = analyze_image(
        image,
        patch_size=patch_size
    )

    patches = divide_into_patches(
        image,
        patch_size,
        stride
    )

    records = []

    for patch, y, x in patches:
        detection = find_best_detection(
            y, x, patch_size, detections
        )

        if detection is None:
            continue

        best_intensity = detection.intensity

        weighted_intensity = calculate_weighted_intensity(
            y, x, patch_size, detections
        )

        records.append({
            "y": y,
            "x": x,
            "best": best_intensity,
            "weighted": weighted_intensity,
            "difference": weighted_intensity - best_intensity,
            "has_peaks": len(detection.peaks) > 0
        })

    increased = sum(
        r["difference"] > 1e-6 for r in records
    )

    decreased = sum(
        r["difference"] < -1e-6 for r in records
    )

    unchanged = len(records) - increased - decreased

    print("Total mapped patches:", len(records))
    print("Increased intensity:", increased)
    print("Decreased intensity:", decreased)
    print("Unchanged intensity:", unchanged)

    differences = np.array([
        r["difference"] for r in records
    ])

    print("\nIntensity difference statistics")
    print("Mean:", differences.mean())
    print("Mean absolute:", np.abs(differences).mean())
    print("Minimum:", differences.min())
    print("Maximum:", differences.max())

    # Count patches that actually satisfy the filter condition.
    best_active = sum(
        r["best"] >= threshold and r["has_peaks"]
        for r in records
    )

    weighted_active = sum(
        r["weighted"] >= threshold and r["has_peaks"]
        for r in records
    )

    print("\nFiltering activation")
    print("Best active patches:", best_active)
    print("Weighted active patches:", weighted_active)


    print("\nActivation changes")

    best_only = []
    weighted_only = []

    for r in records:
        best_is_active = (
            r["best"] >= threshold and r["has_peaks"]
        )

        weighted_is_active = (
            r["weighted"] >= threshold and r["has_peaks"]
        )

        if best_is_active and not weighted_is_active:
            best_only.append(r)

        elif weighted_is_active and not best_is_active:
            weighted_only.append(r)

    print("Best only:", len(best_only))
    print("Weighted only:", len(weighted_only))

    print("\nBest-only patches:")
    for r in best_only:
        print(
            f"({r['y']}, {r['x']}) "
            f"best={r['best']:.4f}, "
            f"weighted={r['weighted']:.4f}"
        )

    print("\nWeighted-only patches:")
    for r in weighted_only:
        print(
            f"({r['y']}, {r['x']}) "
            f"best={r['best']:.4f}, "
            f"weighted={r['weighted']:.4f}"
        )

    # Find the largest intensity changes.
    records.sort(
        key=lambda r: abs(r["difference"]),
        reverse=True
    )

    print("\nTop 10 intensity differences")

    for r in records[:10]:
        print(
            f"({r['y']}, {r['x']}) "
            f"best={r['best']:.4f}, "
            f"weighted={r['weighted']:.4f}, "
            f"diff={r['difference']:+.4f}, "
            f"peaks={r['has_peaks']}"
        )


if __name__ == "__main__":
    main()
