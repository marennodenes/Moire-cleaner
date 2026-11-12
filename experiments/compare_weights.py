from pathlib import Path

import numpy as np

from moire.detection import analyze_image
from moire.patch_processing import (
    divide_into_patches,
    merge_patches,
)
from moire.adaptive import (
    find_best_detection,
    filter_patch_with_detection,
)
from experiments.compare_single_patch import (
    load_image,
    save_image,
)


def main():
    image_path = Path("data/samples/test_image.png")
    output_dir = Path("experiments/results/weight_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)

    patch_size = 512
    stride = 256
    sigma = 2.0
    threshold = 0.2

    # Detect moire only once.
    print("Running detection...")
    detections = analyze_image(image, patch_size=patch_size)

    # Divide image into overlapping patches.
    patches = divide_into_patches(
        image, patch_size, stride
    )

    # Filter each patch only once.
    filtered_patches = []
    active_count = 0

    print("Filtering patches...")

    for patch, y, x in patches:
        detection = find_best_detection(
            y, x, patch_size, detections
        )

        if (
            detection is not None
            and detection.peaks
            and detection.intensity >= threshold
        ):
            active_count += 1

        filtered = filter_patch_with_detection(
            patch,
            detection,
            sigma=sigma,
            threshold=threshold
        )

        filtered_patches.append((filtered, y, x))

    print("Total patches:", len(filtered_patches))
    print("Active patches:", active_count)
    # Merge the same filtered patches using three methods.
    results = {}

    for mode in ("uniform", "triangular", "hann"):
        print(f"Running {mode} merging...")

        merged = merge_patches(
            filtered_patches,
            image.shape,
            weight_mode=mode
        )

        results[mode] = merged

        save_image(
            output_dir / f"{mode}.png",
            merged
        )

    # Compare results without uint8 rounding.
    pairs = [
        ("uniform", "triangular"),
        ("uniform", "hann"),
        ("triangular", "hann"),
    ]

    print("\nWeight Comparison Results")

    for mode_a, mode_b in pairs:
        difference = np.abs(
            results[mode_a] - results[mode_b]
        )

        mean_diff = difference.mean()
        max_diff = difference.max()

        print(f"\n{mode_a} vs {mode_b}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  Maximum difference: {max_diff:.6f}")

        # Amplify differences for visualization.
        amplified = np.clip(
            difference * 20, 0, 255
        ).astype(np.uint8)

        save_image(
            output_dir / f"diff_{mode_a}_{mode_b}.png",
            amplified
        )

    print("\nResults saved to:", output_dir)

if __name__ == "__main__":
    main()