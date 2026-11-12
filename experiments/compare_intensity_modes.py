
from pathlib import Path

import numpy as np

from moire.detection import analyze_image
from moire.adaptive import process_overlapping_patches

from experiments.compare_single_patch import (
    load_image,
    save_image,
)
from experiments.evaluate_seams import evaluate_seams


def main():
    image_path = Path("data/samples/test_image.png")
    output_dir = Path("experiments/results/intensity_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)

    patch_size = 512
    stride = 256
    sigma = 2.0
    threshold = 0.2

    # Run detection only once.
    print("Running detection...")
    detections = analyze_image(
        image,
        patch_size=patch_size
    )

    results = {}

    for mode in ("best", "weighted"):
        print(f"\nRunning {mode} filtering...")

        processed = process_overlapping_patches(
            image,
            detections,
            patch_size=patch_size,
            stride=stride,
            sigma=sigma,
            threshold=threshold,
            weight_mode="triangular",
            intensity_mode=mode
        )

        results[mode] = processed

        # Save the result.
        save_image(
            output_dir / f"{mode}.png",
            processed
        )

        # Quantify changes from the original.
        change = np.abs(
            processed - image.astype(np.float64)
        )

        print("Mean change:", change.mean())
        print("Maximum change:", change.max())

        # Evaluate patch boundary discontinuities
        # using the same uint8 representation as saved.
        processed_uint8 = np.clip(
            np.rint(processed), 0, 255
        ).astype(np.uint8)

        seam, nearby = evaluate_seams(
            image,
            processed_uint8,
            patch_size,
            stride
        )

        print("Seam change:", seam)
        print("Nearby change:", nearby)
        print("Seam excess:", seam - nearby)

    # Compare best vs weighted directly.
    difference = np.abs(
        results["best"] - results["weighted"]
    )

    print("\nBest vs Weighted")
    print("Mean difference:", difference.mean())
    print("Maximum difference:", difference.max())

    amplified = np.clip(
        difference * 20, 0, 255
    ).astype(np.uint8)

    save_image(
        output_dir / "difference.png",
        amplified
    )

    print("\nResults saved to:", output_dir)


if __name__ == "__main__":
    main()
