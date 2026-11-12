from pathlib import Path

import numpy as np

from moire.detection import analyze_image
from experiments.compare_single_patch import (
    load_image,
    save_image,
    apply_full_notch,
    mean_change,
)


def main():
    image_path = Path("data/samples/test_image.png")
    output_dir = Path("experiments/results/sigma_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    detections = analyze_image(image, patch_size=512)

    candidates = [result for result in detections if result.peaks]

    if not candidates:
        raise ValueError("No detected peaks found.")

    selected = max(
        candidates,
        key=lambda result: result.intensity
    )

    y, x, size = selected.y, selected.x, selected.size
    patch = image[y:y + size, x:x + size].copy()

    save_image(output_dir / "original.png", patch)

    print("Selected Patch:", selected.patch_id)
    print("Intensity:", selected.intensity)
    print()

    for sigma in (1.0, 2.0, 4.0, 8.0):
        filtered = apply_full_notch(
            patch,
            selected.peaks,
            sigma=sigma
        )

        save_image(
            output_dir / f"notch_sigma_{sigma:.0f}.png",
            filtered
        )

        print(
            f"Sigma={sigma:.1f}, "
            f"Mean change={mean_change(patch, filtered):.4f}"
        )


    # Diagnostic experiment: exclude peaks close to DC.
    min_dc_distance = 20

    high_frequency_peaks = [
        peak for peak in selected.peaks
        if np.hypot(
            peak.fy * size,
            peak.fx * size
        ) >= min_dc_distance
    ]

    print()
    print("Original peak count:", len(selected.peaks))
    print("Selected peak count:", len(high_frequency_peaks))

    if high_frequency_peaks:
        filtered = apply_full_notch(
            patch,
            high_frequency_peaks,
            sigma=4.0
        )

        save_image(
            output_dir / "notch_sigma_4_highfreq.png",
            filtered
        )

        print(
            "Sigma=4.0, high-frequency peaks only, "
            f"Mean change={mean_change(patch, filtered):.4f}"
        )


    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
