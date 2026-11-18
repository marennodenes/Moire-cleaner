
import csv
from pathlib import Path

import numpy as np

from moire.detection import analyze_image
from moire.adaptive import (
    find_best_detection,
    get_detection_overlaps,
    merge_local_detection_peaks,
    calculate_peak_distance,
)
from moire.patch_processing import (
    get_patch_positions,
    decide_filter_strength,
)

from experiments.evaluate_synthetic_moire import (
    create_clean_image,
)
from experiments.evaluate_multiple_moire_conditions import (
    create_scenarios,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "peak_mapping_analysis"
)


def count_new_frequencies(
    best_peaks,
    merged_peaks,
    patch_size=512,
    max_distance_bins=1.0
):
    """
    Count merged peaks that have no nearby
    equivalent in the best peak list.
    """
    count = 0

    for merged_peak in merged_peaks:
        found = any(
            calculate_peak_distance(
                merged_peak, best_peak
            ) * patch_size <= max_distance_bins + 1e-9
            for best_peak in best_peaks
        )

        if not found:
            count += 1

    return count


def analyze_peak_mapping(
    image,
    detection_results,
    patch_size=512,
    stride=256,
    threshold=0.2
):
    """
    Inspect best and merged peak selection
    for each filtering patch.

    Does not perform FFT or modify input images.
    """
    height, width = image.shape[:2]

    y_positions = get_patch_positions(
        height, patch_size, stride
    )
    x_positions = get_patch_positions(
        width, patch_size, stride
    )

    records = []

    for y in y_positions:
        for x in x_positions:
            best_detection = find_best_detection(
                y, x, patch_size, detection_results
            )

            overlaps = get_detection_overlaps(
                y, x, patch_size, detection_results
            )

            local_detections = [
                result for result, area, ratio in overlaps
            ]

            if best_detection is None:
                best_peaks = []
                merged_peaks = []
                strength = 0.0
            else:
                best_peaks = best_detection.peaks

                merged_peaks = merge_local_detection_peaks(
                    local_detections,
                    patch_size=patch_size,
                    max_distance_bins=1.0
                )

                strength = decide_filter_strength(
                    best_detection.intensity,
                    threshold
                )

            new_count = count_new_frequencies(
                best_peaks,
                merged_peaks,
                patch_size=patch_size
            )

            records.append({
                "patch_y": y,
                "patch_x": x,
                "overlap_regions": len(local_detections),
                "best_peak_count": len(best_peaks),
                "merged_peak_count": len(merged_peaks),
                "new_frequency_count": new_count,
                "best_active": int(
                    strength > 0 and len(best_peaks) > 0
                ),
                "merged_active": int(
                    strength > 0 and len(merged_peaks) > 0
                ),
                "crosses_center": int(
                    x < width // 2 < x + patch_size
                ),
            })

    return records


def summarize_mapping(records):
    """Summarize peak selection across patches."""
    return {
        "total_patches": len(records),
        "best_active_patches": sum(
            r["best_active"] for r in records
        ),
        "merged_active_patches": sum(
            r["merged_active"] for r in records
        ),
        "total_best_peaks": sum(
            r["best_peak_count"] for r in records
        ),
        "total_merged_peaks": sum(
            r["merged_peak_count"] for r in records
        ),
        "total_new_frequencies": sum(
            r["new_frequency_count"] for r in records
        ),
        "patches_with_new_frequencies": sum(
            r["new_frequency_count"] > 0
            for r in records
        ),
    }


def save_csv(path, records):
    """Save per-patch analysis to a CSV file."""
    if not records:
        raise ValueError("No records to save.")

    with open(
        path, "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )

        writer.writeheader()
        writer.writerows(records)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean = create_clean_image(size=1024)
    scenarios = create_scenarios(clean)

    for name, image in scenarios.items():
        print(f"\n=== {name} ===")

        detections = analyze_image(
            image, patch_size=512
        )

        records = analyze_peak_mapping(
            image,
            detections,
            patch_size=512,
            stride=256,
            threshold=0.2
        )

        summary = summarize_mapping(records)

        for key, value in summary.items():
            print(f"{key}: {value}")

        csv_path = OUTPUT_DIR / f"{name}.csv"
        save_csv(csv_path, records)

        if name == "C_spatially_varying":
            print("\n--- Center Boundary Patches ---")

            for record in records:
                if record["crosses_center"]:
                    print(record)

    print("\nAnalysis completed.")
    print("Saved CSV files to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
